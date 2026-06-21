from __future__ import annotations

import argparse
import csv
import sqlite3
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


RANK_TO_COLUMN = {
    "kingdom": "kingdom",
    "phylum": "phylum_division",
    "class": "class",
    "order": "tax_order",
    "family": "family",
    "genus": "genus",
    "species": "species",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export or fill missing species taxonomy from NCBI Taxonomy.")
    parser.add_argument("--db", required=True, type=Path, help="SQLite database containing the species table.")
    parser.add_argument("--output", required=True, type=Path, help="CSV audit output.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--email", default="", help="Optional contact email for NCBI E-utilities.")
    parser.add_argument("--sleep", type=float, default=0.34, help="Delay between NCBI requests.")
    parser.add_argument(
        "--apply-missing",
        action="store_true",
        help="Fill only blank taxonomy fields in species from successful NCBI lookups.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as conn, args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        conn.row_factory = sqlite3.Row
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "species_number",
                "latin_name",
                "ncbi_taxid",
                "lookup_taxid",
                "lookup_status",
                "kingdom",
                "phylum_division",
                "class",
                "tax_order",
                "family",
                "genus",
                "species",
            ],
        )
        writer.writeheader()
        for row in load_species_with_gaps(conn, limit=args.limit):
            result = lookup_taxonomy(row["ncbi_taxid"], row["latin_name"], email=args.email)
            output = {
                "species_number": row["species_number"],
                "latin_name": row["latin_name"],
                "ncbi_taxid": row["ncbi_taxid"],
                **result,
            }
            writer.writerow(output)
            if args.apply_missing and result["lookup_status"] == "ok":
                apply_missing_taxonomy(conn, row, result)
            time.sleep(max(float(args.sleep), 0.0))
        if args.apply_missing:
            conn.commit()


def load_species_with_gaps(conn: sqlite3.Connection, *, limit: int | None) -> list[sqlite3.Row]:
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    return conn.execute(
        f"""
        SELECT species_number, latin_name, ncbi_taxid,
               kingdom, phylum_division, class, tax_order, family, genus, species
        FROM species
        WHERE latin_name IS NOT NULL
          AND TRIM(latin_name) <> ''
          AND (
              phylum_division IS NULL OR TRIM(phylum_division) = ''
              OR class IS NULL OR TRIM(class) = ''
              OR tax_order IS NULL OR TRIM(tax_order) = ''
              OR family IS NULL OR TRIM(family) = ''
              OR genus IS NULL OR TRIM(genus) = ''
          )
        ORDER BY species_number
        {limit_clause}
        """
    ).fetchall()


def lookup_taxonomy(ncbi_taxid: object, latin_name: object, *, email: str) -> dict[str, object]:
    taxid = clean_text(ncbi_taxid)
    if not taxid:
        taxid = search_taxid(clean_text(latin_name), email=email)
    if not taxid:
        return empty_result("not_found")
    try:
        return fetch_taxonomy(taxid, email=email)
    except Exception as exc:  # pragma: no cover - network failure path
        result = empty_result(f"error:{exc.__class__.__name__}")
        result["lookup_taxid"] = taxid
        return result


def search_taxid(latin_name: str, *, email: str) -> str:
    if not latin_name:
        return ""
    params = {
        "db": "taxonomy",
        "term": f"{latin_name}[Scientific Name]",
        "retmode": "json",
    }
    if email:
        params["email"] = email
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
    text = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
    import json

    ids = json.loads(text).get("esearchresult", {}).get("idlist", [])
    return str(ids[0]) if ids else ""


def fetch_taxonomy(taxid: str, *, email: str) -> dict[str, object]:
    params = {"db": "taxonomy", "id": taxid, "retmode": "xml"}
    if email:
        params["email"] = email
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(params)
    text = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
    root = ET.fromstring(text)
    taxon = root.find(".//Taxon")
    if taxon is None:
        return empty_result("not_found")
    result = empty_result("ok")
    result["lookup_taxid"] = taxid
    lineage = taxon.findall(".//LineageEx/Taxon")
    for item in lineage:
        rank = clean_text(item.findtext("Rank")).lower()
        column = RANK_TO_COLUMN.get(rank)
        if column:
            result[column] = clean_text(item.findtext("ScientificName"))
    own_rank = clean_text(taxon.findtext("Rank")).lower()
    own_column = RANK_TO_COLUMN.get(own_rank)
    if own_column:
        result[own_column] = clean_text(taxon.findtext("ScientificName"))
    return result


def empty_result(status: str) -> dict[str, object]:
    return {
        "lookup_taxid": "",
        "lookup_status": status,
        "kingdom": "",
        "phylum_division": "",
        "class": "",
        "tax_order": "",
        "family": "",
        "genus": "",
        "species": "",
    }


def apply_missing_taxonomy(conn: sqlite3.Connection, row: sqlite3.Row, result: dict[str, object]) -> None:
    updates = {}
    for column in RANK_TO_COLUMN.values():
        current = clean_text(row[column])
        replacement = clean_text(result.get(column))
        if not current and replacement:
            updates[column] = replacement
    if not clean_text(row["ncbi_taxid"]) and clean_text(result.get("lookup_taxid")):
        updates["ncbi_taxid"] = clean_text(result.get("lookup_taxid"))
    if not updates:
        return
    assignments = ", ".join(f'"{column}" = ?' for column in updates)
    conn.execute(
        f'UPDATE species SET {assignments} WHERE species_number = ?',
        [*updates.values(), row["species_number"]],
    )


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
