from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager
from matplotlib.lines import Line2D
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


RDLogger.DisableLog("rdApp.warning")

DEFAULT_CHEMICAL_SPACE = Path("实验汇总/09_主线训练空间与作图数据/task_train_chemical_space.csv")
DEFAULT_SPECIES_EMBEDDING = Path("实验汇总/09_主线训练空间与作图数据/species_embedding_lookup.csv.gz")
DEFAULT_TASK_SPECIES_SPACE = Path("实验汇总/09_主线训练空间与作图数据/task_train_species_embedding_space.csv.gz")
DEFAULT_OUT_DIR = Path("outputs/paper_figures/fig_model_results_integrated_20260709/training_space")
DEFAULT_PREFIX = "compound_species_embedding_tsne_starry"
DEFAULT_PALETTE = Path(
    "outputs/paper_figures/fig_model_results_integrated_20260709/performance/figure_palette_starry_night_v1.yaml"
)

RANDOM_SEED = 20260713
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
TSNE_MAX_ITER = 1200

STARRY_DEFAULTS = {
    "midnight_navy": "#182D4F",
    "ultramarine": "#264A8A",
    "cobalt_blue": "#2F6DB3",
    "swirl_cyan": "#6AAED6",
    "star_yellow": "#F2C94C",
    "moon_gold": "#D8A528",
    "warm_orange": "#D8892B",
    "cypress_green": "#315B4B",
    "lavender_shadow": "#7C6FA6",
    "night_gray": "#7B8491",
    "ivory": "#F7F1D0",
    "paper": "#FFFFFF",
    "grid": "#DDE3EA",
    "charcoal": "#20262B",
}

BACKGROUND = STARRY_DEFAULTS["paper"]
GRID = STARRY_DEFAULTS["grid"]
TEXT = STARRY_DEFAULTS["charcoal"]
MUTED = STARRY_DEFAULTS["night_gray"]

FIG_TITLE_SIZE = 11.0
PANEL_TITLE_SIZE = 9.8
AXIS_LABEL_SIZE = 7.8
TICK_LABEL_SIZE = 7.0
LEGEND_TEXT_SIZE = 6.6
NOTE_TEXT_SIZE = 6.4

CHEMICAL_CLASS_COLORS = {
    "Metals/inorganics": "#315B4B",
    "PFAS/highly fluorinated": "#F2C94C",
    "Chlorinated organics": "#2F6DB3",
    "Other halogenated organics": "#6AAED6",
    "Organophosphorus": "#D8892B",
    "Organosulfur": "#7C6FA6",
    "PAHs/aromatic hydrocarbons": "#182D4F",
    "Phenols/anilines/bisphenols": "#D8A528",
    "Carboxylic/ester/amide organics": "#8FB9A8",
    "Nitrogen-containing organics": "#264A8A",
    "Hydrocarbons/solvents": "#7B8491",
    "Oxygenated organics": "#A9CFE5",
    "Other organics": "#B8A6C9",
    "Invalid/no structure": "#D8DDE6",
}

SPECIES_GROUP_COLORS = {
    "fish": "#2F6DB3",
    "crustacean": "#6AAED6",
    "insect": "#D8892B",
    "vascular_plant": "#315B4B",
    "algae": "#F2C94C",
    "mollusk": "#7C6FA6",
    "worm": "#D8A528",
    "amphibian": "#264A8A",
    "fungi": "#8FB9A8",
    "cyanobacteria": "#A9CFE5",
    "unclassified": "#7B8491",
    "Other": "#B8A6C9",
}


@dataclass(frozen=True)
class ChemicalClassResult:
    chemical_class: str
    rule_id: str
    rule_basis_zh: str
    evidence_flags: str
    canonical_smiles: str
    valid_structure: bool
    c_count: int
    n_count: int
    o_count: int
    f_count: int
    cl_count: int
    br_count: int
    i_count: int
    p_count: int
    s_count: int
    metal_count: int
    aromatic_ring_count: int


METALS = {
    "Li",
    "Na",
    "K",
    "Rb",
    "Cs",
    "Fr",
    "Be",
    "Mg",
    "Ca",
    "Sr",
    "Ba",
    "Ra",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "Al",
    "Si",
}

SMARTS = {
    "perfluoro_chain": Chem.MolFromSmarts("[CX4](F)(F)[CX4](F)(F)"),
    "carboxylic_acid": Chem.MolFromSmarts("[CX3](=O)[OX1H0-,OX2H1]"),
    "ester": Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]"),
    "amide": Chem.MolFromSmarts("[NX3][CX3](=O)[#6]"),
    "phenol": Chem.MolFromSmarts("c[OX2H]"),
    "aniline": Chem.MolFromSmarts("c[NX3;H2,H1,H0]"),
}

CLASS_RULEBOOK = [
    (
        "R00",
        "Invalid/no structure",
        "SMILES 缺失或 RDKit 无法解析；不参与 Morgan fingerprint t-SNE。",
    ),
    (
        "R01",
        "Metals/inorganics",
        "含金属/类金属元素，或无碳结构；优先于卤代有机物规则，避免将金属氯化物误判为有机氯。",
    ),
    (
        "R02",
        "PFAS/highly fluorinated",
        "含多氟碳链或名称含 perfluoro/PFAS/PFOA/PFOS 等线索，且 F 原子数较高。",
    ),
    (
        "R03",
        "Chlorinated organics",
        "含碳且含 Cl；PFAS 和金属/无机规则已优先排除。",
    ),
    (
        "R04",
        "Other halogenated organics",
        "含碳且含 Br/I/F，但不满足 PFAS 或有机氯规则。",
    ),
    ("R05", "Organophosphorus", "含碳且含 P，通常覆盖有机磷农药、磷酸酯或膦酸酯。"),
    ("R06", "Organosulfur", "含碳且含 S，且未被 PFAS/卤代/有机磷等高优先级规则捕获。"),
    (
        "R07",
        "PAHs/aromatic hydrocarbons",
        "含两个及以上芳香环，且主要由 C/H 构成，不含常见杂原子或卤素。",
    ),
    (
        "R08",
        "Phenols/anilines/bisphenols",
        "含酚羟基、苯胺结构，或化学名提示 phenol/bisphenol/aniline。",
    ),
    (
        "R09",
        "Carboxylic/ester/amide organics",
        "含羧酸、酯或酰胺结构，且未被前序更特异规则捕获。",
    ),
    ("R10", "Nitrogen-containing organics", "含碳且含 N，且未被前序更特异规则捕获。"),
    ("R11", "Hydrocarbons/solvents", "只含 C/H 或以烃类/非卤代溶剂结构为主。"),
    ("R12", "Oxygenated organics", "含碳且含 O，未被前序规则捕获。"),
    ("R99", "Other organics", "含碳但未匹配上述规则。"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compound t-SNE and species embedding-space overview figures.")
    parser.add_argument("--chemical-space", default=str(DEFAULT_CHEMICAL_SPACE), help="Task-level training chemical space CSV.")
    parser.add_argument("--species-embedding", default=str(DEFAULT_SPECIES_EMBEDDING), help="Species embedding lookup CSV/CSV.GZ.")
    parser.add_argument(
        "--task-species-space",
        default=str(DEFAULT_TASK_SPECIES_SPACE),
        help="Task-level species training coverage CSV/CSV.GZ.",
    )
    parser.add_argument("--palette", default=str(DEFAULT_PALETTE), help="Starry Night palette YAML.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Output file prefix.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG DPI.")
    parser.add_argument("--max-compounds", type=int, default=0, help="Optional cap for debugging; 0 uses all valid compounds.")
    parser.add_argument("--max-species", type=int, default=0, help="Optional cap for debugging; 0 uses all species.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = load_starry_palette(Path(args.palette))
    apply_palette(palette)
    configure_matplotlib()

    chemical_space = pd.read_csv(args.chemical_space)
    species_embedding = pd.read_csv(args.species_embedding)
    task_species_space = pd.read_csv(args.task_species_space)

    compounds = build_compound_table(chemical_space)
    compounds = classify_compounds(compounds)
    compound_embedding = build_compound_tsne(compounds, max_compounds=args.max_compounds)

    species_table = build_species_table(species_embedding, task_species_space)
    species_embedding_plot = build_species_tsne(species_table, max_species=args.max_species)

    outputs: list[str] = []
    outputs.extend(
        plot_composite(
            compound_embedding,
            species_embedding_plot,
            out_base=out_dir / args.prefix,
            dpi=args.dpi,
        )
    )

    compound_classification_path = out_dir / f"{args.prefix}_compound_classification_all.csv"
    compounds.to_csv(compound_classification_path, index=False, encoding="utf-8-sig")
    outputs.append(str(compound_classification_path))

    compound_plot_path = out_dir / f"{args.prefix}_compound_tsne_plot_data.csv"
    compound_embedding.to_csv(compound_plot_path, index=False, encoding="utf-8-sig")
    outputs.append(str(compound_plot_path))

    species_plot_path = out_dir / f"{args.prefix}_species_embedding_tsne_plot_data.csv"
    species_embedding_plot.to_csv(species_plot_path, index=False, encoding="utf-8-sig")
    outputs.append(str(species_plot_path))

    rulebook_path = out_dir / f"{args.prefix}_chemical_class_rulebook.csv"
    pd.DataFrame(CLASS_RULEBOOK, columns=["rule_id", "chemical_class", "rule_basis_zh"]).to_csv(
        rulebook_path, index=False, encoding="utf-8-sig"
    )
    outputs.append(str(rulebook_path))

    caption_path = write_caption_doc(
        out_dir=out_dir,
        prefix=args.prefix,
        compounds=compound_embedding,
        all_compounds=compounds,
        species=species_embedding_plot,
        chemical_space_path=Path(args.chemical_space),
        species_embedding_path=Path(args.species_embedding),
        task_species_space_path=Path(args.task_species_space),
    )
    outputs.append(str(caption_path))

    manifest_path = out_dir / f"{args.prefix}_manifest.json"
    manifest = {
        "chemical_space": str(args.chemical_space),
        "species_embedding": str(args.species_embedding),
        "task_species_space": str(args.task_species_space),
        "compound_n_total": int(len(compounds)),
        "compound_n_tsne": int(compound_embedding["tsne_x"].notna().sum()),
        "species_n_total": int(len(species_table)),
        "species_n_tsne": int(species_embedding_plot["tsne_x"].notna().sum()),
        "morgan": {"radius": MORGAN_RADIUS, "n_bits": MORGAN_BITS},
        "tsne": {
            "random_seed": RANDOM_SEED,
            "max_iter": TSNE_MAX_ITER,
            "compound_preprocessing": "Morgan fingerprint -> TruncatedSVD(50) -> t-SNE",
            "species_preprocessing": "species embedding -> StandardScaler -> PCA(50) -> t-SNE",
        },
        "classification_note": "Chemical classes are mutually exclusive primary labels assigned by a priority-ordered rulebook; evidence columns are retained in the classification table.",
        "classification_table": str(compound_classification_path),
        "compound_plot_data": str(compound_plot_path),
        "outputs": outputs,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append(str(manifest_path))

    print(json.dumps({"outputs": outputs}, ensure_ascii=False, indent=2))


def load_starry_palette(path: Path) -> dict[str, str]:
    colors = dict(STARRY_DEFAULTS)
    if path.exists():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        colors.update(payload.get("colors", {}) or {})
    return colors


def apply_palette(colors: dict[str, str]) -> None:
    global BACKGROUND, GRID, TEXT, MUTED
    BACKGROUND = colors["paper"]
    GRID = colors["grid"]
    TEXT = colors["charcoal"]
    MUTED = colors["night_gray"]
    for key in list(STARRY_DEFAULTS):
        if key in colors:
            STARRY_DEFAULTS[key] = colors[key]


def configure_matplotlib() -> None:
    font_path = first_existing([Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/msyh.ttc")])
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def build_compound_table(chemical_space: pd.DataFrame) -> pd.DataFrame:
    data = chemical_space.copy()
    data["smiles"] = data["smiles"].fillna("").astype(str)
    data["dtxsid"] = data["dtxsid"].fillna("").astype(str)
    data["cas_number"] = data["cas_number"].fillna("").astype(str)
    data["chemical_name"] = data["chemical_name"].fillna("").astype(str)
    data["chemical_key"] = np.where(data["dtxsid"].str.len() > 0, data["dtxsid"], data["cas_number"])
    data["chemical_key"] = np.where(data["chemical_key"].astype(str).str.len() > 0, data["chemical_key"], data["smiles"])

    grouped = (
        data.groupby(["chemical_key", "smiles"], dropna=False)
        .agg(
            dtxsid=("dtxsid", first_nonempty),
            cas_number=("cas_number", first_nonempty),
            chemical_name=("chemical_name", first_nonempty),
            n_training_rows=("n_rows", "sum"),
            n_task_rows=("task_head", "size"),
            n_task_heads=("task_head", "nunique"),
            n_task_families=("task_family", "nunique"),
            task_families=("task_family", unique_join),
            medium_domains=("medium_domains", unique_join),
            unique_species_count_max=("unique_species_count", "max"),
            target_value_mean_min=("target_value_mean_min", "min"),
            target_value_mean_max=("target_value_mean_max", "max"),
        )
        .reset_index()
    )
    return grouped


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = str(value)
        if text and text.lower() != "nan":
            return text
    return ""


def unique_join(values: pd.Series) -> str:
    tokens: set[str] = set()
    for value in values.dropna().astype(str):
        for part in value.replace(";", ",").split(","):
            part = part.strip()
            if part and part.lower() != "nan":
                tokens.add(part)
    return ";".join(sorted(tokens))


def classify_compounds(compounds: pd.DataFrame) -> pd.DataFrame:
    results = [classify_one(row.smiles, row.chemical_name) for row in compounds.itertuples(index=False)]
    payload = pd.DataFrame([result.__dict__ for result in results])
    return pd.concat([compounds.reset_index(drop=True), payload], axis=1)


def classify_one(smiles: str, chemical_name: str) -> ChemicalClassResult:
    mol = Chem.MolFromSmiles(str(smiles)) if str(smiles).strip() else None
    if mol is None:
        return ChemicalClassResult(
            "Invalid/no structure",
            "R00",
            "SMILES 缺失或 RDKit 无法解析；不参与 Morgan fingerprint t-SNE。",
            "invalid_smiles",
            "",
            False,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    canonical = Chem.MolToSmiles(mol, canonical=True)
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    c_count = counts.get("C", 0)
    n_count = counts.get("N", 0)
    o_count = counts.get("O", 0)
    f_count = counts.get("F", 0)
    cl_count = counts.get("Cl", 0)
    br_count = counts.get("Br", 0)
    i_count = counts.get("I", 0)
    p_count = counts.get("P", 0)
    s_count = counts.get("S", 0)
    metal_count = sum(count for element, count in counts.items() if element in METALS)
    aromatic_ring_count = sum(1 for ring in mol.GetRingInfo().AtomRings() if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring))
    name_lower = str(chemical_name).lower()

    flags = {
        "metal_or_metalloid": metal_count > 0,
        "no_carbon": c_count == 0,
        "pfas_name_cue": any(token in name_lower for token in ["perfluoro", "pfos", "pfoa", "pfas", "fluorotelomer"]),
        "perfluoro_chain": has_substructure(mol, "perfluoro_chain"),
        "carboxylic_acid": has_substructure(mol, "carboxylic_acid"),
        "ester": has_substructure(mol, "ester"),
        "amide": has_substructure(mol, "amide"),
        "phenol": has_substructure(mol, "phenol"),
        "aniline": has_substructure(mol, "aniline"),
        "name_phenol_aniline": any(token in name_lower for token in ["phenol", "bisphenol", "aniline"]),
        "halogenated": (f_count + cl_count + br_count + i_count) > 0,
        "hydrocarbon_only": c_count > 0 and (n_count + o_count + f_count + cl_count + br_count + i_count + p_count + s_count + metal_count) == 0,
    }
    evidence = ";".join(key for key, value in flags.items() if value)

    if metal_count > 0 or c_count == 0:
        cls, rule, basis = rule_lookup("R01")
    elif c_count > 0 and f_count >= 4 and (flags["pfas_name_cue"] or flags["perfluoro_chain"] or f_count >= max(4, c_count)):
        cls, rule, basis = rule_lookup("R02")
    elif c_count > 0 and cl_count > 0:
        cls, rule, basis = rule_lookup("R03")
    elif c_count > 0 and (br_count + i_count + f_count) > 0:
        cls, rule, basis = rule_lookup("R04")
    elif c_count > 0 and p_count > 0:
        cls, rule, basis = rule_lookup("R05")
    elif c_count > 0 and s_count > 0:
        cls, rule, basis = rule_lookup("R06")
    elif c_count > 0 and aromatic_ring_count >= 2 and (n_count + o_count + f_count + cl_count + br_count + i_count + p_count + s_count) == 0:
        cls, rule, basis = rule_lookup("R07")
    elif flags["phenol"] or flags["aniline"] or flags["name_phenol_aniline"]:
        cls, rule, basis = rule_lookup("R08")
    elif c_count > 0 and (flags["carboxylic_acid"] or flags["ester"] or flags["amide"]):
        cls, rule, basis = rule_lookup("R09")
    elif c_count > 0 and n_count > 0:
        cls, rule, basis = rule_lookup("R10")
    elif flags["hydrocarbon_only"]:
        cls, rule, basis = rule_lookup("R11")
    elif c_count > 0 and o_count > 0:
        cls, rule, basis = rule_lookup("R12")
    else:
        cls, rule, basis = rule_lookup("R99")

    return ChemicalClassResult(
        cls,
        rule,
        basis,
        evidence,
        canonical,
        True,
        c_count,
        n_count,
        o_count,
        f_count,
        cl_count,
        br_count,
        i_count,
        p_count,
        s_count,
        metal_count,
        aromatic_ring_count,
    )


def has_substructure(mol: Chem.Mol, key: str) -> bool:
    pattern = SMARTS[key]
    return bool(pattern is not None and mol.HasSubstructMatch(pattern))


def rule_lookup(rule_id: str) -> tuple[str, str, str]:
    for rid, label, basis in CLASS_RULEBOOK:
        if rid == rule_id:
            return label, rid, basis
    return "Other organics", "R99", "含碳但未匹配上述规则。"


def build_compound_tsne(compounds: pd.DataFrame, *, max_compounds: int = 0) -> pd.DataFrame:
    valid = compounds[compounds["valid_structure"]].copy()
    valid = valid.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    if max_compounds and len(valid) > max_compounds:
        valid = valid.sort_values("n_training_rows", ascending=False).head(max_compounds).reset_index(drop=True)

    fingerprints = np.zeros((len(valid), MORGAN_BITS), dtype=np.float32)
    keep_rows: list[int] = []
    for row_idx, row in enumerate(valid.itertuples(index=False)):
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
        arr = np.zeros((MORGAN_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints[row_idx] = arr
        keep_rows.append(row_idx)

    valid = valid.iloc[keep_rows].reset_index(drop=True)
    fingerprints = fingerprints[keep_rows]
    reduced = reduce_sparse_features(fingerprints, n_components=min(50, max(2, len(valid) - 1)))
    coords = run_tsne(reduced, random_state=RANDOM_SEED)
    valid["tsne_x"] = coords[:, 0]
    valid["tsne_y"] = coords[:, 1]
    valid["marker_size"] = scaled_marker_size(valid["n_training_rows"], min_size=13, max_size=52)
    return valid


def reduce_sparse_features(features: np.ndarray, *, n_components: int) -> np.ndarray:
    if features.shape[0] <= 3:
        return features
    n_components = min(n_components, features.shape[0] - 1, features.shape[1] - 1)
    if n_components < 2:
        return features
    return TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED).fit_transform(features)


def run_tsne(features: np.ndarray, *, random_state: int) -> np.ndarray:
    n = features.shape[0]
    if n < 4:
        return PCA(n_components=2, random_state=random_state).fit_transform(features)
    perplexity = min(45.0, max(5.0, (n - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=TSNE_MAX_ITER,
        metric="euclidean",
        random_state=random_state,
        method="barnes_hut",
        angle=0.5,
    ).fit_transform(features)


def build_species_table(species_embedding: pd.DataFrame, task_species_space: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        task_species_space.groupby("species_number", dropna=False)
        .agg(
            species_training_rows=("n_rows", "sum"),
            n_task_heads=("task_head", "nunique"),
            n_task_families=("task_family", "nunique"),
            task_families=("task_family", unique_join),
            medium_domains=("medium_domains", unique_join),
            unique_chemical_count_sum=("unique_chemical_count", "sum"),
            unique_chemical_count_max=("unique_chemical_count", "max"),
        )
        .reset_index()
    )
    table = species_embedding.merge(coverage, on="species_number", how="left").copy()
    table["species_training_rows"] = table["species_training_rows"].fillna(0)
    table["taxon_plot_group"] = table["taxon_group_l2"].fillna("unclassified").replace("", "unclassified")
    major = set(table["taxon_plot_group"].value_counts().head(10).index)
    table["taxon_plot_group"] = table["taxon_plot_group"].where(table["taxon_plot_group"].isin(major), "Other")
    return table


def build_species_tsne(species_table: pd.DataFrame, *, max_species: int = 0) -> pd.DataFrame:
    emb_cols = [col for col in species_table.columns if col.startswith("emb_")]
    data = species_table.dropna(subset=emb_cols).copy().reset_index(drop=True)
    if max_species and len(data) > max_species:
        data = data.sort_values("species_training_rows", ascending=False).head(max_species).reset_index(drop=True)
    features = data[emb_cols].to_numpy(dtype=np.float32)
    scaled = StandardScaler().fit_transform(features)
    n_components = min(50, scaled.shape[1], max(2, scaled.shape[0] - 1))
    reduced = PCA(n_components=n_components, random_state=RANDOM_SEED).fit_transform(scaled)
    coords = run_tsne(reduced, random_state=RANDOM_SEED + 1)
    data["tsne_x"] = coords[:, 0]
    data["tsne_y"] = coords[:, 1]
    data["marker_size"] = scaled_marker_size(data["species_training_rows"], min_size=12, max_size=48)
    return data


def scaled_marker_size(values: pd.Series, *, min_size: float, max_size: float) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float)
    transformed = np.sqrt(np.maximum(numeric, 0.0))
    if np.nanmax(transformed) <= 0:
        return np.full_like(transformed, (min_size + max_size) / 2.0)
    scaled = transformed / np.nanmax(transformed)
    return min_size + (max_size - min_size) * scaled


def plot_composite(compounds: pd.DataFrame, species: pd.DataFrame, *, out_base: Path, dpi: int) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=mm_to_inch(183, 132), facecolor=BACKGROUND)
    plot_compound_panel(axes[0], compounds)
    plot_species_panel(axes[1], species)
    fig.subplots_adjust(left=0.065, right=0.990, top=0.925, bottom=0.405, wspace=0.155)
    add_combined_legend(fig, compounds, species)
    return save_figure(fig, out_base, dpi=dpi)


def plot_compound_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    ordered_classes = ordered_present_classes(data["chemical_class"], CHEMICAL_CLASS_COLORS)
    for cls in ordered_classes:
        subset = data[data["chemical_class"] == cls]
        ax.scatter(
            subset["tsne_x"],
            subset["tsne_y"],
            s=subset["marker_size"],
            c=CHEMICAL_CLASS_COLORS.get(cls, "#B8A6C9"),
            alpha=0.68,
            edgecolors=matplotlib.colors.to_rgba(BACKGROUND, 0.86),
            linewidths=0.32,
            rasterized=True,
        )
    style_2d_axis(ax, "(a) Compound chemical-space t-SNE", "t-SNE 1", "t-SNE 2")


def plot_species_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    ordered_groups = ordered_present_classes(data["taxon_plot_group"], SPECIES_GROUP_COLORS)
    for group in ordered_groups:
        subset = data[data["taxon_plot_group"] == group]
        ax.scatter(
            subset["tsne_x"],
            subset["tsne_y"],
            s=subset["marker_size"],
            c=SPECIES_GROUP_COLORS.get(group, "#B8A6C9"),
            alpha=0.62,
            edgecolors=matplotlib.colors.to_rgba(BACKGROUND, 0.88),
            linewidths=0.30,
            rasterized=True,
        )
    style_2d_axis(ax, "(b) Learned species-embedding t-SNE", "t-SNE 1", "t-SNE 2")


def style_2d_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=PANEL_TITLE_SIZE, fontweight="bold", color=TEXT, pad=8)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, fontweight="bold", color=TEXT)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, fontweight="bold", color=TEXT)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, colors=MUTED, width=0.6, length=2.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, color=matplotlib.colors.to_rgba(GRID, 0.78), linewidth=0.62)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(matplotlib.colors.to_rgba(MUTED, 0.72))
        ax.spines[spine].set_linewidth(0.75)


def ordered_present_classes(values: pd.Series, color_map: dict[str, str]) -> list[str]:
    present = list(values.value_counts().index)
    ordered = [key for key in color_map if key in present]
    ordered.extend([value for value in present if value not in ordered])
    return ordered


def add_combined_legend(fig: plt.Figure, compounds: pd.DataFrame, species: pd.DataFrame) -> None:
    compound_classes = ordered_present_classes(compounds["chemical_class"], CHEMICAL_CLASS_COLORS)
    species_groups = ordered_present_classes(species["taxon_plot_group"], SPECIES_GROUP_COLORS)
    compound_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CHEMICAL_CLASS_COLORS.get(cls, "#B8A6C9"),
            markeredgecolor=BACKGROUND,
            markeredgewidth=0.45,
            markersize=5.8,
            label=f"{short_label(cls)} ({int((compounds['chemical_class'] == cls).sum())})",
        )
        for cls in compound_classes
    ]
    species_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SPECIES_GROUP_COLORS.get(group, "#B8A6C9"),
            markeredgecolor=BACKGROUND,
            markeredgewidth=0.45,
            markersize=5.8,
            label=f"{group} ({int((species['taxon_plot_group'] == group).sum())})",
        )
        for group in species_groups
    ]
    fig.text(
        0.5,
        0.318,
        "Compound class",
        ha="center",
        va="center",
        fontsize=LEGEND_TEXT_SIZE,
        color=TEXT,
        fontweight="bold",
    )
    legend_a = fig.legend(
        compound_handles,
        [handle.get_label() for handle in compound_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.145),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_TEXT_SIZE,
        handletextpad=0.26,
        columnspacing=0.74,
        labelspacing=0.27,
    )
    fig.text(
        0.5,
        0.112,
        "Species group",
        ha="center",
        va="center",
        fontsize=LEGEND_TEXT_SIZE,
        color=TEXT,
        fontweight="bold",
    )
    legend_b = fig.legend(
        species_handles,
        [handle.get_label() for handle in species_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.028),
        ncol=5,
        frameon=False,
        fontsize=LEGEND_TEXT_SIZE,
        handletextpad=0.26,
        columnspacing=0.80,
        labelspacing=0.27,
    )
    for legend in (legend_a, legend_b):
        for text in legend.get_texts():
            text.set_color(TEXT)
            text.set_fontweight("bold")


def short_label(label: str) -> str:
    replacements = {
        "PFAS/highly fluorinated": "PFAS/high-F",
        "PAHs/aromatic hydrocarbons": "PAHs/arom. HC",
        "Carboxylic/ester/amide organics": "Carboxyl/ester/amide",
        "Nitrogen-containing organics": "N organics",
        "Phenols/anilines/bisphenols": "Phenol/aniline",
        "Other halogenated organics": "Other halogenated",
    }
    return replacements.get(label, label)


def mm_to_inch(width_mm: float, height_mm: float) -> tuple[float, float]:
    return width_mm / 25.4, height_mm / 25.4


def save_figure(fig: plt.Figure, out_base: Path, *, dpi: int) -> list[str]:
    png_path = out_base.with_suffix(".png")
    svg_path = out_base.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(svg_path)
    plt.close(fig)
    return [str(png_path), str(svg_path)]


def write_caption_doc(
    *,
    out_dir: Path,
    prefix: str,
    compounds: pd.DataFrame,
    all_compounds: pd.DataFrame,
    species: pd.DataFrame,
    chemical_space_path: Path,
    species_embedding_path: Path,
    task_species_space_path: Path,
) -> Path:
    doc_path = out_dir / f"{prefix}_中文图注与解读.md"
    chem_counts = class_count_table(all_compounds, "chemical_class")
    chem_plot_counts = class_count_table(compounds, "chemical_class")
    species_counts = class_count_table(species, "taxon_plot_group")
    content = f"""# 化合物类别 t-SNE 与物种 embedding 空间总览图注与解读

## 图件与数据

- 主图：`{prefix}.png` / `{prefix}.svg`
- 化合物训练空间：`{chemical_space_path}`
- 物种 embedding：`{species_embedding_path}`
- 物种训练覆盖：`{task_species_space_path}`
- 全量化合物分类表：{len(all_compounds):,} 个化合物/结构记录；其中 {int(all_compounds["valid_structure"].sum()):,} 个具有 RDKit 可解析结构。
- 入图化合物：{len(compounds):,} 个去重后的有效结构化合物。
- 物种数量：{len(species):,} 个训练后 species embedding。

## 方法说明

化合物面板使用 Morgan fingerprint（半径 {MORGAN_RADIUS}，{MORGAN_BITS} bits）表示分子结构，经 TruncatedSVD 降到 50 维后进行 t-SNE。图中每个点代表一个去重后的化合物，颜色表示互斥化学主类，点大小表示该化合物在训练空间中的记录数。

化合物类别采用优先级规则赋予单一主标签。优先级从无效/无结构、金属/无机、PFAS/高度氟化、有机氯、其他卤代、有机磷、有机硫、PAH、酚/苯胺/双酚、羧酸/酯/酰胺、含氮有机物、烃类、含氧有机物到其他有机物。完整分类依据、元素计数、SMARTS/名称线索和规则编号已保存在分类表中。

物种面板使用训练后导出的 193 维 species embedding，经标准化、PCA 降到 50 维后进行 t-SNE。图中每个点代表一个物种，颜色表示分类学组，点大小表示该物种在训练空间中的记录数。该图用于展示模型学习到的物种向量空间结构，只能作为表示诊断，不能直接解释为生理机制。

## 化合物主类分布

{chem_counts}

## 入图化合物主类分布

{chem_plot_counts}

## 物种组分布

{species_counts}

## 图注

**图 (a) 化合物 Morgan fingerprint t-SNE。**
该图展示训练空间中化合物结构的二维分布。不同颜色表示基于结构元素、SMARTS 片段和名称线索得到的互斥化学主类，可用于说明模型训练空间是否覆盖 PFAS/高度氟化有机物、有机氯、金属/无机物、有机磷、PAH 等环境污染物类型。

**图 (b) 物种 embedding t-SNE。**
该图展示训练后 species embedding 的二维投影。颜色表示分类学组，点大小反映训练记录覆盖。若相近分类组在空间中形成相对连续区域，说明模型的物种/分类学表示与数据中的分类结构具有一定一致性；但 embedding 坐标本身不应被解释为具体生物机制。
"""
    doc_path.write_text(content, encoding="utf-8")
    return doc_path


def class_count_table(data: pd.DataFrame, column: str) -> str:
    total = len(data)
    lines = ["| 类别 | 数量 | 占比 |", "|---|---:|---:|"]
    for label, count in data[column].value_counts().items():
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"| {label} | {int(count):,} | {pct:.1f}% |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
