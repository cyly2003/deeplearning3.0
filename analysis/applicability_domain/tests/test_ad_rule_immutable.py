from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_outer_test_evaluation_did_not_modify_locked_rule() -> None:
    integrity = json.loads((ROOT / "ad_rule_integrity.json").read_text(encoding="utf-8"))
    current = file_hash(ROOT / "ad_rule_locked.json")
    assert integrity["unchanged"] is True
    assert integrity["sha256_before_test_evaluation"] == integrity["sha256_after_test_evaluation"] == current


def test_figure_outputs_are_traceable_to_saved_source_data() -> None:
    manifest = json.loads((ROOT / "figure_source_data" / "figure_source_manifest.json").read_text(encoding="utf-8"))
    assert manifest
    for figure in manifest.values():
        for output in figure["outputs"]:
            assert Path(output).exists()
        for source in figure["source_data"]:
            path = Path(source)
            assert path.exists() and path.stat().st_size > 0
