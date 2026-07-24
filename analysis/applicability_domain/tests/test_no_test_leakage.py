from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_calibration_function_has_no_test_input_or_test_target_reference() -> None:
    source_path = ROOT / "04_calibrate_ad_on_validation.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    evaluate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_candidates")
    argument_names = [argument.arg for argument in evaluate.args.args]
    names = {node.id for node in ast.walk(evaluate) if isinstance(node, ast.Name)}
    assert argument_names == ["validation", "config"]
    assert "test" not in names


def test_locked_rule_hashes_validation_identities_only() -> None:
    rule = json.loads((ROOT / "ad_rule_locked.json").read_text(encoding="utf-8"))
    assert "validation_record_id_sha256" in rule
    assert not any("test" in key.casefold() for key in rule if key.endswith("sha256"))
    records = pd.read_parquet(ROOT / "ad_record_level.parquet")
    assert set(records["analysis_split"]) == {"validation", "test"}


def test_context_fit_metadata_is_training_derived() -> None:
    ranges = json.loads((ROOT / "context_training_ranges.json").read_text(encoding="utf-8"))
    payload = json.dumps(ranges).casefold()
    assert "test_target" not in payload
    assert "validation_target" not in payload
