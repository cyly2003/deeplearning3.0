from __future__ import annotations

from qsar_tl.training.toxicity_binning import _molecular_weight


def test_textual_missing_smiles_do_not_reach_rdkit_weight_parsing() -> None:
    for value in (None, "nan", "NaN", "NULL", "<NA>", " n/a "):
        assert _molecular_weight({"smiles": value}, descriptor_mol_weight=None) is None
