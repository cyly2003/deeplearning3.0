from scripts.analyze_v1_2_48_random8_2_similarity_coverage import (
    Interval,
    consolidate_bins,
    empirical_intervals,
    metric_values,
)


def _row(index: int, score: float, task: str = "T") -> dict[str, object]:
    return {
        "aggregate_id": str(index),
        "canonical_smiles": f"canonical_{index}",
        "resampling_cluster": f"cluster_{index}",
        "task_head": task,
        "y_true": float(index),
        "y_pred": float(index) + 0.1,
        "abs_error": 0.1,
        "smax_all_fit": score,
        "smax_stage3_fit": score,
    }


def test_interval_includes_one_only_in_final_bin() -> None:
    intervals = [
        Interval(0.0, 0.8, False, ("low",)),
        Interval(0.8, 1.0, True, ("high",)),
    ]
    assert not intervals[0].contains(1.0)
    assert intervals[1].contains(1.0)
    assert intervals[1].label == "Smax >= 0.80"


def test_full_interval_label_is_not_a_strict_less_than_claim() -> None:
    interval = Interval(0.0, 1.0, True, ("all",))
    assert interval.label == "0.00 <= Smax <= 1.00"


def test_support_merge_uses_adjacent_intervals_without_metrics() -> None:
    rows = [_row(index, 0.1) for index in range(12)] + [
        _row(index + 12, 0.9) for index in range(120)
    ]
    bins, fixed = consolidate_bins(
        rows,
        score_field="smax_all_fit",
        min_rows=100,
        min_canonicals=10,
    )
    assert len(fixed) == 5
    assert len(bins) == 1
    assert bins[0].contains(0.1)
    assert bins[0].contains(0.9)


def test_metric_values_reports_perfect_within_task_fit() -> None:
    rows = [_row(1, 0.1, "A"), _row(2, 0.1, "A"), _row(3, 0.1, "B"), _row(4, 0.1, "B")]
    for row in rows:
        row["y_pred"] = row["y_true"]
        row["abs_error"] = 0.0
    values = metric_values(rows, min_task_rows=2)
    assert values["mae"] == 0.0
    assert values["within_task_centered_r2"] == 1.0


def test_empirical_intervals_partition_canonical_similarity_distribution() -> None:
    rows = [_row(index, index / 8.0) for index in range(1, 9)]
    intervals = empirical_intervals(rows, "smax_all_fit", count=4)
    assert len(intervals) == 4
    assert all(
        sum(interval.contains(row["smax_all_fit"]) for interval in intervals) == 1
        for row in rows
    )
    assert intervals[-1].upper_inclusive
