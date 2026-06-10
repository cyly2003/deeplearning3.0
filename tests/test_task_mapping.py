from __future__ import annotations

import unittest

from qsar_tl.data.task_mapping import map_task_head, parse_effect_level_x


class TaskMappingTests(unittest.TestCase):
    def test_lc50_mortality_maps_to_ecx_mortality(self) -> None:
        result = map_task_head(endpoint="LC50", effect="MOR", measurement="MORT")

        self.assertEqual(result.task_status, "included")
        self.assertEqual(result.task_head, "ECx_Mortality")
        self.assertEqual(result.task_family, "ECx")
        self.assertEqual(result.effect_family, "Mortality")
        self.assertEqual(result.effect_level_x, 50.0)

    def test_lc10_survival_maps_to_mortality(self) -> None:
        result = map_task_head(endpoint="LC10/", effect="MOR/", measurement="SURV/")

        self.assertEqual(result.task_status, "included")
        self.assertEqual(result.task_head, "ECx_Mortality")
        self.assertEqual(result.effect_level_x, 10.0)

    def test_noec_growth_has_no_effect_level(self) -> None:
        result = map_task_head(endpoint="NOEC", effect="GRO", measurement="LGTH")

        self.assertEqual(result.task_status, "included")
        self.assertEqual(result.task_head, "NOEC_Growth")
        self.assertIsNone(result.effect_level_x)

    def test_loec_reproduction_has_no_effect_level(self) -> None:
        result = map_task_head(endpoint="LOEC", effect="REP", measurement="PROG/")

        self.assertEqual(result.task_status, "included")
        self.assertEqual(result.task_head, "LOEC_Reproduction")
        self.assertIsNone(result.effect_level_x)

    def test_excluded_endpoint_is_reported(self) -> None:
        result = map_task_head(endpoint="BCFD", effect="ACC", measurement="RSDE")

        self.assertEqual(result.task_status, "excluded")
        self.assertEqual(result.task_excluded_reason, "excluded_endpoint:BCFD")

    def test_oral_target_is_excluded_before_endpoint_mapping(self) -> None:
        result = map_task_head(
            endpoint="LC50",
            effect="MOR",
            measurement="MORT",
            target_name="neg_log10_mg_kg_bw_day",
            target_basis="mg/kg/day",
        )

        self.assertEqual(result.task_status, "excluded")
        self.assertEqual(result.task_excluded_reason, "excluded_oral_target")

    def test_effect_level_parser(self) -> None:
        self.assertEqual(parse_effect_level_x("EC05"), 5.0)
        self.assertEqual(parse_effect_level_x("LC99.9"), 99.9)
        self.assertIsNone(parse_effect_level_x("NOEC"))


if __name__ == "__main__":
    unittest.main()
