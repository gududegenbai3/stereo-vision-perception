"""Coverage, fixed-domain and source-weighting tests for the new evaluation."""

import unittest
import numpy as np

from src.stereo_fair_metrics import evaluation_domains, paired_bootstrap, scene_source_summary, score_prediction


class FairEvaluationContracts(unittest.TestCase):
    def test_missing_and_wrong_outputs_are_not_removed_from_primary_score(self) -> None:
        truth: np.ndarray = np.array([[1., 2., 3., 4.]], np.float32)
        prediction: np.ndarray = np.array([[1., np.nan, 6., -100.]], np.float32)
        result: dict[str, object] = score_prediction(prediction, truth, np.ones_like(truth, bool))
        self.assertEqual(result["reference"], 4)
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["bad2_or_missing"], .75)
        self.assertEqual(result["coverage"], .75)

    def test_search_roi_is_independent_of_prediction_and_known_masks(self) -> None:
        truth: np.ndarray = np.ones((32, 256), np.float32)*4
        arrays: dict[str, np.ndarray] = {"disparity": truth, "known": np.ones_like(truth, bool), "non_occluded": np.ones_like(truth, bool)}
        domains: dict[str, np.ndarray] = evaluation_domains(arrays, 192)
        self.assertFalse(domains["fixed_roi"][:, :192].any())
        self.assertTrue(domains["fixed_roi"][4:-4, 196:-4].all())
        self.assertTrue(np.all(~domains["fixed_roi"] | domains["non_occluded"]))

    def test_source_equal_macro_is_not_dominated_by_repeated_observations(self) -> None:
        records: list[dict[str, object]] = [{"dataset": "large", "group": "same", "metrics": {"fixed_roi": {"bad2_or_missing": 1.}}} for _ in range(100)]
        records.append({"dataset": "small", "group": "other", "metrics": {"fixed_roi": {"bad2_or_missing": 0.}}})
        result: dict[str, object] = scene_source_summary(records, "fixed_roi", "bad2_or_missing")
        self.assertEqual(result["source_equal_macro"], .5)

    def test_bootstrap_is_paired_and_flags_mismatched_groups(self) -> None:
        first: dict = {"a": {"one": .1, "two": .2}, "b": {"three": .3}}
        second: dict = {"a": {"one": .2, "two": .3}, "b": {"three": .4}}
        result: dict = paired_bootstrap(first, second, 7, 100)
        self.assertAlmostEqual(result["difference_source_equal_macro"], -.1)
        self.assertEqual(result["component_wins"], 3)
        with self.assertRaises(ValueError):
            paired_bootstrap(first, {"a": {"other": .1}, "b": {"three": .4}}, 7, 100)


if __name__ == "__main__":
    unittest.main()
