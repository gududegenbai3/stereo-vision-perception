"""Independent checks against split leakage, crop geometry and masked loss errors."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.banet import environment_dll_handle
from src.banet_training import fixed_domain_metrics, set_training_mode, supervised_loss
from src.stereo_training_data import StereoTrainingSample, freeze_split, read_training_pfm, training_crop

_dll_handle: object | None = environment_dll_handle()
if importlib.util.find_spec("torch") is not None:
    import torch
else:
    torch = None


class BanetTrainingContracts(unittest.TestCase):
    def sample(self) -> StereoTrainingSample:
        image: np.ndarray = np.arange(256 * 512 * 3, dtype=np.uint8).reshape(256, 512, 3)
        truth: np.ndarray = np.full((256, 512), 4., np.float32)
        mask: np.ndarray = np.ones((256, 512), bool)
        return StereoTrainingSample("forest_1s", "forest", image, image.copy(), truth, mask, mask.copy())

    def test_split_rejects_same_site_in_two_sets_and_missing_sites(self) -> None:
        names: list[str] = ["forest_1s", "forest_2s", "terrains_1l", "terrains_1s", "facade_1s"]
        split: dict = freeze_split(names, {"train": ["forest"], "validation": ["terrains"], "test": ["facade"]})
        self.assertEqual(split["train"], ["forest_1s", "forest_2s"])
        with self.assertRaisesRegex(ValueError, "leakage"):
            freeze_split(names, {"train": ["forest"], "validation": ["forest", "terrains"], "test": ["facade"]})
        with self.assertRaisesRegex(ValueError, "Unassigned"):
            freeze_split(names, {"train": ["forest"], "validation": ["unknown"], "test": ["facade"]})

    def test_pfm_bottom_up_and_pixel_units(self) -> None:
        expected: np.ndarray = np.array([[1.25, np.inf], [3.5, 4.75]], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path: Path = Path(directory) / "gt.pfm"
            path.write_bytes(b"Pf\n2 2\n-1\n" + expected[::-1].astype("<f4").tobytes())
            np.testing.assert_array_equal(read_training_pfm(path, (2, 2)), expected)
            with self.assertRaises(ValueError):
                read_training_pfm(path, (3, 2))

    def test_shared_crop_retains_disparity_and_excludes_unseen_correspondence(self) -> None:
        left, right, target, valid = training_crop(self.sample(), (256, 512), 192, np.random.default_rng(1), (1, 1), (1, 1))
        np.testing.assert_array_equal(left, right)
        self.assertEqual(left.shape, (3, 256, 512))
        self.assertEqual(left.dtype, np.float32)
        self.assertFalse(valid[..., :4].any())
        self.assertTrue(valid[..., 4:].all())
        np.testing.assert_array_equal(target[valid], np.full(256 * 508, 4, np.float32))

    @unittest.skipIf(torch is None, "Optional training tests require requirements-banet.txt")
    def test_loss_excludes_masked_finite_and_infinite_targets_before_arithmetic(self) -> None:
        final = torch.tensor([[[[1., 100., 100.]]]], requires_grad=True)
        auxiliary = torch.tensor([[[[4., 100., 100.]]]], requires_grad=True)
        target = torch.tensor([[[[2., float("inf"), 5.]]]])
        valid = torch.tensor([[[[True, False, False]]]])
        loss = supervised_loss([final, auxiliary], target, valid, 192, .3)
        self.assertAlmostEqual(float(loss.detach()), .95, places=6)
        loss.backward()
        np.testing.assert_array_equal(final.grad.numpy(), [[[[-1., 0., 0.]]]])
        np.testing.assert_allclose(auxiliary.grad.numpy(), [[[[.3, 0., 0.]]]])
        with self.assertRaisesRegex(ValueError, "No valid"):
            supervised_loss([final, auxiliary], target, torch.zeros_like(valid), 192, .3)

    @unittest.skipIf(torch is None, "Optional training tests require requirements-banet.txt")
    def test_frozen_bn_statistics_keep_affine_parameters_trainable(self) -> None:
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3))
        before = model[1].running_mean.clone()
        set_training_mode(model, True)
        model(torch.ones(2, 3, 8, 8)).sum().backward()
        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        self.assertTrue(model[1].weight.requires_grad)
        self.assertIsNotNone(model[1].weight.grad)
        torch.testing.assert_close(model[1].running_mean, before)

    def test_dense_evaluation_does_not_drop_bad_predictions(self) -> None:
        sample: StereoTrainingSample = self.sample()
        prediction: np.ndarray = sample.disparity_px.copy()
        prediction[:, :100] = 150  # Deliberately wrong/out-of-image correspondence.
        result: dict = fixed_domain_metrics(prediction, sample, 192)
        self.assertEqual(result["pixels"], 256 * 512)
        self.assertEqual(result["bad1_pixels"], 256 * 100)
        self.assertAlmostEqual(result["mae_px"], 146 * 100 / 512)


if __name__ == "__main__":
    unittest.main()
