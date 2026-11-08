"""Behavioral tests for domain adversarial training and deployment."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from d4rnet.data import DriverDataset, TeacherLogits
from d4rnet.grl import GRLController
from d4rnet.losses import classification_loss, distillation_weight
from d4rnet.metrics import ClassificationMetrics
from d4rnet.models import D4RNet, GradientReversal
from d4rnet.runtime import (
    capture_rng,
    load_student,
    load_torch,
    restore_rng,
    save_checkpoint,
    seed_everything,
)

torch.set_num_threads(2)


class GradientTests(unittest.TestCase):
    def test_reversal_changes_only_backward(self):
        value = torch.tensor([2.0, -3.0], requires_grad=True)
        result = GradientReversal(0.25)(value)
        torch.testing.assert_close(result, value)
        result.sum().backward()
        torch.testing.assert_close(value.grad, torch.full_like(value, -0.25))

    def test_legacy_preserves_threshold_order_and_epoch_reset(self):
        controller = GRLController(5)
        controller.begin_epoch(3)
        self.assertFalse(controller.enabled)
        controller.begin_epoch(4)
        controller.update(0.15, 4)  # >0.1 and <1/5: original first branch wins.
        self.assertTrue(controller.enabled)
        self.assertAlmostEqual(controller.strength, 1.05e-4)
        controller.update(0.0, 4)
        self.assertFalse(controller.enabled)
        controller.begin_epoch(5)
        self.assertTrue(controller.enabled)
        self.assertEqual(controller.strength, 1e-4)

    def test_algorithm1_uses_ema_and_retains_state_across_epochs(self):
        controller = GRLController(20, mode="algorithm1")
        controller.begin_epoch(4)
        controller.update(0.9, 4)
        self.assertAlmostEqual(controller.accuracy_ema, 0.09)
        strength = controller.strength
        controller.begin_epoch(5)
        self.assertEqual(controller.strength, strength)
        restored = GRLController(**controller.state_dict())
        self.assertEqual(restored.state_dict(), controller.state_dict())

    def test_kd_has_correct_direction_temperature_and_teacher_detachment(self):
        student = torch.tensor([[1.0, -1.0], [0.2, 0.8]], requires_grad=True)
        teacher = torch.tensor([[0.4, 0.6], [1.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        hard, soft = classification_loss(student, labels, teacher, temperature=3, beta=0.7)
        target = torch.softmax(teacher.detach() / 3, 1)
        expected = (target * (target.log() - torch.log_softmax(student / 3, 1))).sum() / 2 * 9
        torch.testing.assert_close(soft, 0.7 * expected)
        (hard + soft).backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_beta_schedule_and_hard_only_ablation(self):
        self.assertEqual(distillation_weight(10), 0.7)
        self.assertAlmostEqual(distillation_weight(11), 0.66)
        self.assertEqual(distillation_weight(30), 0.1)
        logits = torch.randn(3, 22)
        labels = torch.tensor([0, 1, 2])
        hard, soft = classification_loss(logits, labels)
        torch.testing.assert_close(hard, nn.functional.cross_entropy(logits, labels))
        self.assertEqual(soft.item(), 0)


class ModelTests(unittest.TestCase):
    def test_published_feature_shapes_and_auxiliary_backward(self):
        model = D4RNet(num_domains=14).eval()
        captured, handles = {}, []
        for index in (3, 6, 13, 18):

            def record(module, inputs, output, index=index):
                captured[index] = tuple(output.shape[1:])

            handles.append(model.features[index].register_forward_hook(record))
        model.set_grl(True, 1e-4)
        main, auxiliary = model(torch.randn(2, 3, 224, 224), return_aux=True)
        self.assertEqual(
            captured, {3: (24, 56, 56), 6: (32, 28, 28), 13: (96, 14, 14), 18: (1280, 7, 7)}
        )
        self.assertEqual(main.shape, (2, 22))
        self.assertEqual(auxiliary.shape, (2, 14))
        (main.square().mean() + auxiliary.square().mean()).backward()
        self.assertIsNotNone(model.features[0][0].weight.grad)
        self.assertIsNotNone(model.aux_classifier.weight.grad)
        for handle in handles:
            handle.remove()

    def test_v3_auxiliary_branch(self):
        model = D4RNet("mobilenet_v3_large", num_domains=65).eval()
        with torch.no_grad():
            main, auxiliary = model(torch.randn(1, 3, 64, 64), return_aux=True)
        self.assertEqual(main.shape, (1, 22))
        self.assertEqual(auxiliary.shape, (1, 65))

    def test_training_checkpoint_and_deploy_model_produce_identical_logits(self):
        model = D4RNet(num_domains=14).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save({"model": model.state_dict(), "config": {"backbone": "mobilenet_v2"}}, path)
            deploy = load_student(path).eval()
            inputs = torch.randn(2, 3, 64, 64)
            with torch.no_grad():
                torch.testing.assert_close(model(inputs), deploy(inputs), rtol=0, atol=0)
            self.assertFalse(any(key.startswith("aux_") for key in deploy.state_dict()))
            weights = model.state_dict()
            del weights["main_classifier.0.weight"]
            torch.save(weights, path)
            with self.assertRaises(RuntimeError):
                load_student(path)

    def test_inactive_reversal_retains_ordinary_auxiliary_gradients(self):
        model = D4RNet(num_domains=14).eval()
        image = torch.randn(2, 3, 64, 64)
        model.set_grl(False, 1e-4)
        _, domain = model(image, return_aux=True)
        domain.sum().backward()
        ordinary = model.aux_projection[0].weight.grad.clone()
        classifier = model.aux_classifier.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        model.set_grl(True, 0.25)
        _, domain = model(image, return_aux=True)
        domain.sum().backward()
        torch.testing.assert_close(model.aux_projection[0].weight.grad, -0.25 * ordinary)
        torch.testing.assert_close(model.aux_classifier.weight.grad, classifier)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        Image.new("L", (30, 45), 100).save(self.root / "gray.png")
        Image.new("RGB", (35, 25), (120, 30, 80)).save(self.root / "rgb.png")
        (self.root / "train.txt").write_text("42\tgray.png\t0\t0\n42\trgb.png\t1\t1\n")
        spec = {
            "image_subdir": ".",
            "train": "train.txt",
            "auxiliary_train": "train.txt",
            "val": "train.txt",
            "test": "train.txt",
            "num_domains": 2,
        }
        self.patch = patch("d4rnet.data.dataset_specs", return_value={"fixture": spec})
        self.patch.start()
        self.dataset = DriverDataset(self.root, "fixture", auxiliary=True, splits_root=self.root)

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def cache(self, **overrides):
        value = dict(
            logits=torch.arange(44).float().reshape(2, 22).flip(0),
            labels=torch.tensor([1, 0]),
            indices=torch.tensor([1, 0]),
            sample_fingerprint=self.dataset.fingerprint,
            image_size=224,
        )
        value.update(overrides)
        path = self.root / "logits.pt"
        torch.save(value, path)
        return path

    def test_rgb_resize_and_zero_based_position_ignore_manifest_id(self):
        sample = self.dataset[0]
        self.assertEqual(sample["image"].shape, (3, 224, 224))
        self.assertEqual(sample["index"], 0)
        self.assertTrue(torch.equal(self.dataset[0]["image"], self.dataset[0]["image"]))
        cache = TeacherLogits(self.cache(), self.dataset)
        torch.testing.assert_close(
            cache[torch.tensor([1, 0])], torch.arange(44).float().reshape(2, 22).flip(0)
        )

    def test_logit_cache_rejects_mismatch_duplicate_indices_and_legacy_by_default(self):
        for overrides in [
            dict(sample_fingerprint="wrong"),
            dict(indices=torch.tensor([0, 0])),
            dict(labels=torch.tensor([0, 1])),
            dict(image_size=320),
        ]:
            with self.assertRaises(ValueError):
                TeacherLogits(self.cache(**overrides), self.dataset)
        path = self.cache()
        value = load_torch(path)
        del value["sample_fingerprint"]
        torch.save(value, path)
        with self.assertRaises(ValueError):
            TeacherLogits(path, self.dataset)
        TeacherLogits(path, self.dataset, allow_legacy=True)

    def test_shuffled_loader_uses_the_matching_teacher_row(self):
        cache = TeacherLogits(self.cache(), self.dataset)
        loader = DataLoader(
            self.dataset, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(42)
        )
        for batch in loader:
            expected = 22 * batch["label"].item()
            self.assertEqual(cache[batch["index"]][0, 0].item(), expected)


class MetricTests(unittest.TestCase):
    def test_metrics_match_hand_computed_macro_results(self):
        metric = ClassificationMetrics(3)
        metric.update(
            torch.tensor([[9.0, 0, 0], [0.0, 8, 0], [0.0, 7, 1], [0.0, 0, 5]]),
            torch.tensor([0, 0, 1, 2]),
        )
        result = metric.compute()
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["top1"], 75)
        self.assertAlmostEqual(result["precision_macro"], 100 * 2.5 / 3)
        self.assertAlmostEqual(result["recall_macro"], 100 * 2.5 / 3)

    def test_rng_state_roundtrip_uses_safe_checkpoint_types(self):
        seed_everything(42)
        state = capture_rng()
        expected = (torch.rand(2), np.random.rand(2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pt"
            save_checkpoint(path, state)
            restore_rng(load_torch(path))
        torch.testing.assert_close(torch.rand(2), expected[0])
        np.testing.assert_array_equal(np.random.rand(2), expected[1])


if __name__ == "__main__":
    unittest.main()
