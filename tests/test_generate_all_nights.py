"""Whole-night generation for every participant (CPU, fake sampler)."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from night_cover import assemble, default_batch_size, plan_cover, subject_seed
from scripts import generate_all_nights as gan
from tests import night_fixture as nf
from watchsleep_loader import SSDataset


def fake_sampler(model, clean, schedule, args, device):
    """Generated seconds = stage mean (+ event shift when the event channel exists)."""
    stages = clean[:, :, 1:5].argmax(dim=2).cpu().numpy()
    seconds = nf.STAGE_IBI[stages]
    if clean.shape[2] == 11:
        seconds = seconds + nf.APNEA_SHIFT * clean[:, :, 10].cpu().numpy()
    return torch.as_tensor(seconds / 2.0 * 2.0 - 1.0), {}           # minmax [0, 2] s -> [-1, 1]


def fake_loader(args, checkpoint, device):
    return None, dict(path=str(checkpoint), sha256="fake")


class PlanTests(unittest.TestCase):
    def test_thirty_second_cover_is_one_window_per_valid_epoch(self):
        valid = np.array([1, 1, 0, 1, 1], dtype=bool)
        windows, uncovered = plan_cover(valid, np.ones(4, dtype=bool), 1)
        self.assertEqual(windows, [(0, [0]), (1, [0]), (3, [0]), (4, [0])])
        self.assertEqual(uncovered, [])

    def test_two_minute_cover_fills_tail_and_holes_once(self):
        valid = np.ones(18, dtype=bool)
        windows, uncovered = plan_cover(valid, np.ones(17, dtype=bool), 4)
        self.assertEqual(windows[:4], [(0, [0, 1, 2, 3]), (4, [0, 1, 2, 3]), (8, [0, 1, 2, 3]), (12, [0, 1, 2, 3])])
        self.assertEqual(windows[4], (14, [2, 3]))
        self.assertEqual(uncovered, [])
        valid[6] = False                                    # breaks the tile 4..7
        windows, uncovered = plan_cover(valid, np.ones(17, dtype=bool), 4)
        self.assertEqual(windows, [(0, [0, 1, 2, 3]), (2, [2, 3]), (7, [0]), (8, [0, 1, 2, 3]),
                                   (12, [0, 1, 2, 3]), (14, [2, 3])])
        self.assertEqual(uncovered, [])
        adjacent = np.ones(17, dtype=bool)
        adjacent[9] = False                                 # time gap between epochs 9 and 10
        windows, uncovered = plan_cover(np.ones(18, dtype=bool), adjacent, 4)
        self.assertEqual(windows, [(0, [0, 1, 2, 3]), (4, [0, 1, 2, 3]), (6, [2, 3]), (10, [0, 1]),
                                   (12, [0, 1, 2, 3]), (14, [2, 3])])
        self.assertEqual(uncovered, [])
        # An isolated valid epoch shorter than one window cannot be generated and is reported.
        windows, uncovered = plan_cover(np.array([1, 1, 0, 1, 1, 1, 1], dtype=bool), np.ones(6, dtype=bool), 4)
        self.assertEqual(windows, [(3, [0, 1, 2, 3])])
        self.assertEqual(uncovered, [0, 1])

    def test_assemble_and_helpers(self):
        windows = [(0, [0, 1]), (1, [1])]
        rows = np.arange(2 * 1500, dtype=np.float64).reshape(2, 1500)
        night, window_id = assemble(windows, rows, 3, 2)
        np.testing.assert_array_equal(night[:1500], rows[0])
        np.testing.assert_array_equal(night[1500:], rows[1, 750:])
        self.assertEqual(window_id[0], 0)
        self.assertEqual(window_id[-1], 1)
        with self.assertRaises(ValueError):
            assemble([(0, [0, 1]), (1, [0])], rows, 3, 2)
        self.assertEqual(default_batch_size(750), 32)
        self.assertEqual(default_batch_size(3000), 8)
        self.assertEqual(subject_seed(1, "S002"), subject_seed(1, "S002"))
        self.assertNotEqual(subject_seed(1, "S002"), subject_seed(1, "S003"))


class GenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.checkpoint = self.base / "model.pth"
        self.checkpoint.write_bytes(b"fake checkpoint")

    def run_generation(self, config, output, *extra):
        cli = gan.parse_args(["--checkpoint", str(self.checkpoint), "--config", str(config),
                              "--output-dir", str(output), *extra])
        with contextlib.redirect_stdout(io.StringIO()):
            return gan.run(cli, sampler=fake_sampler, model_loader=fake_loader, device=torch.device("cpu"))

    def test_thirty_second_event_model_generates_every_night(self):
        config, records, labels = nf.write_dataset(self.base, window_epochs=1)
        output = self.base / "out30"
        report = self.run_generation(config, output, "--label", "30s_apnea")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["model_channels"], 11)
        self.assertEqual(report["batch_size"], 32)
        self.assertEqual(report["completed_subjects"], len(nf.SUBJECTS))
        for sid, *_ in nf.SUBJECTS:
            with np.load(output / "subjects" / f"{sid}.npz") as saved:
                self.assertTrue(saved["covered"].all())
                self.assertEqual(len(saved["window_starts"]), len(nf.RAW_STAGES))
                flags = nf.combined_flags(labels, sid)
                expected = nf.STAGE_IBI[np.repeat(nf.FOUR, 750)] + nf.APNEA_SHIFT * flags
                np.testing.assert_allclose(saved["generated_ibi_seconds"], expected, atol=1e-6)
                np.testing.assert_array_equal(saved["apnea_samples"], flags)
            meta = json.loads((output / "subjects" / f"{sid}.json").read_text())
            self.assertEqual(meta["status"], "completed")
            self.assertAlmostEqual(meta["apnea_effect"]["mean_change_event_samples_ms"], nf.APNEA_SHIFT * 1000, places=3)
            self.assertAlmostEqual(meta["apnea_effect"]["mean_change_non_event_samples_ms"], 0.0, places=6)
            exported = SSDataset(output / "night_npz", "dreamt_ibi", file_list=[f"{sid}.npz"]).read_record(0)
            self.assertEqual(exported["ibi_epochs"].shape, (len(nf.RAW_STAGES), 750))
            self.assertTrue(exported["synthetic"])
        summary_rows = (output / "subjects_summary.csv").read_text().splitlines()
        self.assertEqual(len(summary_rows), 1 + len(nf.SUBJECTS))
        # A resubmitted job skips completed participants with the same checkpoint and seed.
        again = self.run_generation(config, output)
        self.assertEqual(again["completed_subjects"], len(nf.SUBJECTS))
        self.assertTrue(all(json.loads((output / "subjects" / f"{sid}.json").read_text())["status"] == "completed"
                            for sid, *_ in nf.SUBJECTS))

    def test_two_minute_model_without_events_covers_the_tail(self):
        config, _, _ = nf.write_dataset(self.base, window_epochs=4, apnea=False)
        output = self.base / "out2"
        report = self.run_generation(config, output, "--subjects", "S092", "S001", "--label", "2min")
        self.assertEqual(report["model_channels"], 10)
        self.assertEqual(report["batch_size"], 8)
        with np.load(output / "subjects" / "S092.npz") as saved:
            self.assertTrue(saved["covered"].all())
            self.assertEqual(saved["window_starts"].tolist(), [0, 3000, 6000, 9000, 10500])
            self.assertNotIn("apnea_samples", saved.files)
            np.testing.assert_allclose(saved["generated_ibi_seconds"], nf.STAGE_IBI[np.repeat(nf.FOUR, 750)], atol=1e-6)
        meta = json.loads((output / "subjects" / "S092.json").read_text())
        self.assertNotIn("apnea_effect", meta)
        self.assertGreater(meta["pooled_w1_ms"], 0)

    def test_check_only_plans_without_a_model(self):
        config, _, _ = nf.write_dataset(self.base, window_epochs=1)
        cli = gan.parse_args(["--checkpoint", str(self.checkpoint), "--config", str(config),
                              "--output-dir", str(self.base / "plan"), "--check-only"])
        with contextlib.redirect_stdout(io.StringIO()):
            report = gan.run(cli, model_loader=lambda *a: self.fail("model must not load"))
        self.assertEqual(report["status"], "checked_only")
        self.assertEqual(report["total_windows"], len(nf.SUBJECTS) * len(nf.RAW_STAGES))


if __name__ == "__main__":
    unittest.main()
