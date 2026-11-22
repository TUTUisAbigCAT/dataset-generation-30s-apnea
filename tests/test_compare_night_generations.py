"""Paired comparison of whole-night generations (CPU, fake samplers)."""
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

from scripts import compare_night_generations as cng
from scripts import generate_all_nights as gan
from tests import night_fixture as nf


AHI = np.array([row[1] for row in nf.SUBJECTS], dtype=np.float64)


def noisy_sampler(noise_seconds, use_events):
    """Stage mean (+ AHI and event terms of the fixture) plus Gaussian noise, in [-1, 1] units."""
    def sample(model, clean, schedule, args, device):
        stages = clean[:, :, 1:5].argmax(dim=2).cpu().numpy()
        seconds = nf.STAGE_IBI[stages] + noise_seconds * torch.randn(stages.shape, dtype=torch.float64).numpy()
        if use_events:  # the better model also follows the AHI condition (z-score over all four participants)
            seconds = seconds + 0.002 * (clean[:, :, 5].cpu().numpy() * AHI.std() + AHI.mean())
        if use_events and clean.shape[2] == 11:
            seconds = seconds + nf.APNEA_SHIFT * clean[:, :, 10].cpu().numpy()
        return torch.as_tensor(seconds - 1.0), {}
    return sample


def fake_loader(args, checkpoint, device):
    return None, dict(path=str(checkpoint), sha256="fake")


class CompareTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.checkpoint = self.base / "model.pth"
        self.checkpoint.write_bytes(b"fake")
        (self.base / "d30").mkdir()
        (self.base / "d2").mkdir()
        self.config30, _, self.labels = nf.write_dataset(self.base / "d30", window_epochs=1, repeat=3)
        self.config2, _, _ = nf.write_dataset(self.base / "d2", window_epochs=4, apnea=False, repeat=3)
        self.out2, self.out30 = self.base / "gen2", self.base / "gen30"
        for config, output, sampler in ((self.config2, self.out2, noisy_sampler(0.10, False)),
                                        (self.config30, self.out30, noisy_sampler(0.02, True))):
            cli = gan.parse_args(["--checkpoint", str(self.checkpoint), "--config", str(config),
                                  "--output-dir", str(output)])
            with contextlib.redirect_stdout(io.StringIO()):
                gan.run(cli, sampler=sampler, model_loader=fake_loader, device=torch.device("cpu"))

    def compare(self, *extra):
        cli = cng.parse_args(["--run", f"2min={self.out2}", "--run", f"30s_apnea={self.out30}",
                              "--output-dir", str(self.base / "compare"), *extra])
        with contextlib.redirect_stdout(io.StringIO()):
            return cng.run(cli)

    def test_better_model_wins_on_every_paired_measure(self):
        summary = self.compare("--apnea-label-dir", str(self.labels))
        self.assertEqual(summary["subjects"], len(nf.SUBJECTS))
        old, new = summary["models"]["2min"], summary["models"]["30s_apnea"]
        self.assertLess(new["pooled"]["w1_ms"], old["pooled"]["w1_ms"])
        self.assertLess(new["per_epoch_w1_ms"]["median"], old["per_epoch_w1_ms"]["median"])
        paired = new["paired_vs_baseline"]["subject_w1_ms"]
        self.assertEqual(paired["improved"], len(nf.SUBJECTS))
        self.assertGreater(new["paired_vs_baseline"]["epoch_w1_ms"]["improved_share"], .5)
        self.assertEqual(set(new["stages"]), {"W", "Light", "REM", "Deep"})
        self.assertIn("event_epochs", new["events"])
        # The synthetic real IBI drops by 150 ms during events; only the event model reproduces it.
        self.assertAlmostEqual(new["event_effect"]["real_mean_ms"], nf.APNEA_SHIFT * 1000, delta=40)
        self.assertLess(new["event_effect"]["mean_abs_error_ms"], old["event_effect"]["mean_abs_error_ms"])
        self.assertAlmostEqual(new["counterfactual_event_channel"]["mean_change_event_samples_ms"],
                               nf.APNEA_SHIFT * 1000, delta=5)
        self.assertNotIn("counterfactual_event_channel", old)
        for band in ("VLF", "LF", "HF"):
            self.assertIn(band, new["spectrum_generated_minus_real"])
        self.assertGreater(new["seams"]["boundaries"], old["seams"]["boundaries"])  # 30-s windows: 4x more seams
        folder = self.base / "compare"
        for name in ("summary.json", "per_subject.csv", "report.md", "distribution_all.png", "spectrum_all.png",
                     "subject_w1_2min_vs_30s_apnea.png", "w1_by_stage_and_event.png", "event_effect.png"):
            self.assertTrue((folder / name).is_file(), name)
        report = (folder / "report.md").read_text()
        self.assertIn("| pooled W1 (ms) |", report)
        json.loads((folder / "summary.json").read_text())

    def test_event_strata_fall_back_to_the_event_model_and_inputs_are_checked(self):
        summary = self.compare("--no-figures")
        self.assertEqual(summary["event_strata_source"], "apnea_samples of an event model")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cng.parse_args(["--run", f"a={self.out2}", "--output-dir", str(self.base / "x")])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cng.parse_args(["--run", f"a={self.out2}", "--run", f"a={self.out30}", "--output-dir", "x"])

    def test_runs_on_different_real_data_are_refused(self):
        path = self.out30 / "subjects" / "S001.npz"
        with np.load(path) as saved:
            arrays = dict(saved)
        arrays["reference_ibi_seconds"] = arrays["reference_ibi_seconds"] + 0.01
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "disagree on the real IBI"):
            self.compare("--no-figures")


if __name__ == "__main__":
    unittest.main()
