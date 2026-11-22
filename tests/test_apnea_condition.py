"""Optional eleventh condition channel: respiratory events from the sidecar folder."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import apnea_labels as al
from data_utils import APNEA_INDEX, Preprocessor
from night_checkpoint import load_inference_config
from night_data import prepare_night
import subject_windows as sw
from tests import night_fixture as nf
from tests.test_validation_sampling import TinyDenoiser
from training_utils import fetchModel
from validation_sampling import _sample_windows_from_original, model_input_channels


def processor(config_path):
    return Preprocessor("DREAMT", data_config=json.loads(Path(config_path).read_text()))


class ApneaChannelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.config, self.records, self.labels = nf.write_dataset(self.base)

    def test_disabled_keeps_the_ten_historical_channels(self):
        config = json.loads(self.config.read_text())
        del config["apnea_condition"]
        plain = Preprocessor("DREAMT", data_config=config)
        self.assertEqual(len(plain.columns), 10)
        self.assertNotIn("apnea_condition", plain.data_identity())
        self.assertEqual(plain.window_array("S001.npz", 0, 750).shape, (750, 10))

    def test_eleventh_channel_carries_per_sample_events(self):
        pre = processor(self.config)
        self.assertEqual(list(pre.columns)[-1], "apnea_event")
        self.assertEqual(APNEA_INDEX, 10)
        self.assertEqual(pre.hierarchical_features_cyclic, list(pre.columns[1:]))
        flags = nf.combined_flags(self.labels, "S002")
        for start in (0, 750 * 5, 750 * 17):
            window = pre.window_array("S002.npz", start, 750)
            self.assertEqual(window.shape, (750, 11))
            np.testing.assert_array_equal(window[:, APNEA_INDEX], flags[start:start + 750])
        self.assertTrue(set(np.unique(flags)) == {0, 1})
        training = pre.make_training_dataset(750, 750)
        self.assertEqual(len(training), 4 * len(nf.RAW_STAGES))       # 30-s windows, every epoch
        self.assertEqual(tuple(training[0].shape), (750, 11))
        identity = pre.data_identity()
        self.assertEqual(identity["columns"][-1], "apnea_event")
        self.assertEqual(identity["apnea_condition"]["event_columns"], list(al.EVENT_COLUMNS))
        restored = Preprocessor.from_identity(identity)
        np.testing.assert_array_equal(restored.window_array("S002.npz", 750, 750),
                                      pre.window_array("S002.npz", 750, 750))

    def test_event_column_subset_and_relative_label_dir(self):
        config = json.loads(self.config.read_text())
        config["apnea_condition"] = dict(label_dir="labels", event_columns=["Hypopnea"])
        path = self.base / "subset.json"
        path.write_text(json.dumps(config))
        _, data_config, _ = load_inference_config(path)
        self.assertEqual(data_config["apnea_condition"]["label_dir"], str(self.labels.resolve()))
        pre = Preprocessor("DREAMT", data_config=data_config)
        window = pre.window_array("S003.npz", 0, 750 * 3)
        np.testing.assert_array_equal(window[:, APNEA_INDEX], nf.combined_flags(self.labels, "S003", ["Hypopnea"])[:2250])

    def test_changed_npz_or_missing_sidecar_is_refused(self):
        with np.load(self.records / "S001.npz") as saved:
            arrays = dict(saved)
        arrays["data"] = arrays["data"] * 1.0001
        np.savez_compressed(self.records / "S001.npz", **arrays)
        with self.assertRaisesRegex(ValueError, "different NPZ"):
            processor(self.config)
        shutil.rmtree(self.labels)
        with self.assertRaises(FileNotFoundError):
            processor(self.config)

    def test_night_conditions_and_window_helpers_include_the_event_channel(self):
        pre = processor(self.config)
        prepared = prepare_night(pre, "S092", allow_time_gaps=True)
        self.assertEqual(prepared["conditions"].shape, (len(nf.RAW_STAGES) * 750, 10))
        flags = nf.combined_flags(self.labels, "S092")
        np.testing.assert_array_equal(prepared["conditions"][:, 9], flags)
        np.testing.assert_array_equal(prepared["apnea_samples"], flags)
        self.assertEqual(prepared["provenance"]["condition_columns"][-1], "apnea_event")
        self.assertEqual(sw.check_preprocessor_columns(pre), 11)
        clean = sw.build_clean_windows(prepared["conditions"], [0, 750], 750)
        self.assertEqual(clean.shape, (2, 750, 11))
        np.testing.assert_array_equal(clean[1, :, 10], flags[750:1500])
        np.testing.assert_array_equal(clean[:, :, 0], 0)

    def test_sampler_treats_the_event_channel_as_a_clean_condition(self):
        steps, length = 4, 6
        args = SimpleNamespace(window_size=length, stride=length, timesteps=steps, s4_lmax=length)
        betas = torch.linspace(.0001, .02, steps).reshape(steps, 1)
        schedule = {"T": steps, "betas": betas, "alphas": 1 - betas, "alpha_bars": torch.cumprod(1 - betas, dim=0)}
        clean = torch.rand(3, length, 11)
        clean[:, :, 10] = torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.float32)
        generated, metadata = _sample_windows_from_original(TinyDenoiser().eval(), clean, schedule, args, "cpu")
        self.assertEqual(tuple(generated.shape), (3, length))
        self.assertTrue(metadata["condition_columns_bitwise_preserved"])
        with self.assertRaises(ValueError):
            _sample_windows_from_original(TinyDenoiser().eval(), torch.rand(3, length, 12), schedule, args, "cpu")

    def test_model_width_follows_the_channel_count(self):
        args, _, _ = load_inference_config(self.config)
        args.s4_lmax = args.window_size = 750
        for channels in (10, 11):
            model = fetchModel(channels, 1, args)
            self.assertEqual(model_input_channels(model), channels)
        steps = 2
        betas = torch.linspace(.0001, .02, steps).reshape(steps, 1)
        schedule = {"T": steps, "betas": betas, "alphas": 1 - betas, "alpha_bars": torch.cumprod(1 - betas, dim=0)}
        small = SimpleNamespace(window_size=750, stride=750, timesteps=steps, s4_lmax=750)
        with self.assertRaisesRegex(ValueError, "10 or 11 channels"):
            _sample_windows_from_original(fetchModel(10, 1, args).eval(), torch.zeros(1, 750, 11), schedule, small, "cpu")

    def test_frozen_training_config_keeps_the_apnea_block(self):
        from scripts import run_dreamt
        seen = []

        def no_model_subprocess(command, **kwargs):
            seen.append(json.loads(Path(kwargs["env"]["DREAMT_CONFIG_PATH"]).read_text(encoding="utf-8")))
            return subprocess.CompletedProcess(command, 0)

        args = ["run_dreamt.py", "--config", str(self.config), "--output-dir", str(self.base / "run"),
                "--epochs", "2", "--threads", "2", "--skip-generation"]
        with mock.patch.object(sys, "argv", args), mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(run_dreamt, "require_cuda", return_value="cuda:0"), \
             mock.patch.object(run_dreamt, "cuda_environment", return_value={"synthetic": True}), \
             mock.patch.object(run_dreamt.subprocess, "run", side_effect=no_model_subprocess), \
             contextlib.redirect_stdout(io.StringIO()):
            run_dreamt.main()
        frozen = seen[0]
        self.assertEqual(frozen["window_epochs"], 1)
        self.assertEqual(frozen["apnea_condition"], dict(label_dir=str(self.labels.resolve()),
                                                         event_columns=list(al.EVENT_COLUMNS)))
        preflight = json.loads((self.base / "run" / "preflight.json").read_text())
        self.assertEqual(preflight["training_input_shape"], [750, 11])
        restored = Preprocessor("DREAMT", data_config=frozen)
        self.assertEqual(list(restored.columns)[-1], "apnea_event")


if __name__ == "__main__":
    unittest.main()
