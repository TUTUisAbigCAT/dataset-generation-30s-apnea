import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import apnea_labels as al
from scripts import build_apnea_labels as builder

NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}


def random_stage_labels(rng, epochs, epoch_samples):
    """Stage strings with leading/trailing wake, odd offsets, Missing/P rows and NaN."""
    labels = []
    labels += ["P"] * int(rng.integers(0, 5))
    labels += ["W"] * int(rng.integers(epoch_samples, 3 * epoch_samples))
    for _ in range(epochs):
        stage = NAMES[int(rng.integers(0, 5))]
        labels += [stage] * int(rng.integers(1, 3 * epoch_samples))
        if rng.random() < .15:
            labels += [rng.choice(["Missing", "P", float("nan")])] * int(rng.integers(1, 4))
    labels += ["W"] * int(rng.integers(0, 2 * epoch_samples))
    return labels


class ReplayTests(unittest.TestCase):
    def test_map_stages_matches_dict_get(self):
        raw = np.array(["W", "N1", "N2", "N3", "R", "Missing", "P", float("nan"), "X"], dtype=object)
        self.assertEqual(al.map_stages(raw).tolist(), [0, 1, 2, 3, 4, -1, -1, -1, -1])

    def test_replay_equals_literal_original_loop(self):
        rng = np.random.default_rng(7)
        for trial in range(60):
            epoch_samples = int(rng.choice([8, 12, 20]))
            labels = random_stage_labels(rng, int(rng.integers(3, 25)), epoch_samples)
            mapped = al.map_stages(np.asarray(labels, dtype=object))
            if not np.isin(mapped, [1, 2, 3, 4]).any():
                continue
            try:
                expected_rows, expected_stages = al.reference_epoch_selection(
                    mapped, epoch_samples=epoch_samples, downsample=4)
            except ValueError:  # the original fails on an empty concatenate
                with self.assertRaises(ValueError):
                    al.replay_epoch_selection(mapped, epoch_samples=epoch_samples, downsample=4)
                continue
            rows, stages, details = al.replay_epoch_selection(mapped, epoch_samples=epoch_samples, downsample=4)
            np.testing.assert_array_equal(rows, expected_rows, err_msg=f"trial {trial}")
            np.testing.assert_array_equal(stages, expected_stages, err_msg=f"trial {trial}")
            self.assertEqual(len(rows), details["kept_epochs"] * epoch_samples // 4)

    def test_no_sleep_raises(self):
        with self.assertRaises(ValueError):
            al.replay_epoch_selection(np.array([0, 0, 0, -1]), epoch_samples=2, downsample=1)

    def test_event_flags_numeric_nan_and_text(self):
        frame = pd.DataFrame({"Obstructive_Apnea": [np.nan, 1, 0, np.nan],
                              "Central_Apnea": [np.nan, np.nan, np.nan, 1.0],
                              "Hypopnea": ["x", np.nan, np.nan, np.nan],
                              "Multiple_Events": [np.nan] * 4})
        flags, raw = al.event_flags(frame)
        self.assertEqual(flags.tolist(), [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        self.assertIn("x", raw["Hypopnea"])
        with self.assertRaises(ValueError):
            al.event_flags(frame.drop(columns="Hypopnea"))

    def test_count_runs(self):
        self.assertEqual(al.count_runs([0, 1, 1, 0, 1, 0, 0, 1]), 3)
        self.assertEqual(al.count_runs([]), 0)


def write_subject(folder, sid, rng, *, epochs=14):
    """Raw 100 Hz CSV plus an NPZ built by the literal original pipeline."""
    labels = random_stage_labels(rng, epochs, al.SOURCE_EPOCH_SAMPLES)
    n = len(labels)
    events = {column: np.full(n, np.nan) for column in al.EVENT_COLUMNS}
    for column in al.EVENT_COLUMNS:
        for _ in range(3):
            start = int(rng.integers(0, n - 2000))
            events[column][start:start + int(rng.integers(1000, 2000))] = 1.0
    frame = pd.DataFrame(dict(TIMESTAMP=np.arange(n) / 100.0, BVP=rng.normal(size=n),
                              Sleep_Stage=labels, **events))
    raw_dir, npz_dir = folder / "raw", folder / "npz"
    raw_dir.mkdir(exist_ok=True)
    npz_dir.mkdir(exist_ok=True)
    frame.to_csv(raw_dir / f"{sid}_whole_df.csv", index=False)
    mapped = al.map_stages(frame["Sleep_Stage"].to_numpy(dtype=object))
    rows, stages = al.reference_epoch_selection(mapped, epoch_samples=al.SOURCE_EPOCH_SAMPLES, downsample=4)
    ibi = (rows.astype(np.float32) / 1e6)[:, None]  # any IBI; only length and stages matter
    np.savez(npz_dir / f"{sid}.npz", data=ibi, fs=25, ahi=5.0, stages=stages)
    return frame, rows, raw_dir, npz_dir


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.rng = np.random.default_rng(3)

    def tearDown(self):
        self.temporary.cleanup()

    def test_sidecar_reads_events_at_source_rows_and_binds_to_npz(self):
        frame, rows, raw_dir, npz_dir = write_subject(self.folder, "S001", self.rng)
        arrays, report = al.build_subject_labels(raw_dir / "S001_whole_df.csv", npz_dir / "S001.npz", ahi=5.0)
        expected = np.stack([frame[c].fillna(0).to_numpy()[rows] != 0 for c in al.EVENT_COLUMNS], axis=1)
        np.testing.assert_array_equal(arrays["event_flags"], expected.astype(np.uint8))
        np.testing.assert_array_equal(arrays["source_row"], rows)
        np.testing.assert_array_equal(arrays["apnea_any"], expected.any(axis=1).astype(np.uint8))
        self.assertTrue(report["stages_match"])
        self.assertGreater(report["event_sample_fraction"], 0)
        np.savez(self.folder / "S001.npz", **arrays)
        sha = al.sha256_file(npz_dir / "S001.npz")
        obstructive = al.load_sample_labels(self.folder, "S001.npz", npz_sha256=sha, expected_samples=len(rows),
                                            event_columns=["Obstructive_Apnea"])
        np.testing.assert_array_equal(obstructive, expected[:, 0].astype(np.uint8))
        every = al.load_sample_labels(self.folder, "S001.npz", npz_sha256=sha, expected_samples=(len(rows) + 1) // 2,
                                      event_columns=list(al.EVENT_COLUMNS), downsample_rate=2)
        np.testing.assert_array_equal(every, expected.any(axis=1)[::2].astype(np.uint8))
        with self.assertRaises(ValueError):
            al.load_sample_labels(self.folder, "S001.npz", npz_sha256="0" * 64, expected_samples=len(rows),
                                  event_columns=["Hypopnea"])
        with self.assertRaises(ValueError):
            al.load_sample_labels(self.folder, "S001.npz", npz_sha256=sha, expected_samples=len(rows) + 1,
                                  event_columns=["Hypopnea"])

    def test_stage_or_length_disagreement_is_refused(self):
        _, rows, raw_dir, npz_dir = write_subject(self.folder, "S002", self.rng)
        with np.load(npz_dir / "S002.npz") as saved:
            stages, data = saved["stages"].copy(), saved["data"]
        stages[len(stages) // 2] = (stages[len(stages) // 2] + 1) % 5
        np.savez(npz_dir / "S002.npz", data=data, fs=25, stages=stages)
        with self.assertRaisesRegex(ValueError, "stages differ"):
            al.build_subject_labels(raw_dir / "S002_whole_df.csv", npz_dir / "S002.npz")
        np.savez(npz_dir / "S002.npz", data=data[:-750], fs=25, stages=stages[:-750])
        with self.assertRaisesRegex(ValueError, "replay gives"):
            al.build_subject_labels(raw_dir / "S002_whole_df.csv", npz_dir / "S002.npz")

    def test_script_writes_reports_keeps_existing_and_lists_failures(self):
        for sid in ("S001", "S002", "S003"):
            write_subject(self.folder, sid, self.rng, epochs=8)
        (self.folder / "raw" / "S003_whole_df.csv").unlink()
        pd.DataFrame(dict(SID=["S001", "S002", "S003"], AHI=[5.0, 12.0, 30.0])).to_csv(
            self.folder / "npz" / "participant_info.csv", index=False)
        before = {p.name: p.read_bytes() for p in (self.folder / "npz").glob("*.npz")}
        out = self.folder / "labels"
        summary = builder.run(builder.parse_args(["--raw-dir", str(self.folder / "raw"),
                                                  "--npz-dir", str(self.folder / "npz"),
                                                  "--output-dir", str(out), "--workers", "2"]))
        self.assertEqual(summary["subjects_failed"], ["S003"])
        self.assertEqual(summary["subjects_ok"], 2)
        self.assertTrue((out / "S001.npz").is_file() and not (out / "S003.npz").exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.folder / "npz").glob("*.npz")})
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["subjects"]["S001"]["participant_ahi"], 5.0)
        stamp = (out / "S001.npz").stat().st_mtime_ns
        again = builder.run(builder.parse_args(["--raw-dir", str(self.folder / "raw"),
                                                "--npz-dir", str(self.folder / "npz"),
                                                "--output-dir", str(out), "--subjects", "S001"]))
        self.assertEqual((out / "S001.npz").stat().st_mtime_ns, stamp)
        self.assertEqual(again["subjects"]["S001"]["status"], "kept_existing")
        self.assertEqual(again["subjects"]["S002"]["status"], "written")  # earlier entries survive a subset run
        self.assertEqual(again["subjects"]["S001"]["participant_ahi"], 5.0)
        self.assertEqual(again["subjects_failed"], ["S003"])
        with self.assertRaises(ValueError):
            builder.run(builder.parse_args(["--raw-dir", str(self.folder / "raw"),
                                            "--npz-dir", str(self.folder / "npz"),
                                            "--output-dir", str(self.folder / "npz")]))


class RawFolderTests(unittest.TestCase):
    def test_missing_or_csv_free_raw_folder_fails_before_any_subject(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                builder.csv_index(folder / "absent")
            (folder / "data_64Hz").mkdir()
            (folder / "notes.txt").write_text("x")
            with self.assertRaisesRegex(FileNotFoundError, "No CSV files directly inside"):
                builder.csv_index(folder)
            (folder / "S002_whole_df.csv.gz").write_bytes(b"")
            (folder / "S003_PSG_df.csv").write_text("")
            self.assertEqual(sorted(builder.csv_index(folder)), ["S002", "S003"])

    def test_64_hz_csv_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            frame, _, raw_dir, npz_dir = write_subject(folder, "S001", np.random.default_rng(5), epochs=6)
            frame["TIMESTAMP"] = np.arange(len(frame)) / 64.0
            frame.to_csv(raw_dir / "S001_whole_df.csv", index=False)
            with self.assertRaisesRegex(ValueError, "not 100 Hz"):
                al.build_subject_labels(raw_dir / "S001_whole_df.csv", npz_dir / "S001.npz")


class OptionTests(unittest.TestCase):
    def test_condition_options(self):
        self.assertIsNone(al.condition_options({}))
        self.assertEqual(al.condition_options(dict(apnea_condition=dict(label_dir="x")))["event_columns"],
                         list(al.EVENT_COLUMNS))
        chosen = al.condition_options(dict(apnea_condition=dict(
            label_dir="x", event_columns=["Hypopnea", "Obstructive_Apnea"])))
        self.assertEqual(chosen["event_columns"], ["Obstructive_Apnea", "Hypopnea"])
        for bad in (dict(), dict(label_dir=""), dict(label_dir="x", event_columns=[]),
                    dict(label_dir="x", event_columns=["Snore"]), dict(label_dir="x", other=1), "x"):
            with self.assertRaises(ValueError):
                al.condition_options(dict(apnea_condition=bad))


if __name__ == "__main__":
    unittest.main()
