"""Synthetic DREAMT folder with respiratory-event sidecars (shared by the 30-s/apnea tests)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import apnea_labels as al

STAGE_IBI = np.array([0.80, 1.00, 0.90, 1.10])        # W, Light, REM, Deep (seconds)
APNEA_SHIFT = -0.15                                    # synthetic IBI change during an event
RAW_STAGES = np.array([0, 2, 2, 4, 3, 3, 2, 0, 1, 4, 3, 2, 2, 0, 4, 2, 1, 1])
FOUR = np.array([0, 1, 1, 2, 3, 3, 1, 0, 1, 2, 3, 1, 1, 0, 2, 1, 1, 1])
SUBJECTS = (("S001", 5, 22, 30, "F"), ("S002", 40, 30, 55, "M"), ("S003", 15, 26, 45, "F"),
            ("S092", 60, 33, 60, "M"))


def event_flags(index, samples):
    """Four event columns; subject-dependent events, some crossing epoch edges."""
    flags = np.zeros((samples, 4), dtype=np.uint8)
    rng = np.random.default_rng(100 + index)
    for column in range(4):
        for _ in range(2 + index):
            start = int(rng.integers(0, samples - 600))
            flags[start:start + int(rng.integers(250, 600)), column] = 1
    return flags


def write_dataset(directory, *, window_epochs=1, apnea=True, event_columns=None, tail_samples=0, repeat=1):
    """repeat tiles the 18-epoch stage sequence (repeat=3 gives nights long enough for a 10-min Welch PSD)."""
    directory = Path(directory)
    raw_stages, four = np.tile(RAW_STAGES, repeat), np.tile(FOUR, repeat)
    records, labels = directory / "records", directory / "labels"
    records.mkdir()
    labels.mkdir()
    for index, (sid, ahi, *_rest) in enumerate(SUBJECTS):
        samples = len(raw_stages) * 750
        flags = event_flags(index, samples + tail_samples)
        values = np.repeat(STAGE_IBI[four], 750).astype(np.float64) + 0.002 * ahi
        values = np.concatenate([values, np.full(tail_samples, 1.0)])
        values = values + APNEA_SHIFT * flags.max(axis=1)
        values[::750] = 0.0                               # one missing marker per epoch
        stages = np.concatenate([np.repeat(raw_stages, 750), np.full(tail_samples, raw_stages[-1])])
        npz = records / f"{sid}.npz"
        np.savez_compressed(npz, data=values.astype(np.float32)[:, None], stages=stages,
                            fs=np.array(25), ahi=np.array(float(ahi)))
        np.savez_compressed(labels / f"{sid}.npz", schema=np.array(al.SCHEMA), event_flags=flags,
                            event_columns=np.array(al.EVENT_COLUMNS), apnea_any=flags.max(axis=1),
                            stages=stages.astype(np.int8), source_row=np.arange(len(stages)) * 4,
                            source_npz_name=np.array(npz.name), source_npz_sha256=np.array(al.sha256_file(npz)),
                            source_csv_name=np.array("synthetic.csv"), source_csv_sha256=np.array("0" * 64),
                            replay=np.array("{}"), fs=np.array(25))
    with (records / "participant_info.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["SID", "AHI", "BMI", "AGE", "GENDER"])
        writer.writeheader()
        for sid, ahi, bmi, age, gender in SUBJECTS:
            writer.writerow(dict(SID=sid, AHI=ahi, BMI=bmi, AGE=age, GENDER=gender))
    config = dict(data_dir=str(records), window_epochs=window_epochs, missing_value_policy="preserve",
                  signal_normalization=dict(method="minmax", min_seconds=0.0, max_seconds=2.0),
                  evaluation_protocol=dict(mode="all_subjects_train_single_subject_eval", subject="S092",
                                           expected_subjects=len(SUBJECTS)))
    if apnea:
        config["apnea_condition"] = dict(label_dir=str(labels))
        if event_columns is not None:
            config["apnea_condition"]["event_columns"] = list(event_columns)
    path = directory / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, records, labels


def combined_flags(labels, sid, columns=al.EVENT_COLUMNS):
    with np.load(Path(labels) / f"{sid}.npz") as saved:
        flags = saved["event_flags"]
    return flags[:, [al.EVENT_COLUMNS.index(c) for c in columns]].max(axis=1)
