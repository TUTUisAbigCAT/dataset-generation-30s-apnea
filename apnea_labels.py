"""Per-sample respiratory-event labels aligned to the WatchSleepNet DREAMT NPZ files.

The DREAMT_PIBI_SE NPZ files hold IBI and sleep stages only. They were made by
WatchSleepNet ``dreamt_extract_ibi_se_ppg_ibi.py`` from the raw 100 Hz CSVs:

1. map Sleep_Stage with {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "Missing": -1}
   (any other label, such as "P", also maps to -1) and drop every -1 row;
2. trim with the script's own indices: start = (first non-wake row) // 3000,
   end = last non-wake row; the signal keeps [start, end), the stages [start, end];
3. scan 30-second (3000-row) epochs: an epoch whose stage is constant is kept
   and the scan advances 3000 rows, otherwise it advances one row;
4. keep every 4th row of the concatenated epochs (100 Hz -> 25 Hz).

This module replays steps 1-4 on the stage column only, which gives the CSV row
behind every stored 25 Hz sample, and reads the four event columns at those
rows. The NPZ files are never modified: labels are written to a separate
sidecar folder, and each sidecar is accepted only if its replayed stage
sequence and length equal the NPZ exactly. Each sidecar records the NPZ
SHA-256, and the training loader refuses a sidecar whose NPZ has changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "dreamt_respiratory_event_sidecar_v1"
EVENT_COLUMNS = ("Obstructive_Apnea", "Central_Apnea", "Hypopnea", "Multiple_Events")
STAGE_MAPPING = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "Missing": -1}
SOURCE_FS = 100
SOURCE_EPOCH_SAMPLES = 3000
DOWNSAMPLE = 4
CHANNEL_NAME = "apnea_event"


def map_stages(raw_stages):
    """Exactly ``stage_mapping.get(stage, -1)`` for every raw label (NaN -> -1)."""
    return np.fromiter((STAGE_MAPPING.get(stage, -1) for stage in raw_stages),
                       dtype=np.int64, count=len(raw_stages))


def replay_epoch_rows(mapped_stages, *, epoch_samples=SOURCE_EPOCH_SAMPLES):
    """Replay trimming and the epoch scan; return (row_view, stage_view, starts, details).

    Kept epoch k covers row_view[starts[k]:starts[k] + epoch_samples] at the
    source rate. Rows are original CSV row indices (0 = first data row).
    """
    mapped_stages = np.asarray(mapped_stages, dtype=np.int64)
    rows = np.flatnonzero(mapped_stages != -1)
    stages = mapped_stages[rows]
    non_wake = np.flatnonzero(np.isin(stages, [1, 2, 3, 4]))
    if not non_wake.size:
        raise ValueError("No valid sleep stages found in the data.")
    start_idx = max(int(non_wake[0]) // epoch_samples, 0)
    end_idx = min(int(non_wake[-1]), len(stages))
    if start_idx < end_idx:
        signal_length = end_idx - start_idx
        stage_view = stages[start_idx:end_idx + 1]
        row_view = rows[start_idx:end_idx + 1]
        trimmed = True
    else:
        signal_length = len(stages)
        stage_view, row_view, trimmed = stages, rows, False
    # changes[k] counts stage changes up to position k, so the epoch [i, i+E)
    # is constant exactly when changes[i + E - 1] == changes[i].
    changes = np.concatenate(([0], np.cumsum(stage_view[1:] != stage_view[:-1])))
    change_positions = np.flatnonzero(stage_view[1:] != stage_view[:-1]) + 1
    starts = []
    i = 0
    while i + epoch_samples <= signal_length:
        if changes[i + epoch_samples - 1] == changes[i]:
            starts.append(i)
            i += epoch_samples
        else:
            # Every start in (i, c) still contains the last change c inside
            # [i, i+E), so the original one-row steps all fail until c.
            last = np.searchsorted(change_positions, i + epoch_samples - 1, side="right") - 1
            i = max(i + 1, int(change_positions[last]))
    if not starts:
        raise ValueError("No constant-stage epochs were found")
    starts = np.asarray(starts, dtype=np.int64)
    details = dict(valid_rows=int(len(rows)), start_idx=start_idx, end_idx=end_idx, trimmed=trimmed,
                   signal_length=int(signal_length), kept_epochs=int(len(starts)),
                   first_kept_row=int(row_view[starts[0]]),
                   last_kept_row=int(row_view[starts[-1] + epoch_samples - 1]))
    return row_view, stage_view, starts, details


def replay_epoch_selection(mapped_stages, *, epoch_samples=SOURCE_EPOCH_SAMPLES, downsample=DOWNSAMPLE):
    """Return (source_rows, stages, details) for every stored output sample, in stored order."""
    row_view, stage_view, starts, details = replay_epoch_rows(mapped_stages, epoch_samples=epoch_samples)
    positions = (starts[:, None] + np.arange(epoch_samples, dtype=np.int64)[None, :]).reshape(-1)[::downsample]
    return row_view[positions], stage_view[positions], details


def reference_epoch_selection(mapped_stages, *, epoch_samples, downsample):
    """Literal transcription of the original loop (slow; used by tests only)."""
    mapped_stages = np.asarray(mapped_stages)
    valid = mapped_stages != -1
    index = np.arange(len(mapped_stages))[valid]
    stages = mapped_stages[valid]
    non_wake = np.where(np.isin(stages, [1, 2, 3, 4]))[0]
    start_idx = non_wake[0] // epoch_samples
    end_idx = non_wake[-1]
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, len(index))
    if start_idx < end_idx:
        index = index[start_idx:end_idx]
        stages = stages[start_idx:end_idx + 1]
    kept_index, kept_stages = [], []
    i = 0
    while i + epoch_samples <= len(index):
        epoch_stages = stages[i:i + epoch_samples]
        if np.all(epoch_stages == epoch_stages[0]):
            kept_index.append(index[i:i + epoch_samples])
            kept_stages.extend([epoch_stages[0]] * epoch_samples)
            i += epoch_samples
        else:
            i += 1
    kept_index = np.concatenate(kept_index)[::downsample]
    kept_stages = np.asarray(kept_stages)[::downsample]
    return kept_index, kept_stages


def event_flags(frame, columns=EVENT_COLUMNS):
    """uint8 [rows, len(columns)]: 1 where the column holds a nonzero number or any text."""
    import pandas as pd

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV lacks respiratory-event columns {missing}; found {list(frame.columns)}")
    flags = np.zeros((len(frame), len(columns)), dtype=np.uint8)
    raw_values = {}
    for k, column in enumerate(columns):
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        text = raw.notna() & numeric.isna()
        flags[:, k] = ((numeric.fillna(0).to_numpy() != 0) | text.to_numpy()).astype(np.uint8)
        values = pd.unique(raw.dropna())
        raw_values[column] = [str(value) for value in values[:10]]
    return flags, raw_values


def count_runs(flag):
    flag = np.asarray(flag, dtype=bool)
    if not flag.size:
        return 0
    return int(flag[0]) + int(np.count_nonzero(flag[1:] & ~flag[:-1]))


def sha256_file(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def watchsleep_ppg_ibi(segment, fs=SOURCE_FS):
    """WatchSleepNet calculate_ibi_segment(use_package="neurokit") for one epoch."""
    import warnings
    import neurokit2 as nk

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", nk.misc.NeuroKitWarning)
        _, info = nk.ppg_process(segment, sampling_rate=fs)
        for warning in caught:
            if "Too few peaks detected to compute the rate" in str(warning.message):
                return np.zeros_like(segment)
        peaks = info["PPG_Peaks"]
        values = np.diff(peaks) / fs
    ibi = np.zeros_like(segment)
    for k in range(len(peaks) - 1):
        ibi[peaks[k]:peaks[k + 1]] = values[k]
    ibi[(ibi < 0.3) | (ibi >= 2.0)] = 0.0
    return ibi


def check_ibi_epochs(bvp, row_view, starts, stored, count, *, epoch_samples=SOURCE_EPOCH_SAMPLES):
    """Recompute IBI for `count` evenly spaced kept epochs and compare with the NPZ."""
    import neurokit2 as nk

    picks = np.unique(np.linspace(0, len(starts) - 1, min(count, len(starts))).round().astype(np.int64))
    out_epoch = epoch_samples // DOWNSAMPLE
    matches = []
    for k in picks:
        segment = np.asarray(bvp[row_view[starts[k]:starts[k] + epoch_samples]])
        recomputed = watchsleep_ppg_ibi(segment)[::DOWNSAMPLE].astype(np.float32)
        saved = stored[k * out_epoch:(k + 1) * out_epoch]
        matches.append(bool(np.allclose(recomputed, saved, rtol=0.0, atol=1e-6)))
    return dict(epochs_checked=len(picks), epochs_matching=int(sum(matches)),
                checked_epoch_indices=picks.tolist(), neurokit2_version=nk.__version__)


def build_subject_labels(csv_path, npz_path, *, ahi=None, verify_ibi_epochs=0):
    """Return (arrays, report) for one subject or raise ValueError on any mismatch."""
    import pandas as pd

    csv_path, npz_path = Path(csv_path), Path(npz_path)
    wanted = ("Sleep_Stage",) + EVENT_COLUMNS + (("BVP",) if verify_ibi_epochs else ())
    frame = pd.read_csv(csv_path, usecols=lambda name: name in wanted, low_memory=False)
    if "Sleep_Stage" not in frame.columns:
        raise ValueError(f"{csv_path.name} has no Sleep_Stage column")
    mapped = map_stages(frame["Sleep_Stage"].to_numpy(dtype=object))
    row_view, stage_view, starts, details = replay_epoch_rows(mapped)
    positions = (starts[:, None] + np.arange(SOURCE_EPOCH_SAMPLES, dtype=np.int64)[None, :]).reshape(-1)[::DOWNSAMPLE]
    source_rows, replay_stages = row_view[positions], stage_view[positions]
    with np.load(npz_path, allow_pickle=False) as saved:
        stored_ibi = np.asarray(saved["data"]).reshape(-1)
        npz_samples = int(stored_ibi.size)
        npz_stages = np.asarray(saved["stages"]).reshape(-1)
    if npz_stages.size != npz_samples:
        raise ValueError(f"{npz_path.name}: expected sample-level stages")
    if len(replay_stages) != npz_samples:
        raise ValueError(f"{npz_path.name}: replay gives {len(replay_stages)} samples, NPZ has {npz_samples}")
    mismatch = np.flatnonzero(replay_stages != npz_stages.astype(np.int64))
    if mismatch.size:
        raise ValueError(f"{npz_path.name}: replayed stages differ from the NPZ at {mismatch.size} samples "
                         f"(first {mismatch[:5].tolist()})")
    flags_all, raw_values = event_flags(frame)
    flags = flags_all[source_rows]
    any_event = flags.max(axis=1).astype(np.uint8)
    sleep_rows = np.isin(mapped, [1, 2, 3, 4])
    sleep_hours = float(sleep_rows.sum()) / SOURCE_FS / 3600.0
    any_all = flags_all.max(axis=1)
    runs_by_type = {column: count_runs(flags_all[:, k]) for k, column in enumerate(EVENT_COLUMNS)}
    total_runs = sum(runs_by_type.values())
    epochs = npz_samples // 750
    epoch_any = any_event[:epochs * 750].reshape(epochs, 750).max(axis=1) if epochs else np.zeros(0)
    report = dict(
        subject=npz_path.stem, csv=csv_path.name, samples=npz_samples, epochs=epochs,
        stages_match=True, length_match=True, replay=details,
        event_sample_fraction=float(any_event.mean()),
        event_epoch_fraction=float(epoch_any.mean()) if epochs else 0.0,
        event_runs_in_csv=runs_by_type, event_runs_total_in_csv=total_runs,
        event_runs_any_in_stored_timeline=count_runs(any_event),
        event_rows_in_csv=int(any_all.sum()), event_samples_in_stored_timeline=int(any_event.sum()),
        sleep_hours_in_csv=sleep_hours,
        events_per_sleep_hour=(total_runs / sleep_hours) if sleep_hours > 0 else None,
        participant_ahi=None if ahi is None else float(ahi),
        raw_event_values=raw_values,
    )
    if verify_ibi_epochs:
        report["ibi_check"] = check_ibi_epochs(frame["BVP"].to_numpy(), row_view, starts, stored_ibi,
                                               verify_ibi_epochs)
    arrays = dict(
        schema=np.array(SCHEMA), event_flags=flags, event_columns=np.array(EVENT_COLUMNS),
        apnea_any=any_event, stages=npz_stages.astype(np.int8), source_row=source_rows.astype(np.int64),
        source_npz_name=np.array(npz_path.name), source_npz_sha256=np.array(sha256_file(npz_path)),
        source_csv_name=np.array(csv_path.name), source_csv_sha256=np.array(sha256_file(csv_path)),
        replay=np.array(json.dumps(details, sort_keys=True)), fs=np.array(25),
    )
    return arrays, report


def condition_options(config):
    """Validated ``apnea_condition`` block, or None when the channel is disabled."""
    options = config.get("apnea_condition") if isinstance(config, dict) else None
    if options is None:
        return None
    if not isinstance(options, dict) or not set(options) <= {"label_dir", "event_columns"} or "label_dir" not in options:
        raise ValueError("apnea_condition requires label_dir and optionally event_columns")
    label_dir = options["label_dir"]
    if not isinstance(label_dir, str) or not label_dir.strip():
        raise ValueError("apnea_condition.label_dir must be a nonempty path")
    columns = options.get("event_columns", list(EVENT_COLUMNS))
    if (not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)
            or any(column not in EVENT_COLUMNS for column in columns)):
        raise ValueError(f"apnea_condition.event_columns must be a nonempty subset of {list(EVENT_COLUMNS)}")
    return dict(label_dir=label_dir, event_columns=[column for column in EVENT_COLUMNS if column in columns])


def load_sample_labels(label_dir, filename, *, npz_sha256, expected_samples, event_columns, downsample_rate=1):
    """0/1 per stored sample after the loader's downsample stride, bound to one NPZ by SHA-256.

    expected_samples is the NPZ length after that stride (complete epochs plus tail).
    """
    path = Path(label_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing respiratory-event sidecar {path}; run scripts/build_apnea_labels.py")
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["schema"]) != SCHEMA:
            raise ValueError(f"{path.name}: unexpected sidecar schema {saved['schema']}")
        if str(saved["source_npz_sha256"]) != npz_sha256:
            raise ValueError(f"{path.name}: sidecar was built from a different NPZ (SHA-256 differs)")
        stored_columns = [str(value) for value in saved["event_columns"]]
        flags = np.asarray(saved["event_flags"])
    if flags.ndim != 2 or flags.shape[1] != len(stored_columns):
        raise ValueError(f"{path.name}: malformed event flags")
    selected = [stored_columns.index(column) for column in event_columns]
    combined = flags[:, selected].max(axis=1).astype(np.uint8)[::downsample_rate]
    if combined.shape != (expected_samples,):
        raise ValueError(f"{path.name}: event flags do not match the NPZ length")
    return combined
