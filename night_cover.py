"""Cover a whole night with independent trained-length windows (evaluation and export only).

Windows are tiled without overlap from the first stored epoch. A tile that
crosses an invalid epoch or a known time gap is skipped, and every valid epoch
it left uncovered (plus the short tail at the end of the night) is filled by
the legal window containing it that covers the most still-uncovered epochs.
From such a filling window only the uncovered epochs are kept, so every epoch
is generated exactly once. With 30-second windows the cover is simply one
window per valid epoch.
"""
from __future__ import annotations

import zlib

import numpy as np

from window_config import EPOCH_SAMPLES


def _legal(valid, adjacent, start, count):
    end = start + count
    return (start >= 0 and end <= len(valid) and bool(valid[start:end].all())
            and (count == 1 or bool(adjacent[start:end - 1].all())))


def plan_cover(valid_epochs, adjacent, window_epochs):
    """Return (windows, uncovered): windows is a list of (first_epoch, kept_epoch_offsets)."""
    valid = np.asarray(valid_epochs, dtype=bool)
    adjacent = np.asarray(adjacent, dtype=bool)
    epochs, count = len(valid), int(window_epochs)
    if count < 1:
        raise ValueError("window_epochs must be positive")
    if len(adjacent) != max(0, epochs - 1):
        raise ValueError("adjacent must have one entry per neighbouring epoch pair")
    covered = np.zeros(epochs, dtype=bool)
    windows = []
    for first in range(0, epochs - count + 1, count):
        if _legal(valid, adjacent, first, count):
            windows.append((first, list(range(count))))
            covered[first:first + count] = True
    for epoch in range(epochs):
        if covered[epoch] or not valid[epoch]:
            continue
        best = None
        for first in range(max(0, epoch - count + 1), min(epoch, epochs - count) + 1):
            if not _legal(valid, adjacent, first, count):
                continue
            gain = int((~covered[first:first + count]).sum())
            if best is None or gain > best[0] or (gain == best[0] and first > best[1]):
                best = (gain, first)
        if best is None:
            continue
        first = best[1]
        kept = [offset for offset in range(count) if not covered[first + offset]]
        windows.append((first, kept))
        covered[[first + offset for offset in kept]] = True
    windows.sort(key=lambda item: item[0])
    uncovered = np.flatnonzero(valid & ~covered).tolist()
    return windows, uncovered


def assemble(windows, generated_windows, total_epochs, window_epochs):
    """Place generated windows [n, window_epochs*750] on the night timeline.

    Returns (night, window_id): NaN and -1 where no window was kept.
    """
    generated_windows = np.asarray(generated_windows, dtype=np.float64)
    size = window_epochs * EPOCH_SAMPLES
    if generated_windows.shape != (len(windows), size):
        raise ValueError("generated_windows must hold one trained-length row per planned window")
    night = np.full(total_epochs * EPOCH_SAMPLES, np.nan)
    window_id = np.full(total_epochs * EPOCH_SAMPLES, -1, dtype=np.int32)
    for index, ((first, kept), row) in enumerate(zip(windows, generated_windows)):
        for offset in kept:
            target = slice((first + offset) * EPOCH_SAMPLES, (first + offset + 1) * EPOCH_SAMPLES)
            if window_id[target][0] != -1:
                raise ValueError("An epoch was planned twice")
            night[target] = row[offset * EPOCH_SAMPLES:(offset + 1) * EPOCH_SAMPLES]
            window_id[target] = index
    return night, window_id


def subject_seed(base_seed, subject):
    """Reproducible per-participant seed, independent of processing order."""
    return (int(base_seed) + zlib.crc32(str(subject).encode("utf-8"))) % (2 ** 32)


def default_batch_size(window_size, samples_per_batch=24000):
    """Same samples per call as the validated training-time sampler (8 x 2 min, 32 x 30 s)."""
    return max(1, samples_per_batch // int(window_size))
