"""Generate the whole night of every participant with one frozen checkpoint.

Works for any trained window length (30-s or 2-min models) and for models with
or without the respiratory-event condition: the run's own frozen
dataset_config.json decides. Every valid epoch is generated exactly once from
independent windows (night_cover.py); the real IBI never enters the sampler.
Each participant gets its own seed derived from --seed and the participant ID.

Outputs (in --output-dir; reruns skip participants already completed with the
same checkpoint and seed, so a timed-out job can simply be resubmitted):
  subjects/<SID>.npz        generated_ibi_seconds, reference_ibi_seconds, covered, window_id,
                            stage_epochs (W/Light/REM/Deep), apnea_samples (event models)
  subjects/<SID>.json       per-participant metrics at real-observed positions
  subjects/<SID>_apnea_effect.npz   optional: event windows generated with the event
                            channel as recorded and set to 0, with identical noise
  night_npz/<SID>.npz       DREAMT/WatchSleepNet-format synthetic night (values below
                            --missing-threshold-seconds written as the 0 marker)
  subjects_summary.csv, summary.json

Example:
  python scripts/generate_all_nights.py --checkpoint RUN/training/best_kl_model.pth \
      --config RUN/workspace/dataset_config.json --label 30s_apnea --output-dir OUT
"""
from __future__ import annotations

import argparse
import csv
from functools import partial
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

SCHEMA = "dreamt_whole_night_generation_v1"
SUMMARY_FIELDS = ("subject", "status", "epochs", "windows", "uncovered_epochs", "observed_samples",
                  "pooled_w1_ms", "pooled_ks", "mean_bias_ms", "std_ratio", "apnea_effect_ms", "seconds", "error")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="The run's workspace/dataset_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default=None, help="Name stored in the outputs, for example 2min or 30s_apnea")
    parser.add_argument("--subjects", nargs="*", default=None, help="Default: every configured participant")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Windows per sampler call (default: 24000 samples, i.e. 32 x 30 s or 8 x 2 min)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--missing-threshold-seconds", type=float, default=0.3,
                        help="For night_npz only: generated values below this are written as the 0 marker")
    parser.add_argument("--apnea-counterfactual-windows", type=int, default=40,
                        help="Event models: per participant, regenerate up to this many event windows with "
                             "the event channel set to 0 (same noise); 0 disables")
    parser.add_argument("--check-only", action="store_true", help="Plan every night without a model or GPU")
    cli = parser.parse_args(argv)
    if (cli.batch_size is not None and cli.batch_size < 1) or cli.threads < 1 or cli.apnea_counterfactual_windows < 0:
        parser.error("--batch-size and --threads must be positive, --apnea-counterfactual-windows nonnegative")
    if not 0 <= cli.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    return cli


def save_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_npz(path, **arrays):
    temporary = Path(path).with_name("." + Path(path).stem + ".partial.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def all_subjects(preprocessor):
    return sorted(Path(name).stem for names in preprocessor.splits.values() for name in names)


def finite_or_none(value):
    return None if value is None or not np.isfinite(value) else float(value)


def apnea_effect(context, prepared, windows, clean, seed, limit):
    """Event windows with the channel as recorded vs set to 0, generated with identical noise."""
    import subject_windows as sw
    from data_utils import APNEA_INDEX

    flags = clean[:, :, APNEA_INDEX]
    candidates = np.flatnonzero(flags.max(axis=1) > 0)
    if not candidates.size or not limit:
        return None, None
    chosen = candidates[np.unique(np.linspace(0, len(candidates) - 1, min(limit, len(candidates))).round().astype(int))]
    recorded = clean[chosen]
    zeroed = recorded.copy()
    zeroed[:, :, APNEA_INDEX] = 0.0
    kwargs = dict(batch_size=context["batch_size"], seed=seed, sampler=context["sampler"])
    with_events = sw.to_seconds(sw.sample_windows(context["model"], recorded, context["schedule"], context["args"],
                                                  context["device"], **kwargs), context["scaler"])
    without = sw.to_seconds(sw.sample_windows(context["model"], zeroed, context["schedule"], context["args"],
                                              context["device"], **kwargs), context["scaler"])
    event = recorded[:, :, APNEA_INDEX] > 0
    reference = np.stack([prepared["reference_seconds"][windows[k][0] * 750:windows[k][0] * 750 + clean.shape[1]]
                          for k in chosen])
    difference = with_events - without
    summary = dict(windows=int(len(chosen)), event_windows_available=int(len(candidates)),
                   mean_change_event_samples_ms=float(difference[event].mean() * 1000.0),
                   mean_change_non_event_samples_ms=(float(difference[~event].mean() * 1000.0)
                                                     if (~event).any() else None),
                   definition="generated(event channel as recorded) - generated(event channel 0), same noise")
    arrays = dict(window_starts=np.asarray([windows[k][0] * 750 for k in chosen], dtype=np.int64),
                  generated_with_events=with_events.astype(np.float32),
                  generated_events_zeroed=without.astype(np.float32),
                  event=event, reference_ibi_seconds=reference.astype(np.float32), seed=np.array(seed))
    return arrays, summary


def generate_subject(context, sid, cli, output):
    import subject_windows as sw
    from distribution_metrics import pooled_distribution_metrics
    from night_cover import assemble, plan_cover, subject_seed
    from night_data import prepare_night, save_night_npz

    started = time.time()
    prepared = prepare_night(context["preprocessor"], sid, allow_time_gaps=True)
    record, args = prepared["record"], context["args"]
    window_epochs = args.window_size // 750
    stage_epochs = np.asarray(prepared["stage_epochs_four"], dtype=np.int64)
    windows, uncovered = plan_cover(record["valid_epochs"], record["adjacent"], window_epochs)
    meta = dict(schema=SCHEMA, subject=sid, label=cli.label, epochs=int(len(stage_epochs)),
                windows=len(windows), uncovered_epochs=len(uncovered), uncovered_epoch_indices=uncovered[:50],
                window_epochs=window_epochs, checkpoint_sha256=context["checkpoint_sha256"],
                base_seed=cli.seed, seed=subject_seed(cli.seed, sid),
                role=prepared["provenance"]["subject_role"])
    if not windows:
        meta.update(status="no_windows", seconds=round(time.time() - started, 1))
        return meta
    starts = [first * 750 for first, _ in windows]
    clean = sw.build_clean_windows(prepared["conditions"], starts, args.window_size)
    standardized = sw.sample_windows(context["model"], clean, context["schedule"], args, context["device"],
                                     batch_size=context["batch_size"], seed=meta["seed"], sampler=context["sampler"])
    generated_windows = sw.to_seconds(standardized, context["scaler"])
    night, window_id = assemble(windows, generated_windows, len(stage_epochs), window_epochs)
    reference = np.asarray(prepared["reference_seconds"], dtype=np.float64)
    covered = window_id >= 0
    observed = covered & np.isfinite(reference) & (reference > 0)
    stage_samples = np.repeat(stage_epochs, 750)
    arrays = dict(schema=np.array(SCHEMA), subject=np.array(sid),
                  generated_ibi_seconds=night.astype(np.float32), reference_ibi_seconds=reference.astype(np.float32),
                  covered=covered, window_id=window_id, window_starts=np.asarray(starts, dtype=np.int64),
                  window_epochs=np.array(window_epochs), stage_epochs=stage_epochs.astype(np.int8),
                  sample_rate_hz=np.array(25), seed=np.array(meta["seed"]))
    if prepared.get("apnea_samples") is not None:
        arrays["apnea_samples"] = prepared["apnea_samples"].astype(np.uint8)
    if observed.sum() >= 2:
        metrics = pooled_distribution_metrics(night[covered], reference[covered], observed[covered],
                                              stage_samples[covered])
        meta.update(observed_samples=metrics["samples"], pooled_w1_ms=metrics["pooled_w1_ms"],
                    pooled_ks=metrics["pooled_ks"], mean_bias_ms=metrics["mean_bias_ms"],
                    std_ratio=finite_or_none(metrics["std_ratio"]), per_stage=metrics["per_stage"])
    save_npz(output / "subjects" / f"{sid}.npz", **arrays)
    if context["apnea"] and cli.apnea_counterfactual_windows:
        effect_arrays, effect = apnea_effect(context, prepared, windows, clean,
                                             subject_seed(cli.seed + 1, sid), cli.apnea_counterfactual_windows)
        if effect_arrays is not None:
            save_npz(output / "subjects" / f"{sid}_apnea_effect.npz", **effect_arrays)
            meta.update(apnea_effect=effect, apnea_effect_ms=effect["mean_change_event_samples_ms"])
    if not uncovered and covered.all():
        target = output / "night_npz" / f"{sid}.npz"
        partial_path = target.with_name("." + sid + ".partial.npz")
        for path in (partial_path, target):
            if path.exists():
                path.unlink()
        exported = save_night_npz(partial_path, prepared, night,
                                  missing_threshold_seconds=cli.missing_threshold_seconds)
        os.replace(partial_path, target)
        exported["path"] = str(target)
        meta["night_npz"] = exported
    meta.update(status="completed", seconds=round(time.time() - started, 1))
    return meta


def run(cli, *, sampler=None, model_loader=None, device=None):
    """Testable body. sampler/model_loader/device default to the CUDA production path."""
    from data_utils import Preprocessor
    from diffusion_schedule import diffusion_identity
    from night_checkpoint import load_inference_config
    from night_cover import default_batch_size, plan_cover
    from night_data import prepare_night
    from scripts.evaluate_denoising import sha256
    import subject_windows as sw

    output = cli.output_dir.expanduser().resolve()
    (output / "subjects").mkdir(parents=True, exist_ok=True)
    (output / "night_npz").mkdir(exist_ok=True)
    checkpoint = cli.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint file does not exist: " + str(checkpoint))
    args, data_config, config_provenance = load_inference_config(cli.config)
    preprocessor = Preprocessor("DREAMT", data_config=data_config)
    channels = sw.check_preprocessor_columns(preprocessor)
    subjects = cli.subjects or all_subjects(preprocessor)
    batch_size = cli.batch_size or default_batch_size(args.window_size)
    report = dict(schema=SCHEMA, status="starting", label=cli.label, check_only=cli.check_only,
                  checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  config_provenance=config_provenance, model_args=vars(args), model_channels=channels,
                  columns=list(preprocessor.columns), window_size=args.window_size,
                  window_epochs=args.window_size // 750, batch_size=batch_size, seed=cli.seed,
                  diffusion_identity=diffusion_identity(args), scaler=preprocessor.scaler,
                  reference_ibi_passed_to_sampler=False, subjects_requested=len(subjects), output_dir=str(output))
    save_json(output / "summary.json", report)
    if cli.check_only:
        plans = {}
        for sid in subjects:
            prepared = prepare_night(preprocessor, sid, allow_time_gaps=True)
            windows, uncovered = plan_cover(prepared["record"]["valid_epochs"], prepared["record"]["adjacent"],
                                            args.window_size // 750)
            plans[sid] = dict(epochs=len(prepared["stage_epochs_four"]), windows=len(windows),
                              uncovered_epochs=len(uncovered))
        report.update(status="checked_only", plans=plans,
                      total_windows=sum(p["windows"] for p in plans.values()),
                      total_epochs=sum(p["epochs"] for p in plans.values()))
        save_json(output / "summary.json", report)
        print(json.dumps(dict(status="checked_only", subjects=len(plans), total_windows=report["total_windows"])),
              flush=True)
        return report

    import torch
    from diffusion_schedule import build_diffusion_config
    if device is None:
        from cuda_runtime import require_cuda
        device = require_cuda()
    torch.set_num_threads(cli.threads)
    loader = partial(sw.load_frozen_model, channels=channels) if model_loader is None else model_loader
    model, checkpoint_identity = loader(args, checkpoint, device)
    schedule = {key: value.to(device) if torch.is_tensor(value) else value
                for key, value in build_diffusion_config(args).items()}
    context = dict(preprocessor=preprocessor, args=args, model=model, schedule=schedule, device=device,
                   sampler=sampler, batch_size=batch_size, scaler=preprocessor.scaler,
                   checkpoint_sha256=report["checkpoint_sha256"], apnea=channels == 11)
    report.update(status="running", checkpoint_identity=checkpoint_identity)
    save_json(output / "summary.json", report)
    results = {}
    for sid in subjects:
        meta_path = output / "subjects" / f"{sid}.json"
        if meta_path.is_file():
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            if (previous.get("status") == "completed" and previous.get("checkpoint_sha256") == report["checkpoint_sha256"]
                    and previous.get("base_seed") == cli.seed):
                results[sid] = dict(previous, skipped_existing=True)
                continue
        try:
            meta = generate_subject(context, sid, cli, output)
        except Exception as error:  # continue with the other participants
            meta = dict(schema=SCHEMA, subject=sid, status="failed", error=str(error),
                        traceback=traceback.format_exc(), checkpoint_sha256=report["checkpoint_sha256"])
        save_json(meta_path, meta)
        results[sid] = meta
        print(json.dumps({key: meta.get(key) for key in ("subject", "status", "windows", "pooled_w1_ms",
                                                          "apnea_effect_ms", "seconds", "error")}), flush=True)
    with (output / "subjects_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for sid in subjects:
            writer.writerow(results[sid])
    failed = [sid for sid in subjects if results[sid].get("status") == "failed"]
    report.update(status="completed" if not failed else "completed_with_failures", failed_subjects=failed,
                  completed_subjects=sum(results[sid].get("status") == "completed" for sid in subjects))
    save_json(output / "summary.json", report)
    return report


def main(argv=None):
    report = run(parse_args(argv))
    if report.get("failed_subjects"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
