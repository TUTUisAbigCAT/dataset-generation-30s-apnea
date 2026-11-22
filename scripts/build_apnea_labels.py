"""Build per-sample respiratory-event sidecars for every DREAMT_PIBI_SE NPZ.

For each NPZ (S002.npz, ...) the matching raw CSV (S002_*.csv in --raw-dir) is
read, the WatchSleepNet epoch selection is replayed on its Sleep_Stage column
(see apnea_labels.py), and the four event columns Obstructive_Apnea,
Central_Apnea, Hypopnea and Multiple_Events are read at the CSV row behind
every stored 25 Hz sample. A sidecar is written only if the replayed stages
and length equal the NPZ exactly; the NPZ folder is opened read-only.

Outputs (in --output-dir):
  <SID>.npz        event_flags [samples, 4] uint8, apnea_any, stages, source_row, SHA-256 of NPZ/CSV
  report.json      per-subject checks, event fractions, events per sleep hour vs participant AHI
  report.csv       one row per subject

Existing sidecars are kept (the job can be resubmitted after a timeout).

Example:
  python scripts/build_apnea_labels.py \
      --raw-dir /home/yxtao/scratch/confirmed_datasets/physionet.org/files/dreamt/2.1.0/data_100Hz \
      --npz-dir /home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT_PIBI_SE \
      --output-dir /home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT_PIBI_SE_APNEA
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import apnea_labels as al


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="DREAMT data_100Hz folder with the raw CSVs")
    parser.add_argument("--npz-dir", type=Path, required=True, help="Existing DREAMT_PIBI_SE NPZ folder (read only)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New sidecar folder")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="participant_info.csv (default: NPZ folder, then the raw folder's parent)")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subset, for example S002 S092")
    parser.add_argument("--verify-ibi-epochs", type=int, default=0,
                        help="Also recompute IBI with neurokit2 for this many epochs per subject (needs neurokit2)")
    parser.add_argument("--workers", type=int, default=1, help="Subjects processed in parallel")
    cli = parser.parse_args(argv)
    if cli.verify_ibi_epochs < 0 or cli.workers < 1:
        parser.error("--verify-ibi-epochs must be nonnegative and --workers positive")
    return cli


def process_subject(task):
    """Build and atomically write one sidecar; returns (sid, report). Never raises."""
    sid, csv_path, npz, target, ahi, verify = task
    started = time.time()
    try:
        if csv_path is None:
            raise FileNotFoundError(f"No raw CSV named {sid}_*.csv in the raw folder")
        arrays, report = al.build_subject_labels(csv_path, npz, ahi=ahi, verify_ibi_epochs=verify)
        temporary = Path(target).with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
        report["status"] = "written"
        report["seconds"] = round(time.time() - started, 1)
        return sid, report
    except Exception as error:  # the summary lists every failure
        return sid, dict(subject=sid, status="failed", error=str(error), traceback=traceback.format_exc())


def csv_index(raw_dir):
    index = {}
    for path in sorted(Path(raw_dir).glob("*.csv")):
        sid = path.stem.split("_")[0]
        if sid in index:
            raise ValueError(f"Two raw CSVs for {sid}: {index[sid].name}, {path.name}")
        index[sid] = path
    return index


def participant_ahi(metadata_path):
    if metadata_path is None or not Path(metadata_path).is_file():
        return {}
    import pandas as pd
    frame = pd.read_csv(metadata_path, dtype={"SID": str})
    return {str(sid).strip(): float(ahi) for sid, ahi in zip(frame["SID"], frame["AHI"])}


REPORT_FIELDS = ("subject", "status", "samples", "epochs", "event_sample_fraction", "event_epoch_fraction",
                 "event_runs_total_in_csv", "sleep_hours_in_csv", "events_per_sleep_hour", "participant_ahi",
                 "ibi_epochs_checked", "ibi_epochs_matching", "error")


def run(cli):
    raw_dir, npz_dir, output = (cli.raw_dir.expanduser().resolve(), cli.npz_dir.expanduser().resolve(),
                                cli.output_dir.expanduser().resolve())
    if output == npz_dir:
        raise ValueError("--output-dir must differ from the NPZ folder; NPZ files are never rewritten")
    output.mkdir(parents=True, exist_ok=True)
    metadata = cli.metadata
    if metadata is None:
        for candidate in (npz_dir / "participant_info.csv", raw_dir.parent / "participant_info.csv"):
            if candidate.is_file():
                metadata = candidate
                break
    ahi = participant_ahi(metadata)
    csvs = csv_index(raw_dir)
    npzs = sorted(path for path in npz_dir.glob("*.npz") if path.is_file())
    if cli.subjects:
        wanted = {s.strip().removesuffix(".npz").casefold() for s in cli.subjects}
        npzs = [path for path in npzs if path.stem.casefold() in wanted]
        if len(npzs) != len(wanted):
            raise ValueError("Some requested subjects have no NPZ in --npz-dir")
    previous_report = output / "report.json"
    reports = {}
    if previous_report.is_file():
        reports = json.loads(previous_report.read_text(encoding="utf-8")).get("subjects", {})
    tasks = []
    for npz in npzs:
        sid, target = npz.stem, output / npz.name
        if target.exists():
            reports.setdefault(sid, dict(subject=sid))["status"] = "kept_existing"
            continue
        tasks.append((sid, csvs.get(sid), npz, target, ahi.get(sid), cli.verify_ibi_epochs))

    def record(result):
        sid, report = result
        reports[sid] = report
        print(json.dumps({key: report.get(key) for key in ("subject", "status", "event_sample_fraction",
                                                            "events_per_sleep_hour", "participant_ahi", "error")}),
              flush=True)

    if cli.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(cli.workers, len(tasks))) as pool:
            for result in pool.map(process_subject, tasks):
                record(result)
    else:
        for task in tasks:
            record(process_subject(task))
    rows = []
    for sid in sorted(reports):
        report = reports[sid]
        row = {key: report.get(key) for key in REPORT_FIELDS if key in report}
        if "ibi_check" in report:
            row.update(ibi_epochs_checked=report["ibi_check"]["epochs_checked"],
                       ibi_epochs_matching=report["ibi_check"]["epochs_matching"])
        rows.append(row)
    paired = [(r["events_per_sleep_hour"], r["participant_ahi"]) for r in reports.values()
              if r.get("events_per_sleep_hour") is not None and r.get("participant_ahi") is not None]
    correlation = None
    if len(paired) >= 3:
        values = np.asarray(paired, dtype=np.float64)
        if values[:, 0].std() > 0 and values[:, 1].std() > 0:
            correlation = float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
    summary = dict(schema=al.SCHEMA + "_report", raw_dir=str(raw_dir), npz_dir=str(npz_dir),
                   output_dir=str(output), metadata=None if metadata is None else str(metadata),
                   event_columns=list(al.EVENT_COLUMNS), subjects_total=len(rows),
                   subjects_ok=sum(r["status"] in ("written", "kept_existing") for r in rows),
                   subjects_failed=[r["subject"] for r in rows if r["status"] == "failed"],
                   events_per_sleep_hour_vs_ahi_pearson_r=correlation, subjects=reports)
    (output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(event="done", subjects_ok=summary["subjects_ok"], subjects_total=len(rows),
                          failed=summary["subjects_failed"], ahi_r=correlation)), flush=True)
    return summary


def main(argv=None):
    summary = run(parse_args(argv))
    if summary["subjects_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
