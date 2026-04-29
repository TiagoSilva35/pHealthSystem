#!/usr/bin/env python3
"""
Evaluate beat (R-peak) detection on the MIT-BIH Arrhythmia Database.

Example:
    python src/run_mitdb.py \
        --database mitdb \
        --output mitdb_results \
        --sample-record 208 \
        --tolerance-ms 50
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from src.extract_features import extract_extrasystole_features, save_peak_time_plot
from src.helpers.signal_processing import preprocess_ecg_for_arrhythmia
import wfdb

NON_BEAT_ANNOTATION_SYMBOLS = {
    "+", "~", "|", "[", "]", "x", "(", ")", "!", "p", "t", "u",
    "`", "'", "^", "=", "s", "T", "*", "@", "J", "a", "S", "e", "j", "F"
}


def list_records(db_path: Path):
    """Return all WFDB record basenames found in a directory."""
    return sorted(p.stem for p in db_path.glob("*.hea"))


def load_physiobank_record(record_path: Path):
    """
    Load a WFDB record and return:
        times (seconds), signal (1D numpy array), sampling_rate
    """
    record = wfdb.rdrecord(str(record_path))
    if record.p_signal is None:
        raise ValueError(f"{record_path} has no physical signal (p_signal).")

    signal = record.p_signal[:, 0]  # lead I / channel 0
    sampling_rate = float(record.fs)
    times = np.arange(len(signal), dtype=float) / sampling_rate


    return times, signal, sampling_rate


def load_reference_beats(record_path: Path):
    """
    Load beat annotations from the MIT-BIH atr file.

    Returns:
        ref_samples: np.ndarray[int]
        ref_symbols: list[str]
    """
    ann = wfdb.rdann(str(record_path), "atr")

    ref_samples = []
    ref_symbols = []

    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in NON_BEAT_ANNOTATION_SYMBOLS:
            continue
        ref_samples.append(int(sample))
        ref_symbols.append(symbol)

    return np.asarray(ref_samples, dtype=int), ref_symbols


def match_detections_to_references(ref_samples, det_samples, tol_samples):
    """
    Greedy one-to-one matching between detected peaks and reference beats.

    A detection matches the closest unmatched reference beat within tolerance.

    Returns:
        tp, fp, fn, signed_errors_samples
    """
    ref_samples = np.asarray(ref_samples, dtype=int)
    det_samples = np.asarray(det_samples, dtype=int)

    if len(ref_samples) == 0:
        return 0, int(len(det_samples)), 0, np.asarray([], dtype=float)

    used_ref = np.zeros(len(ref_samples), dtype=bool)
    signed_errors = []

    tp = 0
    fp = 0

    for det in det_samples:
        distances = np.abs(ref_samples - det)
        nearest_idx = int(np.argmin(distances))
        nearest_error = int(det - ref_samples[nearest_idx])

        if distances[nearest_idx] <= tol_samples and not used_ref[nearest_idx]:
            used_ref[nearest_idx] = True
            tp += 1
            signed_errors.append(nearest_error)
        else:
            fp += 1

    fn = int(np.sum(~used_ref))
    return tp, fp, fn, np.asarray(signed_errors, dtype=float)


def evaluate_record(
    record_path: Path,
    min_peak_distance_s=0.06,
    refractory_s=0.12,
    prematurity_threshold=0.95,
    qrs_width_threshold_ms=95.0,
    tolerance_ms=50.0,
):
    """
    Detect beats, compare them against the reference annotations, and return
    a summary dict plus the feature rows.
    """
    times, signal, sampling_rate = load_physiobank_record(record_path)

    preprocessed = preprocess_ecg_for_arrhythmia(
        signal,
        sampling_rate,
        notch_hz=60.0,  
    )
    signal = preprocessed["cleaned"]
    ref_samples, ref_symbols = load_reference_beats(record_path)

    features = extract_extrasystole_features(
        times,
        signal,
        sampling_rate=sampling_rate,
        min_peak_distance_s=min_peak_distance_s,
        prematurity_threshold=prematurity_threshold,
        qrs_width_threshold_ms=qrs_width_threshold_ms,
        refractory_s=refractory_s,
    )

    detected_peak_times_s = np.asarray([row["peak_time_s"] for row in features], dtype=float)
    detected_samples = np.asarray(np.round(detected_peak_times_s * sampling_rate), dtype=int)

    tol_samples = int(round((tolerance_ms / 1000.0) * sampling_rate))
    tp, fp, fn, signed_errors = match_detections_to_references(
        ref_samples, detected_samples, tol_samples
    )

    reference_beats = int(len(ref_samples))
    detected_beats = int(len(detected_samples))

    sensitivity = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2.0 * tp / (2 * tp + fp + fn) * 100.0 if (2 * tp + fp + fn) > 0 else 0.0

    mean_abs_error_ms = (
        float(np.mean(np.abs(signed_errors)) * 1000.0 / sampling_rate)
        if len(signed_errors) > 0
        else np.nan
    )
    median_abs_error_ms = (
        float(np.median(np.abs(signed_errors)) * 1000.0 / sampling_rate)
        if len(signed_errors) > 0
        else np.nan
    )

    summary = {
        "record": record_path.name,
        "reference_beats": reference_beats,
        "detected_beats": detected_beats,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "SE_percent": sensitivity,
        "PPV_percent": ppv,
        "F1_percent": f1,
        "mean_abs_error_ms": mean_abs_error_ms,
        "median_abs_error_ms": median_abs_error_ms,
        "sampling_rate_hz": sampling_rate,
        "duration_s": float(times[-1]) if len(times) else 0.0,
    }

    return summary, features, times, signal


def batch_process_database(
    database_path,
    output_dir,
    skip_plots=False,
    sample_record=None,
    show_plots=False,
    min_peak_distance_s=0.06,
    refractory_s=0.12,
    prematurity_threshold=0.95,
    qrs_width_threshold_ms=95.0,
    tolerance_ms=50.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(database_path)
    records = list_records(db_path)

    if not records:
        raise FileNotFoundError(
            f"No .hea records found in {db_path}. Make sure this is a WFDB database folder."
        )

    print(f"[INFO] Found {len(records)} records in {database_path}")

    summaries = []

    for record_name in records:
        record_path = db_path / record_name

        try:
            print(f"[PROC] {record_name} ...", end=" ", flush=True)

            summary, features, times, signal = evaluate_record(
                record_path,
                min_peak_distance_s=min_peak_distance_s,
                refractory_s=refractory_s,
                prematurity_threshold=prematurity_threshold,
                qrs_width_threshold_ms=qrs_width_threshold_ms,
                tolerance_ms=tolerance_ms,
            )

            summaries.append(summary)

            print(
                f"OK | ref={summary['reference_beats']} "
                f"det={summary['detected_beats']} "
                f"TP={summary['TP']} FP={summary['FP']} FN={summary['FN']} "
                f"SE={summary['SE_percent']:.2f}% PPV={summary['PPV_percent']:.2f}%"
            )

            output_csv = output_dir / f"{record_name}_beat_eval.csv"
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
                writer.writeheader()
                writer.writerow(summary)

            # Save detected beat feature rows too, if you want to inspect them later.
            features_csv = output_dir / f"{record_name}_detected_beats.csv"
            if features:
                feat_df = pd.DataFrame(features)
                feat_df.to_csv(features_csv, index=False)

            if not skip_plots:
                plot_file = output_dir / f"{record_name}_detected_peaks.png"
                save_peak_time_plot(times, signal, features, str(plot_file), show_plot=show_plots)

        except Exception as e:
            print(f"ERROR: {e}")
            summaries.append(
                {
                    "record": record_name,
                    "reference_beats": None,
                    "detected_beats": None,
                    "TP": None,
                    "FP": None,
                    "FN": None,
                    "SE_percent": None,
                    "PPV_percent": None,
                    "F1_percent": None,
                    "mean_abs_error_ms": None,
                    "median_abs_error_ms": None,
                    "sampling_rate_hz": None,
                    "duration_s": None,
                    "error": str(e),
                }
            )

    summary_df = pd.DataFrame(summaries)

    # Add global totals
    ok_df = summary_df.dropna(subset=["reference_beats", "detected_beats", "TP", "FP", "FN"]).copy()
    if not ok_df.empty:
        total_ref = int(ok_df["reference_beats"].sum())
        total_det = int(ok_df["detected_beats"].sum())
        total_tp = int(ok_df["TP"].sum())
        total_fp = int(ok_df["FP"].sum())
        total_fn = int(ok_df["FN"].sum())

        global_se = 100.0 * total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        global_ppv = 100.0 * total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        global_f1 = 2.0 * total_tp / (2 * total_tp + total_fp + total_fn) * 100.0 if (2 * total_tp + total_fp + total_fn) > 0 else 0.0

        global_row = {
            "record": "GLOBAL",
            "reference_beats": total_ref,
            "detected_beats": total_det,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "SE_percent": global_se,
            "PPV_percent": global_ppv,
            "F1_percent": global_f1,
            "mean_abs_error_ms": float(ok_df["mean_abs_error_ms"].mean()),
            "median_abs_error_ms": float(ok_df["median_abs_error_ms"].median()),
            "sampling_rate_hz": "",
            "duration_s": float(ok_df["duration_s"].sum()),
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([global_row])], ignore_index=True)

    summary_csv = output_dir / "database_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n[INFO] Saved summary to {summary_csv}")
    print(f"[INFO] Processed {len(records)} records")

    print("\n[SUMMARY]")
    print(summary_df[[
        "record",
        "reference_beats",
        "detected_beats",
        "TP",
        "FP",
        "FN",
        "SE_percent",
        "PPV_percent",
        "F1_percent",
    ]].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIT-BIH beat detection evaluation"
    )
    parser.add_argument("--database", required=True, help="Path to MIT-BIH WFDB database directory")
    parser.add_argument("--output", default="mitdb_results", help="Output directory for results")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--sample-record", default=None, help="Record name to visualize with extra plots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument(
        "--min-peak-distance",
        type=float,
        default=0.06,
        help="Minimum distance between detected peaks in seconds",
    )
    parser.add_argument(
        "--refractory",
        type=float,
        default=0.12,
        help="Refractory period after an accepted QRS in seconds",
    )
    parser.add_argument(
        "--prematurity-threshold",
        type=float,
        default=0.95,
        help="Prematurity threshold used by your detector",
    )
    parser.add_argument(
        "--qrs-width-threshold-ms",
        type=float,
        default=95.0,
        help="QRS width threshold used by your detector",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=50.0,
        help="Matching tolerance between detected and reference beats in ms",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batch_process_database(
        args.database,
        args.output,
        skip_plots=args.skip_plots,
        sample_record=args.sample_record,
        show_plots=args.show,
        min_peak_distance_s=args.min_peak_distance,
        refractory_s=args.refractory,
        prematurity_threshold=args.prematurity_threshold,
        qrs_width_threshold_ms=args.qrs_width_threshold_ms,
        tolerance_ms=args.tolerance_ms,
    )


if __name__ == "__main__":
    main()