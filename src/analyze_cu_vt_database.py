#!/usr/bin/env python3
"""Analyze CU Ventricular Tachyarrhythmia records for PVC candidates.

This script provides two workflows:
1) Pan-Tompkins peak detection + RR prematurity PVC candidates.
2) Post-processed ECG + QRS-width-based PVC candidates.

It can process the full database and optionally visualize one selected sample
(1-35) with detected peaks, candidates, and QRS width measurements.
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.helpers.physiobank_loader import list_records, load_physiobank_record
from src.helpers.signal_processing import estimate_qrs_width_ms, preprocess_ecg_for_arrhythmia


def sample_to_record(sample_number):
    if sample_number < 1 or sample_number > 35:
        raise ValueError("Sample number must be between 1 and 35")
    return f"cu{sample_number:02d}"


def _build_rr_prematurity_mask(peak_times_s, prematurity_threshold):
    n = peak_times_s.size
    if n < 3:
        return np.zeros(n, dtype=bool), np.full(n, np.nan, dtype=float), np.nan

    rr_prev = np.full(n, np.nan, dtype=float)
    rr_values = np.diff(peak_times_s)
    rr_prev[1:] = rr_values

    rr_baseline = float(np.median(rr_values))
    if rr_baseline <= 0:
        return np.zeros(n, dtype=bool), rr_prev, rr_baseline

    prematurity_index = rr_prev / rr_baseline
    is_pvc_candidate = np.isfinite(prematurity_index) & (prematurity_index < prematurity_threshold)
    return is_pvc_candidate, rr_prev, rr_baseline


def _load_pan_tompkins_detector(sampling_rate):
    """Load PanTompkinsQRS class from src/pan-thompkins/pan_thompkins.py."""
    module_path = Path(__file__).resolve().parent / "pan-thompkins" / "pan_thompkins.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Pan-Tompkins implementation not found: {module_path}")

    spec = importlib.util.spec_from_file_location("pan_thompkins_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Pan-Tompkins module from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PanTompkinsQRS(fs=sampling_rate)


def _enforce_min_peak_distance(peaks, min_distance_samples):
    if peaks.size == 0:
        return peaks

    selected = [int(peaks[0])]
    for peak in peaks[1:]:
        if int(peak) - selected[-1] >= min_distance_samples:
            selected.append(int(peak))
    return np.asarray(selected, dtype=int)


def analyze_pan_tompkins(signal, times, sampling_rate, min_peak_distance_s, prematurity_threshold):
    detector = _load_pan_tompkins_detector(sampling_rate)
    signal_df = pd.DataFrame({"time_s": times, "ecg": signal})
    result = detector.solve(signal_df, use_preprocessing=True)

    peaks = np.asarray(result.get("r_peaks", []), dtype=int)
    min_distance = max(1, int(min_peak_distance_s * sampling_rate))
    peaks = _enforce_min_peak_distance(peaks, min_distance)

    peak_times = times[peaks] if peaks.size else np.array([], dtype=float)

    candidate_mask, rr_prev, rr_baseline = _build_rr_prematurity_mask(peak_times, prematurity_threshold)

    return {
        "peaks": peaks,
        "peak_times_s": peak_times,
        "peak_values": signal[peaks] if peaks.size else np.array([], dtype=float),
        "candidate_mask": candidate_mask,
        "rr_prev_s": rr_prev,
        "rr_baseline_s": rr_baseline,
        "num_peaks": int(peaks.size),
        "num_candidates": int(np.sum(candidate_mask)),
    }


def analyze_qrs_width(signal, times, sampling_rate, min_peak_distance_s, prematurity_threshold, qrs_width_threshold_ms):
    processed = preprocess_ecg_for_arrhythmia(signal, sampling_rate)
    cleaned = processed["cleaned"]

    detector = _load_pan_tompkins_detector(sampling_rate)
    signal_df = pd.DataFrame({"time_s": times, "ecg": signal})
    pan_result = detector.solve(signal_df, use_preprocessing=True)
    peaks = np.asarray(pan_result.get("r_peaks", []), dtype=int)
    min_distance = max(1, int(min_peak_distance_s * sampling_rate))
    peaks = _enforce_min_peak_distance(peaks, min_distance)

    if peaks.size < 3:
        return {
            "cleaned_signal": cleaned,
            "features": [],
            "peak_indices": np.array([], dtype=int),
            "peak_times_s": np.array([], dtype=float),
            "peak_values": np.array([], dtype=float),
            "qrs_width_ms": np.array([], dtype=float),
            "wide_qrs_mask": np.array([], dtype=bool),
            "combined_mask": np.array([], dtype=bool),
            "num_beats": 0,
            "num_wide_qrs_candidates": 0,
            "num_combined_candidates": 0,
        }

    peak_times = times[peaks]
    rr_prev = np.full(peaks.size, np.nan, dtype=float)
    rr_values = np.diff(peak_times)
    rr_prev[1:] = rr_values
    rr_baseline = float(np.median(rr_values)) if rr_values.size else np.nan

    qrs_widths = np.asarray(
        [estimate_qrs_width_ms(cleaned, int(peak), sampling_rate) for peak in peaks],
        dtype=float,
    )
    prematurity = rr_prev / rr_baseline if np.isfinite(rr_baseline) and rr_baseline > 0 else np.full(peaks.size, np.nan)

    premature_mask = np.isfinite(prematurity) & (prematurity < prematurity_threshold)
    wide_qrs_mask = np.isfinite(qrs_widths) & (qrs_widths > qrs_width_threshold_ms)
    combined_mask = premature_mask & wide_qrs_mask

    features = []
    for i, peak in enumerate(peaks):
        features.append(
            {
                "beat_index": i,
                "peak_time_s": float(times[peak]),
                "rr_prev_s": float(rr_prev[i]) if np.isfinite(rr_prev[i]) else np.nan,
                "rr_next_s": float(rr_values[i]) if i < rr_values.size else np.nan,
                "rr_baseline_s": rr_baseline,
                "prematurity_index": float(prematurity[i]) if np.isfinite(prematurity[i]) else np.nan,
                "qrs_width_ms": float(qrs_widths[i]) if np.isfinite(qrs_widths[i]) else np.nan,
                "is_pvc_candidate": int(bool(combined_mask[i])),
            }
        )

    peak_indices = peaks
    peak_values = cleaned[peak_indices] if peak_indices.size else np.array([], dtype=float)

    return {
        "cleaned_signal": cleaned,
        "features": features,
        "peak_indices": peak_indices,
        "peak_times_s": peak_times,
        "peak_values": peak_values,
        "qrs_width_ms": qrs_widths,
        "wide_qrs_mask": wide_qrs_mask,
        "combined_mask": combined_mask,
        "num_beats": int(len(features)),
        "num_wide_qrs_candidates": int(np.sum(wide_qrs_mask)),
        "num_combined_candidates": int(np.sum(combined_mask)),
    }


def plot_pan_sample(record_name, times, signal, pan_result, output_file=None, show_plot=True):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, signal, color="#2a9d8f", linewidth=0.9, label="ECG")

    if pan_result["num_peaks"] > 0:
        ax.scatter(
            pan_result["peak_times_s"],
            pan_result["peak_values"],
            s=24,
            color="#1f77b4",
            label="Detected peaks",
            zorder=3,
        )

    if pan_result["num_candidates"] > 0:
        mask = pan_result["candidate_mask"]
        ax.scatter(
            pan_result["peak_times_s"][mask],
            pan_result["peak_values"][mask],
            s=42,
            color="#d62728",
            label="PVC candidates (premature RR)",
            zorder=4,
        )

    ax.set_title(f"{record_name} - Pan-Tompkins Peaks and PVC Candidates")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=220, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)


def plot_qrs_width_sample(record_name, times, qrs_result, qrs_width_threshold_ms, output_file=None, show_plot=True):
    cleaned = qrs_result["cleaned_signal"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    axes[0].plot(times, cleaned, color="#2a9d8f", linewidth=0.9, label="Cleaned ECG")

    if qrs_result["peak_times_s"].size:
        axes[0].scatter(
            qrs_result["peak_times_s"],
            qrs_result["peak_values"],
            s=24,
            color="#1f77b4",
            label="Detected beats",
            zorder=3,
        )

    if qrs_result["num_wide_qrs_candidates"] > 0:
        wide = qrs_result["wide_qrs_mask"]
        axes[0].scatter(
            qrs_result["peak_times_s"][wide],
            qrs_result["peak_values"][wide],
            s=42,
            color="#d62728",
            label="Wide-QRS candidates",
            zorder=4,
        )

    axes[0].set_title(f"{record_name} - Post-processed ECG with Detected Beats")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    if qrs_result["peak_times_s"].size:
        axes[1].plot(
            qrs_result["peak_times_s"],
            qrs_result["qrs_width_ms"],
            color="#2ca02c",
            linewidth=1.0,
            marker="o",
            markersize=3,
            label="QRS width (ms)",
        )

    axes[1].axhline(
        qrs_width_threshold_ms,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.0,
        label=f"Threshold ({qrs_width_threshold_ms:.1f} ms)",
    )

    if qrs_result["num_wide_qrs_candidates"] > 0:
        wide = qrs_result["wide_qrs_mask"]
        axes[1].scatter(
            qrs_result["peak_times_s"][wide],
            qrs_result["qrs_width_ms"][wide],
            s=36,
            color="#d62728",
            label="Wide-QRS candidates",
            zorder=4,
        )

    axes[1].set_title(f"{record_name} - Detected QRS Width")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("QRS width (ms)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=220, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)


def save_summary_csv(rows, output_csv):
    fieldnames = [
        "record",
        "sample_number",
        "sampling_rate_hz",
        "num_samples",
        "pan_detected_peaks",
        "pan_pvc_candidates",
        "qrs_detected_beats",
        "qrs_wide_candidates",
        "qrs_combined_pvc_candidates",
        "error",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def parse_sample_number(record_name):
    suffix = record_name[2:] if record_name.startswith("cu") else ""
    try:
        return int(suffix)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze CU VT database with Pan-Tompkins and QRS-width workflows")
    parser.add_argument(
        "--database",
        default="cu-ventricular-tachyarrhythmia-database-1.0.0",
        help="Path to CU VT database folder",
    )
    parser.add_argument(
        "--output-dir",
        default="cu_analysis_results",
        help="Directory to store summary CSV and plots",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample number to visualize (1-35)",
    )
    parser.add_argument(
        "--min-peak-distance",
        type=float,
        default=0.25,
        help="Minimum detected beat spacing in seconds",
    )
    parser.add_argument(
        "--prematurity-threshold",
        type=float,
        default=0.80,
        help="PVC prematurity threshold (RRprev / RRbaseline)",
    )
    parser.add_argument(
        "--qrs-width-threshold-ms",
        type=float,
        default=110.0,
        help="QRS width threshold for wide-QRS candidates",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    database_path = Path(args.database)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = list_records(database_path)
    if not records:
        raise ValueError(f"No records found in {database_path}")

    summary_rows = []

    for record_name in records:
        record_path = database_path / record_name
        sample_number = parse_sample_number(record_name)

        try:
            times, signal, sampling_rate = load_physiobank_record(record_path)

            pan_result = analyze_pan_tompkins(
                signal,
                times,
                sampling_rate,
                min_peak_distance_s=args.min_peak_distance,
                prematurity_threshold=args.prematurity_threshold,
            )
            qrs_result = analyze_qrs_width(
                signal,
                times,
                sampling_rate,
                min_peak_distance_s=args.min_peak_distance,
                prematurity_threshold=args.prematurity_threshold,
                qrs_width_threshold_ms=args.qrs_width_threshold_ms,
            )

            summary_rows.append(
                {
                    "record": record_name,
                    "sample_number": sample_number,
                    "sampling_rate_hz": f"{sampling_rate:.2f}",
                    "num_samples": len(signal),
                    "pan_detected_peaks": pan_result["num_peaks"],
                    "pan_pvc_candidates": pan_result["num_candidates"],
                    "qrs_detected_beats": qrs_result["num_beats"],
                    "qrs_wide_candidates": qrs_result["num_wide_qrs_candidates"],
                    "qrs_combined_pvc_candidates": qrs_result["num_combined_candidates"],
                    "error": "",
                }
            )

            print(
                f"[OK] {record_name}: "
                f"Pan candidates={pan_result['num_candidates']} | "
                f"QRS-wide candidates={qrs_result['num_wide_qrs_candidates']} | "
                f"QRS-combined candidates={qrs_result['num_combined_candidates']}"
            )

            if args.sample is not None and sample_number == args.sample:
                plot_pan_sample(record_name, times, signal, pan_result, show_plot=True)
                plot_qrs_width_sample(
                    record_name,
                    times,
                    qrs_result,
                    args.qrs_width_threshold_ms,
                    show_plot=True,
                )

                print(f"[INFO] Displayed Matplotlib plots for sample {args.sample} ({record_name})")

        except Exception as exc:
            summary_rows.append(
                {
                    "record": record_name,
                    "sample_number": sample_number,
                    "sampling_rate_hz": "",
                    "num_samples": "",
                    "pan_detected_peaks": "",
                    "pan_pvc_candidates": "",
                    "qrs_detected_beats": "",
                    "qrs_wide_candidates": "",
                    "qrs_combined_pvc_candidates": "",
                    "error": str(exc),
                }
            )
            print(f"[ERROR] {record_name}: {exc}")

    summary_csv = output_dir / "cu_database_summary.csv"
    save_summary_csv(summary_rows, summary_csv)

    valid_rows = [row for row in summary_rows if not row.get("error")]
    total_pan = int(sum(int(row["pan_pvc_candidates"]) for row in valid_rows))
    total_qrs_wide = int(sum(int(row["qrs_wide_candidates"]) for row in valid_rows))
    total_qrs_combined = int(sum(int(row["qrs_combined_pvc_candidates"]) for row in valid_rows))

    print("\n[SUMMARY]")
    print(f"Records processed successfully: {len(valid_rows)}/{len(summary_rows)}")
    print(f"Pan-Tompkins PVC candidates (database total): {total_pan}")
    print(f"QRS-width PVC candidates (database total): {total_qrs_wide}")
    print(f"Combined premature+wide PVC candidates (database total): {total_qrs_combined}")
    print(f"Saved summary: {summary_csv}")

    if args.sample is not None:
        sample_record = sample_to_record(args.sample)
        if sample_record not in [row["record"] for row in summary_rows]:
            print(f"[WARN] Requested sample {args.sample} ({sample_record}) was not found.")


if __name__ == "__main__":
    main()
