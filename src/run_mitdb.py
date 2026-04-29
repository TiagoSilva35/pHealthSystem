#!/usr/bin/env python3
#run it: python src/run_mitdb.py --database mitdb --output mitdb_results --sample-record 208
#we kinda need to mess around with parmeters + its normal that we do not reach the value son the db since we are only using 2 features..


"""Batch process PhysioBank database records through the PVC extractor.

Usage:
    python src/run_mitdb.py --database <path> --output <dir>
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from helpers.run_cu_db import load_physiobank_record, list_records
from src.extract_features import (
    extract_extrasystole_features,
    save_peak_time_plot,
)


NON_BEAT_ANNOTATION_SYMBOLS = {'+', '~', '|'}
ANNOTATION_SPLIT_SECONDS = 5 * 60


def summarize_annotations(record_path, sampling_rate):
    atr_path = Path(record_path).with_suffix('.atr')
    if not atr_path.exists():
        return None

    try:
        import wfdb
    except ImportError:
        return None

    ann = wfdb.rdann(str(record_path), 'atr')
    cutoff_sample = int(ANNOTATION_SPLIT_SECONDS * sampling_rate)

    summary = {
        'annotated_beats': 0,
        'annotated_normal_beats': 0,
        'annotated_pvc_beats': 0,
        'annotated_other_beats': 0,
        'annotated_before_5min': 0,
        'annotated_after_5min': 0,
        'annotated_before_normal': 0,
        'annotated_after_normal': 0,
        'annotated_before_pvc': 0,
        'annotated_after_pvc': 0,
    }

    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in NON_BEAT_ANNOTATION_SYMBOLS:
            continue

        summary['annotated_beats'] += 1
        if sample < cutoff_sample:
            summary['annotated_before_5min'] += 1
        else:
            summary['annotated_after_5min'] += 1

        if symbol == 'N':
            summary['annotated_normal_beats'] += 1
            if sample < cutoff_sample:
                summary['annotated_before_normal'] += 1
            else:
                summary['annotated_after_normal'] += 1
        elif symbol == 'V':
            summary['annotated_pvc_beats'] += 1
            if sample < cutoff_sample:
                summary['annotated_before_pvc'] += 1
            else:
                summary['annotated_after_pvc'] += 1
        else:
            summary['annotated_other_beats'] += 1

    return summary


def plot_premature_r_peaks(record_name, times, signal, features, prematurity_threshold, output_file=None, show_plot=True):
    if not features:
        return

    peak_times = np.asarray([row["peak_time_s"] for row in features], dtype=float)
    peak_indices = np.asarray([int(np.argmin(np.abs(times - peak_time))) for peak_time in peak_times], dtype=int)
    peak_values = signal[peak_indices]
    premature_mask = np.asarray([
        np.isfinite(row["prematurity_index"]) and row["prematurity_index"] < prematurity_threshold
        for row in features
    ], dtype=bool)

    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(times, signal, linewidth=0.9, color="#2a9d8f", label="ECG")
    axis.scatter(peak_times, peak_values, s=28, color="#1f77b4", label="Detected peaks", zorder=3)

    if np.any(premature_mask):
        axis.scatter(
            peak_times[premature_mask],
            peak_values[premature_mask],
            s=46,
            color="#d62728",
            label="Premature R-peaks",
            zorder=4,
        )

    axis.set_title(f"{record_name} - Detected peaks and premature R-peaks")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()

    if output_file:
        figure.savefig(output_file, dpi=220, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(figure)


def plot_qrs_width_sample(record_name, features, qrs_width_threshold_ms, output_file=None, show_plot=True):
    if not features:
        return

    peak_times = np.asarray([row["peak_time_s"] for row in features], dtype=float)
    qrs_widths = np.asarray([row["qrs_width_ms"] for row in features], dtype=float)
    wide_mask = np.isfinite(qrs_widths) & (qrs_widths > qrs_width_threshold_ms)

    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(
        peak_times,
        qrs_widths,
        color="#2ca02c",
        linewidth=1.0,
        marker="o",
        markersize=3,
        label="QRS width (ms)",
    )
    axis.axhline(
        qrs_width_threshold_ms,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.0,
        label=f"Threshold ({qrs_width_threshold_ms:.1f} ms)",
    )

    if np.any(wide_mask):
        axis.scatter(
            peak_times[wide_mask],
            qrs_widths[wide_mask],
            s=46,
            color="#d62728",
            label="Wide QRS beats",
            zorder=4,
        )

    axis.set_title(f"{record_name} - Detected QRS width")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("QRS width (ms)")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()

    if output_file:
        figure.savefig(output_file, dpi=220, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(figure)


def batch_process_database(
    database_path,
    output_dir,
    skip_plots=False,
    sample_record=None,
    show_plots=False,
    min_peak_distance_s=0.25,
    refractory_s=0.30,
    prematurity_threshold=0.90,
    qrs_width_threshold_ms=100.0,
):
    """Process all records in a PhysioBank database.
    
    Args:
        database_path: path to the database directory
        output_dir: directory to save results
        skip_plots: if True, skip generating individual plots to save time
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = Path(database_path)
    records = list_records(db_path)
    
    print(f"[INFO] Found {len(records)} records in {database_path}")
    
    all_results = []
    summary_data = []
    
    for record_name in records:
        record_path = db_path / record_name
        
        try:
            print(f"[PROC] Loading {record_name}...", end=" ", flush=True)
            times, signal, sampling_rate = load_physiobank_record(record_path)
            print(f"OK ({len(signal)} samples @ {sampling_rate} Hz)")
            
            print(f"[PROC] Extracting PVC features...", end=" ", flush=True)
            features = extract_extrasystole_features(
                times,
                signal,
                sampling_rate=sampling_rate,
                min_peak_distance_s=min_peak_distance_s,
                prematurity_threshold=prematurity_threshold,
                qrs_width_threshold_ms=qrs_width_threshold_ms,
                refractory_s=refractory_s,
            )
            print(f"OK ({len(features)} beats detected)")

            annotation_summary = summarize_annotations(record_path, sampling_rate)
            if annotation_summary is not None:
                print(
                    "[INFO] Annotation ground truth: "
                    f"beats={annotation_summary['annotated_beats']} | "
                    f"normal={annotation_summary['annotated_normal_beats']} | "
                    f"PVC={annotation_summary['annotated_pvc_beats']} | "
                    f"other={annotation_summary['annotated_other_beats']}"
                )
            
            # Save per-record feature CSV
            output_csv = output_dir / f"{record_name}_pvc_features.csv"
            fieldnames = [
                "beat_index",
                "peak_time_s",
                "rr_prev_s",
                "rr_next_s",
                "rr_baseline_s",
                "prematurity_index",
                "qrs_width_ms",
                "is_pvc_candidate",
            ]
            
            with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(features)
            
            # Generate plot if requested
            if not skip_plots:
                output_plot = output_dir / f"{record_name}_pvc_peaks.png"
                # Pass show_plots so user can request interactive display instead of PNG
                save_peak_time_plot(times, signal, features, str(output_plot), show_plot=show_plots)

                if sample_record is not None and record_name == sample_record:
                    plot_premature_r_peaks(
                        record_name,
                        times,
                        signal,
                        features,
                        prematurity_threshold=prematurity_threshold,
                        output_file=str(output_dir / f"{record_name}_premature_r_peaks.png"),
                        show_plot=show_plots,
                    )
                    plot_qrs_width_sample(
                        record_name,
                        features,
                        qrs_width_threshold_ms=qrs_width_threshold_ms,
                        output_file=str(output_dir / f"{record_name}_qrs_width.png"),
                        show_plot=show_plots,
                    )
            
            # Aggregate statistics
            n_pvcs = sum(1 for row in features if row["is_pvc_candidate"])
            summary_data.append({
                "record": record_name,
                "num_beats": len(features),
                "num_pvcs": n_pvcs,
                "sampling_rate_hz": sampling_rate,
                "duration_s": times[-1] if len(times) > 0 else 0,
                "pvc_rate_percent": 100.0 * n_pvcs / len(features) if len(features) > 0 else 0,
                "annotated_beats": annotation_summary["annotated_beats"] if annotation_summary else None,
                "annotated_normal_beats": annotation_summary["annotated_normal_beats"] if annotation_summary else None,
                "annotated_pvc_beats": annotation_summary["annotated_pvc_beats"] if annotation_summary else None,
                "annotated_other_beats": annotation_summary["annotated_other_beats"] if annotation_summary else None,
                "annotated_before_5min": annotation_summary["annotated_before_5min"] if annotation_summary else None,
                "annotated_after_5min": annotation_summary["annotated_after_5min"] if annotation_summary else None,
                "annotated_before_normal": annotation_summary["annotated_before_normal"] if annotation_summary else None,
                "annotated_after_normal": annotation_summary["annotated_after_normal"] if annotation_summary else None,
                "annotated_before_pvc": annotation_summary["annotated_before_pvc"] if annotation_summary else None,
                "annotated_after_pvc": annotation_summary["annotated_after_pvc"] if annotation_summary else None,
            })
            
            print(f"[DONE] {record_name}: {n_pvcs} PVC candidates out of {len(features)} beats\n")
            
        except Exception as e:
            print(f"ERROR: {e}\n")
            summary_data.append({
                "record": record_name,
                "num_beats": None,
                "num_pvcs": None,
                "sampling_rate_hz": None,
                "duration_s": None,
                "pvc_rate_percent": None,
                "annotated_beats": None,
                "annotated_normal_beats": None,
                "annotated_pvc_beats": None,
                "annotated_other_beats": None,
                "annotated_before_5min": None,
                "annotated_after_5min": None,
                "annotated_before_normal": None,
                "annotated_after_normal": None,
                "annotated_before_pvc": None,
                "annotated_after_pvc": None,
                "error": str(e),
            })
    
    # Save summary CSV
    summary_csv = output_dir / "database_summary.csv"
    summary_fieldnames = [
        "record",
        "num_beats",
        "num_pvcs",
        "sampling_rate_hz",
        "duration_s",
        "pvc_rate_percent",
        "annotated_beats",
        "annotated_normal_beats",
        "annotated_pvc_beats",
        "annotated_other_beats",
        "annotated_before_5min",
        "annotated_after_5min",
        "annotated_before_normal",
        "annotated_after_normal",
        "annotated_before_pvc",
        "annotated_after_pvc",
        "error",
    ]
    
    with open(summary_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames, restval="")
        writer.writeheader()
        writer.writerows(summary_data)
    
    print(f"\n[INFO] Saved database summary to {summary_csv}")
    print(f"[INFO] Processed {len(records)} records")
    
    # Print statistics
    successful = [s for s in summary_data if s["num_beats"] is not None]
    if successful:
        total_beats = sum(s["num_beats"] for s in successful)
        total_pvcs = sum(s["num_pvcs"] for s in successful)
        avg_pvc_rate = 100.0 * total_pvcs / total_beats if total_beats > 0 else 0
        
        print(f"\n[SUMMARY]")
        print(f"  Records processed: {len(successful)}/{len(records)}")
        print(f"  Total beats detected: {total_beats}")
        print(f"  Total PVC candidates: {total_pvcs}")
        print(f"  Average PVC rate: {avg_pvc_rate:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process PhysioBank database records through PVC extractor"
    )
    parser.add_argument(
        "--database", required=True, help="Path to PhysioBank database directory"
    )
    parser.add_argument(
        "--output", default="batch_results", help="Output directory for results"
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip generating individual plots"
    )
    parser.add_argument(
        "--sample-record", default=None, help="Record name to visualize with extra plots (e.g. 208)"
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the sample plots interactively"
    )
    parser.add_argument(
        "--min-peak-distance",
        type=float,
        default=0.06,
        help="Minimum distance between detected candidate peaks (seconds)",
    )
    parser.add_argument(
        "--refractory",
        type=float,
        default=0.12,
        help="Refractory period after an accepted QRS (seconds)",
    )
    parser.add_argument(
        "--prematurity-threshold",
        type=float,
        default=0.95,
        help="Prematurity index threshold for PVC candidates",
    )
    parser.add_argument(
        "--qrs-width-threshold-ms",
        type=float,
        default=95.0,
        help="QRS width threshold in ms for PVC candidates",
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
    )


if __name__ == "__main__":
    main()
