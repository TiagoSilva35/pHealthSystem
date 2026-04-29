import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.helpers.signal_processing import estimate_qrs_width_ms
from src.algorithms.pan_thompkins import PanThompkinsQRS

"""PVC-focused feature extraction from ECG.

Glossary:
- R-peak: highest (or lowest, depending on lead polarity) point of a heartbeat.
- RR interval: time between consecutive R-peaks.
- QRS complex: fast ventricular depolarization segment around the R-peak.
- Prematurity index: how early a beat occurs relative to the baseline RR.
"""


def load_cleaned_ecg_csv(csv_path):
    """Load ECG data and infer sampling rate from the time column.

    The loader accepts either a cleaned signal column named ``ecg_cleaned`` or
    a raw signal column named ``ecg`` so the extractor can be tested directly on
    the raw acquisition CSV.
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.dtype.names is None:
        raise ValueError("CSV must include headers")

    columns = set(data.dtype.names)
    if "time_s" not in columns:
        raise ValueError("CSV must contain a 'time_s' column")

    if "ecg_cleaned" in columns:
        signal = np.asarray(data["ecg_cleaned"], dtype=float)
    elif "ecg" in columns:
        signal = np.asarray(data["ecg"], dtype=float)
    else:
        raise ValueError("CSV must contain either 'ecg_cleaned' or 'ecg' column")

    times = np.asarray(data["time_s"], dtype=float)
    if times.size < 5:
        raise ValueError("CSV must contain at least 5 samples")

    dt = np.median(np.diff(times))
    if dt <= 0:
        raise ValueError("Invalid timestamps in CSV")

    sampling_rate = 1.0 / dt
    return times, signal, sampling_rate


def robust_std(values):
    """Robust spread estimate via MAD (Mitral annular disjunction), less sensitive to spikes/outliers."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return 1.4826 * mad + 1e-9


def detect_r_peaks(ecg, sampling_rate, min_peak_distance_s=0.25, prominence_factor=1.0, refractory_s=0.30):
    """Detect heartbeat peaks using the Pan-Tompkins algorithm.

    The Pan-Tompkins algorithm is specifically designed for ECG QRS detection and
    is more robust than generic peak detection. The prominence_factor parameter
    is kept for API compatibility but is not used in Pan-Tompkins.
    """
    detector = PanThompkinsQRS(fs=sampling_rate)
    signal_df = pd.DataFrame({"time_s": np.arange(ecg.size) / sampling_rate, "ecg": ecg})
    result = detector.solve(signal_df, use_preprocessing=True)
    return np.asarray(result.get("r_peaks", []), dtype=int)


def extract_beat_windows(ecg, peaks, sampling_rate, pre_s=0.20, post_s=0.40):
    """Cut fixed windows around each R-peak for morphology comparison."""
    pre = int(pre_s * sampling_rate)
    post = int(post_s * sampling_rate)

    valid_peaks = []
    windows = []
    for peak in peaks:
        start = peak - pre
        end = peak + post
        if start < 0 or end >= ecg.size:
            continue
        valid_peaks.append(int(peak))
        windows.append(ecg[start:end])

    if not windows:
        return np.empty((0,), dtype=int), np.empty((0, pre + post), dtype=float)

    return np.asarray(valid_peaks, dtype=int), np.vstack(windows)


def compute_local_shape_features(ecg, peak_idx, sampling_rate, window_s=0.06):
    """Compute local morphology descriptors around one beat."""
    half = int(window_s * sampling_rate)
    start = max(0, peak_idx - half)
    end = min(ecg.size - 1, peak_idx + half)
    local = ecg[start : end + 1]
    if local.size < 3:
        return np.nan, np.nan, np.nan

    peak_to_peak = float(np.max(local) - np.min(local))
    centered = local - np.median(local)
    qrs_area = float(np.sum(np.abs(centered)) / sampling_rate)
    max_slope = float(np.max(np.abs(np.diff(local))) * sampling_rate)
    return peak_to_peak, qrs_area, max_slope


def compute_pvc_rule(
    prematurity_index,
    qrs_width_ms,
    prematurity_threshold=0.80,
    qrs_width_threshold_ms=130.0,
    detection_rule="and",
):
    """Apply PVC detection rule and return (candidate_flag, score).
    
    Supported rules:
      - "and": beat is BOTH premature AND wide (strict, original logic)
      - "or":  beat is premature OR wide (looser, catches more)
      - "weighted": probabilistic scoring (0.0 to 1.0)
    """
    cond_premature = np.isfinite(prematurity_index) and prematurity_index < prematurity_threshold
    cond_wide = np.isfinite(qrs_width_ms) and qrs_width_ms > qrs_width_threshold_ms
    
    if detection_rule == "and":
        # Strict: must match both conditions
        candidate = bool(cond_premature and cond_wide)
        # Score ranges 0-1: 0 if neither, 0.5 if one, 1 if both
        score = float((cond_premature + cond_wide) / 2.0)
    elif detection_rule == "or":
        # Loose: match either condition
        candidate = bool(cond_premature or cond_wide)
        score = float((cond_premature + cond_wide) / 2.0)
    elif detection_rule == "weighted":
        # Probabilistic: blend prematurity and QRS width evidence
        prem_score = 0.0
        qrs_score = 0.0
        
        if np.isfinite(prematurity_index):
            # Closer to threshold = higher score
            prem_score = max(0.0, min(1.0, 1.0 - (prematurity_index / prematurity_threshold)))
        
        if np.isfinite(qrs_width_ms):
            # Wider = higher score
            qrs_score = min(1.0, max(0.0, (qrs_width_ms - prematurity_threshold) / 50.0))
        
        score = 0.6 * prem_score + 0.4 * qrs_score  # Prematurity weighted more heavily
        candidate = score > 0.5
    else:
        raise ValueError(f"Unknown detection_rule: {detection_rule}")
    
    return int(candidate), score


def extract_extrasystole_features(
    times,
    ecg_cleaned,
    sampling_rate,
    min_peak_distance_s=0.25,
    prominence_factor=1.0,
    prematurity_threshold=0.80,
    qrs_width_threshold_ms=130.0,
    refractory_s=0.30,
    detection_rule="and",
):
    """Build per-beat PVC-focused features and a simple candidate flag.

    Candidate rule uses two signals together:
    1) beat is early (premature),
    2) QRS is wide.
    
    Parameters:
      - detection_rule: "and" (both conditions), "or" (either), "weighted" (probabilistic)
    """
    peaks = detect_r_peaks(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        min_peak_distance_s=min_peak_distance_s,
        prominence_factor=prominence_factor,
        refractory_s=refractory_s,
    )

    if peaks.size < 3:
        return []

    rr_prev = np.full(peaks.size, np.nan, dtype=float)
    rr_next = np.full(peaks.size, np.nan, dtype=float)

    peak_times = times[peaks]
    rr_values = np.diff(peak_times)
    rr_prev[1:] = rr_values
    rr_next[:-1] = rr_values

    # Baseline RR approximates the patient's local "normal" cycle length.
    rr_baseline = float(np.median(rr_values)) if rr_values.size else np.nan

    features = []
    for i, peak in enumerate(peaks):
        peak_time = float(times[peak])
        rr_p = float(rr_prev[i]) if np.isfinite(rr_prev[i]) else np.nan
        rr_n = float(rr_next[i]) if np.isfinite(rr_next[i]) else np.nan

        prematurity_index = rr_p / rr_baseline if np.isfinite(rr_p) and rr_baseline > 0 else np.nan
        qrs_width_ms = estimate_qrs_width_ms(ecg_cleaned, int(peak), sampling_rate)

        # Apply PVC detection rule
        is_candidate, pvc_score = compute_pvc_rule(
            prematurity_index,
            qrs_width_ms,
            prematurity_threshold=prematurity_threshold,
            qrs_width_threshold_ms=qrs_width_threshold_ms,
            detection_rule=detection_rule,
        )

        features.append(
            {
                "beat_index": i,
                "peak_time_s": peak_time,
                "rr_prev_s": rr_p,
                "rr_next_s": rr_n,
                "rr_baseline_s": rr_baseline,
                "prematurity_index": prematurity_index,
                "qrs_width_ms": qrs_width_ms,
                "pvc_score": pvc_score,
                "is_pvc_candidate": is_candidate,
            }
        )

    return features


def save_features_csv(features, output_csv):
    """Persist extracted beat features to a CSV table."""
    fieldnames = [
        "beat_index",
        "peak_time_s",
        "rr_prev_s",
        "rr_next_s",
        "rr_baseline_s",
        "prematurity_index",
        "qrs_width_ms",
        "pvc_score",
        "is_pvc_candidate",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)


def save_peak_time_plot(times, ecg, features, output_plot=None, show_plot=False):
    """Save an ECG plot highlighting the samples used as peak_time_s."""
    if not features:
        return

    peak_times = np.asarray([row["peak_time_s"] for row in features], dtype=float)
    peak_indices = np.asarray([int(np.argmin(np.abs(times - peak_time))) for peak_time in peak_times], dtype=int)
    peak_values = ecg[peak_indices]
    candidates = np.asarray([row["is_pvc_candidate"] for row in features], dtype=int) > 0

    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(times, ecg, linewidth=0.9, color="#2a9d8f", label="ECG signal")
    axis.scatter(
        peak_times,
        peak_values,
        s=28,
        color="#1f77b4",
        label="Detected peaks",
        zorder=3,
    )

    if np.any(candidates):
        axis.scatter(
            peak_times[candidates],
            peak_values[candidates],
            s=46,
            color="#d62728",
            label="PVC candidates",
            zorder=4,
        )

    for peak_time, peak_value in zip(peak_times[candidates], peak_values[candidates]):
        axis.axvline(peak_time, color="#d62728", alpha=0.16, linewidth=1.0)
        axis.annotate(
            f"{peak_time:.3f}s",
            xy=(peak_time, peak_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#d62728",
        )

    axis.set_title("ECG with peak_time_s markers for PVC-focused analysis")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    # If show_plot requested, display interactively instead of saving to PNG
    if show_plot:
        plt.show()
    else:
        if output_plot:
            figure.savefig(output_plot, dpi=220, bbox_inches="tight")
        plt.close(figure)


def parse_args():
    """CLI options for extraction and rule thresholds."""
    parser = argparse.ArgumentParser(description="Extract PVC-focused features from ECG CSV")
    parser.add_argument("--input", default="ecg_samples_cleaned.csv", help="Input cleaned ECG CSV")
    parser.add_argument("--output", default="pvc_features.csv", help="Output features CSV")
    parser.add_argument("--plot", default="extrasystole_peak_times.png", help="Output ECG plot with peak markers")
    parser.add_argument("--show", action="store_true", help="Display the peak-time plot interactively instead of saving a PNG")
    parser.add_argument("--min-peak-distance", type=float, default=0.25, help="Minimum R-peak spacing in seconds")
    parser.add_argument("--prominence-factor", type=float, default=1.0, help="R-peak prominence multiplier")
    parser.add_argument("--prematurity-threshold", type=float, default=0.80, help="Prematurity index threshold")
    parser.add_argument("--qrs-width-threshold-ms", type=float, default=110.0, help="Wide QRS threshold in ms")
    parser.add_argument("--refractory", type=float, default=0.3, help="Refractory period after an accepted QRS in seconds")
    parser.add_argument("--detection-rule", choices=["and", "or", "weighted"], default="and",
                        help="PVC detection rule: 'and' (both premature AND wide, strict), 'or' (either premature OR wide, loose), 'weighted' (probabilistic)")
    return parser.parse_args()


def main():
    """Run feature extraction end-to-end from cleaned ECG CSV."""
    args = parse_args()
    times, ecg_cleaned, sampling_rate = load_cleaned_ecg_csv(args.input)

    features = extract_extrasystole_features(
        times,
        ecg_cleaned,
        sampling_rate=sampling_rate,
        min_peak_distance_s=args.min_peak_distance,
        prominence_factor=args.prominence_factor,
        prematurity_threshold=args.prematurity_threshold,
        qrs_width_threshold_ms=args.qrs_width_threshold_ms,
        refractory_s=args.refractory,
        detection_rule=args.detection_rule,
    )

    save_features_csv(features, args.output)
    save_peak_time_plot(times, ecg_cleaned, features, args.plot, show_plot=args.show)

    n_candidates = int(sum(row["is_pvc_candidate"] for row in features))
    print(f"[INFO] Sampling rate inferred: {sampling_rate:.2f} Hz")
    print(f"[INFO] Detection rule: {args.detection_rule}")
    print(f"[INFO] Detected beats: {len(features)}")
    print(f"[INFO] PVC candidates: {n_candidates}")
    print(f"[INFO] Saved feature table to {args.output}")
    if args.show:
        print("[INFO] Displayed peak-time plot interactively (no PNG saved)")
    else:
        print(f"[INFO] Saved peak-time plot to {args.plot}")


if __name__ == "__main__":
    main()
