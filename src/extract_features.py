import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

"""Extrasystole feature extraction from cleaned ECG.

Glossary:
- R-peak: highest (or lowest, depending on lead polarity) point of a heartbeat.
- RR interval: time between consecutive R-peaks.
- QRS complex: fast ventricular depolarization segment around the R-peak.
- Prematurity index: how early a beat occurs relative to the baseline RR.
- Compensatory pause: longer pause that can follow an ectopic beat.
"""


def load_cleaned_ecg_csv(csv_path):
    """Load cleaned ECG and infer sampling rate from the time column."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.dtype.names is None:
        raise ValueError("CSV must include headers")

    required = {"time_s", "ecg_cleaned"}
    if not required.issubset(set(data.dtype.names)):
        raise ValueError("CSV must contain 'time_s' and 'ecg_cleaned' columns")

    times = np.asarray(data["time_s"], dtype=float)
    cleaned = np.asarray(data["ecg_cleaned"], dtype=float)
    if times.size < 5:
        raise ValueError("CSV must contain at least 5 samples")

    dt = np.median(np.diff(times))
    if dt <= 0:
        raise ValueError("Invalid timestamps in CSV")

    sampling_rate = 1.0 / dt
    return times, cleaned, sampling_rate


def robust_std(values):
    """Robust spread estimate via MAD (Mitral annular disjunction), less sensitive to spikes/outliers."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return 1.4826 * mad + 1e-9


def detect_r_peaks(ecg, sampling_rate, min_peak_distance_s=0.25, prominence_factor=1.0):
    """Detect heartbeat peaks.

    We test both positive and negative polarity because ECG lead orientation can
    invert the waveform; the detector keeps the polarity with stronger peaks.
    """
    centered = ecg - np.median(ecg)
    min_distance = max(1, int(min_peak_distance_s * sampling_rate))
    min_prominence = prominence_factor * robust_std(centered)

    pos_peaks, pos_props = find_peaks(centered, distance=min_distance, prominence=min_prominence)
    neg_peaks, neg_props = find_peaks(-centered, distance=min_distance, prominence=min_prominence)

    pos_score = float(np.mean(pos_props["prominences"])) if len(pos_peaks) else -np.inf
    neg_score = float(np.mean(neg_props["prominences"])) if len(neg_peaks) else -np.inf

    if pos_score >= neg_score:
        return np.asarray(pos_peaks, dtype=int)
    return np.asarray(neg_peaks, dtype=int)


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


def estimate_qrs_width_ms(ecg, peak_idx, sampling_rate, search_half_window_s=0.12):
    """Approximate QRS width around one beat in milliseconds.

    This estimates the duration of the main ventricular deflection by threshold
    crossings on the local envelope.
    """
    half = int(search_half_window_s * sampling_rate)
    start = max(0, peak_idx - half)
    end = min(ecg.size - 1, peak_idx + half)
    segment = ecg[start : end + 1]
    if segment.size < 5:
        return np.nan

    edge = max(1, int(0.02 * sampling_rate))
    baseline = np.median(np.concatenate((segment[:edge], segment[-edge:])))
    detrended = segment - baseline
    envelope = np.abs(detrended)

    peak_env = float(np.max(envelope))
    if peak_env <= 1e-9:
        return np.nan

    # Search the dominant deflection near the detected R peak.
    center = peak_idx - start
    anchor_half = max(1, int(0.02 * sampling_rate))
    local_left = max(0, center - anchor_half)
    local_right = min(envelope.size, center + anchor_half + 1)
    anchor = local_left + int(np.argmax(envelope[local_left:local_right]))

    thr = 0.10 * peak_env
    left = anchor
    right = anchor

    while left > 0 and envelope[left] >= thr:
        left -= 1
    while right < envelope.size - 1 and envelope[right] >= thr:
        right += 1

    width_samples = right - left
    if width_samples <= 1:
        return np.nan
    return 1000.0 * width_samples / sampling_rate


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


def correlation_with_template(beat_window, template):
    """Morphology similarity between one beat and a normal-beat template."""
    if beat_window.size != template.size:
        return np.nan
    if np.std(beat_window) < 1e-9 or np.std(template) < 1e-9:
        return np.nan
    return float(np.corrcoef(beat_window, template)[0, 1])


def extract_extrasystole_features(
    times,
    ecg_cleaned,
    sampling_rate,
    min_peak_distance_s=0.25,
    prominence_factor=1.0,
    prematurity_threshold=0.80,
    compensatory_threshold=1.10,
    qrs_width_threshold_ms=110.0,
    template_corr_threshold=0.90,
):
    """Build per-beat extrasystole features and a simple candidate flag.

    Candidate rule uses four signals together:
    1) beat is early (premature),
    2) followed by pause,
    3) QRS is wide,
    4) morphology differs from normal template.
    """
    peaks = detect_r_peaks(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        min_peak_distance_s=min_peak_distance_s,
        prominence_factor=prominence_factor,
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

    valid_peaks, beat_windows = extract_beat_windows(ecg_cleaned, peaks, sampling_rate)
    beat_map = {int(p): beat_windows[i] for i, p in enumerate(valid_peaks)}

    # Normal template is built from beats whose RR is near baseline.
    template = None
    if beat_windows.shape[0] > 0:
        normal_idxs = np.where((rr_prev >= 0.85 * rr_baseline) & (rr_prev <= 1.15 * rr_baseline))[0]
        template_candidates = []
        for idx in normal_idxs:
            peak = int(peaks[idx])
            if peak in beat_map:
                template_candidates.append(beat_map[peak])
        if not template_candidates:
            template_candidates = [beat_map[int(p)] for p in valid_peaks]
        template = np.median(np.vstack(template_candidates), axis=0)

    features = []
    for i, peak in enumerate(peaks):
        peak_value = float(ecg_cleaned[peak])
        peak_time = float(times[peak])
        rr_p = float(rr_prev[i]) if np.isfinite(rr_prev[i]) else np.nan
        rr_n = float(rr_next[i]) if np.isfinite(rr_next[i]) else np.nan

        prematurity_index = rr_p / rr_baseline if np.isfinite(rr_p) and rr_baseline > 0 else np.nan
        compensatory_index = rr_n / rr_baseline if np.isfinite(rr_n) and rr_baseline > 0 else np.nan

        qrs_width_ms = estimate_qrs_width_ms(ecg_cleaned, int(peak), sampling_rate)
        peak_to_peak_local, qrs_area, max_slope = compute_local_shape_features(ecg_cleaned, int(peak), sampling_rate)

        corr = np.nan
        if template is not None and int(peak) in beat_map:
            corr = correlation_with_template(beat_map[int(peak)], template)

        cond_premature = np.isfinite(prematurity_index) and prematurity_index < prematurity_threshold
        cond_pause = np.isfinite(compensatory_index) and compensatory_index > compensatory_threshold
        cond_wide = np.isfinite(qrs_width_ms) and qrs_width_ms > qrs_width_threshold_ms
        cond_shape = np.isnan(corr) or corr < template_corr_threshold

        is_extrasystole_candidate = bool(cond_premature and cond_pause and cond_wide and cond_shape)

        features.append(
            {
                "beat_index": i,
                "peak_sample": int(peak),
                "peak_time_s": peak_time,
                "peak_value": peak_value,
                "rr_prev_s": rr_p,
                "rr_next_s": rr_n,
                "rr_baseline_s": rr_baseline,
                "prematurity_index": prematurity_index,
                "compensatory_pause_index": compensatory_index,
                "qrs_width_ms": qrs_width_ms,
                "qrs_peak_to_peak": peak_to_peak_local,
                "qrs_area_abs": qrs_area,
                "qrs_max_slope": max_slope,
                "template_corr": corr,
                "is_extrasystole_candidate": int(is_extrasystole_candidate),
            }
        )

    return features


def save_features_csv(features, output_csv):
    """Persist extracted beat features to a CSV table."""
    fieldnames = [
        "beat_index",
        "peak_sample",
        "peak_time_s",
        "peak_value",
        "rr_prev_s",
        "rr_next_s",
        "rr_baseline_s",
        "prematurity_index",
        "compensatory_pause_index",
        "qrs_width_ms",
        "qrs_peak_to_peak",
        "qrs_area_abs",
        "qrs_max_slope",
        "template_corr",
        "is_extrasystole_candidate",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)


def parse_args():
    """CLI options for extraction and rule thresholds."""
    parser = argparse.ArgumentParser(description="Extract extrasystole-oriented features from cleaned ECG CSV")
    parser.add_argument("--input", default="ecg_samples_cleaned.csv", help="Input cleaned ECG CSV")
    parser.add_argument("--output", default="extrasystole_features.csv", help="Output features CSV")
    parser.add_argument("--min-peak-distance", type=float, default=0.25, help="Minimum R-peak spacing in seconds")
    parser.add_argument("--prominence-factor", type=float, default=1.0, help="R-peak prominence multiplier")
    parser.add_argument("--prematurity-threshold", type=float, default=0.80, help="Prematurity index threshold")
    parser.add_argument("--compensatory-threshold", type=float, default=1.10, help="Compensatory pause threshold")
    parser.add_argument("--qrs-width-threshold-ms", type=float, default=110.0, help="Wide QRS threshold in ms")
    parser.add_argument("--template-corr-threshold", type=float, default=0.90, help="Template correlation threshold")
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
        compensatory_threshold=args.compensatory_threshold,
        qrs_width_threshold_ms=args.qrs_width_threshold_ms,
        template_corr_threshold=args.template_corr_threshold,
    )

    save_features_csv(features, args.output)

    n_candidates = int(sum(row["is_extrasystole_candidate"] for row in features))
    print(f"[INFO] Sampling rate inferred: {sampling_rate:.2f} Hz")
    print(f"[INFO] Detected beats: {len(features)}")
    print(f"[INFO] Extrasystole candidates: {n_candidates}")
    print(f"[INFO] Saved feature table to {args.output}")


if __name__ == "__main__":
    main()
