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


def compute_local_shape_features(ecg, peak_idx, sampling_rate, window_s=0.08):
    """
    Computes morphology descriptors including QRS width (ms) in addition
    to the Area‑to‑Amplitude ratio.
    """
    half = int(window_s * sampling_rate)
    start = max(0, peak_idx - half)
    end = min(ecg.size - 1, peak_idx + half)
    local = ecg[start : end + 1]

    if local.size < 3:
        return 0.0, 0.0, 0.0, np.nan

    # Peak-to-peak amplitude
    peak_to_peak = float(np.max(local) - np.min(local))

    # Area (Integral of absolute centred signal)
    centred = local - np.median(local)
    qrs_area = float(np.sum(np.abs(centred)) / sampling_rate)

    # Area-to-Amplitude Ratio
    aa_ratio = qrs_area / (peak_to_peak + 1e-9)

    # True QRS width (ms) – uses the existing reliable estimator
    qrs_width = estimate_qrs_width_ms(ecg, peak_idx, sampling_rate)

    return peak_to_peak, qrs_area, aa_ratio, qrs_width


def robust_baseline(values):
    """
    Robust central tendency: median of the values between the 25th and 75th percentile.
    This ignores frequent outliers (e.g. PVCs) that would otherwise corrupt a simple median.
    """
    if len(values) < 2:
        return np.nan
    q1, q3 = np.percentile(values, [25, 75])
    central = values[(values >= q1) & (values <= q3)]
    if central.size == 0:
        return np.median(values)
    return np.median(central)


def compute_pvc_rule(
    prematurity_index,
    qrs_width_ms,
    morphology_score=None,
    prematurity_threshold=0.80,
    qrs_width_threshold_ms=130.0,
    detection_rule="and",
    morph_threshold=0.30,
):
    """Apply PVC detection rule and return (candidate_flag, score).
    
    Supported rules:
      - "and": beat is BOTH premature AND wide (strict, original logic)
      - "or":  beat is premature OR wide (looser, catches more)
    - "weighted": probabilistic scoring (0.0 to 1.0) with continuous evidence accumulation
    """
    cond_premature = np.isfinite(prematurity_index) and prematurity_index < prematurity_threshold
    cond_wide = np.isfinite(qrs_width_ms) and qrs_width_ms > qrs_width_threshold_ms
    cond_morph = False
    morph_evidence_score = 0.0
    if morphology_score is not None and np.isfinite(morphology_score):
        # morphology_score is similarity to a median-normal QRS (1.0 identical, 0.0 opposite)
        morph_evidence_score = float(1.0 - float(morphology_score))
        cond_morph = morph_evidence_score > float(morph_threshold)
    
    if detection_rule == "and":
        # Strict: must match both primary conditions. If morphology is available, include it in the
        # aggregated score but keep strict candidate logic based on prematurity+width.
        candidate = bool(cond_premature and cond_wide)
        denom = 2.0 + (1.0 if (morphology_score is not None and np.isfinite(morphology_score)) else 0.0)
        score = float((float(cond_premature) + float(cond_wide) + (float(cond_morph) if denom > 2.0 else 0.0)) / denom)
    elif detection_rule == "or":
        # Loose: match either primary condition. Average available evidence for score.
        candidate = bool(cond_premature or cond_wide)
        denom = 2.0 + (1.0 if (morphology_score is not None and np.isfinite(morphology_score)) else 0.0)
        score = float((float(cond_premature) + float(cond_wide) + (float(cond_morph) if denom > 2.0 else 0.0)) / denom)
    elif detection_rule == "weighted":
        # Continuous evidence accumulation across prematurity, width and morphology-evidence.
        prem_scale = max(0.05, 0.15 * prematurity_threshold)
        qrs_scale = max(5.0, 0.30 * qrs_width_threshold_ms)

        prem_score = 0.0
        if np.isfinite(prematurity_index):
            prem_score = 1.0 / (1.0 + np.exp((prematurity_index - prematurity_threshold) / prem_scale))

        qrs_score = 0.0
        if np.isfinite(qrs_width_ms):
            qrs_score = 1.0 / (1.0 + np.exp(-(qrs_width_ms - qrs_width_threshold_ms) / qrs_scale))

        # morphology contributes as "evidence of abnormal morphology" = 1 - similarity
        morph_score = 0.0
        if morphology_score is not None and np.isfinite(morphology_score):
            morph_score = morph_evidence_score

        # weights chosen to prioritise prematurity, then width, then morphology
        score = 0.55 * prem_score + 0.30 * qrs_score + 0.15 * morph_score
        candidate = score > 0.50
    else:
        raise ValueError(f"Unknown detection rule: {detection_rule}")

    return int(candidate), score


def extract_extrasystole_features(
    times,
    ecg_cleaned,
    sampling_rate,
    min_peak_distance_s=0.25,
    prominence_factor=1.0,
    refractory_s=0.30,
    detection_rule="weighted",
    window_size=20
):
    """Build per-beat PVC-focused features and a simple candidate flag.

    Candidate rule uses two signals together:
    1) beat is early (premature),
    2) QRS is wide.
    3) Optional morphology evidence (weighted in "weighted" mode)
    
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

    peak_times = times[peaks]
    n_beats = len(peaks)
    rr_values = np.diff(peak_times)  # length n_beats-1

    # 2. Pre‑compute morphology for every beat
    aa_ratios = np.zeros(n_beats)
    qrs_widths = np.full(n_beats, np.nan)
    for i, p in enumerate(peaks):
        _, _, aa_ratio_i, qrs_w = compute_local_shape_features(ecg_cleaned, p, sampling_rate)
        aa_ratios[i] = aa_ratio_i
        qrs_widths[i] = qrs_w


    global_rr_baseline = robust_baseline(rr_values)
    global_aa_baseline = robust_baseline(aa_ratios)
    global_width_baseline = robust_baseline(qrs_widths[~np.isnan(qrs_widths)])

    # Compute per-beat QRS morphology similarity to a median template built from the same signal.
    # Use iterative template refinement and small-shift alignment to make the template robust
    # against outliers and minor alignment errors.
    morph_window_s = 0.12
    half_samples = int(morph_window_s * sampling_rate)
    seg_len = 2 * half_samples + 1
    segments = []
    valid = []
    for peak in peaks:
        start = int(peak) - half_samples
        end = int(peak) + half_samples
        if start >= 0 and end < ecg_cleaned.size:
            segments.append(np.asarray(ecg_cleaned[start : end + 1], dtype=float))
            valid.append(True)
        else:
            segments.append(np.full(seg_len, np.nan))
            valid.append(False)

    segments = np.asarray(segments)
    valid_mask = np.asarray(valid, dtype=bool)
    morph_scores = np.full(peaks.size, np.nan)

    if np.sum(valid_mask) > 0:
        # Parameters for refinement
        max_iters = 3
        refine_frac = 0.25  # drop bottom 25% of segments by similarity each iter
        max_shift = max(1, int(0.02 * sampling_rate))  # allow +-20 ms alignment

        # Start with the raw valid segments
        aligned_segments = [segments[i].copy() for i in range(len(segments)) if valid_mask[i]]

        for it in range(max_iters):
            template = np.median(np.vstack(aligned_segments), axis=0)
            tmpl = template - np.mean(template)
            tmpl_norm = float(np.linalg.norm(tmpl))

            corrs = []
            new_aligned = []
            for seg in aligned_segments:
                seg_z = seg - np.mean(seg)
                best_corr = -1.0
                best_shift = 0
                # search small shifts to compensate for mis-centering
                for shift in range(-max_shift, max_shift + 1):
                    s_shift = np.roll(seg_z, shift)
                    s_norm = float(np.linalg.norm(s_shift))
                    if tmpl_norm > 0.0 and s_norm > 0.0:
                        c = float(np.dot(s_shift, tmpl) / (s_norm * tmpl_norm))
                    else:
                        c = 0.0
                    if c > best_corr:
                        best_corr = c
                        best_shift = shift

                # apply best shift to the original (not zero-mean) seg and store mean-removed version
                seg_aligned = np.roll(seg, best_shift)
                seg_aligned = seg_aligned - np.mean(seg_aligned)
                new_aligned.append(seg_aligned)
                corrs.append(best_corr)

            corrs = np.asarray(corrs, dtype=float)
            # If refinement requested, drop lowest similarity fraction and continue
            if refine_frac > 0 and it < max_iters - 1 and corrs.size > 0:
                thresh = np.percentile(corrs, 100.0 * refine_frac)
                keep_mask = corrs >= thresh
                if np.all(keep_mask):
                    aligned_segments = new_aligned
                    break
                aligned_segments = [s for k, s in zip(keep_mask, new_aligned) if k]
                if len(aligned_segments) == 0:
                    # can't refine further
                    aligned_segments = new_aligned
                    break
            else:
                aligned_segments = new_aligned
                break

        # Final template and per-segment similarity (map from -1..1 to 0..1)
        final_template = np.median(np.vstack(aligned_segments), axis=0)
        final_t = final_template - np.mean(final_template)
        final_t_norm = float(np.linalg.norm(final_t))

        # compute score for each original valid segment (with alignment)
        idx_valid = np.flatnonzero(valid_mask)
        for idx_pos, seg_idx in enumerate(idx_valid):
            seg = segments[seg_idx]
            seg_z = seg - np.mean(seg)
            # find best small shift against final template
            best_corr = -1.0
            for shift in range(-max_shift, max_shift + 1):
                s_shift = np.roll(seg_z, shift)
                s_norm = float(np.linalg.norm(s_shift))
                if final_t_norm > 0.0 and s_norm > 0.0:
                    c = float(np.dot(s_shift, final_t) / (s_norm * final_t_norm))
                else:
                    c = 0.0
                if c > best_corr:
                    best_corr = c

            # map correlation [-1,1] -> [0,1]; higher = more similar
            morph_scores[seg_idx] = float((best_corr + 1.0) / 2.0) if best_corr is not None else np.nan

    features = []
    for i in range(n_beats):
        # Current RR (preceding beat)
        curr_rr = rr_values[i - 1] if i > 0 else np.nan
        # Compensatory pause: RR to next beat
        curr_next_rr = rr_values[i] if i < n_beats - 1 else np.nan
        curr_aa = aa_ratios[i]
        curr_width = qrs_widths[i]

        # Indices relative to global robust baseline
        prem_index = curr_rr / global_rr_baseline if not np.isnan(curr_rr) and global_rr_baseline > 0 else np.nan
        morph_index = curr_aa / (global_aa_baseline + 1e-9)
        width_index = curr_width / (global_width_baseline + 1e-9) if not np.isnan(curr_width) and global_width_baseline > 0 else np.nan
        pause_index = curr_next_rr / global_rr_baseline if not np.isnan(curr_next_rr) and global_rr_baseline > 0 else np.nan

        # Vote using the chosen rule
        is_candidate, pvc_score = compute_pvc_rule(
            prematurity_index,
            qrs_width_ms,
            morphology_score=(float(morph_scores[i]) if np.isfinite(morph_scores[i]) else None),
            prematurity_threshold=prematurity_threshold,
            qrs_width_threshold_ms=qrs_width_threshold_ms,
            detection_rule=detection_rule,
        )

        features.append(
            {
                "beat_index": i,
                "peak_time_s": float(times[peaks[i]]),
                "rr_prev_s": float(curr_rr) if not np.isnan(curr_rr) else None,
                "rr_next_s": float(curr_next_rr) if not np.isnan(curr_next_rr) else None,
                "rr_baseline_s": global_rr_baseline,
                "prematurity_index": prem_index,
                "aa_ratio": curr_aa,
                "morph_index": morph_index,
                "qrs_width_ms": float(curr_width) if not np.isnan(curr_width) else None,
                "width_index": width_index,
                "pause_index": pause_index,
                "pvc_score": pvc_score,
                "morphology_score": (float(morph_scores[i]) if np.isfinite(morph_scores[i]) else np.nan),
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
        "morphology_score",
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
