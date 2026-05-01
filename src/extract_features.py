import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.algorithms.mlp_pvc import FEATURE_KEYS, get_mlp_predictor
from src.helpers.clinical_metrics import compute_pvc_burden_metrics
from src.helpers.signal_processing import estimate_qrs_width_ms
from src.algorithms.pan_thompkins import PanThompkinsQRS

"""PVC-focused feature extraction from ECG.

Glossary:
- R-peak: highest (or lowest, depending on lead polarity) point of a heartbeat.
- RR interval: time between consecutive R-peaks.
- QRS complex: fast ventricular depolarization segment around the R-peak.
- Prematurity index: how early a beat occurs relative to the baseline RR.
"""

_MLP_PREDICTOR = None


def _get_mlp_model():
    global _MLP_PREDICTOR
    if _MLP_PREDICTOR is None:
        _MLP_PREDICTOR = get_mlp_predictor()
    return _MLP_PREDICTOR


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

    qrs_width = estimate_qrs_width_ms(ecg, peak_idx, sampling_rate, 
                                  search_half_window_s=0.25, threshold_fraction=0.15)

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
    prem_index, width_index, morphology_score=None, detection_rule="weighted",
    prem_thr=0.92, width_thr=1.15, morph_sim_thr=0.85, pause_index=None, pause_thr=1.2,
    rr_prev_s=np.nan, rr_next_s=np.nan, rr_baseline_s=np.nan, aa_ratio=np.nan,
    morph_index=np.nan, qrs_width_ms=np.nan,
):
    if detection_rule == "mlp":
        predict = _get_mlp_model()
        feature_vector = np.array(
            [
                rr_prev_s,
                rr_next_s,
                rr_baseline_s,
                prem_index,
                aa_ratio,
                morph_index,
                qrs_width_ms,
                width_index,
                pause_index,
                morphology_score,
            ],
            dtype=float,
        )
        if feature_vector.size != len(FEATURE_KEYS):
            raise ValueError(f"MLP feature vector must have {len(FEATURE_KEYS)} values.")
        probability = float(predict(feature_vector))
        return int(probability > 0.5), probability

    cond_prem = np.isfinite(prem_index) and prem_index < prem_thr
    cond_wide = np.isfinite(width_index) and width_index > width_thr
    cond_morph = morphology_score is not None and np.isfinite(morphology_score) and morphology_score < morph_sim_thr
    cond_pause = np.isfinite(pause_index) and pause_index > pause_thr if pause_index else False

    wide_or_abnormal = cond_wide or cond_morph

    if detection_rule == "weighted":
        prem_score = 1.0 / (1.0 + np.exp((prem_index - prem_thr) / 0.15)) if np.isfinite(prem_index) else 0.0
        qrs_score = 1.0 / (1.0 + np.exp(-(width_index - width_thr) / 0.25)) if np.isfinite(width_index) else 0.0
        morph_score = 1.0 - morphology_score if morphology_score is not None else 0.0
        p_score = 1.0 / (1.0 + np.exp(-(pause_index - pause_thr) / 0.10)) if np.isfinite(pause_index) else 0.0

        score = (0.35 * prem_score) + (0.30 * qrs_score) + (0.20 * morph_score) + (0.15 * p_score)
        candidate = score > 0.50
    else:
        if detection_rule == "and":
            # Any two of the four indicators
            score_count = sum([cond_prem, cond_wide, cond_morph, cond_pause])
            candidate = score_count >= 2
        elif detection_rule == "or":
            candidate = cond_prem or wide_or_abnormal or cond_pause
        score = 1.0 if candidate else 0.0

    return int(candidate), score


def extract_extrasystole_features(
    times,
    ecg_cleaned,
    sampling_rate,
    min_peak_distance_s=0.25,
    prominence_factor=1.0,
    refractory_s=0.30,
    prematurity_threshold=0.80,
    qrs_width_threshold_ms=110.0,
    detection_rule="weighted",
    window_size=10
):
    """
    Full PVC feature extraction incorporating local RR baselines and relative width.
    Integrated from extract_features_2.py.
    """
    # 1. Detect R-peaks using Pan-Tompkins
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

    aa_ratios = np.zeros(n_beats)
    qrs_widths = np.full(n_beats, np.nan)
    for i, p in enumerate(peaks):
        _, _, aa_ratio_i, qrs_w = compute_local_shape_features(ecg_cleaned, p, sampling_rate)
        aa_ratios[i] = aa_ratio_i
        qrs_widths[i] = qrs_w

    global_rr_baseline = robust_baseline(rr_values)
    global_aa_baseline = robust_baseline(aa_ratios)
    global_width_baseline = robust_baseline(qrs_widths[~np.isnan(qrs_widths)])

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
        max_iters = 3
        refine_frac = 0.25 
        max_shift = max(1, int(0.02 * sampling_rate)) 
        aligned_segments = [segments[i].copy() for i in range(len(segments)) if valid_mask[i]]

        for it in range(max_iters):
            template = np.median(np.vstack(aligned_segments), axis=0)
            tmpl = template - np.mean(template)
            tmpl_norm = float(np.linalg.norm(tmpl))
            new_aligned = []
            corrs = []
            for seg in aligned_segments:
                seg_z = seg - np.mean(seg)
                best_corr = -1.0
                best_shift = 0
                for shift in range(-max_shift, max_shift + 1):
                    s_shift = np.roll(seg_z, shift)
                    s_norm = float(np.linalg.norm(s_shift))
                    c = float(np.dot(s_shift, tmpl) / (s_norm * tmpl_norm + 1e-9))
                    if c > best_corr:
                        best_corr = c
                        best_shift = shift
                new_aligned.append(np.roll(seg, best_shift) - np.mean(seg))
                corrs.append(best_corr)

            if refine_frac > 0 and it < max_iters - 1:
                thresh = np.percentile(corrs, 100.0 * refine_frac)
                aligned_segments = [s for k, s in zip(corrs >= thresh, new_aligned) if k]
                if len(aligned_segments) == 0:
                    aligned_segments = new_aligned
                    break
            else:
                aligned_segments = new_aligned
                break

        final_template = np.median(np.vstack(aligned_segments), axis=0)
        final_t = final_template - np.mean(final_template)
        final_t_norm = float(np.linalg.norm(final_t))

        for idx_valid_pos in np.flatnonzero(valid_mask):
            seg_z = segments[idx_valid_pos] - np.mean(segments[idx_valid_pos])
            best_corr = max([float(np.dot(np.roll(seg_z, sh), final_t) / (np.linalg.norm(np.roll(seg_z, sh)) * final_t_norm + 1e-9)) 
                             for sh in range(-max_shift, max_shift + 1)])
            morph_scores[idx_valid_pos] = float((best_corr + 1.0) / 2.0)

    features = []
    rr_history = []
    
    for i in range(n_beats):
        curr_rr = rr_values[i - 1] if i > 0 else np.nan
        curr_next_rr = rr_values[i] if i < n_beats - 1 else np.nan
        
        if not np.isnan(curr_rr):
            rr_history.append(curr_rr)
            if len(rr_history) > window_size:
                rr_history.pop(0)
        
        local_rr_baseline = np.median(rr_history) if len(rr_history) >= 5 else global_rr_baseline
        
        prem_index = curr_rr / local_rr_baseline if local_rr_baseline > 0 else np.nan
        width_index = qrs_widths[i] / (global_width_baseline + 1e-9)
        pause_index = curr_next_rr / local_rr_baseline if local_rr_baseline > 0 else np.nan
        morph_index = aa_ratios[i] / (global_aa_baseline + 1e-9)

        is_candidate, pvc_score = compute_pvc_rule(
            prem_index=prem_index,
            width_index=width_index,
            morphology_score=morph_scores[i],
            detection_rule=detection_rule,
            prem_thr=prematurity_threshold,
            width_thr=1.15,
            morph_sim_thr=0.85,
            pause_index=pause_index,
            rr_prev_s=curr_rr,
            rr_next_s=curr_next_rr,
            rr_baseline_s=local_rr_baseline,
            aa_ratio=aa_ratios[i],
            morph_index=morph_index,
            qrs_width_ms=qrs_widths[i],
        )

        features.append({
            "beat_index": i,
            "peak_time_s": float(times[peaks[i]]),
            "rr_prev_s": float(curr_rr) if not np.isnan(curr_rr) else None,
            "rr_next_s": float(curr_next_rr) if not np.isnan(curr_next_rr) else None,
            "rr_baseline_s": local_rr_baseline,
            "prematurity_index": prem_index,
            "aa_ratio": aa_ratios[i],
            "morph_index": morph_index,
            "qrs_width_ms": float(qrs_widths[i]) if not np.isnan(qrs_widths[i]) else None,
            "width_index": width_index,
            "pause_index": pause_index,
            "pvc_score": pvc_score,
            "morphology_score": float(morph_scores[i]) if np.isfinite(morph_scores[i]) else np.nan,
            "is_pvc_candidate": is_candidate,
        })
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
    parser.add_argument("--detection-rule", choices=["and", "or", "weighted", "mlp"], default="and",
                        help="PVC detection rule: 'and' (strict), 'or' (loose), 'weighted' (heuristic score), 'mlp' (trained neural baseline)")
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
    duration_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    burden_metrics = compute_pvc_burden_metrics(n_candidates, len(features), duration_s=duration_s)
    print(f"[INFO] Sampling rate inferred: {sampling_rate:.2f} Hz")
    print(f"[INFO] Detection rule: {args.detection_rule}")
    print(f"[INFO] Detected beats: {len(features)}")
    print(f"[INFO] PVC candidates: {n_candidates}")
    print(f"[INFO] PVC burden: {burden_metrics['pvc_burden_percent']:.2f}%")
    print(f"[INFO] PVC rate: {burden_metrics['pvc_rate_per_hour']:.2f}/hour")
    print(f"[INFO] Saved feature table to {args.output}")
    if args.show:
        print("[INFO] Displayed peak-time plot interactively (no PNG saved)")
    else:
        print(f"[INFO] Saved peak-time plot to {args.plot}")


if __name__ == "__main__":
    main()
