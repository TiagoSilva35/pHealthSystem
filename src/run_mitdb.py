#!/usr/bin/env python3
"""
MIT-BIH beat / PVC detection evaluation.

Examples:
  # Beat detection performance (default)
  python src/run_mitdb.py --database mitdb

  # PVC detection performance (matched beats)
  python src/run_mitdb.py --database mitdb --evaluation-mode pvc

  # PVC detection performance on all reference beats (ignoring detector)
  python src/run_mitdb.py --database mitdb --evaluation-mode pvc --pvc-eval-ref

  # VT rhythm detection performance against MIT-BIH rhythm references
  python src/run_mitdb.py --database mitdb --evaluation-mode vt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from src.extract_features import extract_extrasystole_features, save_peak_time_plot
from src.algorithms.mlp_pvc import DS2_RECORDS
from src.helpers.clinical_metrics import compute_pvc_burden_metrics
from src.helpers.plot_signals import plot_signals
from src.helpers.signal_processing import (
    estimate_qrs_width_ms,
    load_ecg_csv,
    preprocess_ecg_for_arrhythmia,
)

VT_BURDEN_THRESHOLD_PERCENT = 10.0
VT_RHYTHM_NOTES = {"(VT"}

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
NON_BEAT_ANNOTATION_SYMBOLS = {
    "+", "~", "|", "[", "]", "x", "(", ")", "!", "p", "t", "u",
    "`", "'", "^", "=", "s", "T", "*", "@", "J", "a", "S", "e", "j", "F",
}
PVC_ANNOTATION_SYMBOLS = {"V"}

# ----------------------------------------------------------------------
# Database I/O (unchanged)
# ----------------------------------------------------------------------
def list_records(db_path: Path):
    return sorted(p.stem for p in db_path.glob("*.hea"))

def load_physiobank_record(record_path: Path):
    record = wfdb.rdrecord(str(record_path))
    if record.p_signal is None:
        raise ValueError(f"{record_path} has no physical signal (p_signal).")
    signal = record.p_signal[:, 0]
    sampling_rate = float(record.fs)
    times = np.arange(len(signal), dtype=float) / sampling_rate
    return times, signal, sampling_rate

def load_reference_beats(record_path: Path):
    ann = wfdb.rdann(str(record_path), "atr")
    ref_samples, ref_symbols, pvc_labels = [], [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in NON_BEAT_ANNOTATION_SYMBOLS:
            continue
        ref_samples.append(int(sample))
        ref_symbols.append(symbol)
        pvc_labels.append(1 if symbol in PVC_ANNOTATION_SYMBOLS else 0)
    ref_samples = np.asarray(ref_samples, dtype=int)
    vt_labels = build_reference_vt_labels(ann, ref_samples)
    return ref_samples, ref_symbols, np.asarray(pvc_labels, dtype=int), vt_labels


def normalize_aux_note(note):
    if not note:
        return ""
    if isinstance(note, bytes):
        text = note.decode("latin1", "ignore")
    else:
        text = str(note)
    return text.replace("\x00", "").strip()


def build_reference_vt_labels(annotation, beat_samples):
    beat_samples = np.asarray(beat_samples, dtype=int)
    if beat_samples.size == 0:
        return np.asarray([], dtype=int)

    rhythm_changes = []
    rhythm_states = []
    for sample, symbol, aux_note in zip(annotation.sample, annotation.symbol, annotation.aux_note):
        if symbol != "+":
            continue
        note = normalize_aux_note(aux_note)
        if not note:
            continue
        rhythm_changes.append(int(sample))
        rhythm_states.append(note in VT_RHYTHM_NOTES)

    if not rhythm_changes:
        return np.zeros(beat_samples.size, dtype=int)

    rhythm_changes = np.asarray(rhythm_changes, dtype=int)
    rhythm_states = np.asarray(rhythm_states, dtype=bool)
    state_idx = np.searchsorted(rhythm_changes, beat_samples, side="right") - 1
    active = np.zeros(beat_samples.size, dtype=bool)
    valid = state_idx >= 0
    active[valid] = rhythm_states[state_idx[valid]]
    return active.astype(int)

def match_detections_to_references(ref_samples, det_samples, tol_samples):
    """Greedy matching, returns tp, fp, fn, signed_errors, matched_ref_idx (index or -1)."""
    ref_samples = np.asarray(ref_samples, dtype=int)
    det_samples = np.asarray(det_samples, dtype=int)
    if len(ref_samples) == 0:
        return 0, len(det_samples), 0, np.array([], dtype=float), np.full(len(det_samples), -1)

    used_ref = np.zeros(len(ref_samples), dtype=bool)
    signed_errors = []
    matched_ref_idx = []
    tp = fp = 0

    for det in det_samples:
        distances = np.abs(ref_samples - det)
        nearest_idx = int(np.argmin(distances))
        nearest_error = int(det - ref_samples[nearest_idx])
        if distances[nearest_idx] <= tol_samples and not used_ref[nearest_idx]:
            used_ref[nearest_idx] = True
            tp += 1
            signed_errors.append(nearest_error)
            matched_ref_idx.append(nearest_idx)
        else:
            fp += 1
            matched_ref_idx.append(-1)

    fn = int(np.sum(~used_ref))
    return tp, fp, fn, np.asarray(signed_errors, dtype=float), np.array(matched_ref_idx, dtype=int)

# ----------------------------------------------------------------------
# Per‑record metric helpers
# ----------------------------------------------------------------------
def compute_beat_detection_metrics(
    record_name,
    tp_beats, fp_beats, fn_beats,
    ref_count, det_count,
    sampling_rate,
    duration_s,
):
    """Return summary dict for QRS detection."""
    se = 100.0 * tp_beats / (tp_beats + fn_beats) if (tp_beats + fn_beats) > 0 else 0.0
    ppv = 100.0 * tp_beats / (tp_beats + fp_beats) if (tp_beats + fp_beats) > 0 else 0.0
    f1 = 2.0 * tp_beats / (2 * tp_beats + fp_beats + fn_beats) * 100.0 if (2 * tp_beats + fp_beats + fn_beats) > 0 else 0.0

    return {
        "record": record_name,
        "reference_beats": ref_count,
        "detected_beats": det_count,
        "TP": tp_beats,
        "FP": fp_beats,
        "FN": fn_beats,
        "SE_percent": se,
        "PPV_percent": ppv,
        "F1_percent": f1,
        "sampling_rate_hz": sampling_rate,
        "duration_s": duration_s,
    }

def compute_pvc_detection_metrics_matched(
    record_name,
    is_pvc_pred,          # array of 0/1 for all detections
    matched_ref_idx,      # index to ref_samples or -1
    pvc_labels,           # true PVC labels of all reference beats
    ref_count,
    sampling_rate,
    duration_s,
):
    """Evaluate PVC classification only on correctly detected beats (matched)."""
    matched_mask = matched_ref_idx != -1
    pred_matched = is_pvc_pred[matched_mask]
    true_matched = pvc_labels[matched_ref_idx[matched_mask]]

    tp = int(np.sum((pred_matched == 1) & (true_matched == 1)))
    fp = int(np.sum((pred_matched == 1) & (true_matched == 0)))
    fn = int(np.sum(pvc_labels) - tp)  # total true PVCs minus those correctly predicted
    tn = (len(pvc_labels) - int(np.sum(pvc_labels))) - fp  # total non-PVCs minus those incorrectly predicted as PVC

    sens = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc  = 100.0 * (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    ppv  = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2.0 * tp / (2 * tp + fp + fn) * 100.0 if (2 * tp + fp + fn) > 0 else 0.0

    detected_burden = compute_pvc_burden_metrics(
        int(np.sum(is_pvc_pred)),
        len(is_pvc_pred),
        duration_s,
        vt_burden_threshold_percent=VT_BURDEN_THRESHOLD_PERCENT,
    )
    reference_burden = compute_pvc_burden_metrics(
        int(np.sum(pvc_labels)),
        ref_count,
        duration_s,
        vt_burden_threshold_percent=VT_BURDEN_THRESHOLD_PERCENT,
    )

    return {
        "record": record_name,
        "reference_pvc_beats": int(np.sum(pvc_labels)),
        "detected_pvc_candidates": int(np.sum(is_pvc_pred)),
        "reference_beats": int(ref_count),
        "detected_beats": int(len(is_pvc_pred)),
        "TP_pvc": tp,
        "FP_pvc": fp,
        "FN_pvc": fn,
        "TN_pvc": tn,
        "Sensitivity_pvc_percent": sens,
        "Specificity_pvc_percent": spec,
        "Accuracy_pvc_percent": acc,
        "PPV_pvc_percent": ppv,
        "F1_pvc_percent": f1,
        "reference_pvc_burden_percent": reference_burden["pvc_burden_percent"],
        "reference_pvc_rate_per_hour": reference_burden["pvc_rate_per_hour"],
        "reference_possible_vt": reference_burden["possible_vt_from_burden"],
        "detected_pvc_burden_percent": detected_burden["pvc_burden_percent"],
        "detected_pvc_rate_per_hour": detected_burden["pvc_rate_per_hour"],
        "detected_possible_vt": detected_burden["possible_vt_from_burden"],
        "sampling_rate_hz": sampling_rate,
        "duration_s": duration_s,
    }

def compute_pvc_detection_metrics_reference(
    record_name,
    clean_signal,
    times,
    ref_samples,
    pvc_labels,
    prematurity_threshold,
    qrs_width_threshold_ms,
    sampling_rate,
    duration_s,
):
    """Evaluate PVC rule directly on all reference beats (ignores the QRS detector)."""
    peak_times = times[ref_samples]
    rr = np.diff(peak_times)
    rr_median = np.median(rr) if len(rr) > 0 else np.nan

    tp = fp = fn = tn = 0
    for i, sample in enumerate(ref_samples):
        # Prematurity index
        if i > 0 and np.isfinite(rr[i-1]) and rr_median > 0:
            prem_index = rr[i-1] / rr_median
        else:
            prem_index = np.nan

        # QRS width
        qrs_width = estimate_qrs_width_ms(clean_signal, int(sample), sampling_rate)

        # Rule
        cond_prem = np.isfinite(prem_index) and prem_index < prematurity_threshold
        cond_wide = np.isfinite(qrs_width) and qrs_width > qrs_width_threshold_ms
        pred = 1 if cond_prem and cond_wide else 0

        true = pvc_labels[i]
        if true == 1 and pred == 1: tp += 1
        elif true == 0 and pred == 1: fp += 1
        elif true == 1 and pred == 0: fn += 1
        else: tn += 1

    sens = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc  = 100.0 * (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    ppv  = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2.0 * tp / (2 * tp + fp + fn) * 100.0 if (2 * tp + fp + fn) > 0 else 0.0

    detected_burden = compute_pvc_burden_metrics(
        tp + fp,
        len(pvc_labels),
        duration_s,
        vt_burden_threshold_percent=VT_BURDEN_THRESHOLD_PERCENT,
    )
    reference_burden = compute_pvc_burden_metrics(
        int(np.sum(pvc_labels)),
        len(pvc_labels),
        duration_s,
        vt_burden_threshold_percent=VT_BURDEN_THRESHOLD_PERCENT,
    )

    return {
        "record": record_name,
        "reference_pvc_beats": int(np.sum(pvc_labels)),
        "detected_pvc_candidates": tp + fp,  # number of reference beats classified as PVC
        "reference_beats": int(len(pvc_labels)),
        "detected_beats": int(len(pvc_labels)),
        "TP_pvc": tp,
        "FP_pvc": fp,
        "FN_pvc": fn,
        "TN_pvc": tn,
        "Sensitivity_pvc_percent": sens,
        "Specificity_pvc_percent": spec,
        "Accuracy_pvc_percent": acc,
        "PPV_pvc_percent": ppv,
        "F1_pvc_percent": f1,
        "reference_pvc_burden_percent": reference_burden["pvc_burden_percent"],
        "reference_pvc_rate_per_hour": reference_burden["pvc_rate_per_hour"],
        "reference_possible_vt": reference_burden["possible_vt_from_burden"],
        "detected_pvc_burden_percent": detected_burden["pvc_burden_percent"],
        "detected_pvc_rate_per_hour": detected_burden["pvc_rate_per_hour"],
        "detected_possible_vt": detected_burden["possible_vt_from_burden"],
        "sampling_rate_hz": sampling_rate,
        "duration_s": duration_s,
    }


def count_positive_runs(binary_labels):
    labels = np.asarray(binary_labels, dtype=int)
    if labels.size == 0:
        return 0
    starts = np.where((labels == 1) & (np.r_[0, labels[:-1]] == 0))[0]
    return int(starts.size)


def detect_vt_candidates(
    detected_samples,
    is_pvc_pred,
    sampling_rate,
    min_consecutive_beats=3,
    max_rr_s=0.60,
    max_gap_beats=0,   
):
    detected_samples = np.asarray(detected_samples, dtype=int)
    is_pvc_pred = np.asarray(is_pvc_pred, dtype=int)
    n = detected_samples.size
    if n == 0:
        return np.asarray([], dtype=int)

    vt_pred = np.zeros(n, dtype=int)
    if n < min_consecutive_beats:
        return vt_pred

    rr = np.diff(detected_samples) / float(sampling_rate)

    # 1st pass: strict runs of consecutive PVCs with RR ≤ max_rr_s
    runs = []          # list of (start_idx, end_idx)
    i = 0
    while i < n:
        if is_pvc_pred[i] != 1:
            i += 1
            continue
        j = i
        while (j + 1 < n and is_pvc_pred[j + 1] == 1
               and np.isfinite(rr[j]) and rr[j] <= max_rr_s):
            j += 1
        if j - i + 1 >= min_consecutive_beats:
            runs.append((i, j))
        i = j + 1

    # Gap merging (single pass, left to right)
    if max_gap_beats > 0 and len(runs) > 1:
        merged_runs = []
        prev_start, prev_end = runs[0]
        for start, end in runs[1:]:
            gap_size = start - prev_end - 1   # number of beats between the two runs
            if gap_size <= max_gap_beats:
                # Check that the RRs across the gap satisfy rate criterion
                # We need the RR from prev_end to prev_end+1 (first gap beat)
                # and the RR from the last gap beat to start (i.e., start-1 to start)
                # All those indices must exist.
                valid = True
                # check RR from prev_end to next beat (first gap beat)
                if prev_end < n - 1 and np.isfinite(rr[prev_end]) and rr[prev_end] > max_rr_s:
                    valid = False
                # check RR from each gap beat to its next
                for k in range(prev_end + 1, start):
                    if k < n - 1 and np.isfinite(rr[k]) and rr[k] > max_rr_s:
                        valid = False
                # check RR from last gap beat (start-1) to start (which is covered by rr[start-1])
                if start > 0 and np.isfinite(rr[start-1]) and rr[start-1] > max_rr_s:
                    valid = False
                if valid:
                    # Merge: extend prev_end to end
                    prev_end = end
                    continue
            # can't merge, push previous run and start new
            merged_runs.append((prev_start, prev_end))
            prev_start, prev_end = start, end
        merged_runs.append((prev_start, prev_end))
        runs = merged_runs

    # Apply runs to vt_pred
    for start, end in runs:
        vt_pred[start : end + 1] = 1

    return vt_pred


def compute_vt_detection_metrics_matched(
    record_name,
    is_vt_pred,
    matched_ref_idx,
    vt_labels,
    ref_count,
    det_count,
    sampling_rate,
    duration_s,
):
    matched_mask = matched_ref_idx != -1
    pred_matched = is_vt_pred[matched_mask]
    true_matched = vt_labels[matched_ref_idx[matched_mask]]

    tp = int(np.sum((pred_matched == 1) & (true_matched == 1)))
    fp = int(np.sum((pred_matched == 1) & (true_matched == 0)))
    fn = int(np.sum(vt_labels) - tp)
    tn = int((len(vt_labels) - int(np.sum(vt_labels))) - fp)

    sens = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = 100.0 * (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    ppv = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2.0 * tp / (2 * tp + fp + fn) * 100.0 if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "record": record_name,
        "reference_vt_beats": int(np.sum(vt_labels)),
        "detected_vt_candidates": int(np.sum(is_vt_pred)),
        "reference_vt_episodes": count_positive_runs(vt_labels),
        "detected_vt_episodes": count_positive_runs(is_vt_pred),
        "reference_beats": int(ref_count),
        "detected_beats": int(det_count),
        "TP_vt": tp,
        "FP_vt": fp,
        "FN_vt": fn,
        "TN_vt": tn,
        "Sensitivity_vt_percent": sens,
        "Specificity_vt_percent": spec,
        "Accuracy_vt_percent": acc,
        "PPV_vt_percent": ppv,
        "F1_vt_percent": f1,
        "sampling_rate_hz": sampling_rate,
        "duration_s": duration_s,
    }

# ----------------------------------------------------------------------
# Global aggregation helpers
# ----------------------------------------------------------------------
def aggregate_beat_metrics(summaries):
    """Given list of beat‑detection summary dicts, add a GLOBAL row and return DataFrame."""
    df = pd.DataFrame(summaries)
    ok = df.dropna(subset=["reference_beats", "detected_beats", "TP", "FP", "FN"]).copy()
    if ok.empty:
        return df
    total_ref = int(ok["reference_beats"].sum())
    total_det = int(ok["detected_beats"].sum())
    total_tp  = int(ok["TP"].sum())
    total_fp  = int(ok["FP"].sum())
    total_fn  = int(ok["FN"].sum())

    global_se  = 100.0 * total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    global_ppv = 100.0 * total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_f1  = 2.0 * total_tp / (2 * total_tp + total_fp + total_fn) * 100.0 if (2 * total_tp + total_fp + total_fn) > 0 else 0.0

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
        "sampling_rate_hz": "",
        "duration_s": float(ok["duration_s"].sum()),
    }
    return pd.concat([df, pd.DataFrame([global_row])], ignore_index=True)

def aggregate_pvc_metrics(summaries):
    """Given list of PVC‑detection summary dicts, add a GLOBAL row and return DataFrame."""
    # Filter out error records (those with only "record" and "error" keys)
    valid_summaries = [s for s in summaries if "TP_pvc" in s]
    df = pd.DataFrame(valid_summaries) if valid_summaries else pd.DataFrame()
    
    if df.empty:
        # Return all summaries if no valid ones (preserve error records for visibility)
        return pd.DataFrame(summaries)
    
    ok = df.copy()
    total_tp = int(ok["TP_pvc"].sum())
    total_fp = int(ok["FP_pvc"].sum())
    total_fn = int(ok["FN_pvc"].sum())
    total_tn = int(ok["TN_pvc"].sum())
    total_reference_beats = int(ok["reference_beats"].sum())
    total_detected_beats = int(ok["detected_beats"].sum())
    total_duration_s = float(ok["duration_s"].sum())

    global_sens = 100.0 * total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    global_spec = 100.0 * total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    global_acc  = 100.0 * (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0.0
    global_ppv  = 100.0 * total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_f1   = 2.0 * total_tp / (2 * total_tp + total_fp + total_fn) * 100.0 if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
    global_reference_burden = compute_pvc_burden_metrics(
        int(ok["reference_pvc_beats"].sum()),
        total_reference_beats,
        total_duration_s,
        vt_burden_threshold_percent=VT_BURDEN_THRESHOLD_PERCENT,
    )
    global_detected_burden = compute_pvc_burden_metrics(
        int(ok["detected_pvc_candidates"].sum()),
        total_detected_beats,
        total_duration_s,
        vt_burden_threshold_percent=VT_BURDEN_THRESHOLD_PERCENT,
    )

    global_row = {
        "record": "GLOBAL",
        "reference_pvc_beats": int(ok["reference_pvc_beats"].sum()),
        "detected_pvc_candidates": int(ok["detected_pvc_candidates"].sum()),
        "reference_beats": total_reference_beats,
        "detected_beats": total_detected_beats,
        "TP_pvc": total_tp,
        "FP_pvc": total_fp,
        "FN_pvc": total_fn,
        "TN_pvc": total_tn,
        "Sensitivity_pvc_percent": global_sens,
        "Specificity_pvc_percent": global_spec,
        "Accuracy_pvc_percent": global_acc,
        "PPV_pvc_percent": global_ppv,
        "F1_pvc_percent": global_f1,
        "reference_pvc_burden_percent": global_reference_burden["pvc_burden_percent"],
        "reference_pvc_rate_per_hour": global_reference_burden["pvc_rate_per_hour"],
        "reference_possible_vt": global_reference_burden["possible_vt_from_burden"],
        "detected_pvc_burden_percent": global_detected_burden["pvc_burden_percent"],
        "detected_pvc_rate_per_hour": global_detected_burden["pvc_rate_per_hour"],
        "detected_possible_vt": global_detected_burden["possible_vt_from_burden"],
        "sampling_rate_hz": "",
        "duration_s": total_duration_s,
    }
    return pd.concat([df, pd.DataFrame([global_row])], ignore_index=True)


def aggregate_vt_metrics(summaries):
    valid_summaries = [s for s in summaries if "TP_vt" in s]
    df = pd.DataFrame(valid_summaries) if valid_summaries else pd.DataFrame()

    if df.empty:
        return pd.DataFrame(summaries)

    ok = df.copy()
    total_tp = int(ok["TP_vt"].sum())
    total_fp = int(ok["FP_vt"].sum())
    total_fn = int(ok["FN_vt"].sum())
    total_tn = int(ok["TN_vt"].sum())
    total_reference_beats = int(ok["reference_beats"].sum())
    total_detected_beats = int(ok["detected_beats"].sum())
    total_duration_s = float(ok["duration_s"].sum())

    global_sens = 100.0 * total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    global_spec = 100.0 * total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    global_acc = 100.0 * (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0.0
    global_ppv = 100.0 * total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_f1 = 2.0 * total_tp / (2 * total_tp + total_fp + total_fn) * 100.0 if (2 * total_tp + total_fp + total_fn) > 0 else 0.0

    global_row = {
        "record": "GLOBAL",
        "reference_vt_beats": int(ok["reference_vt_beats"].sum()),
        "detected_vt_candidates": int(ok["detected_vt_candidates"].sum()),
        "reference_vt_episodes": int(ok["reference_vt_episodes"].sum()),
        "detected_vt_episodes": int(ok["detected_vt_episodes"].sum()),
        "reference_beats": total_reference_beats,
        "detected_beats": total_detected_beats,
        "TP_vt": total_tp,
        "FP_vt": total_fp,
        "FN_vt": total_fn,
        "TN_vt": total_tn,
        "Sensitivity_vt_percent": global_sens,
        "Specificity_vt_percent": global_spec,
        "Accuracy_vt_percent": global_acc,
        "PPV_vt_percent": global_ppv,
        "F1_vt_percent": global_f1,
        "sampling_rate_hz": "",
        "duration_s": total_duration_s,
    }
    return pd.concat([df, pd.DataFrame([global_row])], ignore_index=True)

# ----------------------------------------------------------------------
# Single CSV evaluation (ecg_samples.csv flow)
# ----------------------------------------------------------------------
def evaluate_ecg_csv(
    csv_path,
    output_dir,
    skip_plots=False,
    show_plots=False,
    min_peak_distance_s=0.25,
    refractory_s=0.30,
    prematurity_threshold=0.80,
    qrs_width_threshold_ms=110.0,
    detection_rule="and",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(csv_path)
    times, signal, sampling_rate = load_ecg_csv(str(input_path))
    preprocessed = preprocess_ecg_for_arrhythmia(signal, sampling_rate, notch_hz=60.0)
    clean_signal = preprocessed["cleaned"]

    features = extract_extrasystole_features(
        times,
        clean_signal,
        sampling_rate=sampling_rate,
        min_peak_distance_s=min_peak_distance_s,
        refractory_s=refractory_s,
        prematurity_threshold=prematurity_threshold,
        qrs_width_threshold_ms=qrs_width_threshold_ms,
        detection_rule=detection_rule,
    )

    detected_beats = len(features)
    pvc_candidates = int(sum(row["is_pvc_candidate"] for row in features))
    duration_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    pvc_burden = compute_pvc_burden_metrics(pvc_candidates, detected_beats, duration_s)

    summary = {
        "record": input_path.name,
        "sampling_rate_hz": float(sampling_rate),
        "duration_s": duration_s,
        "detected_beats": detected_beats,
        "detected_pvc_candidates": pvc_candidates,
        "pvc_burden_percent": pvc_burden["pvc_burden_percent"],
        "pvc_rate_per_hour": pvc_burden["pvc_rate_per_hour"],
        "pvc_ratio_percent": pvc_burden["pvc_burden_percent"],
        "detection_rule": detection_rule,
    }

    stem = input_path.stem
    summary_csv = output_dir / f"{stem}_csv_eval.csv"
    features_csv = output_dir / f"{stem}_detected_beats.csv"
    final_plot = output_dir / f"{stem}_detected_peaks.png"

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    if features:
        pd.DataFrame(features).to_csv(features_csv, index=False)
    else:
        pd.DataFrame(
            columns=[
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
        ).to_csv(features_csv, index=False)

    if not skip_plots:
        save_peak_time_plot(times, clean_signal, features, str(final_plot), show_plot=show_plots)

    print(f"[INFO] Processed ECG CSV: {csv_path}")
    print(f"[INFO] Sampling rate inferred: {sampling_rate:.2f} Hz")
    print(f"[INFO] Detected beats: {detected_beats}")
    print(f"[INFO] PVC burden: {pvc_burden['pvc_burden_percent']:.2f}%")
    print(f"[INFO] PVC rate: {pvc_burden['pvc_rate_per_hour']:.2f}/hour")
    print(f"[INFO] Extrasystole candidates: {pvc_candidates}")
    print(f"[INFO] Saved summary to {summary_csv}")
    print(f"[INFO] Saved beat features to {features_csv}")
    if skip_plots:
        print("[INFO] Plot generation skipped")
    elif show_plots:
        print("[INFO] Displayed final beats/extrasystoles plot interactively")
    else:
        print(f"[INFO] Saved final beats/extrasystoles plot to {final_plot}")

# ----------------------------------------------------------------------
# Main record evaluation (dispatches to helpers)
# ----------------------------------------------------------------------
def evaluate_record(
    record_path,
    min_peak_distance_s,
    refractory_s,
    prematurity_threshold,
    qrs_width_threshold_ms,
    tolerance_ms,
    evaluation_mode,
    detection_rule="weighted",
    pvc_eval_ref=False,   # new flag: if True, evaluate PVC on all reference beats
    vt_min_consecutive_beats=3,
    vt_max_rr_s=0.60,
):
    # record_name = record_path.stem
    # if record_name != "207":
    #     return None, None, None, None  # skip all but record 207 for now (for quick testing)
    times, signal, sampling_rate = load_physiobank_record(record_path)
    # plot_signals(signal, sampling_rate)
    preprocessed = preprocess_ecg_for_arrhythmia(signal, sampling_rate, notch_hz=60.0)
    clean_signal = preprocessed["cleaned"]
    # plot_signals(clean_signal, sampling_rate)

    ref_samples, ref_symbols, pvc_labels, vt_labels = load_reference_beats(record_path)

    # Run the detector and extract features (always needed for beat detection;
    # for PVC‑reference mode we could skip the detector, but we keep it for consistency)
    features = extract_extrasystole_features(
        times, clean_signal,
        sampling_rate,
        min_peak_distance_s=min_peak_distance_s,
        refractory_s=refractory_s,
        detection_rule=detection_rule,
    )

    detected_peak_times_s = np.asarray([row["peak_time_s"] for row in features], dtype=float)
    detected_samples = np.asarray(np.round(detected_peak_times_s * sampling_rate), dtype=int)
    is_pvc_pred = np.asarray([row["is_pvc_candidate"] for row in features], dtype=int)
    is_vt_pred = detect_vt_candidates(
        detected_samples,
        is_pvc_pred,
        sampling_rate,
        min_consecutive_beats=vt_min_consecutive_beats,
        max_rr_s=vt_max_rr_s,
        max_gap_beats=1,
    )

    tol_samples = int(round((tolerance_ms / 1000.0) * sampling_rate))
    tp_beats, fp_beats, fn_beats, signed_errors, matched_ref_idx = match_detections_to_references(
        ref_samples, detected_samples, tol_samples
    )

    duration_s = float(times[-1]) if len(times) else 0.0

    if evaluation_mode == "beats":
        summary = compute_beat_detection_metrics(
            record_path.name,
            tp_beats, fp_beats, fn_beats,
            len(ref_samples), len(detected_samples),
            sampling_rate, duration_s,
        )
    elif evaluation_mode == "pvc":
        if pvc_eval_ref:
            # Use reference beats directly (ignore detector)
            summary = compute_pvc_detection_metrics_reference(
                record_path.name,
                clean_signal, times,
                ref_samples, pvc_labels,
                prematurity_threshold, qrs_width_threshold_ms,
                sampling_rate, duration_s,
            )
        else:
            # Use matched beats only
            summary = compute_pvc_detection_metrics_matched(
                record_path.name,
                is_pvc_pred, matched_ref_idx, pvc_labels,
                len(ref_samples), sampling_rate, duration_s,
            )
    else:
        summary = compute_vt_detection_metrics_matched(
            record_path.name,
            is_vt_pred,
            matched_ref_idx,
            vt_labels,
            len(ref_samples),
            len(detected_samples),
            sampling_rate,
            duration_s,
        )
    return summary, features, times, clean_signal

# ----------------------------------------------------------------------
# Batch processing (now thin)
# ----------------------------------------------------------------------
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
    evaluation_mode="beats",
    detection_rule="weighted",
    pvc_eval_ref=False,
    vt_min_consecutive_beats=3,
    vt_max_rr_s=0.60,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(database_path)
    records = list_records(db_path)
    print(f"[INFO] Found {len(records)} records in {database_path}")
    print(f"[INFO] Evaluation mode: {evaluation_mode}", end="")
    if evaluation_mode == "pvc":
        print(f" ({'reference‑based' if pvc_eval_ref else 'matched‑beats'})", end="")
    print(f" | Detection rule: {detection_rule}")

    summaries = []
    for record_name in records:
        record_path = db_path / record_name

        # For PVC evaluation, skip records with < 10 reference PVC beats.
        # For VT evaluation, ignore records with no VT reference beats.
        if evaluation_mode in {"pvc", "vt"}:
            try:
                _, _, pvc_labels, vt_labels = load_reference_beats(record_path)
                if evaluation_mode == "pvc":
                    if int(np.sum(pvc_labels)) < 10:
                        print(f"[SKIP] {record_name} (only {int(np.sum(pvc_labels))} reference PVCs)")
                        continue
                elif int(np.sum(vt_labels)) == 0:
                    print(f"[SKIP] {record_name} (0 reference VT beats)")
                    continue
            except Exception as e:
                print(f"[SKIP] {record_name} (failed to load reference: {e})")
                continue
        
        try:
            print(f"[PROC] {record_name} ...", end=" ", flush=True)
            summary, features, times, signal = evaluate_record(
                record_path,
                min_peak_distance_s=min_peak_distance_s,
                refractory_s=refractory_s,
                prematurity_threshold=prematurity_threshold,
                qrs_width_threshold_ms=qrs_width_threshold_ms,
                tolerance_ms=tolerance_ms,
                evaluation_mode=evaluation_mode,
                detection_rule=detection_rule,
                pvc_eval_ref=pvc_eval_ref,
                vt_min_consecutive_beats=vt_min_consecutive_beats,
                vt_max_rr_s=vt_max_rr_s,
            )
            summaries.append(summary)

            # Print a brief status
            if evaluation_mode == "beats":
                print(f"OK | ref={summary['reference_beats']} det={summary['detected_beats']} "
                      f"TP={summary['TP']} FP={summary['FP']} FN={summary['FN']} "
                      f"SE={summary['SE_percent']:.2f}% PPV={summary['PPV_percent']:.2f}%")
            elif evaluation_mode == "pvc":
                print(f"OK | refPVC={summary['reference_pvc_beats']} cand={summary['detected_pvc_candidates']} "
                      f"TP={summary['TP_pvc']} FP={summary['FP_pvc']} FN={summary['FN_pvc']} TN={summary['TN_pvc']} "
                      f"Sens={summary['Sensitivity_pvc_percent']:.2f}% "
                      f"Spec={summary['Specificity_pvc_percent']:.2f}% "
                      f"Acc={summary['Accuracy_pvc_percent']:.2f}%")
            else:
                print(f"OK | refVT={summary['reference_vt_beats']} cand={summary['detected_vt_candidates']} "
                      f"TP={summary['TP_vt']} FP={summary['FP_vt']} FN={summary['FN_vt']} TN={summary['TN_vt']} "
                      f"Sens={summary['Sensitivity_vt_percent']:.2f}% "
                      f"Spec={summary['Specificity_vt_percent']:.2f}% "
                      f"Acc={summary['Accuracy_vt_percent']:.2f}%")

            # Save per‑record summary
            if evaluation_mode == "beats":
                prefix = "beat"
            elif evaluation_mode == "pvc":
                prefix = "pvc"
            else:
                prefix = "vt"
            suffix = "_ref" if (evaluation_mode == "pvc" and pvc_eval_ref) else ""
            csv_path = output_dir / f"{record_name}_{prefix}_eval{suffix}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
                writer.writeheader()
                writer.writerow(summary)

            # Save full feature table
            features_csv = output_dir / f"{record_name}_detected_beats.csv"
            if features:
                feat_df = pd.DataFrame(features)
                feat_df.to_csv(features_csv, index=False)

            if not skip_plots:
                plot_file = output_dir / f"{record_name}_detected_peaks.png"
                save_peak_time_plot(times, signal, features, str(plot_file), show_plot=show_plots)

        except Exception as e:
            print(f"ERROR: {e}")
            summaries.append({"record": record_name, "error": str(e)})

    # Global aggregation
    if summaries:
        if evaluation_mode == "beats":
            summary_df = aggregate_beat_metrics(summaries)
        elif evaluation_mode == "pvc":
            summary_df = aggregate_pvc_metrics(summaries)
        else:
            summary_df = aggregate_vt_metrics(summaries)
    else:
        summary_df = pd.DataFrame()

    # Save global CSV
    if evaluation_mode == "beats":
        mode_tag = "beats"
    elif evaluation_mode == "pvc":
        mode_tag = f"pvc{'ref' if pvc_eval_ref else ''}"
    else:
        mode_tag = "vt"
    summary_csv = output_dir / f"database_summary_{mode_tag}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[INFO] Saved summary to {summary_csv}")

    # Print concise table
    if not summary_df.empty:
        print("\n[SUMMARY]")
        if evaluation_mode == "beats":
            cols = ["record", "reference_beats", "detected_beats",
                    "TP", "FP", "FN", "SE_percent", "PPV_percent", "F1_percent"]
        elif evaluation_mode == "pvc":
            cols = ["record", "reference_beats", "detected_beats", "reference_pvc_beats", "detected_pvc_candidates",
                    "TP_pvc", "FP_pvc", "FN_pvc", "TN_pvc",
                    "Sensitivity_pvc_percent", "Specificity_pvc_percent",
                "Accuracy_pvc_percent", "PPV_pvc_percent", "F1_pvc_percent",
                "reference_pvc_burden_percent", "detected_pvc_burden_percent",
            "reference_pvc_rate_per_hour", "detected_pvc_rate_per_hour",
            "reference_possible_vt", "detected_possible_vt"]
        else:
            cols = ["record", "reference_beats", "detected_beats", "reference_vt_beats", "detected_vt_candidates",
                    "reference_vt_episodes", "detected_vt_episodes",
                    "TP_vt", "FP_vt", "FN_vt", "TN_vt",
                    "Sensitivity_vt_percent", "Specificity_vt_percent",
                    "Accuracy_vt_percent", "PPV_vt_percent", "F1_vt_percent"]
        print(summary_df[cols].to_string(index=False))

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MIT-BIH or local ECG CSV beat / PVC detection evaluation")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--database", help="Path to MIT-BIH WFDB database directory")
    input_group.add_argument("--ecg-csv", help="Path to a local ECG CSV file with columns time_s,ecg")
    parser.add_argument("--output", default="mitdb_results", help="Output directory")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--sample-record", default=None, help="Record name to visualise with extra plots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--min-peak-distance", type=float, default=0.06)
    parser.add_argument("--refractory", type=float, default=0.12)
    parser.add_argument("--prematurity-threshold", type=float, default=0.85)
    parser.add_argument("--qrs-width-threshold-ms", type=float, default=120.0)
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    parser.add_argument("--evaluation-mode", choices=["beats", "pvc", "vt"], default="beats",
                        help="Evaluation mode: 'beats' (QRS detection), 'pvc' (PVC detection), or 'vt' (VT rhythm detection)")
    parser.add_argument("--pvc-eval-ref", action="store_true",
                        help="If set, evaluate PVC rule directly on reference beats (ignoring detector). Only used when --evaluation-mode=pvc")
    parser.add_argument("--detection-rule", choices=["and", "or", "weighted", "mlp", "2of4"], default="and",
                        help="PVC detection rule: 'and' (strict), 'or' (loose), 'weighted' (heuristic score), 'mlp' (trained neural baseline)")
    parser.add_argument("--vt-min-consecutive-beats", type=int, default=3,
                        help="Minimum consecutive PVC-like beats for VT candidate runs (used when --evaluation-mode=vt)")
    parser.add_argument("--vt-max-rr-s", type=float, default=0.60,
                        help="Maximum RR interval (s) between consecutive beats inside a VT run (used when --evaluation-mode=vt)")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.ecg_csv:
        evaluate_ecg_csv(
            csv_path=args.ecg_csv,
            output_dir=args.output,
            skip_plots=args.skip_plots,
            show_plots=args.show,
            min_peak_distance_s=args.min_peak_distance,
            refractory_s=args.refractory,
            prematurity_threshold=args.prematurity_threshold,
            qrs_width_threshold_ms=args.qrs_width_threshold_ms,
            detection_rule=args.detection_rule,
        )
        return

    batch_process_database(
        args.database, args.output,
        skip_plots=args.skip_plots,
        sample_record=args.sample_record,
        show_plots=args.show,
        min_peak_distance_s=args.min_peak_distance,
        refractory_s=args.refractory,
        prematurity_threshold=args.prematurity_threshold,
        qrs_width_threshold_ms=args.qrs_width_threshold_ms,
        tolerance_ms=args.tolerance_ms,
        evaluation_mode=args.evaluation_mode,
        detection_rule=args.detection_rule,
        pvc_eval_ref=args.pvc_eval_ref,
        vt_min_consecutive_beats=args.vt_min_consecutive_beats,
        vt_max_rr_s=args.vt_max_rr_s,
    )

if __name__ == "__main__":
    main()
