#!/usr/bin/env python3
"""
Analyze PVC detection performance per sample to identify parameter tuning opportunities.

Usage:
  python src/analyze_pvc_performance.py --database mitdb --output mitdb_analysis
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from src.extract_features import extract_extrasystole_features
from src.helpers.signal_processing import preprocess_ecg_for_arrhythmia

# Constants from run_mitdb.py
NON_BEAT_ANNOTATION_SYMBOLS = {
    "+", "~", "|", "[", "]", "x", "(", ")", "!", "p", "t", "u",
    "`", "'", "^", "=", "s", "T", "*", "@", "J", "a", "S", "e", "j", "F",
}
PVC_ANNOTATION_SYMBOLS = {"V"}


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
    return np.asarray(ref_samples, dtype=int), ref_symbols, np.asarray(pvc_labels, dtype=int)


def analyze_sample(
    record_path: Path,
    min_peak_distance_s=0.06,
    refractory_s=0.12,
    prematurity_threshold=0.95,
    qrs_width_threshold_ms=95.0,
):
    """Analyze a single MITDB sample and return diagnostic features."""
    record_name = record_path.stem
    
    # Load signal and reference
    times, signal, sampling_rate = load_physiobank_record(record_path)
    preprocessed = preprocess_ecg_for_arrhythmia(signal, sampling_rate, notch_hz=60.0)
    clean_signal = preprocessed["cleaned"]
    
    ref_samples, ref_symbols, pvc_labels = load_reference_beats(record_path)
    n_ref_pvcs = int(np.sum(pvc_labels))
    n_ref_normals = len(pvc_labels) - n_ref_pvcs
    
    # Extract features with current parameters
    features = extract_extrasystole_features(
        times, clean_signal,
        sampling_rate=sampling_rate,
        min_peak_distance_s=min_peak_distance_s,
        prematurity_threshold=prematurity_threshold,
        qrs_width_threshold_ms=qrs_width_threshold_ms,
        refractory_s=refractory_s,
    )
    
    # Compute statistics on detected beats
    if features:
        prem_indices = np.array([f["prematurity_index"] for f in features if np.isfinite(f["prematurity_index"])])
        qrs_widths = np.array([f["qrs_width_ms"] for f in features if np.isfinite(f["qrs_width_ms"])])
        rr_baselines = np.array([f["rr_baseline_s"] for f in features if np.isfinite(f["rr_baseline_s"])])
        
        prem_median = np.median(prem_indices) if len(prem_indices) > 0 else np.nan
        prem_std = np.std(prem_indices) if len(prem_indices) > 0 else np.nan
        qrs_median = np.median(qrs_widths) if len(qrs_widths) > 0 else np.nan
        qrs_std = np.std(qrs_widths) if len(qrs_widths) > 0 else np.nan
        rr_baseline_median = np.median(rr_baselines) if len(rr_baselines) > 0 else np.nan
    else:
        prem_median = prem_std = qrs_median = qrs_std = rr_baseline_median = np.nan
    
    n_candidates = int(sum(f["is_pvc_candidate"] for f in features))
    
    # Estimate how many PVCs would be caught if we loosen thresholds
    n_either = 0  # premature OR wide
    n_premature = 0
    n_wide = 0
    for f in features:
        prem = np.isfinite(f["prematurity_index"]) and f["prematurity_index"] < prematurity_threshold
        wide = np.isfinite(f["qrs_width_ms"]) and f["qrs_width_ms"] > qrs_width_threshold_ms
        if prem:
            n_premature += 1
        if wide:
            n_wide += 1
        if prem or wide:
            n_either += 1
    
    return {
        "record": record_name,
        "ref_beats": len(pvc_labels),
        "ref_pvc": n_ref_pvcs,
        "ref_normal": n_ref_normals,
        "detected_beats": len(features),
        "pvc_candidates": n_candidates,
        "candidates_prem_only": n_premature,
        "candidates_wide_only": n_wide,
        "candidates_prem_or_wide": n_either,
        "prem_idx_median": f"{prem_median:.3f}" if np.isfinite(prem_median) else "N/A",
        "prem_idx_std": f"{prem_std:.3f}" if np.isfinite(prem_std) else "N/A",
        "qrs_width_median_ms": f"{qrs_median:.1f}" if np.isfinite(qrs_median) else "N/A",
        "qrs_width_std_ms": f"{qrs_std:.1f}" if np.isfinite(qrs_std) else "N/A",
        "rr_baseline_median_s": f"{rr_baseline_median:.3f}" if np.isfinite(rr_baseline_median) else "N/A",
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze PVC detection parameters per sample")
    parser.add_argument("--database", required=True, help="Path to MIT-BIH database")
    parser.add_argument("--output", default="mitdb_analysis", help="Output directory")
    parser.add_argument("--min-peak-distance", type=float, default=0.06)
    parser.add_argument("--refractory", type=float, default=0.12)
    parser.add_argument("--prematurity-threshold", type=float, default=0.95)
    parser.add_argument("--qrs-width-threshold-ms", type=float, default=95.0)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = Path(args.database)
    records = list_records(db_path)
    
    print(f"[INFO] Analyzing {len(records)} records...")
    analyses = []
    
    for record_name in records:
        record_path = db_path / record_name
        try:
            print(f"[ANALYZE] {record_name} ...", end=" ", flush=True)
            analysis = analyze_sample(
                record_path,
                min_peak_distance_s=args.min_peak_distance,
                refractory_s=args.refractory,
                prematurity_threshold=args.prematurity_threshold,
                qrs_width_threshold_ms=args.qrs_width_threshold_ms,
            )
            analyses.append(analysis)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Save analysis CSV
    if analyses:
        df = pd.DataFrame(analyses)
        output_csv = output_dir / "pvc_analysis.csv"
        df.to_csv(output_csv, index=False)
        
        print(f"\n[INFO] Analysis saved to {output_csv}")
        print("\n[SUMMARY]")
        cols = ["record", "ref_pvc", "detected_beats", "pvc_candidates", 
                "candidates_prem_or_wide", "prem_idx_median", "qrs_width_median_ms"]
        print(df[cols].to_string(index=False))
        
        print("\n[GUIDANCE]")
        print("Columns explained:")
        print("  pvc_candidates: beats matching current AND rule (premature AND wide)")
        print("  candidates_prem_only: beats that are only premature (not wide)")
        print("  candidates_wide_only: beats that are only wide (not premature)")
        print("  candidates_prem_or_wide: beats matching looser OR rule (premature OR wide)")
        print("\nIf 'pvc_candidates' is much lower than 'ref_pvc', try:")
        print("  1. Lower prematurity_threshold (e.g., 0.80 instead of 0.95)")
        print("  2. Lower qrs_width_threshold_ms (e.g., 85 instead of 95)")
        print("  3. Use 'candidates_prem_or_wide' as an alternative detection rule")


if __name__ == "__main__":
    main()
