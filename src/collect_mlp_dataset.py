#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.algorithms.mlp_pvc import FEATURE_KEYS, PACED_RECORDS, feature_dicts_to_matrix
from src.extract_features import extract_extrasystole_features
from src.helpers.signal_processing import preprocess_ecg_for_arrhythmia
from src.run_mitdb import (
    list_records,
    load_physiobank_record,
    load_reference_beats,
    match_detections_to_references,
)


def _labels_from_matches(matched_ref_idx, pvc_labels):
    matched_ref_idx = np.asarray(matched_ref_idx, dtype=int)
    labels = np.zeros(matched_ref_idx.shape[0], dtype=np.int64)
    matched_mask = matched_ref_idx >= 0
    labels[matched_mask] = pvc_labels[matched_ref_idx[matched_mask]]
    return labels


def collect_dataset(
    database_path,
    output_file,
    min_peak_distance_s=0.06,
    refractory_s=0.12,
    prematurity_threshold=0.85,
    qrs_width_threshold_ms=120.0,
    tolerance_ms=50.0,
    detection_rule="and",
):
    db_path = Path(database_path)
    records = list_records(db_path)
    if not records:
        raise ValueError(f"No records found in {database_path}")

    X_all = []
    y_all = []
    record_ids_all = []

    for record_name in records:
        # Skip paced records
        try:
            record_id = int(record_name)
            if record_id in PACED_RECORDS:
                print(f"[SKIP] {record_name} (paced record)")
                continue
        except ValueError:
            pass
        
        record_path = db_path / record_name
        print(f"[PROC] {record_name}")
        times, signal, sampling_rate = load_physiobank_record(record_path)
        cleaned = preprocess_ecg_for_arrhythmia(signal, sampling_rate, notch_hz=60.0)["cleaned"]
        ref_samples, _, pvc_labels = load_reference_beats(record_path)

        features = extract_extrasystole_features(
            times,
            cleaned,
            sampling_rate=sampling_rate,
            min_peak_distance_s=min_peak_distance_s,
            refractory_s=refractory_s,
            prematurity_threshold=prematurity_threshold,
            qrs_width_threshold_ms=qrs_width_threshold_ms,
            detection_rule=detection_rule,
        )

        if not features:
            print(f"[WARN] {record_name}: no detected beats")
            continue

        peak_times = np.asarray([row["peak_time_s"] for row in features], dtype=float)
        detected_samples = np.asarray(np.round(peak_times * sampling_rate), dtype=int)
        tol_samples = int(round((tolerance_ms / 1000.0) * sampling_rate))
        _, _, _, _, matched_ref_idx = match_detections_to_references(ref_samples, detected_samples, tol_samples)

        X_record = feature_dicts_to_matrix(features, feature_keys=FEATURE_KEYS)
        y_record = _labels_from_matches(matched_ref_idx, pvc_labels)

        n = min(X_record.shape[0], y_record.shape[0])
        if n == 0:
            continue
        X_record = X_record[:n]
        y_record = y_record[:n]

        try:
            record_id = int(record_name)
        except ValueError:
            record_id = -1
        record_ids = np.full(n, record_id, dtype=np.int32)

        X_all.append(X_record)
        y_all.append(y_record)
        record_ids_all.append(record_ids)

    if not X_all:
        raise ValueError("No beats were collected; dataset is empty.")

    X = np.vstack(X_all).astype(np.float32)
    y = np.concatenate(y_all).astype(np.int64)
    record_ids = np.concatenate(record_ids_all).astype(np.int32)

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        record_ids=record_ids,
        feature_keys=np.asarray(FEATURE_KEYS, dtype=object),
    )
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    print(f"[INFO] Saved dataset to {output_file}")
    print(f"[INFO] Shape: X={X.shape}, y={y.shape}")
    print(f"[INFO] Class balance: positives={positives}, negatives={negatives}")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect MIT-BIH dataset for PVC MLP training")
    parser.add_argument("--database", default="mitdb", help="Path to MIT-BIH WFDB database")
    parser.add_argument("--output", default="mlp_pvc_dataset.npz", help="Output NPZ dataset file")
    parser.add_argument("--min-peak-distance", type=float, default=0.06)
    parser.add_argument("--refractory", type=float, default=0.12)
    parser.add_argument("--prematurity-threshold", type=float, default=0.85)
    parser.add_argument("--qrs-width-threshold-ms", type=float, default=120.0)
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    parser.add_argument(
        "--detection-rule",
        choices=["and", "or", "weighted"],
        default="and",
        help="Rule used only for generating detected beats before labeling",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_dataset(
        database_path=args.database,
        output_file=args.output,
        min_peak_distance_s=args.min_peak_distance,
        refractory_s=args.refractory,
        prematurity_threshold=args.prematurity_threshold,
        qrs_width_threshold_ms=args.qrs_width_threshold_ms,
        tolerance_ms=args.tolerance_ms,
        detection_rule=args.detection_rule,
    )
