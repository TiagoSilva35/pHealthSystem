import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.helpers.signal_processing import load_ecg_csv, preprocess_ecg_for_arrhythmia


def process_ecg_csv(
    input_csv,
    output_csv,
    output_plot,
    low_hz,
    high_hz,
    notch_hz,
    notch_quality,
):
    times, ecg, sampling_rate = load_ecg_csv(input_csv)
    result = preprocess_ecg_for_arrhythmia(
        ecg,
        sampling_rate=sampling_rate,
        low_hz=low_hz,
        high_hz=high_hz,
        notch_hz=notch_hz,
        notch_quality=notch_quality,
    )
    cleaned = result["cleaned"]

    np.savetxt(
        output_csv,
        np.column_stack((times, ecg, cleaned)),
        delimiter=",",
        header="time_s,ecg_raw,ecg_cleaned",
        comments="",
    )

    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(times, ecg, linewidth=0.8, color="#d1495b")
    axes[0].set_title("Raw ECG")
    axes[0].set_ylabel("ADC")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, cleaned, linewidth=0.8, color="#2a9d8f")
    axes[1].set_title("Cleaned ECG (bandpass + notch + artifact suppression)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_plot, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"[INFO] Sampling rate inferred from CSV: {sampling_rate:.2f} Hz")
    print(f"[INFO] Saved cleaned ECG to {output_csv}")
    print(f"[INFO] Saved comparison plot to {output_plot}")
    print(f"[INFO] Artifact samples corrected: {result['artifact_samples']}")
    print(f"[INFO] Raw quality metrics: {result['quality_raw']}")
    print(f"[INFO] Cleaned quality metrics: {result['quality_cleaned']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean ECG signal from ecg_samples.csv")
    parser.add_argument("--input", default="ecg_samples.csv", help="Input ECG CSV file")
    parser.add_argument("--output", default="ecg_samples_cleaned.csv", help="Output cleaned ECG CSV file")
    parser.add_argument("--plot", default="ecg_cleaning_comparison.png", help="Output comparison plot")
    parser.add_argument("--low-hz", type=float, default=0.5, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--high-hz", type=float, default=40.0, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--notch-hz", type=float, default=55.0, help="Powerline notch frequency (Hz)")
    parser.add_argument("--notch-q", type=float, default=30.0, help="Notch filter quality factor")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_ecg_csv(
        input_csv=args.input,
        output_csv=args.output,
        output_plot=args.plot,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        notch_hz=args.notch_hz,
        notch_quality=args.notch_q,
    )
