import numpy as np
from scipy import signal

from src.helpers.constants import ECG_ANALOG_CHANNEL, ECG_OUTPUT_FILE


def validate_channels(channels):
    if not channels:
        raise ValueError("At least one analog channel must be provided")

    normalized = []
    for channel in channels:
        value = int(channel)
        if value < 0 or value > 5:
            raise ValueError("Analog channels must be in the range [0, 5]")
        normalized.append(value)
    return normalized


def extract_analog_signals(batch, n_channels):
    return batch[:, 5 : 5 + n_channels]


def load_ecg_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if "time_s" not in data.dtype.names or "ecg" not in data.dtype.names:
        raise ValueError("CSV must contain 'time_s' and 'ecg' columns")

    times = np.asarray(data["time_s"], dtype=float)
    ecg = np.asarray(data["ecg"], dtype=float)
    if times.size < 4:
        raise ValueError("ECG CSV must contain at least 4 samples")

    dt = np.median(np.diff(times))
    if dt <= 0:
        raise ValueError("Invalid timestamps in ECG CSV")

    sampling_rate = 1.0 / dt
    return times, ecg, sampling_rate


def _lowpass_for_artifacts(ecg, sampling_rate, cutoff_hz=40.0, order=4):
    nyquist = 0.5 * sampling_rate
    normalized = cutoff_hz / nyquist
    sos = signal.butter(order, normalized, btype="lowpass", output="sos")
    return signal.sosfiltfilt(sos, ecg)


def bandpass_filter_ecg(ecg, sampling_rate, low_hz=0.5, high_hz=40.0, order=4):
    nyquist = 0.5 * sampling_rate
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError("Bandpass frequencies must satisfy 0 < low_hz < high_hz < Nyquist")

    sos = signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, ecg)


def notch_filter_ecg(ecg, sampling_rate, notch_hz=50.0, quality=30.0):
    nyquist = 0.5 * sampling_rate
    if notch_hz <= 0 or notch_hz >= nyquist:
        return ecg.copy()

    b, a = signal.iirnotch(w0=notch_hz, Q=quality, fs=sampling_rate)
    return signal.filtfilt(b, a, ecg)


def suppress_transient_artifacts(ecg, z_threshold=6.0, smooth_window=9):
    if ecg.size < 5:
        return ecg.copy(), 0

    diff = np.diff(ecg, prepend=ecg[0])
    mad = np.median(np.abs(diff - np.median(diff))) + 1e-9
    robust_z = np.abs(diff - np.median(diff)) / (1.4826 * mad)
    artifact_mask = robust_z > z_threshold

    if not np.any(artifact_mask):
        return ecg.copy(), 0

    filtered = ecg.copy()
    kernel = np.ones(smooth_window, dtype=float) / smooth_window
    smoothed = np.convolve(ecg, kernel, mode="same")
    filtered[artifact_mask] = smoothed[artifact_mask]
    return filtered, int(np.sum(artifact_mask))


def compute_signal_quality(ecg):
    if ecg.size == 0:
        return {"std": 0.0, "peak_to_peak": 0.0, "clipping_ratio": 0.0}

    std = float(np.std(ecg))
    peak_to_peak = float(np.max(ecg) - np.min(ecg))
    min_v = float(np.min(ecg))
    max_v = float(np.max(ecg))
    clipping_count = np.sum((ecg <= min_v) | (ecg >= max_v))
    clipping_ratio = float(clipping_count / ecg.size)

    return {
        "std": std,
        "peak_to_peak": peak_to_peak,
        "clipping_ratio": clipping_ratio,
    }


def preprocess_ecg_for_arrhythmia(
    ecg,
    sampling_rate,
    low_hz=0.5,
    high_hz=40.0,
    notch_hz=50.0,
    notch_quality=30.0,
):
    baseline_suppressed = bandpass_filter_ecg(ecg, sampling_rate, low_hz=low_hz, high_hz=high_hz)
    powerline_removed = notch_filter_ecg(
        baseline_suppressed,
        sampling_rate,
        notch_hz=notch_hz,
        quality=notch_quality,
    )
    artifact_reduced, artifact_samples = suppress_transient_artifacts(powerline_removed)

    # Optional low-pass clean-up to reduce remaining broadband noise.
    final = _lowpass_for_artifacts(artifact_reduced, sampling_rate, cutoff_hz=high_hz)

    return {
        "cleaned": final,
        "artifact_samples": artifact_samples,
        "quality_raw": compute_signal_quality(ecg),
        "quality_cleaned": compute_signal_quality(final),
    }


def save_outputs(all_analog_batches, channels, sampling_rate, output_file):
    if not all_analog_batches:
        print("[WARN] No samples were collected; skipping CSV export.")
        return

    analog_samples = np.vstack(all_analog_batches)
    times = np.arange(analog_samples.shape[0]) / sampling_rate

    header = "time_s," + ",".join(f"A{channel}" for channel in channels)
    data = np.column_stack((times, analog_samples))
    np.savetxt(output_file, data, delimiter=",", header=header, comments="")
    print(f"[INFO] Saved analog channels to {output_file}")

    if ECG_ANALOG_CHANNEL in channels:
        ecg_idx = channels.index(ECG_ANALOG_CHANNEL)
        ecg = analog_samples[:, ecg_idx]
        np.savetxt(
            ECG_OUTPUT_FILE,
            np.column_stack((times, ecg)),
            delimiter=",",
            header="time_s,ecg",
            comments="",
        )
        print(f"[INFO] Saved ECG channel (A{ECG_ANALOG_CHANNEL}) to {ECG_OUTPUT_FILE}")