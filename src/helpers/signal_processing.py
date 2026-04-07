import numpy as np

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