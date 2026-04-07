import time
from collections import deque
import socket
import bitalino
import matplotlib.pyplot as plt
import numpy as np
import bluetooth


from src.helpers.constants import (
    BATTERY_THRESHOLD,
    CHANNELS,
    ECG_ANALOG_CHANNEL,
    ECG_OUTPUT_FILE,
    MAC_ADDRESS,
    NSAMPLES,
    RUNNING_TIME,
    SAMPLING_RATE,
    SCAN_DURATION,
    SIGNALS_OUTPUT_FILE,
    WINDOW_SECONDS,
    ENABLE_LIVE_PLOT,
)

BLUETOOTH_IMPORT_ERROR = None


def validate_channels(channels):
    normalized = []
    for channel in channels:
        value = int(channel)
        if value < 0 or value > 5:
            raise ValueError("Analog channels must be in the range [0, 5]")
        normalized.append(value)
    return normalized


def discover_nearby_devices(scan_duration):
    try:
        raw = bluetooth.discover_devices(duration=scan_duration, lookup_names=True)
    except Exception as exc:
        print(f"[WARN] Failed to scan nearby Bluetooth devices: {exc}")
        return []

    normalized = []
    for item in raw:
        if isinstance(item, tuple) and len(item) >= 2:
            addr, name = item[0], item[1]
        else:
            addr, name = item, "Unknown"
        normalized.append((str(addr), str(name)))

    normalized.sort(key=lambda device: 0 if "bitalino" in device[1].lower() else 1)
    return normalized


def choose_device(discovered_devices, fallback_mac):
    if not discovered_devices:
        if fallback_mac:
            print(f"[INFO] No devices discovered. Falling back to configured MAC: {fallback_mac}")
            return fallback_mac
        raise RuntimeError("No Bluetooth devices discovered and no fallback MAC configured.")

    print("\nNearby Bluetooth devices:")
    for idx, (addr, name) in enumerate(discovered_devices, start=1):
        marker = " [BITalino?]" if "bitalino" in name.lower() else ""
        print(f"  {idx}. {name} ({addr}){marker}")

    while True:
        selection = input("Choose device number (Enter = first one): ").strip()
        if not selection:
            return discovered_devices[0][0]
        if not selection.isdigit():
            print("Invalid input. Please enter a valid number.")
            continue
        index = int(selection)
        if 1 <= index <= len(discovered_devices):
            return discovered_devices[index - 1][0]
        print(f"Invalid index. Please choose a value between 1 and {len(discovered_devices)}.")


def extract_analog_signals(batch, n_channels):
    return batch[:, 5 : 5 + n_channels]


def setup_live_plot(channels, sampling_rate, window_seconds):
    max_points = max(1, int(window_seconds * sampling_rate))
    x_axis = np.linspace(-window_seconds, 0, max_points)

    plt.ion()
    figure, axes = plt.subplots(len(channels), 1, figsize=(12, 2.8 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]

    lines = []
    buffers = []
    for axis, channel in zip(axes, channels):
        line, = axis.plot(x_axis, np.full(max_points, np.nan), linewidth=1)
        axis.set_ylabel(f"A{channel}")
        axis.grid(True, alpha=0.3)
        axis.set_ylim(0, 1024)
        lines.append(line)
        buffers.append(deque(maxlen=max_points))

    axes[-1].set_xlabel("Time (s)")
    figure.suptitle("BITalino live analog signals")
    figure.tight_layout()

    return figure, lines, buffers, x_axis


def update_live_plot(lines, buffers, x_axis):
    for line, buffer_values in zip(lines, buffers):
        if not buffer_values:
            continue

        y = np.full(x_axis.shape[0], np.nan)
        values = np.array(buffer_values)
        y[-len(values) :] = values
        line.set_ydata(y)

        axis = line.axes
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        if np.isfinite(ymin) and np.isfinite(ymax):
            if ymin == ymax:
                ymax = ymin + 1
            padding = 0.1 * (ymax - ymin)
            axis.set_ylim(ymin - padding, ymax + padding)

    canvas = lines[0].figure.canvas
    canvas.draw_idle()
    canvas.flush_events()
    plt.pause(0.001)


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


def main():
    channels = validate_channels(CHANNELS)
    discovered = discover_nearby_devices(SCAN_DURATION)
    selected_mac = choose_device(discovered, MAC_ADDRESS)
    print(f"[INFO] Connecting to BITalino at {selected_mac}...")
    device = None
    figure = None
    all_analog_batches = []

    try:
        device = bitalino.BITalino(selected_mac, 10)
        device.battery(BATTERY_THRESHOLD)
        device.start(SAMPLING_RATE, channels)
        print("[INFO] Acquisition started. Press Ctrl+C to stop.")

        if ENABLE_LIVE_PLOT:
            figure, lines, buffers, x_axis = setup_live_plot(channels, SAMPLING_RATE, WINDOW_SECONDS)
        else:
            lines, buffers, x_axis = [], [], None

        start_time = time.time()
        while True:
            if RUNNING_TIME > 0 and (time.time() - start_time) >= RUNNING_TIME:
                break

            batch = device.read(NSAMPLES)
            analog_batch = extract_analog_signals(batch, len(channels))
            all_analog_batches.append(analog_batch)

            if ENABLE_LIVE_PLOT:
                for channel_idx in range(analog_batch.shape[1]):
                    buffers[channel_idx].extend(analog_batch[:, channel_idx].tolist())
                update_live_plot(lines, buffers, x_axis)

    except KeyboardInterrupt:
        print("\n[INFO] Acquisition interrupted by user.")
    finally:
        if device is not None:
            try:
                device.stop()
            except Exception:
                pass
            try:
                device.close()
            except Exception:
                pass

        if figure is not None:
            plt.ioff()
            plt.show()

    save_outputs(all_analog_batches, channels, SAMPLING_RATE, SIGNALS_OUTPUT_FILE)


if __name__ == "__main__":
    main()
