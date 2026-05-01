from collections import deque

import matplotlib.pyplot as plt
import numpy as np


CHANNELS_TO_SENSORS = {
    "0": "EMG",
    "1": "ECG",
    "2": "EDA",
    "3": "EEG",
    "4": "Accelerometer",
}


def plot_signals(signals, sampling_rate, duration_sec=30):
    # Handle both 1D (single channel) and 2D (multi-channel) arrays
    if signals.ndim == 1:
        num_samples = int(duration_sec * sampling_rate)
        signals_subset = signals[:num_samples]
        time = np.arange(signals_subset.shape[0]) / sampling_rate
        plt.figure(figsize=(12, 6))
        plt.plot(time, signals_subset)
        plt.xlabel('Time (s)')
        plt.ylabel('Signal Value')
        plt.title(f'Signal (First {duration_sec}s)')
        plt.grid(True)
        plt.show()
    else:
        num_samples = int(duration_sec * sampling_rate)
        signals_subset = signals[:num_samples]
        time = np.arange(signals_subset.shape[0]) / sampling_rate
        plt.figure(figsize=(12, 6))
        plt.plot(time, signals_subset)
        plt.xlabel('Time (s)')
        plt.ylabel('Signal Value')
        plt.title(f'Bitalino Signals (First {duration_sec}s)')
        plt.grid(True)
        plt.show()
    # plt.legend() # Only use if your signals array has headers/labels
    plt.show()


def setup_live_plot(channels, sampling_rate, window_seconds):

    max_points = max(1, int(window_seconds * sampling_rate))

    plt.ion()
    figure, axes = plt.subplots(len(channels), 1, figsize=(12, 2.8 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]

    lines = []
    buffers = []
    time_buffers = []
    for axis, channel in zip(axes, channels):
        channel = CHANNELS_TO_SENSORS.get(str(channel), f"A{channel}")
        line, = axis.plot([], [], linewidth=1)
        axis.set_ylabel(f"{channel}")
        axis.grid(True, alpha=0.3)
        axis.set_xlim(0, window_seconds)
        axis.set_ylim(0, 1024)
        lines.append(line)
        buffers.append(deque(maxlen=max_points))
        time_buffers.append(deque(maxlen=max_points))

    axes[-1].set_xlabel("Time (s)")
    figure.suptitle("BITalino live analog signals")
    figure.tight_layout()

    return figure, lines, buffers, time_buffers


def update_live_plot(lines, buffers, time_buffers):
    max_time = 0.0
    for line, buffer_values, buffer_times in zip(lines, buffers, time_buffers):
        if not buffer_values:
            continue

        x = np.array(buffer_times)
        y = np.array(buffer_values)
        line.set_data(x, y)
        max_time = max(max_time, float(x[-1]))

    for line in lines:
        axis = line.axes
        xmin = 0.0
        xmax = max(max_time, 1.0)
        axis.set_xlim(xmin, xmax)

    canvas = lines[0].figure.canvas
    canvas.draw_idle()
    canvas.flush_events()
    plt.pause(0.001)


def save_full_acquisition_plot(all_analog_batches, channels, sampling_rate, output_file, show_plot=False):
    if not all_analog_batches:
        return

    signals = np.vstack(all_analog_batches)
    time_axis = np.arange(signals.shape[0]) / sampling_rate

    figure, axes = plt.subplots(len(channels), 1, figsize=(12, 2.8 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]

    for axis, channel_idx in zip(axes, range(len(channels))):
        channel = channels[channel_idx]
        channel_label = CHANNELS_TO_SENSORS.get(str(channel), f"A{channel}")
        axis.plot(time_axis, signals[:, channel_idx], linewidth=1)
        axis.set_ylabel(channel_label)
        axis.grid(True, alpha=0.3)
        axis.set_ylim(0, 1024)

    axes[-1].set_xlabel("Time (s)")
    figure.suptitle("BITalino full acquisition")
    figure.tight_layout()

    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved full acquisition plot to {output_file}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)