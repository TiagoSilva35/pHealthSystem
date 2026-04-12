import time
import bitalino
import matplotlib.pyplot as plt
import numpy as np

from src.helpers.constants import (
    BATTERY_THRESHOLD,
    CHANNELS,
    ECG_ANALOG_CHANNEL,
    ECG_OUTPUT_FILE,
    FULL_PLOT_OUTPUT_FILE,
    MAC_ADDRESS,
    NSAMPLES,
    RUNNING_TIME,
    SAMPLING_RATE,
    SCAN_DURATION,
    SIGNALS_OUTPUT_FILE,
    WINDOW_SECONDS,
    ENABLE_LIVE_PLOT,
)
from src.helpers.bluetooth_utils import choose_device, discover_nearby_devices
from src.helpers.plot_signals import save_full_acquisition_plot, setup_live_plot, update_live_plot
from src.helpers.signal_processing import extract_analog_signals, save_outputs, validate_channels

BLUETOOTH_IMPORT_ERROR = None


def main():
    channels = validate_channels(CHANNELS)
    discovered = discover_nearby_devices(SCAN_DURATION)
    selected_mac = choose_device(discovered, MAC_ADDRESS)
    candidate_macs = [selected_mac]
    for addr, name in discovered:
        if "bitalino" in name.lower() and addr not in candidate_macs:
            candidate_macs.append(addr)
    if MAC_ADDRESS and MAC_ADDRESS not in candidate_macs:
        candidate_macs.append(MAC_ADDRESS)

    if len(candidate_macs) > 1:
        print(f"[INFO] Will try {len(candidate_macs)} BITalino candidate address(es) if needed.")

    device = None
    figure = None
    all_analog_batches = []
    timeout_reached = False

    try:
        last_connect_error = None
        for mac in candidate_macs:
            print(f"[INFO] Connecting to BITalino at {mac}...")
            try:
                device = bitalino.BITalino(mac, 10)
                selected_mac = mac
                break
            except UnicodeDecodeError as exc:
                last_connect_error = exc
                print(f"[WARN] Handshake failed at {mac} (invalid BITalino version response).")
                continue
            except Exception as exc:
                last_connect_error = exc
                print(f"[WARN] Connection failed at {mac}: {exc}")
                continue

        if device is None:
            raise RuntimeError(
                "Unable to establish a valid BITalino session. "
                "Check that the selected MAC is the BITalino board, Bluetooth pairing is correct, "
                "and no other process is connected to the device."
            ) from last_connect_error

        device.battery(BATTERY_THRESHOLD)
        device.start(SAMPLING_RATE, channels)
        print("[INFO] Acquisition started. Press Ctrl+C to stop.")

        if ENABLE_LIVE_PLOT:
            figure, lines, buffers, time_buffers = setup_live_plot(channels, SAMPLING_RATE, WINDOW_SECONDS)
        else:
            lines, buffers, time_buffers = [], [], []

        start_time = time.time()
        total_samples = 0
        while True:
            print(f"Running for {(time.time() - start_time):.1f} seconds... ", end="\r")    
            if RUNNING_TIME > 0 and (time.time() - start_time) >= RUNNING_TIME:
                timeout_reached = True
                print(f"\n[INFO] Reached specified running time of {RUNNING_TIME} seconds. Want to continue? (y/n): ", end="")
                user_input = input().strip().lower()
                if user_input == "n":
                    print("[INFO] Stopping acquisition.")
                    break
                else:
                    if ENABLE_LIVE_PLOT:
                        for b in buffers:
                            b.clear()
                        for tb in time_buffers:
                            tb.clear()
                    total_samples = 0
                    start_time = time.time()
                    print("[INFO] Starting new acquisition period...")
                    continue
                    

            batch = device.read(NSAMPLES)
            analog_batch = extract_analog_signals(batch, len(channels))
            all_analog_batches.append(analog_batch)

            if ENABLE_LIVE_PLOT:
                batch_size = analog_batch.shape[0]
                sample_times = (total_samples + np.arange(batch_size)) / SAMPLING_RATE
                for channel_idx in range(analog_batch.shape[1]):
                    buffers[channel_idx].extend(analog_batch[:, channel_idx].tolist())
                    time_buffers[channel_idx].extend(sample_times.tolist())
                update_live_plot(lines, buffers, time_buffers)

            total_samples += analog_batch.shape[0]

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
            if timeout_reached:
                plot_file = "ecg_live_plot.png"
                figure.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close(figure)
                print(f"\n[INFO] Saved live plot to {plot_file}")
            else:
                plt.show()

    save_outputs(all_analog_batches, channels, SAMPLING_RATE, SIGNALS_OUTPUT_FILE)
    save_full_acquisition_plot(
        all_analog_batches,
        channels,
        SAMPLING_RATE,
        FULL_PLOT_OUTPUT_FILE,
        show_plot=not timeout_reached,
    )


if __name__ == "__main__":
    main()
