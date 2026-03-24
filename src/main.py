import bitalino
import time
import numpy as np
from src.helpers.constants import (
    MAC_ADDRESS,
    BATTERY_THRESHOLD,
    CHANNELS,
    SAMPLING_RATE,
    NSAMPLES,
    DIGITAL_OUTPUT_ON,
    DIGITAL_OUTPUT_OFF,
    RUNNING_TIME,
    ECG_ANALOG_CHANNEL,
    ECG_OUTPUT_FILE,
)
from src.helpers.plot_signals import plot_signals


if __name__ == "__main__":
    device = bitalino.BITalino(MAC_ADDRESS, 10)
    device.battery(BATTERY_THRESHOLD)
    device.start(SAMPLING_RATE, CHANNELS)

    all_samples = []
    start = time.time()
    end = time.time()
    while (end - start) < RUNNING_TIME:
        batch = device.read(NSAMPLES)
        all_samples.append(batch)
        end = time.time()

    device.trigger(DIGITAL_OUTPUT_ON)

    time.sleep(RUNNING_TIME)

    device.trigger(DIGITAL_OUTPUT_OFF)

    device.stop()

    device.close()

    samples = np.vstack(all_samples)
    ecg_column = 5 + ECG_ANALOG_CHANNEL
    ecg = samples[:, ecg_column]
    times = np.arange(ecg.shape[0]) / SAMPLING_RATE
    np.savetxt(
        ECG_OUTPUT_FILE,
        np.column_stack((times, ecg)),
        delimiter=",",
        header="time_s,ecg",
        comments="",
    )
    plot_signals(ecg.reshape(-1, 1), SAMPLING_RATE)
    print(f"Saved ECG to {ECG_OUTPUT_FILE}")
