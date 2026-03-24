import bitalino
import time
from helpers.constants import MAC_ADDRESS, BATTERY_THRESHOLD, CHANNELS, SAMPLING_RATE, NSAMPLES, DIGITAL_OUTPUT_ON, DIGITAL_OUTPUT_OFF, RUNNING_TIME


if __name__ == "__main__":
    device = bitalino.BITalino(MAC_ADDRESS, 10)
    device.battery(BATTERY_THRESHOLD)
    device.start(SAMPLING_RATE, CHANNELS)

    start = time.time()
    end = time.time()
    while (end - start) < RUNNING_TIME:
        print(device.read(NSAMPLES))
        end = time.time()

    device.trigger(DIGITAL_OUTPUT_ON)

    time.sleep(RUNNING_TIME)

    device.trigger(DIGITAL_OUTPUT_OFF)

    device.stop()

    device.close()

