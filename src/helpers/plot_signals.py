import matplotlib.pyplot as plt
import numpy as np

def plot_signals(signals, sampling_rate):
    
    time = np.arange(signals.shape[0]) / sampling_rate
    plt.figure(figsize=(12, 6))
    for i in range(signals.shape[1]):
        plt.plot(time, signals[:, i], label=f'Channel {i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Value')
    plt.title('Bitalino Signals')
    plt.legend()
    plt.grid()
    plt.show()