import numpy as np
import pandas as pd
from scipy import signal
# Assuming SAMPLING_RATE is imported from your constants
from src.helpers.constants import SAMPLING_RATE 
from src.helpers.plot_signals import plot_signals

class Pan_Tompkins_QRS:
    def __init__(self, fs=SAMPLING_RATE):
        self.fs = fs

    def band_pass_filter(self, ecg_signal):
        """
        Applies a 5-15 Hz bandpass filter. 
        Uses filtfilt for zero-phase filtering to avoid peak delay.
        """
        nyq = 0.5 * self.fs
        low = 5 / nyq
        high = 15 / nyq
        # 4th order Butterworth filter 
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_ecg = signal.filtfilt(b, a, ecg_signal)
        return filtered_ecg

    def derivative(self, ecg_signal):
        """Standard Pan-Tompkins derivative filter"""
        return np.diff(ecg_signal, prepend=ecg_signal[0])

    def squaring(self, ecg_signal):
        """Amplify QRS complex and ensure positive values"""
        return np.square(ecg_signal)

    def moving_window_integration(self, ecg_signal):
        """
        Calculates the energy envelope using a moving average window.
        
        """
        window_size = int(0.150 * self.fs)
        # Using convolution for a more efficient moving average [cite: 28]
        return np.convolve(ecg_signal, np.ones(window_size)/window_size, mode='same')

    def detect_peaks(self, mwin_signal):
        """
        Applies adaptive thresholding and physiological constraints.
        """
        peaks = []
        # Initial adaptive threshold: 0.7 * mean of the energy signal 
        threshold = 0.7 * np.mean(mwin_signal)
        # Refractory period: ~0.3s (physiologically impossible to have 2 beats) 
        refractory_samples = int(0.3 * self.fs)
        
        last_peak_idx = -refractory_samples
        
        for i in range(1, len(mwin_signal) - 1):
            # 1. Local peak detection
            if mwin_signal[i] > mwin_signal[i-1] and mwin_signal[i] > mwin_signal[i+1]:
                # 2. Thresholding
                if mwin_signal[i] > threshold:
                    # 3. Refractory period check
                    if (i - last_peak_idx) > refractory_samples:
                        peaks.append(i)
                        last_peak_idx = i
                        # Update threshold adaptively based on the latest peak
                        threshold = 0.5 * threshold + 0.5 * (0.7 * mwin_signal[i])
        
        return np.array(peaks)

    def solve(self, signal_df):
        # 1. Extract raw signal
        raw_signal = signal_df.iloc[:, 1].to_numpy()
        
        # 2. Pre-processing steps
        bpass = self.band_pass_filter(raw_signal)
        der = self.derivative(bpass)
        sqr = self.squaring(der)
        mwin = self.moving_window_integration(sqr)
        
        # 3. Decision Logic (Thresholding)
        peaks_indices = self.detect_peaks(mwin)
        
        # Return the energy envelope for plotting and the detected peak indices
        return mwin, peaks_indices

# Execution
QRS_detector = Pan_Tompkins_QRS(SAMPLING_RATE)
ecg = pd.read_csv('ecg_samples.csv')

# Process signal
mwin_signal, r_peaks = QRS_detector.solve(ecg)

# Heart Rate Calculation: Average of last 5-7 beats 
if len(r_peaks) > 1:
    rr_intervals = np.diff(r_peaks) / SAMPLING_RATE
    # Calculate BPM based on recent history
    avg_bpm = 60 / np.mean(rr_intervals[-7:])
    print(f"Detected Heart Rate: {avg_bpm:.2f} BPM")

# Save and Plot
final_signal = pd.DataFrame({'time_s': ecg.iloc[:, 0], 'ecg_integrated': mwin_signal})
final_signal.to_csv('output_signal.csv', index=False)

# Optional: Visualize R-peaks on the original signal
np_final_signal = final_signal.to_numpy()
plot_signals(np_final_signal, SAMPLING_RATE, 1)