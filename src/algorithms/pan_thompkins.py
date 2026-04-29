import numpy as np
import pandas as pd
from scipy import signal

from src.helpers.constants import SAMPLING_RATE


class PanThompkinsQRS:
    """
    Slide-aligned Pan-Tompkins-style QRS detector.

    Matches the lecture flow:
      1) Optional preprocessing / baseline cleanup
      2) Band-pass filtering for QRS enhancement
      3) Differentiation
      4) Squaring
      5) Moving average (~0.2 s)
      6) Thresholding
      7) Pause / refractory rule (~0.3 s)
      8) Back-search to the original ECG
    """

    def __init__(self, fs=SAMPLING_RATE):
        self.fs = fs

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess_ecg(self, ecg_signal, use_preprocessing=True):
        """
        Keep preprocessing light so the PT chain is not over-filtered.

        The slides discuss baseline removal and noise reduction before
        analysis, but for the actual PT chain we avoid double filtering.
        """
        x = np.asarray(ecg_signal, dtype=float).copy()
        if not use_preprocessing:
            return x

        # Remove DC offset / slow shift.
        return signal.detrend(x, type="constant")

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def lowpass_filter(self, ecg_signal, cutoff_hz=35.0, order=4):
        nyq = 0.5 * self.fs
        if not 0 < cutoff_hz < nyq:
            raise ValueError("Lowpass cutoff must satisfy 0 < cutoff_hz < Nyquist")

        b, a = signal.butter(order, cutoff_hz / nyq, btype="low")
        return signal.filtfilt(b, a, np.asarray(ecg_signal, dtype=float))

    def highpass_filter(self, ecg_signal, cutoff_hz=2.0, order=4):
        nyq = 0.5 * self.fs
        if not 0 < cutoff_hz < nyq:
            raise ValueError("Highpass cutoff must satisfy 0 < cutoff_hz < Nyquist")

        b, a = signal.butter(order, cutoff_hz / nyq, btype="high")
        return signal.filtfilt(b, a, np.asarray(ecg_signal, dtype=float))

    def band_pass_filter(self, ecg_signal, low_hz=5.0, high_hz=15.0, order=4):
        nyq = 0.5 * self.fs
        if not 0 < low_hz < high_hz < nyq:
            raise ValueError("Bandpass frequencies must satisfy 0 < low_hz < high_hz < Nyquist")

        b, a = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
        return signal.filtfilt(b, a, np.asarray(ecg_signal, dtype=float))

    # ------------------------------------------------------------------
    # PT stages
    # ------------------------------------------------------------------
    def derivative(self, ecg_signal):
        """
        5-point derivative approximation.
        """
        x = np.asarray(ecg_signal, dtype=float)
        if x.size < 5:
            return np.zeros_like(x)

        y = np.zeros_like(x, dtype=float)
        for i in range(2, len(x) - 2):
            y[i] = (
                2 * x[i + 1]
                + x[i + 2]
                - 2 * x[i - 1]
                - x[i - 2]
            ) * (self.fs / 8.0)

        y[0] = y[2]
        y[1] = y[2]
        y[-2] = y[-3]
        y[-1] = y[-3]
        return y

    def squaring(self, ecg_signal):
        return np.square(np.asarray(ecg_signal, dtype=float))

    def moving_average(self, ecg_signal, window_sec=0.20):
        """
        Slide match: 0.2 s moving average window.
        """
        window_size = max(1, int(round(window_sec * self.fs)))
        window = np.ones(window_size, dtype=float) / window_size
        return np.convolve(np.asarray(ecg_signal, dtype=float), window, mode="same")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _refine_peak(self, raw_signal, approx_idx, search_ms=120):
        """
        Back-search on the original ECG around the energy peak.
        Pick the strongest positive local maximum in that window.
        """
        x = np.asarray(raw_signal, dtype=float)
        if x.size == 0:
            return int(approx_idx)

        half_window = max(1, int(round((search_ms / 1000.0) * self.fs)))
        start = max(0, int(approx_idx) - half_window)
        end = min(x.size, int(approx_idx) + half_window + 1)

        segment = x[start:end]
        if segment.size == 0:
            return int(approx_idx)

        return start + int(np.argmax(segment))

    def _estimate_heart_rate_bpm(self, r_peaks):
        if len(r_peaks) < 2:
            return np.nan

        rr = np.diff(np.asarray(r_peaks, dtype=float)) / self.fs
        rr = rr[rr > 0]
        if rr.size == 0:
            return np.nan

        return float(60.0 / np.mean(rr))

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect_r_peaks(
        self,
        energy_signal,
        raw_signal,
        candidate_distance_s=0.12,
        pause_s=0.30,
        threshold_ratio=0.70,
    ):
        """
        Detect peaks in the energy signal, then map them back to the ECG.
        Slide-consistent logic:
          - threshold = threshold_ratio * mean(energy)
          - pause/refractory ~ 0.3 s
          - back-search to original ECG
        """
        e = np.asarray(energy_signal, dtype=float)
        raw = np.asarray(raw_signal, dtype=float)

        if e.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int), {}

        candidate_distance = max(1, int(round(candidate_distance_s * self.fs)))
        candidate_peaks, _ = signal.find_peaks(e, distance=candidate_distance)

        if candidate_peaks.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int), {}

        threshold = float(threshold_ratio * np.mean(e))
        pause_samples = int(round(pause_s * self.fs))

        accepted_energy_peaks = []
        r_peaks = []

        last_accept = None

        for p in candidate_peaks:
            if e[p] < threshold:
                continue

            if last_accept is not None and (p - last_accept) < pause_samples:
                continue

            accepted_energy_peaks.append(int(p))
            refined = self._refine_peak(raw, p)
            r_peaks.append(int(refined))
            last_accept = int(p)

        # Remove accidental duplicates after back-search refinement
        if len(r_peaks) > 1:
            r_peaks = sorted(set(r_peaks))

        debug = {
            "candidate_peaks": candidate_peaks,
            "accepted_energy_peaks": np.asarray(accepted_energy_peaks, dtype=int),
            "threshold": threshold,
            "pause_samples": pause_samples,
        }

        return (
            np.asarray(accepted_energy_peaks, dtype=int),
            np.asarray(r_peaks, dtype=int),
            debug,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def solve(
        self,
        signal_df,
        use_preprocessing=True,
        candidate_distance_s=0.12,
        pause_s=0.30,
        threshold_ratio=0.70,
        use_noise_reduction=False,
    ):
        """
        signal_df:
            column 0 -> time in seconds
            column 1 -> ECG amplitude
        """
        time_s = signal_df.iloc[:, 0].to_numpy(dtype=float)
        raw_signal = signal_df.iloc[:, 1].to_numpy(dtype=float)

        # Optional light cleanup before the PT chain.
        base = self.preprocess_ecg(raw_signal, use_preprocessing=use_preprocessing)

        # Optional lecture-style noise reduction / baseline removal.
        # Keep this OFF by default so you do not over-filter.
        if use_noise_reduction:
            base = self.highpass_filter(base, cutoff_hz=2.0, order=4)
            base = self.lowpass_filter(base, cutoff_hz=35.0, order=4)

        # Pan-Tompkins chain
        bpass = self.band_pass_filter(base, low_hz=5.0, high_hz=15.0, order=4)
        der = self.derivative(bpass)
        sqr = self.squaring(der)
        energy = self.moving_average(sqr, window_sec=0.20)

        energy_peaks, r_peaks, debug = self.detect_r_peaks(
            energy_signal=energy,
            raw_signal=raw_signal,
            candidate_distance_s=candidate_distance_s,
            pause_s=pause_s,
            threshold_ratio=threshold_ratio,
        )

        heart_rate_bpm = self._estimate_heart_rate_bpm(r_peaks)

        return {
            "time_s": time_s,
            "raw": raw_signal,
            "preprocessed": base,
            "bandpass": bpass,
            "derivative": der,
            "squared": sqr,
            "energy": energy,
            "energy_peaks": energy_peaks,
            "r_peaks": r_peaks,
            "heart_rate_bpm": heart_rate_bpm,
            "debug": debug,
        }