import numpy as np
import pandas as pd
from scipy import signal

from src.helpers.constants import SAMPLING_RATE


class PanThompkinsQRS:
    """
    Slide-aligned Pan-Tompkins-style QRS detector.

    Matches the lecture flow:
      1) Optional preprocessing / baseline cleanup
      2) Band-pass filtering (3-25 Hz for arrhythmia robustness)
      3) Differentiation
      4) Squaring
      5) Moving average (~0.15 s)
      6) Dual-Thresholding (Signal vs. Noise)
      7) Search-Back mechanism for missed beats (1.66 * RR_avg)
      8) Refractory rule (~0.2 s)
      9) Back-search to the original ECG (Absolute Max for inverted beats)
    """

    def __init__(self, fs=SAMPLING_RATE):
        self.fs = fs

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess_ecg(self, ecg_signal, use_preprocessing=True):
        x = np.asarray(ecg_signal, dtype=float).copy()
        if not use_preprocessing:
            return x
        return signal.detrend(x, type="constant")

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def lowpass_filter(self, ecg_signal, cutoff_hz=35.0, order=4):
        nyq = 0.5 * self.fs
        b, a = signal.butter(order, cutoff_hz / nyq, btype="low")
        return signal.filtfilt(b, a, np.asarray(ecg_signal, dtype=float))

    def highpass_filter(self, ecg_signal, cutoff_hz=2.0, order=4):
        nyq = 0.5 * self.fs
        b, a = signal.butter(order, cutoff_hz / nyq, btype="high")
        return signal.filtfilt(b, a, np.asarray(ecg_signal, dtype=float))

    def band_pass_filter(self, ecg_signal, low_hz=3.0, high_hz=25.0, order=4):
        nyq = 0.5 * self.fs
        b, a = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
        return signal.filtfilt(b, a, np.asarray(ecg_signal, dtype=float))

    # ------------------------------------------------------------------
    # PT stages
    # ------------------------------------------------------------------
    def derivative(self, ecg_signal):
        x = np.asarray(ecg_signal, dtype=float)
        if x.size < 5:
            return np.zeros_like(x)

        y = np.zeros_like(x, dtype=float)
        for i in range(2, len(x) - 2):
            y[i] = (2 * x[i + 1] + x[i + 2] - 2 * x[i - 1] - x[i - 2]) * (self.fs / 8.0)

        y[0] = y[2]
        y[1] = y[2]
        y[-2] = y[-3]
        y[-1] = y[-3]
        return y

    def squaring(self, ecg_signal):
        return np.square(np.asarray(ecg_signal, dtype=float))

    def moving_average(self, ecg_signal, window_sec=0.15):
        window_size = max(1, int(round(window_sec * self.fs)))
        window = np.ones(window_size, dtype=float) / window_size
        return np.convolve(np.asarray(ecg_signal, dtype=float), window, mode="same")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _refine_peak(self, raw_signal, approx_idx, search_ms=120):
        """
        Back-search on the original ECG around the energy peak.
        Uses np.abs to find the absolute maximum (handles inverted QRS like in Record 207).
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

        # Record 207 fix: look for absolute maximum to catch inverted flutter waves
        return start + int(np.argmax(np.abs(segment)))

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
        pause_s=0.20,
    ):
        """
        True Pan-Tompkins Logic with Dual-Thresholds and Search-Back.
        """
        e = np.asarray(energy_signal, dtype=float)
        raw = np.asarray(raw_signal, dtype=float)

        if e.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int), {}

        candidate_distance = max(1, int(round(candidate_distance_s * self.fs)))
        candidate_peaks, _ = signal.find_peaks(e, distance=candidate_distance)

        if candidate_peaks.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int), {}

        # Initialization phase (Estimate signal and noise from first 2 seconds)
        init_window = min(e.size, int(2.0 * self.fs))
        SPKI = np.max(e[:init_window]) if init_window > 0 else 0.0
        NPKI = np.mean(e[:init_window]) if init_window > 0 else 0.0

        pause_samples = int(round(pause_s * self.fs))
        
        # RR interval tracking for search-back
        rr_buffer = []
        recent_rr_avg = self.fs * 1.0  # Default to 1 second / 60 BPM

        accepted_energy_peaks = []
        r_peaks = []
        last_accept = None
        skipped_candidates = []

        for p in candidate_peaks:
            # Calculate current thresholds
            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
            threshold_I2 = 0.5 * threshold_I1  # 50% threshold for search-back

            # 1. SEARCH-BACK MECHANISM
            if last_accept is not None:
                rr_current = p - last_accept
                # If we haven't seen a beat in 166% of the average RR interval
                if rr_current > 1.66 * recent_rr_avg:
                    # Look back at candidates we skipped since the last accepted peak
                    valid_searchback = [sb for sb in skipped_candidates if sb > last_accept and e[sb] > threshold_I2]
                    
                    if valid_searchback:
                        # Find the strongest peak among the valid search-back candidates
                        sb_peak = max(valid_searchback, key=lambda idx: e[idx])
                        
                        accepted_energy_peaks.append(int(sb_peak))
                        refined = self._refine_peak(raw, sb_peak)
                        r_peaks.append(int(refined))

                        # Update Signal Peak with lower weight for search-back
                        SPKI = 0.25 * e[sb_peak] + 0.75 * SPKI
                        
                        # Update RR buffer
                        if len(accepted_energy_peaks) > 1:
                            rr_buffer.append(int(sb_peak) - last_accept)
                            if len(rr_buffer) > 8: rr_buffer.pop(0)
                            recent_rr_avg = np.mean(rr_buffer)

                        last_accept = int(sb_peak)
                        # Recalculate normal threshold after search-back modification
                        threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)

            # 2. NORMAL DETECTION
            if e[p] >= threshold_I1:
                # Enforce refractory period
                if last_accept is None or (p - last_accept) >= pause_samples:
                    accepted_energy_peaks.append(int(p))
                    refined = self._refine_peak(raw, p)
                    r_peaks.append(int(refined))

                    # Update Signal Peak (Standard learning rate)
                    SPKI = 0.125 * e[p] + 0.875 * SPKI

                    # Update RR buffer
                    if last_accept is not None:
                        rr_buffer.append(p - last_accept)
                        if len(rr_buffer) > 8: rr_buffer.pop(0)
                        recent_rr_avg = np.mean(rr_buffer)

                    last_accept = int(p)
                    skipped_candidates = []  # Clear skipped candidates after a successful find
                else:
                    # Treat as noise if inside refractory period
                    NPKI = 0.125 * e[p] + 0.875 * NPKI
            else:
                # 3. NOISE TRACKING
                # Update Noise Peak if it didn't cross threshold
                NPKI = 0.125 * e[p] + 0.875 * NPKI
                skipped_candidates.append(p)

        # Remove accidental duplicates after back-search refinement
        if len(r_peaks) > 1:
            r_peaks = sorted(set(r_peaks))

        debug = {
            "candidate_peaks": candidate_peaks,
            "accepted_energy_peaks": np.asarray(accepted_energy_peaks, dtype=int),
            "final_SPKI": SPKI,
            "final_NPKI": NPKI,
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
        pause_s=0.20,
        use_noise_reduction=False,
    ):
        time_s = signal_df.iloc[:, 0].to_numpy(dtype=float)
        raw_signal = signal_df.iloc[:, 1].to_numpy(dtype=float)

        base = self.preprocess_ecg(raw_signal, use_preprocessing=use_preprocessing)

        if use_noise_reduction:
            base = self.highpass_filter(base, cutoff_hz=2.0, order=4)
            base = self.lowpass_filter(base, cutoff_hz=35.0, order=4)

        bpass = self.band_pass_filter(base, low_hz=3.0, high_hz=25.0, order=4)
        der = self.derivative(bpass)
        sqr = self.squaring(der)
        energy = self.moving_average(sqr, window_sec=0.15)

        # Note: threshold_ratio is removed as we now use true PT dual-thresholding
        energy_peaks, r_peaks, debug = self.detect_r_peaks(
            energy_signal=energy,
            raw_signal=raw_signal,
            candidate_distance_s=candidate_distance_s,
            pause_s=pause_s,
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