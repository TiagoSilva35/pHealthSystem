import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

from src.helpers.constants import SAMPLING_RATE
from src.helpers.plot_signals import plot_signals


class PanTompkinsQRS:
    def __init__(self, fs=SAMPLING_RATE):
        self.fs = fs

    # -----------------------------
    # Preprocessing
    # -----------------------------
    def preprocess_ecg(self, ecg_signal, use_preprocessing=True):
        if not use_preprocessing:
            return ecg_signal.astype(float)

        x = ecg_signal.astype(float).copy()

        # High-pass for baseline removal
        hp_cut = 2.0
        nyq = 0.5 * self.fs
        b_hp, a_hp = signal.butter(4, hp_cut / nyq, btype="high")
        x = signal.filtfilt(b_hp, a_hp, x)

        # Low-pass for high-frequency noise reduction
        lp_cut = 35.0
        b_lp, a_lp = signal.butter(4, lp_cut / nyq, btype="low")
        x = signal.filtfilt(b_lp, a_lp, x)

        return x

    # -----------------------------
    # QRS energy pipeline
    # -----------------------------
    def band_pass_filter(self, ecg_signal):
        nyq = 0.5 * self.fs
        low = 5.0 / nyq
        high = 15.0 / nyq
        b, a = signal.butter(4, [low, high], btype="band")
        return signal.filtfilt(b, a, ecg_signal)

    def derivative(self, ecg_signal):
        """
        5-point derivative approximation.
        """
        y = np.zeros_like(ecg_signal, dtype=float)
        for i in range(2, len(ecg_signal) - 2):
            y[i] = (
                2 * ecg_signal[i + 1]
                + ecg_signal[i + 2]
                - 2 * ecg_signal[i - 1]
                - ecg_signal[i - 2]
            ) * (self.fs / 8.0)

        y[0] = y[2]
        y[1] = y[2]
        y[-2] = y[-3]
        y[-1] = y[-3]
        return y

    def squaring(self, ecg_signal):
        return np.square(ecg_signal)

    def moving_window_integration(self, ecg_signal, window_sec=0.20):
        window_size = max(1, int(round(window_sec * self.fs)))
        window = np.ones(window_size) / window_size
        return np.convolve(ecg_signal, window, mode="same")

    # -----------------------------
    # Peak logic
    # -----------------------------
    def _refine_peak(self, raw_signal, approx_idx, search_ms=120):
        """
        Back-search on the original ECG:
        once a candidate energy peak is found, search around it
        for the true R peak.
        """
        w = max(1, int(round((search_ms / 1000.0) * self.fs)))
        start = max(0, approx_idx - w)
        end = min(len(raw_signal), approx_idx + w + 1)

        segment = raw_signal[start:end]
        if len(segment) == 0:
            return approx_idx

        # If ECG is inverted, abs helps. For standard ECG, this still works well.
        local_idx = int(np.argmax(np.abs(segment)))
        return start + local_idx

    def _peak_slope(self, sig, idx, window_ms=50):
        """
        Slope proxy used to help distinguish QRS from T-wave.
        """
        w = max(1, int(round((window_ms / 1000.0) * self.fs)))
        start = max(0, idx - w)
        end = min(len(sig), idx + w + 1)
        segment = sig[start:end]
        if len(segment) < 2:
            return 0.0
        return float(np.max(np.abs(np.diff(segment))))

    def _initialize_thresholds(self, mwin, candidate_peaks):
        """
        Bootstrap SPKI/NPKI from the first ~2 seconds.
        """
        init_end = int(2.0 * self.fs)
        init_candidates = candidate_peaks[candidate_peaks < init_end]

        if len(init_candidates) == 0:
            init_candidates = candidate_peaks[: min(8, len(candidate_peaks))]

        init_heights = mwin[init_candidates] if len(init_candidates) else np.array([])

        if len(init_heights) == 0:
            spki = 0.0
            npki = 0.0
        elif len(init_heights) == 1:
            spki = float(init_heights[0])
            npki = 0.5 * spki
        else:
            sorted_h = np.sort(init_heights)
            split = max(1, len(sorted_h) // 2)
            npki = float(np.mean(sorted_h[:split]))
            spki = float(np.mean(sorted_h[split:]))
            if spki <= npki:
                spki = float(np.max(sorted_h))
                npki = float(np.median(sorted_h) * 0.5)

        # Lowered initial threshold multiplier for higher sensitivity
        thresh_i1 = npki + 0.10 * (spki - npki)
        thresh_i2 = 0.4 * thresh_i1
        return spki, npki, thresh_i1, thresh_i2

    def detect_r_peaks(self, mwin_signal, raw_signal=None, bpass_signal=None, candidate_distance_s=0.12, refractory_s=0.30):
        """
        Adaptive Pan-Tompkins decision logic:
        - candidate local maxima on the energy envelope
        - adaptive thresholds (SPKI/NPKI)
        - refractory period
        - searchback if a beat is missed
        - back-search to the true R peak
        """
        # Use a shorter candidate distance to allow denser detections (helps PVC-heavy records)
        # Candidate distance in samples (configurable for dense PVC records)
        candidate_distance = max(1, int(candidate_distance_s * self.fs))
        candidate_peaks, _ = signal.find_peaks(
            mwin_signal, distance=candidate_distance
        )

        if len(candidate_peaks) == 0:
            return np.array([], dtype=int), np.array([], dtype=int), {}

        spki, npki, thresh_i1, thresh_i2 = self._initialize_thresholds(
            mwin_signal, candidate_peaks
        )

        qrs_candidates = []
        r_peaks = []

        refractory = int(refractory_s * self.fs)  # slide: pause ~0.3 s (configurable)
        t_wave_limit = int(0.36 * self.fs)

        last_qrs_idx = None
        last_qrs_slope = 0.0
        rr_intervals = []

        debug = {
            "spki": [],
            "npki": [],
            "thresh_i1": [],
            "thresh_i2": [],
            "candidate_peaks": candidate_peaks,
        }

        def rr_stats():
            if len(rr_intervals) == 0:
                return None, None, None
            rr_avg1 = float(np.mean(rr_intervals[-8:]))
            valid = [rr for rr in rr_intervals[-8:] if 0.92 * rr_avg1 <= rr <= 1.16 * rr_avg1]
            rr_avg2 = float(np.mean(valid)) if len(valid) else rr_avg1
            rr_miss = 1.66 * rr_avg2
            return rr_avg1, rr_avg2, rr_miss

        for p in candidate_peaks:
            peak_val = float(mwin_signal[p])

            # Searchback if a beat may have been missed
            if last_qrs_idx is not None and len(rr_intervals) > 0:
                rr_avg1, rr_avg2, rr_miss = rr_stats()
                if rr_miss is not None and (p - last_qrs_idx) > rr_miss:
                    gap_candidates = candidate_peaks[
                        (candidate_peaks > last_qrs_idx)
                        & (candidate_peaks < p)
                        & (mwin_signal[candidate_peaks] > thresh_i2)
                    ]
                    if len(gap_candidates) > 0:
                        sb = int(gap_candidates[np.argmax(mwin_signal[gap_candidates])])

                        qrs_candidates.append(sb)
                        if raw_signal is not None:
                            refined = self._refine_peak(raw_signal, sb)
                        elif bpass_signal is not None:
                            refined = self._refine_peak(bpass_signal, sb)
                        else:
                            refined = sb
                        r_peaks.append(refined)

                        if last_qrs_idx is not None:
                            rr_intervals.append(sb - last_qrs_idx)

                        last_qrs_idx = sb
                        last_qrs_slope = self._peak_slope(
                            raw_signal if raw_signal is not None else bpass_signal,
                            sb
                        )
                        spki = 0.125 * float(mwin_signal[sb]) + 0.875 * spki
                        thresh_i1 = npki + 0.25 * (spki - npki)
                        thresh_i2 = 0.5 * thresh_i1

            accepted = False

            if peak_val >= thresh_i1:
                if last_qrs_idx is None:
                    accepted = True
                else:
                    dt = p - last_qrs_idx

                    if dt > refractory:
                        accepted = True
                    elif dt <= t_wave_limit:
                        # T-wave discrimination: compare slope
                        slope_sig = raw_signal if raw_signal is not None else bpass_signal
                        current_slope = self._peak_slope(slope_sig, p)
                        if current_slope > 0.5 * last_qrs_slope:
                            accepted = True

            if accepted:
                qrs_candidates.append(p)

                if raw_signal is not None:
                    refined = self._refine_peak(raw_signal, p)
                elif bpass_signal is not None:
                    refined = self._refine_peak(bpass_signal, p)
                else:
                    refined = p
                r_peaks.append(refined)

                slope_sig = raw_signal if raw_signal is not None else bpass_signal
                last_qrs_slope = self._peak_slope(slope_sig, p)

                if last_qrs_idx is not None:
                    rr_intervals.append(p - last_qrs_idx)

                last_qrs_idx = p
                spki = 0.125 * peak_val + 0.875 * spki
            else:
                npki = 0.125 * peak_val + 0.875 * npki

            thresh_i1 = npki + 0.25 * (spki - npki)
            thresh_i2 = 0.5 * thresh_i1

            debug["spki"].append(spki)
            debug["npki"].append(npki)
            debug["thresh_i1"].append(thresh_i1)
            debug["thresh_i2"].append(thresh_i2)

        return np.array(qrs_candidates, dtype=int), np.array(r_peaks, dtype=int), debug

    # -----------------------------
    # Main entry point
    # -----------------------------
    def solve(self, signal_df, use_preprocessing=True, min_peak_distance_s=0.12, refractory_s=0.30):
        """
        signal_df:
            column 0 -> time in seconds
            column 1 -> ECG amplitude
        """
        time_s = signal_df.iloc[:, 0].to_numpy(dtype=float)
        raw_signal = signal_df.iloc[:, 1].to_numpy(dtype=float)

        # Slide-aligned preprocessing: noise reduction / baseline removal
        preprocessed = self.preprocess_ecg(raw_signal, use_preprocessing=use_preprocessing)

        # QRS energy estimation
        bpass = self.band_pass_filter(preprocessed)
        der = self.derivative(bpass)
        sqr = self.squaring(der)
        mwin = self.moving_window_integration(sqr, window_sec=0.20)

        # Adaptive peak detection + back-search
        qrs_candidates, r_peaks, debug = self.detect_r_peaks(
            mwin_signal=mwin,
            raw_signal=raw_signal,
            bpass_signal=bpass,
            candidate_distance_s=min_peak_distance_s,
            refractory_s=refractory_s,
        )

        return {
            "time_s": time_s,
            "raw": raw_signal,
            "preprocessed": preprocessed,
            "bandpass": bpass,
            "derivative": der,
            "squared": sqr,
            "mwin": mwin,
            "qrs_candidates": qrs_candidates,
            "r_peaks": r_peaks,
            "debug": debug,
        }


if __name__ == "__main__":
    # -----------------------------
    # Example usage
    # -----------------------------
    ecg = pd.read_csv("ecg_samples.csv")

    detector = PanTompkinsQRS(fs=SAMPLING_RATE)
    result = detector.solve(ecg, use_preprocessing=True)

    time_s = result["time_s"]
    raw = result["raw"]
    mwin_signal = result["mwin"]
    r_peaks = result["r_peaks"]

    peak_timestamps = time_s[r_peaks]
    peak_amplitudes = raw[r_peaks]

    print("\nDetection Summary:")
    print(f"Total Beats Detected: {len(r_peaks)}")

    if len(peak_timestamps) > 1:
        rr_intervals = np.diff(peak_timestamps)

        # Slides mention average of the last 5 or 7 beats
        n = min(7, len(rr_intervals))
        avg_bpm = 60.0 / np.mean(rr_intervals[-n:])
        print(f"Calculated Heart Rate: {avg_bpm:.2f} BPM")

    print("\nPeak Timestamps (s):")
    for i, t in enumerate(peak_timestamps):
        print(f"Beat {i+1}: {t:.3f}s")

    # Save the integrated signal
    final_signal = pd.DataFrame({
        "time_s": time_s,
        "ecg_integrated": mwin_signal
    })
    final_signal.to_csv("output_signal.csv", index=False)

    # Plot integrated signal
    plot_signals(final_signal.to_numpy(), SAMPLING_RATE, 1)

    # Overlay R-peaks on raw ECG
    plt.figure(figsize=(12, 4))
    plt.plot(time_s, raw, label="Raw ECG", alpha=0.7)
    plt.scatter(peak_timestamps, peak_amplitudes, color="red", marker="x", label="R-Peaks")
    plt.title("Pan-Tompkins R-Peak Detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()