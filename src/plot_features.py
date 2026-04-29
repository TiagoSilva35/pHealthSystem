# run it: src/plot_features.py --input pvc_features.csv --output pvc_features_dashboard.png
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_features_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.dtype.names is None:
        raise ValueError("CSV must include headers")

    required = {
        "peak_time_s",
        "rr_prev_s",
        "prematurity_index",
        "qrs_width_ms",
        "is_pvc_candidate",
    }
    if not required.issubset(set(data.dtype.names)):
        raise ValueError("Input CSV does not include all expected feature columns")

    return data


def _safe_array(data, key):
    arr = np.asarray(data[key], dtype=float)
    return arr


def _candidate_mask(data):
    return np.asarray(data["is_pvc_candidate"], dtype=int) > 0


def _finite_mask(*arrays):
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def plot_feature_dashboard(data, output_file, show_plot=False):
    t = _safe_array(data, "peak_time_s")
    rr_prev = _safe_array(data, "rr_prev_s")
    prem = _safe_array(data, "prematurity_index")
    qrs = _safe_array(data, "qrs_width_ms")
    candidates = _candidate_mask(data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: RR interval timeline
    ax = axes[0, 0]
    rr_mask = _finite_mask(t, rr_prev)
    ax.plot(t[rr_mask], rr_prev[rr_mask], color="#1f77b4", linewidth=1.0, label="RR previous")
    if np.any(candidates & rr_mask):
        ax.scatter(
            t[candidates & rr_mask],
            rr_prev[candidates & rr_mask],
            color="#d62728",
            s=26,
            label="Extrasystole candidates",
            zorder=3,
        )
    ax.set_title("RR Interval Over Time")
    ax.set_xlabel("Peak time (s)")
    ax.set_ylabel("RR previous (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Panel 2: QRS width timeline
    ax = axes[0, 1]
    qrs_mask = _finite_mask(t, qrs)
    ax.plot(t[qrs_mask], qrs[qrs_mask], color="#2ca02c", linewidth=1.0, label="QRS width")
    ax.axhline(110.0, color="#ff7f0e", linestyle="--", linewidth=1.0, label="Wide-QRS ref (110 ms)")
    if np.any(candidates & qrs_mask):
        ax.scatter(
            t[candidates & qrs_mask],
            qrs[candidates & qrs_mask],
            color="#d62728",
            s=26,
            label="Extrasystole candidates",
            zorder=3,
        )
    ax.set_title("QRS Width Over Time")
    ax.set_xlabel("Peak time (s)")
    ax.set_ylabel("QRS width (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Panel 3: Prematurity vs QRS width
    ax = axes[1, 0]
    pc_mask = _finite_mask(prem, qrs)
    ax.scatter(prem[pc_mask], qrs[pc_mask], s=22, color="#7f7f7f", alpha=0.7, label="All beats")
    if np.any(candidates & pc_mask):
        ax.scatter(
            prem[candidates & pc_mask],
            qrs[candidates & pc_mask],
            s=34,
            color="#d62728",
            alpha=0.9,
            label="PVC candidates",
            zorder=3,
        )
    ax.axvline(0.80, color="#9467bd", linestyle="--", linewidth=1.0, label="Premature ref (0.80)")
    ax.axhline(110.0, color="#ff7f0e", linestyle="--", linewidth=1.0, label="Wide-QRS ref (110 ms)")
    ax.set_title("PVC Timing/Morphology Space")
    ax.set_xlabel("Prematurity index")
    ax.set_ylabel("QRS width (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Panel 4: Candidate density over time
    ax = axes[1, 1]
    ax.plot(t[qrs_mask], qrs[qrs_mask], color="#2ca02c", linewidth=1.0, label="QRS width")
    if np.any(candidates & qrs_mask):
        ax.scatter(
            t[candidates & qrs_mask],
            qrs[candidates & qrs_mask],
            color="#d62728",
            s=26,
            label="PVC candidates",
            zorder=3,
        )
    ax.axhline(110.0, color="#ff7f0e", linestyle="--", linewidth=1.0, label="Wide-QRS ref (110 ms)")
    ax.set_title("QRS Width With PVC Candidates")
    ax.set_xlabel("Peak time (s)")
    ax.set_ylabel("QRS width (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle("PVC Feature Dashboard", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_file, dpi=220, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PVC feature CSV")
    parser.add_argument("--input", default="pvc_features.csv", help="Input features CSV")
    parser.add_argument("--output", default="pvc_features_dashboard.png", help="Output plot image")
    parser.add_argument("--show", action="store_true", help="Display plot window")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_features_csv(args.input)
    plot_feature_dashboard(data, args.output, show_plot=args.show)
    print(f"[INFO] Saved feature dashboard to {args.output}")


if __name__ == "__main__":
    main()
