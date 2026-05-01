"""Clinical summary metrics for ECG analysis."""


DEFAULT_VT_BURDEN_THRESHOLD_PERCENT = 10.0


def compute_pvc_burden_metrics(
    pvc_count,
    total_beats,
    duration_s=None,
    vt_burden_threshold_percent=DEFAULT_VT_BURDEN_THRESHOLD_PERCENT,
):
    """Return clinically useful PVC burden metrics.

    PVC burden is the percentage of PVCs over the total number of analyzed beats.
    When a recording duration is available, the helper also returns a PVC rate per hour.
    The VT flag is a screening heuristic based on burden and is not a diagnosis.
    """
    pvc_count = max(0, int(pvc_count))
    total_beats = max(0, int(total_beats))

    pvc_burden_percent = 100.0 * pvc_count / total_beats if total_beats > 0 else 0.0

    pvc_rate_per_hour = 0.0
    if duration_s is not None:
        try:
            duration_value = float(duration_s)
        except (TypeError, ValueError):
            duration_value = None
        if duration_value is not None and duration_value > 0:
            pvc_rate_per_hour = 3600.0 * pvc_count / duration_value

    possible_vt = pvc_burden_percent >= float(vt_burden_threshold_percent)

    return {
        "pvc_burden_percent": pvc_burden_percent,
        "pvc_rate_per_hour": pvc_rate_per_hour,
        "possible_vt_from_burden": possible_vt,
        "vt_burden_threshold_percent": float(vt_burden_threshold_percent),
    }