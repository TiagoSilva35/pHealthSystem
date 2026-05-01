from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - handled at runtime
    torch = None
    nn = None


FEATURE_KEYS = (
    "rr_prev_s",
    "rr_next_s",
    "rr_baseline_s",
    "prematurity_index",
    "aa_ratio",
    "morph_index",
    "qrs_width_ms",
    "width_index",
    "pause_index",
    "morphology_score",
)

# de Chazal inter-patient split (standard for MIT-BIH PVC studies)
DS1_RECORDS = frozenset({101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230})
DS2_RECORDS = frozenset({100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234})
PACED_RECORDS = frozenset({102, 104, 107, 217})


class PVCCandidateMLP(nn.Module if nn is not None else object):
    def __init__(self, input_dim=10, hidden_dim1=64, hidden_dim2=32, dropout=0.2):
        if nn is None:
            raise ImportError("PyTorch is required to use the MLP PVC model.")
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x):
        return self.network(x)


def _repo_root():
    return Path(__file__).resolve().parents[2]


def default_model_path():
    return _repo_root() / "mlp_pvc_model.pt"


def default_scaler_path():
    return _repo_root() / "mlp_scaler_params.npz"


def feature_dicts_to_matrix(feature_rows, feature_keys=FEATURE_KEYS):
    if not feature_rows:
        return np.empty((0, len(feature_keys)), dtype=np.float32)

    matrix = np.full((len(feature_rows), len(feature_keys)), np.nan, dtype=np.float32)
    for row_idx, row in enumerate(feature_rows):
        for col_idx, key in enumerate(feature_keys):
            value = row.get(key, np.nan)
            matrix[row_idx, col_idx] = np.nan if value is None else float(value)
    return matrix


def impute_nan_with_medians(X, medians):
    X = np.asarray(X, dtype=np.float32).copy()
    medians = np.asarray(medians, dtype=np.float32)
    nan_mask = ~np.isfinite(X)
    if np.any(nan_mask):
        row_idx, col_idx = np.where(nan_mask)
        X[row_idx, col_idx] = medians[col_idx]
    return X


def normalize_features(X, means, stds):
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)
    stds = np.where(stds <= 1e-9, 1.0, stds)
    return (np.asarray(X, dtype=np.float32) - means) / stds


def load_scaler_params(scaler_path):
    params = np.load(scaler_path, allow_pickle=True)
    medians = np.asarray(params["medians"], dtype=np.float32)
    means = np.asarray(params["means"], dtype=np.float32)
    stds = np.asarray(params["stds"], dtype=np.float32)
    feature_keys = tuple(params.get("feature_keys", FEATURE_KEYS))
    return medians, means, stds, feature_keys


@lru_cache(maxsize=4)
def _load_predictor_cached(model_path_str, scaler_path_str, device_name):
    if torch is None:
        raise ImportError("PyTorch is required for detection_rule='mlp'. Install torch first.")

    model_path = Path(model_path_str)
    scaler_path = Path(scaler_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"MLP model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"MLP scaler file not found: {scaler_path}")

    medians, means, stds, feature_keys = load_scaler_params(scaler_path)
    expected_dim = len(feature_keys)
    device = torch.device(device_name)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        input_dim = int(checkpoint.get("input_dim", expected_dim))
    else:
        state_dict = checkpoint
        input_dim = expected_dim

    model = PVCCandidateMLP(input_dim=input_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    def predict_probability(feature_vector):
        x = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, got {x.shape[1]}")
        x = impute_nan_with_medians(x, medians)
        x = normalize_features(x, means, stds)
        with torch.no_grad():
            xt = torch.from_numpy(x).to(device)
            logit = model(xt)
            prob = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)[0]
        return float(prob)

    return predict_probability


def get_mlp_predictor(model_path=None, scaler_path=None, device_name="cpu"):
    model_path = Path(model_path) if model_path is not None else default_model_path()
    scaler_path = Path(scaler_path) if scaler_path is not None else default_scaler_path()
    return _load_predictor_cached(str(model_path.resolve()), str(scaler_path.resolve()), device_name)
