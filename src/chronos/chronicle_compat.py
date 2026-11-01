"""Code copied from Chronicle so the CHRONOS benchmark runs standalone (no `import chronicle`).

Everything here mirrors a Chronicle function and has to stay in sync with it by hand:

    task_type                     chronicle/config.py       Config.finalize_losses
    cox_loss                      chronicle/losses.py
    LOSS_REGISTRY                 chronicle/config.py
    MultiTaskLoss                 chronicle/losses.py       (static / uncertainty weighting)
    sample_one_token_per_patient  chronicle/metrics.py
    compute_c_index,
    calc_c_index, calc_correlation,
    calc_auc                      chronicle/metrics.py      (single task instead of a dict of tasks)
    FeatureScalers                chronicle/data/standardizer.py  TimeSeriesStandardizer.load /
                                                                  transform / inverse_transform_label
"""

import hashlib
import os
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F


SURVIVAL, REGRESSION, BINARY = "survival", "regression", "binary"

DEFAULT_BINARY_LABEL_COLS = ["label_supportive_care_transfusion_7_days", "label_febrile_neutropenia_21d"]
DEFAULT_RECURRENT_SURVIVAL_TARGETS = ["TTNT"]


def task_type(target: str, binary_label_cols: Sequence[str]) -> str:
    """Config.finalize_losses: binary_label_cols -> BCE, `label*` -> regression, else survival (Cox)."""
    if target in binary_label_cols:
        return BINARY
    if target.startswith("label"):
        return REGRESSION
    return SURVIVAL


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def survival_loss(fn):
    """Mark a loss function as a survival loss (expects ``[N, 2]`` time/event targets)."""
    fn.is_survival = True
    return fn


@survival_loss
def cox_loss(pred, target):
    """Negative Cox partial log-likelihood (Breslow, no tie correction).

    Expects exactly ONE risk score per patient: the risk set is the batch, so
    feeding several timesteps of the same patient makes the partial likelihood
    statistically invalid. Callers must reduce to one token per patient first.

    Args:
        pred: ``[N, 1]`` or ``[N]`` risk scores (higher = higher hazard).
        target: ``[N, 2]`` with ``target[:, 0] = time`` and ``target[:, 1] = event``.
    """
    # logcumsumexp has no float16 kernel — upcast.
    pred = pred.float().view(-1)
    time = target[:, 0].float().view(-1)
    event = target[:, 1].float().view(-1)

    order = torch.argsort(time, descending=True)
    pred = pred[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(pred, dim=0)
    loss = -(pred * event - log_cumsum * event)
    return loss.sum() / event.sum().clamp_min(1.0)


LOSS_REGISTRY = {
    "cox_loss": cox_loss,
    "mse": F.mse_loss,
    "l1": F.l1_loss,
    "huber": F.huber_loss,
    "bce_logits": F.binary_cross_entropy_with_logits,
}


def loss_for(task: str, regression_loss: str = "mse"):
    """Chronicle's defaults: OS_loss cox_loss, regression_loss mse, binary_loss bce_logits."""
    name = {SURVIVAL: "cox_loss", REGRESSION: regression_loss, BINARY: "bce_logits"}[task]
    return name, LOSS_REGISTRY[name]


class MultiTaskLoss:
    """Combine per-task losses into a single scalar.

    Modes:
        ``static``       fixed per-task weights from the config.
        ``uncertainty``  learn a weight per task via homoscedastic uncertainty
                         (Kendall et al. 2018). The log-variance is squashed into
                         ``logvar_bounds`` with a sigmoid so gradients stay alive
                         at the bounds.

    ``uncertainty`` is only meaningful with a reasonably sized batch — and Cox loss
    additionally needs a large risk set — so it refuses batches smaller than
    ``MIN_BATCH_SIZE``.
    """

    MIN_BATCH_SIZE = 16

    def __init__(
        self,
        labels,
        mode="static",
        static_weights=None,
        logvar_bounds=(-8.0, 8.0),
        logvar_prior=0.0,
    ):
        if mode not in ("static", "uncertainty"):
            raise ValueError(f"Unsupported mode {mode!r}. Choose 'static' or 'uncertainty'.")

        self.labels = list(labels)
        self.mode = mode
        self.logvar_prior = float(logvar_prior)

        self._lv_min = float(logvar_bounds[0])
        self._lv_max = float(logvar_bounds[1])
        self._lv_range = self._lv_max - self._lv_min

        if mode == "static":
            self.weights = static_weights or {lbl: 1.0 for lbl in self.labels}
        else:
            self.u = nn.ParameterDict({lbl: nn.Parameter(torch.zeros(())) for lbl in self.labels})

    def parameters(self):
        return list(self.u.parameters()) if self.mode == "uncertainty" else []

    def to(self, device):
        if self.mode == "uncertainty":
            self.u.to(device)
        return self

    def state_dict(self):
        return {lbl: float(p.detach().cpu()) for lbl, p in self.u.items()} if self.mode == "uncertainty" else {}

    def log_var(self, label, dtype=None, device=None):
        """Smooth, bounded log-variance for one task."""
        lv = self._lv_min + self._lv_range * torch.sigmoid(self.u[label])
        if dtype is not None:
            lv = lv.to(dtype)
        if device is not None:
            lv = lv.to(device)
        return lv

    def compute_loss(self, losses, batch_size):
        if self.mode == "uncertainty" and batch_size < self.MIN_BATCH_SIZE:
            raise ValueError(
                f"loss_mode={self.mode!r} needs batch_size >= {self.MIN_BATCH_SIZE} for stable task weighting "
                f"(got {batch_size}). Either raise batch_size or set loss_mode: static."
            )

        if self.mode == "static":
            return sum(self.weights.get(lbl, 1.0) * loss for lbl, loss in losses.items())

        # uncertainty
        total = 0.0
        for lbl, loss in losses.items():
            lv = self.log_var(lbl, dtype=loss.dtype, device=loss.device)
            precision = torch.exp(-lv)
            total = total + 0.5 * (precision * loss + lv)
            if self.logvar_prior > 0:
                total = total + self.logvar_prior * (self.u[lbl] ** 2)
        return total


# --------------------------------------------------------------------------- #
# Sampling and metrics
# --------------------------------------------------------------------------- #

def sample_one_token_per_patient(pred_all, true_all, mask, strategy: str = "random", time_since_id=None):
    """Reduce a per-CLS-token sequence to one prediction per patient.

    Survival losses (and the C-index) need exactly one risk score per patient — the
    risk set is the batch, so several timesteps of the same patient would double-count.
    This picks one valid CLS position per patient.

    Args:
        pred_all: ``[B, T, 1]`` prediction at every CLS token.
        true_all: ``[B, T, D]`` label at every CLS token.
        mask: ``[B, T]`` bool — True where the CLS token is real (not padding).
        strategy: ``random`` samples a uniformly random valid token, which lets the
            model learn to predict risk from partial histories. ``last`` takes the
            final token, i.e. the maximum available history.
        time_since_id: optional ``[B, T]`` (or ``[B, T, 1]``) timestamps; when given,
            the timestamp of each selected token is returned as a third value.

    Returns:
        ``(pred, true)`` — one row per patient that has at least one real token with a
        non-NaN label — plus ``obs_time`` when ``time_since_id`` was supplied.
    """
    # Only draw from labelled tokens: a target can be undefined for part of the trajectory
    # (TTNT before the first therapy), and drawing such a token would drop the patient.
    mask = mask & ~torch.isnan(true_all).any(dim=-1)
    batch, seq_len = mask.shape
    rows = torch.arange(batch, device=mask.device)
    has_any = mask.any(dim=1)

    if strategy == "random":
        n_valid = mask.long().sum(dim=1)
        k = (torch.rand(batch, device=mask.device) * n_valid.float()).long()
        k = k.clamp(max=(n_valid - 1).clamp(min=0))
        # Index of the (k+1)-th True in each row.
        selected = (mask.long().cumsum(dim=1) < (k + 1).unsqueeze(1)).long().sum(dim=1)
    else:
        selected = (seq_len - 1) - mask.flip(dims=[1]).long().argmax(dim=1)

    # A patient with no valid token would index garbage; has_any filters those rows out below.
    selected = selected.clamp(max=seq_len - 1)

    pred = pred_all[rows, selected]
    true = true_all[rows, selected]

    if time_since_id is not None:
        t = time_since_id.squeeze(-1) if time_since_id.dim() == 3 else time_since_id
        return pred[has_any], true[has_any], t[rows, selected][has_any]

    return pred[has_any], true[has_any]


def _concat(tensors):
    if not tensors:
        return torch.empty(0)
    return torch.cat([t.reshape(-1).detach().cpu() for t in tensors], dim=0)


def compute_c_index(pred: torch.Tensor, label: torch.Tensor) -> float:
    """Harrell's C-index.

    Args:
        pred: ``[N]`` risk scores — higher means higher hazard, i.e. shorter survival.
        label: ``[N, 2]`` with time in column 0 and the event indicator in column 1.
    """
    time = label[:, 0].numpy()
    event = label[:, 1].numpy()
    # lifelines expects a predicted *survival time*, so negate the risk score.
    return float(concordance_index(time, -pred.numpy(), event))


def calc_c_index(preds, labels) -> Optional[float]:
    """calc_c_indices for one survival task; None where Chronicle omits the task."""
    if not preds:
        return None
    pred = _concat(preds)
    label = _concat(labels)
    if label.numel() == 0 or label.numel() % 2 != 0:
        return None
    label = label.view(-1, 2)
    if pred.numel() != label.shape[0]:
        return None
    keep = ~torch.isnan(label).any(dim=1) & ~torch.isnan(pred)
    if keep.sum() < 2:
        return None
    # A C-index is undefined without at least one observed event.
    if label[keep][:, 1].sum() == 0:
        return None
    try:
        return compute_c_index(pred[keep], label[keep])
    except (ZeroDivisionError, ValueError):
        return None


def calc_correlation(preds, labels) -> Optional[float]:
    """calc_correlations for one regression task (Pearson r, on the scaled values)."""
    if not preds:
        return None
    pred = _concat(preds).double()
    label = _concat(labels).double()
    if pred.numel() != label.numel() or pred.numel() < 2:
        return None
    keep = ~torch.isnan(pred) & ~torch.isnan(label)
    x, y = pred[keep], label[keep]
    if x.numel() < 2:
        return None
    x_c, y_c = x - x.mean(), y - y.mean()
    denom = torch.linalg.norm(x_c) * torch.linalg.norm(y_c)
    if denom == 0 or torch.isnan(denom):
        return None
    return float((x_c * y_c).sum() / denom)


def calc_auc(preds, labels) -> Optional[float]:
    """calc_auc_scores for one binary task (ROC-AUC of the logits)."""
    if not preds:
        return None
    pred = _concat(preds)
    label = _concat(labels)
    keep = ~torch.isnan(pred) & ~torch.isnan(label)
    y_true = label[keep].numpy().astype(int)
    y_score = pred[keep].numpy()
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


# --------------------------------------------------------------------------- #
# Scalers (TimeSeriesStandardizer, as written by `chronicle-preprocess`)
# --------------------------------------------------------------------------- #

SCALER_FILES = ("feature_scalers.pkl", "label_scalers.pkl", "standardizer_meta.pkl")


def scaler_fingerprint(dir_path: str) -> Optional[Dict[str, str]]:
    """sha256 of every scaler file, so a checkpoint can tell whether it gets its own scalers back."""
    if not os.path.isfile(os.path.join(dir_path, "standardizer_meta.pkl")):
        return None
    out = {}
    for name in SCALER_FILES:
        path = os.path.join(dir_path, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


class FeatureScalers:
    """The fitted state of a TimeSeriesStandardizer: one sklearn scaler per feature / regression target."""

    def __init__(self, dir_path: str):
        if not os.path.isfile(os.path.join(dir_path, "standardizer_meta.pkl")):
            raise FileNotFoundError(f"No fitted scalers under {dir_path}. Expected Chronicle's scalers/ directory.")
        meta = joblib.load(os.path.join(dir_path, "standardizer_meta.pkl"))
        self.dir_path = dir_path
        self.patient_id_col = meta["patient_id_col"]
        self.key_col = meta["key_col"]
        self.value_col = meta["value_col"]
        self.feature_stats = meta.get("feature_stats", {})
        self.scaler_type = meta["scaler_type"]
        self.binary_label_cols = list(meta["binary_label_cols"])
        self.feature_scalers = joblib.load(os.path.join(dir_path, "feature_scalers.pkl"))
        label_path = os.path.join(dir_path, "label_scalers.pkl")
        self.label_scalers = joblib.load(label_path) if os.path.isfile(label_path) else {}

    def transform(self, df: pd.DataFrame, *, audit: bool = False) -> pd.DataFrame:
        """TimeSeriesStandardizer.transform: scale features (and regression targets) row-wise."""
        if not self.feature_scalers:
            raise RuntimeError("The loaded standardizer has no feature scalers.")
        out = df.copy()
        if out.empty:
            raise ValueError("No rows to transform.")

        drifted = []
        for feat, scaler in self.feature_scalers.items():
            rows = out[self.key_col].astype(str) == str(feat)
            values = out.loc[rows, self.value_col]
            usable = values.index[values.notna()]
            if not len(usable):
                continue

            raw = values.loc[usable].to_numpy(dtype=float)
            out.loc[usable, self.value_col] = scaler.transform(raw.reshape(-1, 1)).ravel()

            if audit:
                stats = self.feature_stats.get(str(feat))
                if stats and stats["std"] > 0 and np.nanmax(np.abs((raw - stats["mean"]) / stats["std"])) > 3:
                    drifted.append(str(feat))

        if drifted:
            print(
                f"⚠️ {len(drifted)}/{len(self.feature_scalers)} features contain values >3 sigma from the "
                f"training mean: {sorted(drifted)}. Expected for another cohort; investigate if it is most of them."
            )

        for label, scaler in self.label_scalers.items():
            if label not in out.columns:
                continue
            values = out[label].astype(float)
            usable = values.index[values.notna()]
            if len(usable):
                out.loc[usable, label] = scaler.transform(values.loc[usable].to_numpy().reshape(-1, 1)).ravel()

        return out

    def inverse_transform_label(self, label: str, values: np.ndarray) -> np.ndarray:
        """Map scaled predictions of ``label`` back to the original units (unchanged if never scaled)."""
        scaler = self.label_scalers.get(label)
        if scaler is None:
            return values
        return scaler.inverse_transform(np.asarray(values, dtype=float).reshape(-1, 1)).ravel()
