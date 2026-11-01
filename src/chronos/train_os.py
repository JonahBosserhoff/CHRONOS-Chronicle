"""CHRONOS backbone as a single-target benchmark on Chronicle cohorts.

Same splits, features and scaling as Chronicle (its split CSVs, read unchanged: nothing is split
or scaled here), the
CHRONOS static MLP + GRU + fusion, one head, and Chronicle's loss for the chosen target:

    survival   (OS, TTNT, ...)                       cox_loss on one random labelled day per patient
    regression (label_* not in --binary_label_cols)  mse (or --regression_loss) on every labelled day
    binary     (--binary_label_cols)                 bce_logits on every labelled day

Writes Chronicle-format outputs, so the predictions join like Chronicle's own:

    <out_dir>/metrics_test.json
    <out_dir>/predictions_test/<target>.parquet
    <out_dir>/checkpoints/best_model.pt, last_model.pt
    <out_dir>/history.csv, training_summary.json, config.json, features.csv

Example:
    python src/chronos/train_os.py \
      --data <chronicle>/data/essen_new_labels \
      --global_format <chronicle>/data/global_format_essen_12_03.csv \
      --target OS --out_dir results_os/chronos_OS
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from chronicle_compat import (
    DEFAULT_BINARY_LABEL_COLS,
    DEFAULT_RECURRENT_SURVIVAL_TARGETS,
    BINARY,
    REGRESSION,
    SURVIVAL,
    FeatureScalers,
    MultiTaskLoss,
    calc_auc,
    calc_c_index,
    calc_correlation,
    loss_for,
    sample_one_token_per_patient,
    scaler_fingerprint,
    task_type,
)
from dataset_os import DEFAULT_STATIC_FEATURES, collate_os, densify, load_chronicle_cohort
from model import CHRONOSSingleTaskModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def valid_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    return torch.arange(T)[None, :] < lengths[:, None]


def build_model(ckpt_or_args: Dict, device) -> CHRONOSSingleTaskModel:
    return CHRONOSSingleTaskModel(
        cov_dim=ckpt_or_args["cov_dim"],
        latent_dim=ckpt_or_args["latent_dim"],
        rnn_layers=ckpt_or_args["rnn_layers"],
        dropout=ckpt_or_args["dropout"],
        static_dim=ckpt_or_args["static_dim"],
    ).to(device)


def make_loader(dataset, batch_size: int, shuffle: bool, drop_last: bool, num_workers: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      collate_fn=collate_os, num_workers=num_workers)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def batch_loss(model, batch, layout, spec, loss_fn, device, survival_sampling: str) -> Optional[torch.Tensor]:
    """Chronicle's Trainer._task_losses for one task; None where the batch has no usable label.

    Survival: one labelled day per patient, drawn by Chronicle's sample_one_token_per_patient
    (applied to the day indices, so the draw is exactly Chronicle's). Each patient is then cut
    at its drawn day before the forward pass: the GRU is causal, so the output at that day is
    the one a full pass would give, without running the GRU over the days after it.
    Regression / binary: every labelled day, NaN labels dropped.
    """
    lengths, labels = batch["lengths"], batch["labels"]
    B, T = len(lengths), int(lengths.max())
    mask = valid_mask(lengths, T)

    if spec["task"] == SURVIVAL:
        index = torch.stack(
            [torch.arange(B).view(B, 1).expand(B, T), torch.arange(T).view(1, T).expand(B, T)], dim=-1
        ).float()
        chosen, y_true = sample_one_token_per_patient(index, labels, mask, strategy=survival_sampling)
        if y_true.shape[0] == 0:
            return None
        rows, days = chosen[:, 0].long(), chosen[:, 1].long()
        cut = torch.ones_like(lengths)
        cut[rows] = days + 1
        out = model(densify(batch, layout["static_dim"], layout["n_dynamic"], device, lengths=cut))
        y_pred = out[rows.to(device), days.to(device)].unsqueeze(-1)
        return loss_fn(y_pred, y_true.to(device))

    out = model(densify(batch, layout["static_dim"], layout["n_dynamic"], device)).unsqueeze(-1)
    m = mask.to(device)
    y_true = labels.to(device)[m]
    y_pred = out[m]
    keep = ~torch.isnan(y_true).any(dim=1)
    y_pred, y_true = y_pred[keep], y_true[keep]
    if y_true.shape[0] == 0:
        return None
    return loss_fn(y_pred, y_true)


def train_epoch(model, loader, layout, spec, loss_fn, mtl, opt, device, args) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        opt.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, layout, spec, loss_fn, device, args.survival_sampling)
        if loss is None:
            continue
        loss = mtl.compute_loss({spec["target"]: loss}, batch_size=args.batch_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + mtl.parameters(), 1.0)
        opt.step()
        total += float(loss.detach())
        n += 1
    return total / n if n else float("nan")


# --------------------------------------------------------------------------- #
# Evaluation (shared with validate_os.py)
# --------------------------------------------------------------------------- #

def _prediction_rows(spec, pred, labels, mask, pids, days, scaler) -> pd.DataFrame:
    """Chronicle's _prediction_rows: one row per (patient, day), regression in original units."""
    mask_np = mask.numpy().astype(bool)
    y_pred = pred.numpy()
    y_true = labels.numpy()
    ids = np.repeat(pids.numpy().reshape(-1, 1), mask_np.shape[1], axis=1)[mask_np]
    day = days[:, : mask_np.shape[1]].numpy()[mask_np]
    prediction = y_pred[mask_np].reshape(-1)
    truth = y_true[mask_np]

    frame = {"patient_id": ids, "day": day, "prediction": prediction}
    if spec["task"] == SURVIVAL:
        frame["time"] = truth[:, 0]
        frame["event"] = truth[:, 1]
    else:
        target = truth[:, 0]
        if scaler is not None:
            prediction = scaler.inverse_transform_label(spec["target"], prediction)
            target = scaler.inverse_transform_label(spec["target"], target)
            frame["prediction"] = prediction
        frame["ground_truth"] = target

    df = pd.DataFrame(frame)
    return df.dropna(subset=["ground_truth"]) if "ground_truth" in df else df


@torch.no_grad()
def run_split(model, loader, layout, spec, loss_fn, device, *, c_index_samples: int, survival_sampling: str,
              mtl=None, mtl_batch_size: int = 0, scaler=None, save_predictions: bool = False) -> Dict:
    """Chronicle's evaluate() for one task (and its Trainer._validate loss for model selection).

    One full GRU pass per patient gives the output at every observed day. Survival: loss on one
    random labelled day per patient per batch, C-index mean ± std over `c_index_samples` draws.
    Regression: Pearson r, binary: ROC-AUC, both over every labelled day and on the scaled
    values, as in Chronicle. A metric that cannot be computed is omitted, not NaN.
    """
    model.eval()
    target = spec["target"]
    loss_sum, n_batches = 0.0, 0
    val_total, val_n = 0.0, 0
    surv_preds, surv_labels, surv_masks = [], [], []
    preds_flat, labels_flat = [], []
    frames = []

    for batch in loader:
        lengths = batch["lengths"]
        out = model(densify(batch, layout["static_dim"], layout["n_dynamic"], device)).float().cpu()
        pred = out.unsqueeze(-1)  # [B, T, 1]
        labels = batch["labels"]
        mask = valid_mask(lengths, out.shape[1])

        if spec["task"] == SURVIVAL:
            y_pred, y_true = sample_one_token_per_patient(pred, labels, mask, strategy=survival_sampling)
            surv_preds.append(pred)
            surv_labels.append(labels)
            surv_masks.append(mask)
        else:
            y_true = labels[mask]
            y_pred = pred[mask]
            keep = ~torch.isnan(y_true).any(dim=1)
            y_pred, y_true = y_pred[keep], y_true[keep]
            preds_flat.append(y_pred)
            labels_flat.append(y_true)

        if y_true.shape[0] > 0:
            loss = loss_fn(y_pred, y_true)
            loss_sum += float(loss)
            if mtl is not None:
                val_total += float(mtl.compute_loss({target: loss}, batch_size=mtl_batch_size))
                val_n += 1

        if save_predictions:
            frames.append(_prediction_rows(spec, pred, labels, mask, batch["pids"], batch["days"], scaler))
        n_batches += 1

    metrics = {
        "split_size": len(loader.dataset),
        "num_batches": n_batches,
        "loss": {target: loss_sum / max(n_batches, 1)},
        "c_index": {},
        "c_index_std": {},
        "correlation": {},
        "auc": {},
    }
    if spec["task"] == SURVIVAL:
        draws = []
        for _ in range(max(1, c_index_samples)):
            sp, sl = [], []
            for p, y, m in zip(surv_preds, surv_labels, surv_masks):
                p_s, y_s = sample_one_token_per_patient(p, y, m, strategy=survival_sampling)
                if p_s.shape[0]:
                    sp.append(p_s)
                    sl.append(y_s)
            c = calc_c_index(sp, sl)
            if c is not None:
                draws.append(c)
        if draws:
            metrics["c_index"][target] = float(np.mean(draws))
            metrics["c_index_std"][target] = float(np.std(draws))
    elif spec["task"] == REGRESSION:
        r = calc_correlation(preds_flat, labels_flat)
        if r is not None:
            metrics["correlation"][target] = r
    else:
        a = calc_auc(preds_flat, labels_flat)
        if a is not None:
            metrics["auc"][target] = a

    result = {"metrics": metrics, "val_loss": val_total / val_n if val_n else float("nan")}
    if save_predictions:
        frames = [f for f in frames if len(f)]
        result["predictions"] = pd.concat(frames, ignore_index=True) if frames else None
    return result


def selection_value(spec, result: Dict, selection_metric: str) -> Optional[float]:
    """Chronicle's _select_metric: None (no update) when the task has no usable samples this epoch."""
    if selection_metric == "val_loss":
        return result["val_loss"]
    group = {SURVIVAL: "c_index", REGRESSION: "correlation", BINARY: "auc"}[spec["task"]]
    return result["metrics"][group].get(spec["target"])


def metric_text(spec, metrics: Dict) -> str:
    t = spec["target"]
    if t in metrics["c_index"]:
        return f"c-index {metrics['c_index'][t]:.4f} ± {metrics['c_index_std'][t]:.4f}"
    if t in metrics["correlation"]:
        return f"pearson r {metrics['correlation'][t]:.4f}"
    if t in metrics["auc"]:
        return f"auc {metrics['auc'][t]:.4f}"
    return "metric n/a"


def write_outputs(out_dir: str, split: str, spec, result: Dict):
    with open(os.path.join(out_dir, f"metrics_{split}.json"), "w") as f:
        json.dump(result["metrics"], f, indent=2)
    if result.get("predictions") is not None:
        pred_dir = os.path.join(out_dir, f"predictions_{split}")
        os.makedirs(pred_dir, exist_ok=True)
        result["predictions"].to_parquet(os.path.join(pred_dir, f"{spec['target']}.parquet"), index=False)


# --------------------------------------------------------------------------- #

def main(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    binary_label_cols = split_list(args.binary_label_cols)
    recurrent_targets = split_list(args.recurrent_survival_targets)
    task = task_type(args.target, binary_label_cols)
    loss_name, loss_fn = loss_for(task, args.regression_loss)
    spec = {"target": args.target, "task": task, "loss": loss_name, "recurrent": args.target in recurrent_targets}
    print(f"Target {args.target!r}: {task}, loss {loss_name}"
          + (" (recurrent survival target)" if spec["recurrent"] and task == SURVIVAL else ""))

    datasets, layout = load_chronicle_cohort(
        args.data,
        args.global_format,
        args.target,
        task,
        recurrent=spec["recurrent"],
        static_features=split_list(args.static_features),
        input_features=split_list(args.input_features) or None,
    )

    scaler_dir = args.scaler_dir or os.path.join(args.data, "scalers")
    fingerprint = scaler_fingerprint(scaler_dir)
    scaler = None
    if fingerprint is None:
        print(f"⚠️ No scalers at {scaler_dir}: regression predictions stay in scaled units, "
              "and external validation will need --scaler_dir.")
    else:
        scaler = FeatureScalers(scaler_dir)
        if task == BINARY and args.target in scaler.label_scalers:
            raise ValueError(f"{args.target!r} is binary but was scaled by chronicle-preprocess.")

    n_train = len(datasets["train"])
    if n_train < args.batch_size:
        raise ValueError(
            f"batch_size={args.batch_size} is larger than the train split ({n_train} patients). "
            "Incomplete batches are dropped (and the batch is the Cox risk set), so lower --batch_size."
        )
    eval_bs = args.eval_batch_size or args.batch_size
    train_loader = make_loader(datasets["train"], args.batch_size, True, True, args.num_workers)
    val_loader = make_loader(datasets["val"], eval_bs, False, False, args.num_workers)
    test_loader = make_loader(datasets["test"], eval_bs, False, False, args.num_workers)

    model_cfg = {
        "cov_dim": layout["static_dim"] + layout["n_dynamic"] + 2,
        "static_dim": layout["static_dim"],
        "latent_dim": args.latent_dim,
        "rnn_layers": args.rnn_layers,
        "dropout": args.dropout,
    }
    model = build_model(model_cfg, device)
    mtl = MultiTaskLoss(
        labels=[args.target],
        mode=args.loss_mode,
        static_weights={args.target: 1.0},
        logvar_bounds=tuple(args.logvar_bounds),
        logvar_prior=args.logvar_prior,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: CHRONOSSingleTaskModel, cov_dim={model_cfg['cov_dim']}, static_dim={model_cfg['static_dim']}, "
          f"{n_params:,} parameters | loss_mode {args.loss_mode}")

    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    best_path = os.path.join(args.out_dir, "checkpoints", "best_model.pt")
    last_path = os.path.join(args.out_dir, "checkpoints", "last_model.pt")
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "spec": spec, "model": model_cfg, "scaler_dir": scaler_dir,
                   "scaler_fingerprint": fingerprint, "static": layout["static"],
                   "n_dynamic": layout["n_dynamic"], "presence_only": layout["presence_only"],
                   "vocab": layout["vocab"]}, f, indent=2)
    pd.DataFrame(layout["columns"]).to_csv(os.path.join(args.out_dir, "features.csv"), index=False)

    def checkpoint(epoch):
        return {"model": model.state_dict(), "mtl": mtl.state_dict(), "model_cfg": model_cfg, "layout": layout,
                "spec": spec, "args": vars(args), "global_format": os.path.abspath(args.global_format),
                "scaler_dir": os.path.abspath(scaler_dir), "scaler_fingerprint": fingerprint,
                "binary_label_cols": binary_label_cols, "epoch": epoch}

    opt = torch.optim.AdamW(list(model.parameters()) + mtl.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    mode_max = args.selection_metric != "val_loss"
    best, best_epoch, patience = (-math.inf if mode_max else math.inf), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, layout, spec, loss_fn, mtl, opt, device, args)
        val = run_split(model, val_loader, layout, spec, loss_fn, device, c_index_samples=args.c_index_samples,
                        survival_sampling=args.survival_sampling, mtl=mtl, mtl_batch_size=args.batch_size)
        value = selection_value(spec, val, args.selection_metric)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["val_loss"],
                        "selection_value": value, **{f"val_{k}": v.get(args.target) for k, v in val["metrics"].items()
                                                     if isinstance(v, dict) and k != "loss"}})
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] train loss {train_loss:.4f} | val loss {val['val_loss']:.4f} "
              f"| val {metric_text(spec, val['metrics'])}")

        if value is not None:
            improved = value > best if mode_max else value < best
            if improved:
                best, best_epoch, patience = value, epoch, 0
                torch.save(checkpoint(epoch), best_path)
            else:
                patience += 1
        else:
            print(f"⚠️ {args.selection_metric} not available this epoch — model selection not updated.")
        if args.early_stopping_patience and patience >= args.early_stopping_patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    torch.save(checkpoint(epoch), last_path)
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)

    if best_epoch is None:
        print("⚠️ The selection metric was never available on val — evaluating the LAST model.")
        eval_path = last_path
    else:
        print(f"Best epoch {best_epoch} ({args.selection_metric} {best:.4f}). Evaluating on test...")
        eval_path = best_path
    model.load_state_dict(torch.load(eval_path, map_location=device, weights_only=False)["model"])

    test = run_split(model, test_loader, layout, spec, loss_fn, device, c_index_samples=args.c_index_samples,
                     survival_sampling=args.survival_sampling, scaler=scaler, save_predictions=True)
    write_outputs(args.out_dir, "test", spec, test)
    with open(os.path.join(args.out_dir, "training_summary.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_value": None if best_epoch is None else best,
                   "selection_metric": args.selection_metric, "evaluated_checkpoint": eval_path,
                   "epochs_run": epoch}, f, indent=2)
    print(f"\n✅ Test {metric_text(spec, test['metrics'])} | saved to {args.out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Chronicle cohort dir with <prefix>_{train,val,test}.csv")
    p.add_argument("--global_format", type=str, required=True, help="Chronicle feature schema CSV")
    p.add_argument("--out_dir", type=str, default="results_os")
    p.add_argument("--scaler_dir", type=str, default="", help="scalers of the cohort; empty = <data>/scalers")
    p.add_argument("--seed", type=int, default=42)

    # Target (Chronicle's rules: binary_label_cols -> BCE, label* -> regression, else survival)
    p.add_argument("--target", type=str, default="OS")
    p.add_argument("--binary_label_cols", type=str, default=",".join(DEFAULT_BINARY_LABEL_COLS))
    p.add_argument("--recurrent_survival_targets", type=str, default=",".join(DEFAULT_RECURRENT_SURVIVAL_TARGETS))
    p.add_argument("--regression_loss", type=str, default="mse", choices=["mse", "l1", "huber"])
    p.add_argument("--loss_mode", type=str, default="static", choices=["static", "uncertainty"])
    p.add_argument("--logvar_prior", type=float, default=1.0)
    p.add_argument("--logvar_bounds", type=float, nargs=2, default=[-8.0, 8.0])
    p.add_argument("--survival_sampling", type=str, default="random", choices=["random", "last"])

    # Inputs
    p.add_argument("--static_features", type=str, default=",".join(DEFAULT_STATIC_FEATURES))
    p.add_argument("--input_features", type=str, default="", help="comma list; empty = every schema feature in the data")

    # CHRONOS defaults (src/chronos/train.py)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024, help="for survival targets also the Cox risk set")
    p.add_argument("--eval_batch_size", type=int, default=0, help="0 = batch_size; lower it if long patients run out of memory")
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--rnn_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)

    # Chronicle evaluation protocol
    p.add_argument("--c_index_samples", type=int, default=10)
    p.add_argument("--selection_metric", type=str, default="auto", choices=["auto", "val_loss"],
                   help="auto = c-index (survival) / pearson r (regression) / auc (binary) on val")
    p.add_argument("--early_stopping_patience", type=int, default=0, help="0 = off; the best val epoch is kept either way")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
