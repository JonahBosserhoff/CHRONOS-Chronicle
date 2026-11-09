"""Feature ablation of the models trained by train_os.py: test-split inference with features masked.

For every target of the cohort that has a model (<models_dir>/<target>/checkpoints/best_model.pt,
the layout of train_os.py / slurm/train.sbatch; a target without a model is skipped) and every
masking level p (default 10, 20, ..., 90 %; 0 and 100 are not run):

    1. draw k = round(p/100 · N) of the N features of the cohort, i.e. of the model's vocabulary:
       the schema features that occur anywhere in the cohort (train, val and test), static
       features (age, sex) included unless --no-mask_static
    2. mask them in the test inputs: every observation of them is removed, so they look like a
       feature that was never measured (value 0 and observed-mask 0; presence-only features
       not present; static features [value, observed] = [0, 0])
    3. load the model, run inference on the test split, compute Chronicle's metric
       (C-index / Pearson r / AUC) exactly as train_os.py's test evaluation does

Nothing else changes: same patients, days, labels, time features, history cap (--time_features
and --max_seq_len of the training run), batch size and metric protocol as the test evaluation
of train_os.py. A day whose observations are all masked is kept
(it is predicted from the history so far and the time features), so every level is scored on
the same (patient, day) points as the unmasked model.

Every mask is drawn anew: level p of repeat r masks k features drawn at random from all N, from
its own random stream (--seed, r, p), independently of every other level and repeat (the 20 %
mask does not build on the 10 % one). Targets whose models have the same vocabulary get the same
mask for the same (r, p), so the targets are compared on equal masks. The C-index draws are
re-seeded with --seed before every evaluation, so all levels are scored on the same draws. With
--repeats R, every level is evaluated with R independent masks.

Outputs in --out_dir:

    feature_ablation.csv           one row per (target, repeat, percent): metric, its std over the
                                   C-index draws (survival), loss, n_features, n_masked
    feature_ablation_summary.csv   per (target, percent): mean and std over the repeats
    feature_ablation_masks.csv     the masked features of every (target, repeat, percent)
    feature_ablation_config.json   options, evaluated checkpoints, skipped targets and why

Example (from the repository root), every target and every level in one run:
    python src/chronos/feature_ablation_os.py \
      --data <cohort_dir> \
      --models_dir results_os \
      --out_dir results_os/feature_ablation

or with the options in a YAML file (template: configs/ablation.yaml); options given on the
command line override the file:
    python src/chronos/feature_ablation_os.py --config configs/ablation.yaml [--repeats 5 ...]
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import torch

from cli_config import parse_args as parse_with_config
from chronicle_compat import BINARY, DEFAULT_BINARY_LABEL_COLS, REGRESSION, SURVIVAL, loss_for, scaler_fingerprint
from dataset_os import PATIENT_ID_COL, ChronicleOSDataset, describe, encode_split, read_cohort_csv, resolve_split_csvs
from train_os import DEFAULT_WINDOW_BATCH_SIZE, build_model, make_loader, metric_text, run_split, set_seed, split_list


METRIC = {SURVIVAL: "c_index", REGRESSION: "correlation", BINARY: "auc"}
DEFAULT_PERCENTS = ",".join(str(p) for p in range(10, 100, 10))
OUTPUT_FILES = ("feature_ablation.csv", "feature_ablation_summary.csv", "feature_ablation_masks.csv",
                "feature_ablation_config.json")


# --------------------------------------------------------------------------- #
# Targets and checkpoints
# --------------------------------------------------------------------------- #

def cohort_targets(df: pd.DataFrame, binary_label_cols: Sequence[str]) -> List[str]:
    """Every target the cohort has columns for (Chronicle's rules, see chronicle_compat.task_type).

    Survival: <X>_TIME together with <X>_EVENT. Regression / binary: columns starting with
    `label`, and the --binary_label_cols present in the data.
    """
    cols = [str(c) for c in df.columns]
    present = set(cols)
    found = []
    for c in cols:
        if c.endswith("_TIME") and len(c) > len("_TIME") and c[: -len("_TIME")] + "_EVENT" in present:
            found.append(c[: -len("_TIME")])
        elif c.startswith("label") or c in binary_label_cols:
            found.append(c)
    return list(dict.fromkeys(found))


def checkpoint_path(models_dir: str, target: str, name: str) -> str:
    return os.path.join(models_dir, target, "checkpoints", name)


def check_checkpoint(ckpt: Dict, path: str, target: str, data_dir: str):
    """The checkpoint is the model of `target`, trained on this cohort."""
    if ckpt["spec"]["target"] != target:
        raise ValueError(f"{path} is the model of {ckpt['spec']['target']!r}, not of {target!r}.")

    data_fp = scaler_fingerprint(os.path.join(data_dir, "scalers"))
    if data_fp is not None and ckpt["scaler_fingerprint"] is not None and data_fp != ckpt["scaler_fingerprint"]:
        raise ValueError(
            f"{path} was trained with the scalers at {ckpt['scaler_dir']}, but {data_dir} was scaled with other "
            f"scalers ({os.path.join(data_dir, 'scalers')}). Pass the cohort directory the model was trained on."
        )
    trained_on = ckpt["args"]["data"]
    if os.path.realpath(trained_on) != os.path.realpath(data_dir):
        print(f"   note: {target} was trained on --data {trained_on}; evaluating on {data_dir}.")


def check_test_size(path: str, target: str, n_patients: int):
    """Same test patients as train_os.py's own test evaluation (its metrics_test.json), when present."""
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(path)), "metrics_test.json")
    if not os.path.isfile(metrics_path):
        return
    with open(metrics_path) as f:
        expected = json.load(f).get("split_size")
    if expected is not None and int(expected) != n_patients:
        raise ValueError(
            f"{target}: the test split has {n_patients} patients, but the model was tested on {expected} "
            f"({metrics_path}). Pass the cohort directory the model was trained on."
        )


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #

def parse_percents(raw: str) -> List[int]:
    try:
        percents = [int(p) for p in split_list(raw)]
    except ValueError:
        raise ValueError(f"--percents must be whole numbers, comma-separated (got {raw!r}).") from None
    if not percents:
        raise ValueError("--percents is empty.")
    bad = [p for p in percents if not 0 < p < 100]
    if bad:
        raise ValueError(f"--percents must lie strictly between 0 and 100 (got {bad}).")
    if len(set(percents)) != len(percents):
        raise ValueError(f"--percents has duplicates: {percents}.")
    return sorted(percents)


def n_masked(percent: int, n_features: int) -> int:
    """round(percent/100 · n_features), halves rounded up (exact integer arithmetic)."""
    return (percent * n_features + 50) // 100


def draw_mask(pool: Sequence[str], percent: int, seed: int, repeat: int) -> List[str]:
    """n_masked(percent) features drawn at random from `pool`, without replacement.

    Each (seed, repeat, percent) has its own random stream, so every level and repeat gets a mask
    of its own, and the same mask whatever else is run (other levels, targets or repeats).
    """
    pool = sorted(pool)
    rng = np.random.default_rng([seed, repeat, percent])
    picked = rng.choice(len(pool), size=n_masked(percent, len(pool)), replace=False)
    return sorted(pool[i] for i in picked)


def mask_records(records: List[Dict], layout: Dict, masked: Set[str]) -> List[Dict]:
    """Copies of the test records in which the `masked` features were never observed.

    Dynamic features: their value, observed and present entries are removed (densify then
    leaves 0 in those columns). Static features: [value, observed] become [0, 0]. Days, labels
    and every other entry stay as they are.
    """
    unknown = set(masked) - set(layout["value_col"]) - set(layout["static"])
    if unknown:
        raise ValueError(f"Features without an input column in the model: {sorted(unknown)}.")

    drop = np.zeros(layout["n_dynamic"], dtype=bool)
    for f in masked:
        if f in layout["value_col"]:
            drop[layout["value_col"][f]] = True
        if f in layout["mask_col"]:
            drop[layout["mask_col"][f]] = True
    static_cols = [c for i, f in enumerate(layout["static"]) if f in masked for c in (2 * i, 2 * i + 1)]

    out = []
    for r in records:
        keep = ~drop[r["cols"]]
        static = r["static"].copy()
        static[static_cols] = 0.0
        out.append({**r, "rows": r["rows"][keep], "cols": r["cols"][keep], "vals": r["vals"][keep], "static": static})
    return out


# --------------------------------------------------------------------------- #

def write_results(out_dir: str, rows: List[Dict], masks: List[Dict]):
    results = pd.DataFrame(rows, columns=["target", "task", "metric", "percent", "repeat", "n_features", "n_masked",
                                          "value", "value_std", "loss", "n_patients"])
    results.to_csv(os.path.join(out_dir, "feature_ablation.csv"), index=False)
    pd.DataFrame(masks, columns=["target", "repeat", "percent", "feature"]).to_csv(
        os.path.join(out_dir, "feature_ablation_masks.csv"), index=False)

    summary = (
        results.groupby(["target", "task", "metric", "percent", "n_features", "n_masked"], sort=False)["value"]
        .agg(mean="mean", std="std", n_repeats="count")
        .reset_index()
    )
    summary.to_csv(os.path.join(out_dir, "feature_ablation_summary.csv"), index=False)


def main(args):
    percents = parse_percents(args.percents)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.seed < 0:
        raise ValueError("--seed must be 0 or positive.")
    existing = [f for f in OUTPUT_FILES if os.path.exists(os.path.join(args.out_dir, f))]
    if existing and not args.overwrite:
        raise FileExistsError(f"{args.out_dir} already holds {existing}. Pass another --out_dir, or --overwrite.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_csv = resolve_split_csvs(args.data)["test"]
    test_df = read_cohort_csv(test_csv)
    print(f"📄 test: {test_csv} ({test_df[PATIENT_ID_COL].nunique()} patients, {len(test_df)} rows)")

    targets = split_list(args.targets) or cohort_targets(test_df, split_list(args.binary_label_cols))
    if not targets:
        raise ValueError(f"{test_csv} has no target columns (<X>_TIME/<X>_EVENT or label*); pass --targets.")
    print(f"Targets: {targets} | masking levels: {percents} % | repeats: {args.repeats}")

    # ---- models: every checkpoint is loaded and checked before any inference ---- #
    runs, skipped = [], {}
    for target in targets:
        path = checkpoint_path(args.models_dir, target, args.checkpoint_name)
        if not os.path.isfile(path):
            print(f"⏭️  {target}: no model at {path}, skipped.")
            skipped[target] = f"no model at {path}"
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        check_checkpoint(ckpt, path, target, args.data)
        runs.append((target, path, ckpt))
    if not runs:
        raise FileNotFoundError(
            f"None of the targets {targets} has a model at {args.models_dir}/<target>/checkpoints/{args.checkpoint_name}."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    config = {"args": vars(args), "test_csv": os.path.abspath(test_csv), "percents": percents,
              "device": str(device), "evaluated": {t: os.path.abspath(p) for t, p, _ in runs}, "skipped": skipped}
    with open(os.path.join(args.out_dir, "feature_ablation_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    rows, masks = [], []
    for target, path, ckpt in runs:
        spec, layout, train_args = ckpt["spec"], ckpt["layout"], ckpt["args"]
        task, metric = spec["task"], METRIC[spec["task"]]
        print(f"\n=== {target} ({task}, {metric}) | {path} (epoch {ckpt['epoch']})")

        records = encode_split(test_df, layout, target, task, recurrent=spec["recurrent"], name="test")
        describe("test", ChronicleOSDataset(records))
        check_test_size(path, target, len(records))

        model = build_model(ckpt["model_cfg"], device)
        model.load_state_dict(ckpt["model"])
        _, loss_fn = loss_for(task, train_args["regression_loss"])
        eval_bs = args.eval_batch_size or train_args["eval_batch_size"] or train_args["batch_size"]
        c_index_samples = args.c_index_samples if args.c_index_samples is not None else train_args["c_index_samples"]

        static = set(layout["static"])
        pool = [f for f in layout["vocab"] if args.mask_static or f not in static]
        if not pool:
            raise ValueError(f"{target}: the model has no feature to mask (static features excluded by --no-mask_static).")
        print(f"   {len(pool)} features to mask from"
              + ("" if args.mask_static else f" (static features {sorted(static)} never masked)"))

        for repeat in range(args.repeats):
            for percent in percents:
                masked = draw_mask(pool, percent, args.seed, repeat)
                k = len(masked)
                if k == 0:
                    print(f"⚠️ {percent} % of {len(pool)} features rounds to 0: nothing is masked at this level.")
                dataset = ChronicleOSDataset(mask_records(records, layout, set(masked)))
                loader = make_loader(dataset, eval_bs, False, False, args.num_workers)

                set_seed(args.seed)
                result = run_split(model, loader, layout, spec, loss_fn, device, c_index_samples=c_index_samples,
                                   survival_sampling=train_args["survival_sampling"],
                                   window_batch_size=train_args.get("window_batch_size", DEFAULT_WINDOW_BATCH_SIZE))
                m = result["metrics"]
                value = m[metric].get(target)
                std = m["c_index_std"].get(target) if task == SURVIVAL else None
                rows.append({"target": target, "task": task, "metric": metric, "percent": percent, "repeat": repeat,
                             "n_features": len(pool), "n_masked": k,
                             "value": np.nan if value is None else value, "value_std": np.nan if std is None else std,
                             "loss": m["loss"][target], "n_patients": m["split_size"]})
                masks += [{"target": target, "repeat": repeat, "percent": percent, "feature": f} for f in masked]
                print(f"   repeat {repeat} | {percent:2d} % masked ({k}/{len(pool)}) | {metric_text(spec, m)}")
                write_results(args.out_dir, rows, masks)

    print(f"\n✅ Feature ablation of {[t for t, _, _ in runs]} saved to {args.out_dir}"
          + (f" | skipped (no model): {sorted(skipped)}" if skipped else ""))


REQUIRED = ("data",)


def build_parser() -> argparse.ArgumentParser:
    # No abbreviations: "--conf x" must not be taken as --config by one parser and missed by the other.
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--config", type=str, default="",
                   help="YAML file with any of the options below (keys without '--'); the command line overrides it")
    # required, from the command line or --config (checked after both are merged)
    p.add_argument("--data", type=str, default="",
                   help="(required) Chronicle cohort dir the models were trained on (<prefix>_test.csv is evaluated)")
    p.add_argument("--models_dir", type=str, default="results_os",
                   help="models at <models_dir>/<target>/checkpoints/<checkpoint_name>")
    p.add_argument("--checkpoint_name", type=str, default="best_model.pt")
    p.add_argument("--out_dir", type=str, default="results_os/feature_ablation")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False,
                   help="replace the results of an earlier ablation in --out_dir")
    p.add_argument("--targets", type=str, default="",
                   help="comma list; empty = every target of the test split (<X>_TIME/<X>_EVENT, label*)")
    p.add_argument("--binary_label_cols", type=str, default=",".join(DEFAULT_BINARY_LABEL_COLS),
                   help="only to find targets that do not start with 'label'; each model knows its own task")
    p.add_argument("--percents", type=str, default=DEFAULT_PERCENTS,
                   help="masking levels in %%, each strictly between 0 and 100")
    p.add_argument("--repeats", type=int, default=1, help="independent random masks per level")
    p.add_argument("--mask_static", action=argparse.BooleanOptionalAction, default=True,
                   help="static features (age, sex) can be masked like every other feature")
    p.add_argument("--seed", type=int, default=42, help="masks and C-index draws")
    p.add_argument("--eval_batch_size", type=int, default=0, help="0 = the training run's")
    p.add_argument("--c_index_samples", type=int, default=None, help="default: the training run's")
    p.add_argument("--num_workers", type=int, default=0)
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return parse_with_config(build_parser(), REQUIRED, argv)


if __name__ == "__main__":
    main(parse_args())
