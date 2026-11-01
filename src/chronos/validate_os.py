"""External validation of a model trained by train_os.py on another (raw, unscaled) cohort.

    1. load the checkpoint: weights, input layout, target, and where its scalers live
    2. load those scalers (the training cohort's, from `chronicle-preprocess`), refusing other
       ones unless --allow_other_scalers
    3. scale the external cohort with them, exactly as `chronicle-preprocess --pretrained-scaler-dir`
       does (features and regression targets; categoricals, survival times and 0/1 targets stay)
    4. build the daily sequences with the TRAINING layout and run inference; nothing is fitted

Outputs in --out_dir, in Chronicle's format:

    metrics_external.json
    predictions_external/<target>.parquet
    validation_config.json

The external cohort must be prepared like the training one before `chronicle-preprocess`
(outliers removed, labels created with the same rules), and must have the target's column(s).

Example:
    python src/chronos/validate_os.py \
      --checkpoint results_os/chronos_OS/checkpoints/best_model.pt \
      --data <site_b>/site_b.csv \
      --out_dir results_os/chronos_OS_site_b

or with the options in a YAML file (template: configs/validate_external.yaml); options given on
the command line override the file:
    python src/chronos/validate_os.py --config configs/validate_external.yaml [--seed 1 ...]
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import pandas as pd
import torch

from chronicle_compat import FeatureScalers, loss_for, scaler_fingerprint
from dataset_os import (
    KEY_COL,
    PATIENT_ID_COL,
    TIME_COL,
    VALUE_COL,
    ChronicleOSDataset,
    build_vocab,
    describe,
    encode_split,
    read_cohort_csv,
    read_schema,
    resolve_raw_csv,
)
from train_os import build_model, make_loader, metric_text, run_split, set_seed, write_outputs


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec, layout, train_args = ckpt["spec"], ckpt["layout"], ckpt["args"]
    set_seed(args.seed if args.seed is not None else train_args["seed"])
    print(f"Checkpoint {args.checkpoint} (epoch {ckpt['epoch']}): target {spec['target']!r} ({spec['task']}, "
          f"{spec['loss']})")

    # ---- scalers of the training cohort ----------------------------------- #
    scaler_dir = args.scaler_dir or ckpt["scaler_dir"]
    fingerprint = scaler_fingerprint(scaler_dir)
    if fingerprint is None:
        raise FileNotFoundError(f"No fitted scalers under {scaler_dir}. Pass the training cohort's --scaler_dir.")
    if ckpt["scaler_fingerprint"] is None:
        print("⚠️ The checkpoint recorded no scalers, so it cannot be checked that these are the training ones.")
    elif fingerprint != ckpt["scaler_fingerprint"]:
        if not args.allow_other_scalers:
            raise ValueError(
                f"The scalers at {scaler_dir} are not the ones the model was trained with "
                f"({ckpt['scaler_dir']}). The model only understands inputs scaled like its training data. "
                "Pass --allow_other_scalers to use them anyway."
            )
        print(f"⚠️ Using scalers that differ from the training ones ({ckpt['scaler_dir']}).")
    scalers = FeatureScalers(scaler_dir)
    if (scalers.patient_id_col, scalers.key_col, scalers.value_col) != (PATIENT_ID_COL, KEY_COL, VALUE_COL):
        raise ValueError(
            f"The scalers were fitted on columns {(scalers.patient_id_col, scalers.key_col, scalers.value_col)}, "
            f"this pipeline reads {(PATIENT_ID_COL, KEY_COL, VALUE_COL)}."
        )
    print(f"Scalers: {scaler_dir} ({scalers.scaler_type}, {len(scalers.feature_scalers)} features, "
          f"{len(scalers.label_scalers)} targets)")

    # ---- external cohort -------------------------------------------------- #
    csv_path = resolve_raw_csv(args.data)
    raw = read_cohort_csv(csv_path)
    for col in (PATIENT_ID_COL, KEY_COL, VALUE_COL, TIME_COL):
        if col not in raw.columns:
            raise ValueError(f"{csv_path}: missing required column {col!r}.")
    print(f"📄 external: {csv_path} ({raw[PATIENT_ID_COL].nunique()} patients, {len(raw)} rows)")

    if args.patient_ids:
        ids = pd.read_csv(args.patient_ids)[PATIENT_ID_COL]
        missing = set(ids) - set(raw[PATIENT_ID_COL])
        if missing:
            print(f"⚠️ {len(missing)} ids of {args.patient_ids} are not in the cohort.")
        raw = raw[raw[PATIENT_ID_COL].isin(set(ids))]
        if raw.empty:
            raise ValueError(f"None of the ids in {args.patient_ids} is in {csv_path}.")
        print(f"   restricted to {raw[PATIENT_ID_COL].nunique()} patients of {args.patient_ids}")

    scaled = scalers.transform(raw, audit=True)

    global_format = args.global_format or ckpt["global_format"]
    vocab = build_vocab(read_schema(global_format), [scaled], layout["input_features"])
    model_feats = set(layout["value_col"]) | set(layout["static"])
    unseen = sorted(set(vocab) - model_feats)
    absent = sorted(model_feats - set(vocab))
    if unseen:
        print(f"⚠️ {len(unseen)} features of this cohort were not in the training cohort; they only mark "
              f"observed days, the model has no input for them: {unseen}")
    if absent:
        print(f"⚠️ {len(absent)} model features never occur in this cohort (inputs stay 0 / unobserved): {absent}")

    dataset = ChronicleOSDataset(
        encode_split(scaled, layout, spec["target"], spec["task"], recurrent=spec["recurrent"], vocab=vocab,
                     name="external")
    )
    describe("external", dataset)

    # ---- inference -------------------------------------------------------- #
    model = build_model(ckpt["model_cfg"], device)
    model.load_state_dict(ckpt["model"])
    _, loss_fn = loss_for(spec["task"], train_args["regression_loss"])
    eval_bs = args.eval_batch_size or train_args["eval_batch_size"] or train_args["batch_size"]
    loader = make_loader(dataset, eval_bs, False, False, args.num_workers)

    result = run_split(
        model, loader, layout, spec, loss_fn, device,
        c_index_samples=args.c_index_samples if args.c_index_samples is not None else train_args["c_index_samples"],
        survival_sampling=train_args["survival_sampling"],
        scaler=scalers,
        save_predictions=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    write_outputs(args.out_dir, "external", spec, result)
    with open(os.path.join(args.out_dir, "validation_config.json"), "w") as f:
        json.dump({"config": os.path.abspath(args.config) if args.config else None, "args": vars(args),
                   "checkpoint": os.path.abspath(args.checkpoint), "cohort": os.path.abspath(csv_path),
                   "patient_ids": args.patient_ids or None, "scaler_dir": os.path.abspath(scaler_dir),
                   "scaler_fingerprint": fingerprint, "same_scalers_as_training": fingerprint == ckpt["scaler_fingerprint"],
                   "global_format": os.path.abspath(global_format), "spec": spec,
                   "n_patients": len(dataset), "features_not_in_model": unseen, "model_features_absent": absent},
                  f, indent=2)
    print(f"\n✅ External {metric_text(spec, result['metrics'])} | saved to {args.out_dir}")


REQUIRED = ("checkpoint", "data", "out_dir")


def build_parser() -> argparse.ArgumentParser:
    # No abbreviations: "--conf x" must not be taken as --config by one parser and missed by the other.
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--config", type=str, default="",
                   help="YAML file with any of the options below (keys without '--'); the command line overrides it")
    # required, from the command line or --config (checked after both are merged)
    p.add_argument("--checkpoint", type=str, default="", help="(required) best_model.pt (or last_model.pt) of train_os.py")
    p.add_argument("--data", type=str, default="", help="(required) raw (unscaled) external cohort: CSV file or its directory")
    p.add_argument("--out_dir", type=str, default="", help="(required)")
    p.add_argument("--scaler_dir", type=str, default="", help="empty = the scaler dir recorded in the checkpoint")
    p.add_argument("--allow_other_scalers", action=argparse.BooleanOptionalAction, default=False,
                   help="accept scalers that differ from the training ones")
    p.add_argument("--global_format", type=str, default="", help="empty = the schema recorded in the checkpoint")
    p.add_argument("--patient_ids", type=str, default="", help="optional CSV with a patient_id column to restrict to")
    p.add_argument("--eval_batch_size", type=int, default=0, help="0 = the training run's")
    p.add_argument("--c_index_samples", type=int, default=None, help="default: the training run's")
    p.add_argument("--seed", type=int, default=None, help="default: the training run's")
    p.add_argument("--num_workers", type=int, default=0)
    return p


def config_argv(path: str, parser: argparse.ArgumentParser) -> List[str]:
    """The options of a YAML file as command-line arguments, so argparse converts and checks them.

    Keys are the option names without '--'. null keeps the default. Strings get ~ and $VARS
    expanded; relative paths stay relative to the working directory, as on the command line.
    """
    import yaml  # only needed with --config

    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return []
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a mapping of option: value, got {type(cfg).__name__}.")

    options = {a.dest: a for a in parser._actions if a.option_strings and a.dest not in ("help", "config")}
    unknown = sorted(str(k) for k in cfg if k not in options)
    if unknown:
        raise ValueError(f"{path}: unknown option(s) {unknown}. Valid: {sorted(options)}.")

    argv = []
    for key, value in cfg.items():
        if value is None:
            continue
        action = options[key]
        flag = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ValueError(f"{path}: {key} must be true or false, got {value!r}.")
            argv.append(flag if value else f"--no-{flag[2:]}")
        elif isinstance(value, (bool, list, dict)):
            raise ValueError(f"{path}: {key} takes a single value, got {value!r}.")
        else:
            value = os.path.expandvars(os.path.expanduser(value)) if isinstance(value, str) else str(value)
            argv += [flag, value]
    return argv


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", type=str, default="")
    config = pre.parse_known_args(argv)[0].config
    if config:
        argv = config_argv(config, parser) + argv  # later arguments win: the command line overrides the file
    args = parser.parse_args(argv)
    missing = [name for name in REQUIRED if not getattr(args, name)]
    if missing:
        parser.error(f"missing {', '.join('--' + m for m in missing)} (on the command line or in --config)")
    return args


if __name__ == "__main__":
    main(parse_args())
