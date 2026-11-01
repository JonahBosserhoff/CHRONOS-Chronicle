"""Chronicle cohorts -> daily sequences for the CHRONOS backbone (Chronicle targets benchmark).

Reads Chronicle's long format (one row per observation):

    patient_id,feature_name,feature_value,time_since_initial_diagnosis,OS_TIME,OS_EVENT,label_...,...

either as the split CSVs written by `chronicle-preprocess` (already scaled, scalers fitted on
the train patients) or, for external validation, as a raw cohort scaled with the training
scalers. Every patient becomes one timestep per observed day:

    static     [value, observed] per static feature (age, sex)     -> CHRONOS static MLP
    dynamic    per feature: last value of the day (+ observed mask) -> CHRONOS GRU
    time       years since diagnosis, years since the previous observed day
    label      the target of the day's first row (what Chronicle's CLS token clones):
               [time, event] for survival targets, [value] for regression / binary targets

The days are Chronicle's: every day with an observation of a vocabulary feature (schema ∩
data, optionally narrowed by `input_features`). Columns are ordered by the schema's global_id.
"""

import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from torch.utils.data import Dataset

from chronicle_compat import BINARY, SURVIVAL


PATIENT_ID_COL = "patient_id"
TIME_COL = "time_since_initial_diagnosis"
KEY_COL = "feature_name"
VALUE_COL = "feature_value"

DEFAULT_STATIC_FEATURES = ["age_at_diagnosis", "gender_female"]
SPLITS = ("train", "val", "test")
N_TIME_FEATURES = 2
DAYS_PER_YEAR = 365.25


def resolve_split_csvs(data_path: str) -> Dict[str, str]:
    """`<prefix>_{train,val,test}.csv` under data_path; the directory name wins as prefix."""
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"{data_path} is not a directory with Chronicle split CSVs.")

    def by_prefix(prefix: str) -> Dict[str, str]:
        found = {s: os.path.join(data_path, f"{prefix}_{s}.csv") for s in SPLITS}
        return found if all(os.path.isfile(p) for p in found.values()) else {}

    found = by_prefix(os.path.basename(os.path.normpath(data_path)))
    if found:
        return found

    prefixes = {
        m.group(1)
        for f in os.listdir(data_path)
        for m in [re.match(r"^(.*)_(train|val|test)\.csv$", f)]
        if m
    }
    if len(prefixes) > 1:
        raise FileNotFoundError(f"{data_path} holds split CSVs of more than one cohort: {sorted(prefixes)}.")
    if prefixes:
        found = by_prefix(prefixes.pop())
        if found:
            return found
    raise FileNotFoundError(
        f"No <prefix>_train/_val/_test.csv under {data_path}. Pass the cohort directory Chronicle trains on "
        "(with its split CSVs); CHRONOS does not split or scale a cohort for training."
    )


def resolve_raw_csv(data_path: str) -> str:
    """The raw (unscaled) cohort: a CSV file, or `<dir>/<dir>.csv`, or the one un-split CSV in a dir."""
    if os.path.isfile(data_path):
        return data_path
    if os.path.isdir(data_path):
        own = os.path.join(data_path, os.path.basename(os.path.normpath(data_path)) + ".csv")
        if os.path.isfile(own):
            return own
        loose = sorted(
            f for f in os.listdir(data_path)
            if f.endswith(".csv") and not re.search(r"_(train|val|test)\.csv$", f)
        )
        if len(loose) == 1:
            return os.path.join(data_path, loose[0])
        if len(loose) > 1:
            raise FileNotFoundError(f"{data_path} holds several un-split CSVs {loose}; pass the file itself.")
    raise FileNotFoundError(f"No raw cohort CSV at {data_path}.")


def patient_id_to_int(pid) -> int:
    """Same integer id Chronicle writes to its prediction files (so the two can be joined)."""
    try:
        return int(pid)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(pid))
        if not match:
            raise ValueError(f"Cannot derive a numeric id from patient_id {pid!r}.") from None
        return int(match.group())


def read_cohort_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def read_schema(schema_csv: str) -> pd.DataFrame:
    schema = pd.read_csv(schema_csv)
    for col in (KEY_COL, "global_id", "type"):
        if col not in schema.columns:
            raise ValueError(f"{schema_csv} must have a {col!r} column.")
    schema = schema[schema["type"] != "special_token"].copy()
    schema[KEY_COL] = schema[KEY_COL].astype(str)
    return schema


def build_vocab(
    schema: pd.DataFrame,
    frames: Sequence[pd.DataFrame],
    input_features: Optional[Sequence[str]] = None,
) -> List[str]:
    """Chronicle's vocabulary: schema features that occur in the data (∩ input_features)."""
    data_feats = set()
    for df in frames:
        data_feats |= set(df[KEY_COL].astype(str))
    vocab = set(schema[KEY_COL]) & data_feats
    if input_features:
        missing = set(input_features) - vocab
        if missing:
            print(f"⚠️ Requested input features not in data/schema: {sorted(missing)}")
        vocab &= set(input_features)
    unknown = data_feats - set(schema[KEY_COL])
    if unknown:
        print(f"⚠️ Features in the data but not in the schema (ignored): {sorted(unknown)}")
    if not vocab:
        raise ValueError("No usable input features: data and schema share no feature names.")
    return sorted(vocab)


def build_layout(
    schema_csv: str,
    frames: Dict[str, pd.DataFrame],
    static_features: Sequence[str],
    input_features: Optional[Sequence[str]] = None,
) -> Dict:
    """Decide which feature goes to which input column. Fitted on the training cohort.

    Continuous features get a value and an observed-mask column. Categorical features whose
    train values are all 1 (ICD, OPS, substances, metastases, ...) only exist as "documented",
    so a single presence column carries all their information. Other categoricals (TNM, ECOG,
    gender) keep value + mask, since 0 is a real value there.
    """
    schema = read_schema(schema_csv)
    vocab = build_vocab(schema, list(frames.values()), input_features)

    schema = schema[schema[KEY_COL].isin(vocab)].sort_values("global_id", kind="stable")
    feat_type = dict(zip(schema[KEY_COL], schema["type"]))

    static = [f for f in static_features if f in vocab]
    dynamic = [f for f in schema[KEY_COL] if f not in static]

    train = frames["train"]
    all_one = train.groupby(KEY_COL)[VALUE_COL].agg(lambda v: bool(np.all(v.to_numpy() == 1.0)))

    value_col: Dict[str, int] = {}
    mask_col: Dict[str, int] = {}
    columns: List[Dict] = []
    for f in dynamic:
        presence_only = feat_type[f] == "categorical" and bool(all_one.get(f, False))
        value_col[f] = len(columns)
        columns.append({"column": len(columns), "feature": f, "kind": "present" if presence_only else "value",
                        "schema_type": feat_type[f]})
        if not presence_only:
            mask_col[f] = len(columns)
            columns.append({"column": len(columns), "feature": f, "kind": "observed", "schema_type": feat_type[f]})
    for name in ("years_since_diagnosis", "years_since_previous_day"):
        columns.append({"column": len(columns), "feature": name, "kind": "time", "schema_type": "time"})

    return {
        "vocab": vocab,
        "input_features": list(input_features) if input_features else None,
        "static": static,
        "dynamic": dynamic,
        "value_col": value_col,
        "mask_col": mask_col,
        "presence_only": sorted(f for f in dynamic if f not in mask_col),
        "n_dynamic": len(columns) - N_TIME_FEATURES,
        "static_dim": 2 * len(static),
        "columns": columns,
    }


def label_columns(target: str, task: str) -> List[str]:
    return [f"{target}_TIME", f"{target}_EVENT"] if task == SURVIVAL else [target]


def check_target(df: pd.DataFrame, target: str, task: str, name: str):
    """The target has a column, and a binary target really is 0/1 (Chronicle's _check_binary_labels)."""
    missing = [c for c in label_columns(target, task) if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: target {target!r} has no column {missing}. Survival targets need "
            f"{target}_TIME / {target}_EVENT, regression and binary targets the column {target!r}."
        )
    if task == BINARY:
        values = pd.to_numeric(df[target], errors="coerce").dropna().unique()
        offending = sorted(float(v) for v in values if float(v) not in (0.0, 1.0))
        if offending:
            raise ValueError(
                f"{name}: {target!r} is a binary target but holds values other than 0 and 1 "
                f"(e.g. {offending[:5]}). The usual cause is that `chronicle-preprocess` was not given "
                f"--binary-label-cols {target}, so the column was standardised."
            )


def _check_survival_rows(df: pd.DataFrame, target: str, recurrent: bool):
    """Chronicle's ChronicleDataset._validate on every labelled row.

    `<target>_TIME` is the time remaining, so time + <target>_TIME is constant per patient.
    A recurrent target (TTNT) may restart with a new end point once the previous one is reached.
    NaN rows (TTNT before the first therapy) carry no label and are not checked.
    """
    pid = df[PATIENT_ID_COL].to_numpy()
    times = df[TIME_COL].to_numpy(dtype=np.float32)
    lt = df[f"{target}_TIME"].to_numpy(dtype=np.float32)
    ev = df[f"{target}_EVENT"].to_numpy(dtype=np.float32)
    labelled = ~(np.isnan(lt) | np.isnan(ev))
    pid, times, lt = pid[labelled], times[labelled], lt[labelled]
    if lt.size == 0:
        return
    if lt.min() < 0:
        bad = pid[np.argmax(lt < 0)]
        raise ValueError(f"Patient {bad}: {target}_TIME has negative values.")

    total = times + lt
    same = pid[1:] == pid[:-1]
    ok = np.isclose(total[1:], total[:-1], rtol=1e-5, atol=1e-4)
    if recurrent:
        ok |= (times[1:] > total[:-1]) | np.isclose(times[1:], total[:-1], rtol=1e-5, atol=1e-4)
    bad = same & ~ok
    if bad.any():
        i = int(np.argmax(bad)) + 1
        raise ValueError(
            f"Patient {pid[i]}: time_since_initial_diagnosis + {target}_TIME is not constant "
            f"({total[i - 1]:.1f} -> {total[i]:.1f}). {target}_TIME must be the remaining time from each "
            f"observation, not an absolute date. If {target} restarts per episode (like TTNT per therapy "
            f"line), list it in --recurrent_survival_targets."
        )


def encode_split(
    df: pd.DataFrame,
    layout: Dict,
    target: str,
    task: str,
    *,
    recurrent: bool = False,
    vocab: Optional[Sequence[str]] = None,
    name: str = "split",
) -> List[Dict]:
    """One record per patient: days, per-day labels, sparse dynamic entries, static vector.

    `vocab` decides which rows make a day (default: the layout's, i.e. the training cohort's).
    An external cohort passes its own Chronicle vocabulary, so its days are the ones Chronicle
    would predict on; features the model has no column for then only contribute the day.
    """
    vocab = list(layout["vocab"] if vocab is None else vocab)
    for col in (PATIENT_ID_COL, KEY_COL, VALUE_COL, TIME_COL):
        if col not in df.columns:
            raise ValueError(f"{name}: missing required column {col!r}.")
    if not is_numeric_dtype(df[VALUE_COL]):
        raise TypeError(f"{name}: {VALUE_COL!r} must be numeric; categoricals are integer-coded upstream.")
    check_target(df, target, task, name)
    label_cols = label_columns(target, task)

    n_patients_raw = df[PATIENT_ID_COL].nunique()
    df = df[df[KEY_COL].astype(str).isin(set(vocab))].copy()
    df[KEY_COL] = df[KEY_COL].astype(str)
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="raise")
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="raise")
    if (df[TIME_COL] < 0).any():
        raise ValueError(f"{name}: negative time_since_initial_diagnosis.")
    # Stable sort keeps the file order within a day, which decides the day's label row.
    df = df.sort_values([PATIENT_ID_COL, TIME_COL], kind="stable").reset_index(drop=True)
    if df[PATIENT_ID_COL].nunique() < n_patients_raw:
        print(f"⚠️ {name}: {n_patients_raw - df[PATIENT_ID_COL].nunique()} patients have no known feature and are dropped.")
    if task == SURVIVAL:
        _check_survival_rows(df, target, recurrent)

    n_nan = int(df[VALUE_COL].isna().sum())
    if n_nan:
        print(f"⚠️ {name}: {n_nan} rows with NaN feature_value are treated as not observed.")

    first = df.drop_duplicates([PATIENT_ID_COL, TIME_COL], keep="first").reset_index(drop=True)
    first["_day_pos"] = first.groupby(PATIENT_ID_COL, sort=False).cumcount()
    pid_codes, pids = pd.factorize(first[PATIENT_ID_COL], sort=False)
    code_of = pd.Index(pids)

    obs = df[df[VALUE_COL].notna()]
    dyn = obs[obs[KEY_COL].isin(set(layout["value_col"]))]
    dyn = dyn.drop_duplicates([PATIENT_ID_COL, TIME_COL, KEY_COL], keep="last")
    dyn = dyn.merge(first[[PATIENT_ID_COL, TIME_COL, "_day_pos"]], on=[PATIENT_ID_COL, TIME_COL], how="left")

    presence_only = dyn[KEY_COL].isin(set(layout["presence_only"])).to_numpy()
    not_one = presence_only & (dyn[VALUE_COL].to_numpy() != 1.0)
    if not_one.any():
        feats = sorted(dyn.loc[not_one, KEY_COL].unique())
        print(f"⚠️ {name}: {int(not_one.sum())} values != 1 of presence-only features {feats} are read as present.")

    value_entries = pd.DataFrame({
        "code": code_of.get_indexer(dyn[PATIENT_ID_COL]),
        "row": dyn["_day_pos"].to_numpy(),
        "col": dyn[KEY_COL].map(layout["value_col"]).to_numpy(),
        "val": np.where(presence_only, 1.0, dyn[VALUE_COL].to_numpy()),
    })
    masked = dyn[~presence_only]
    mask_entries = pd.DataFrame({
        "code": code_of.get_indexer(masked[PATIENT_ID_COL]),
        "row": masked["_day_pos"].to_numpy(),
        "col": masked[KEY_COL].map(layout["mask_col"]).to_numpy(),
        "val": 1.0,
    })
    entries = pd.concat([value_entries, mask_entries], ignore_index=True)
    entries = entries.sort_values("code", kind="stable")
    e_code = entries["code"].to_numpy()
    e_bounds = np.searchsorted(e_code, np.arange(len(pids) + 1))
    e_row = np.array(entries["row"].to_numpy(dtype=np.int64), copy=True)
    e_col = np.array(entries["col"].to_numpy(dtype=np.int64), copy=True)
    e_val = np.array(entries["val"].to_numpy(dtype=np.float32), copy=True)

    static_vals = np.zeros((len(pids), layout["static_dim"]), dtype=np.float32)
    st = obs[obs[KEY_COL].isin(set(layout["static"]))].drop_duplicates([PATIENT_ID_COL, KEY_COL], keep="first")
    st_codes = code_of.get_indexer(st[PATIENT_ID_COL])
    st_slot = st[KEY_COL].map({f: i for i, f in enumerate(layout["static"])}).to_numpy()
    static_vals[st_codes, 2 * st_slot] = st[VALUE_COL].to_numpy(dtype=np.float32)
    static_vals[st_codes, 2 * st_slot + 1] = 1.0

    d_bounds = np.searchsorted(pid_codes, np.arange(len(pids) + 1))
    days_all = np.array(first[TIME_COL].to_numpy(dtype=np.float32), copy=True)
    labels_all = np.array(first[label_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32), copy=True)

    records: List[Dict] = []
    seen_ids: Dict[int, object] = {}
    for i, pid in enumerate(pids):
        pid_int = patient_id_to_int(pid)
        if seen_ids.setdefault(pid_int, pid) != pid:
            raise ValueError(f"patient_id {pid!r} and {seen_ids[pid_int]!r} both reduce to {pid_int}.")
        d0, d1 = d_bounds[i], d_bounds[i + 1]
        e0, e1 = e_bounds[i], e_bounds[i + 1]
        records.append({
            "pid": pid_int,
            "days": days_all[d0:d1],
            "labels": labels_all[d0:d1],
            "rows": e_row[e0:e1],
            "cols": e_col[e0:e1],
            "vals": e_val[e0:e1],
            "static": static_vals[i],
        })
    return records


def describe(name: str, ds: "ChronicleOSDataset"):
    n_days = np.array([len(r["days"]) for r in ds.records])
    labelled = np.array([int((~np.isnan(r["labels"]).any(axis=1)).sum()) for r in ds.records])
    print(
        f"   {name}: {len(ds)} patients, days/patient median {np.median(n_days):.0f}, max {n_days.max()}, "
        f"labelled days {labelled.sum()} of {n_days.sum()}"
    )


def load_chronicle_cohort(
    data_path: str,
    schema_csv: str,
    target: str,
    task: str,
    *,
    recurrent: bool = False,
    static_features: Sequence[str] = DEFAULT_STATIC_FEATURES,
    input_features: Optional[Sequence[str]] = None,
):
    """The three splits of `chronicle-preprocess` and the input layout fitted on them."""
    csvs = resolve_split_csvs(data_path)
    frames = {}
    for split, path in csvs.items():
        frames[split] = read_cohort_csv(path)
        print(f"📄 {split}: {path} ({frames[split][PATIENT_ID_COL].nunique()} patients, {len(frames[split])} rows)")

    layout = build_layout(schema_csv, frames, static_features, input_features)
    print(
        f"Vocabulary: {len(layout['vocab'])} features | static: {layout['static']} | "
        f"dynamic columns: {layout['n_dynamic']} ({len(layout['presence_only'])} presence-only) "
        f"+ {N_TIME_FEATURES} time"
    )

    datasets = {
        split: ChronicleOSDataset(encode_split(df, layout, target, task, recurrent=recurrent, name=split))
        for split, df in frames.items()
    }
    for split, ds in datasets.items():
        describe(split, ds)
    return datasets, layout


class ChronicleOSDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


def collate_os(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad per-day arrays; keep the dynamic features sparse until `densify`."""
    lengths = torch.tensor([len(r["days"]) for r in batch], dtype=torch.long)
    B, T = len(batch), int(lengths.max())
    C = batch[0]["labels"].shape[1]

    days = torch.zeros(B, T)
    labels = torch.full((B, T, C), float("nan"))
    for i, r in enumerate(batch):
        n = len(r["days"])
        days[i, :n] = torch.from_numpy(r["days"])
        labels[i, :n] = torch.from_numpy(r["labels"])

    return {
        "pids": torch.tensor([r["pid"] for r in batch], dtype=torch.long),
        "lengths": lengths,
        "days": days,
        "labels": labels,
        "static": torch.from_numpy(np.stack([r["static"] for r in batch])),
        "b_idx": torch.cat([torch.full((len(r["rows"]),), i, dtype=torch.long) for i, r in enumerate(batch)]),
        "t_idx": torch.from_numpy(np.concatenate([r["rows"] for r in batch])),
        "c_idx": torch.from_numpy(np.concatenate([r["cols"] for r in batch])),
        "vals": torch.from_numpy(np.concatenate([r["vals"] for r in batch])),
    }


def densify(batch: Dict[str, torch.Tensor], static_dim: int, n_dynamic: int, device, lengths=None) -> torch.Tensor:
    """Build CHRONOS's `current_covariates` [B, T, static | dynamic | time].

    The static block is only filled at t=0, which is the only step CHRONOSModel.encode reads
    it from. `lengths` cuts patients short (survival training uses the history up to the
    sampled day).
    """
    lengths = batch["lengths"] if lengths is None else lengths
    B, T = len(lengths), int(lengths.max())
    cov = torch.zeros(B, T, static_dim + n_dynamic + N_TIME_FEATURES, device=device)
    cov[:, 0, :static_dim] = batch["static"].to(device)

    b, t, c, v = batch["b_idx"], batch["t_idx"], batch["c_idx"], batch["vals"]
    keep = t < lengths[b]
    cov[b[keep].to(device), t[keep].to(device), (static_dim + c[keep]).to(device)] = v[keep].to(device)

    valid = (torch.arange(T)[None, :] < lengths[:, None]).float()
    days = batch["days"][:, :T]
    delta = torch.zeros_like(days)
    delta[:, 1:] = days[:, 1:] - days[:, :-1]
    cov[:, :, -2] = (days / DAYS_PER_YEAR * valid).to(device)
    cov[:, :, -1] = (delta / DAYS_PER_YEAR * valid).to(device)
    return cov
