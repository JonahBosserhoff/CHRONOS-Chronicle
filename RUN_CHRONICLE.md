# CHRONOS on Chronicle cohorts

The CHRONOS backbone as a single-target benchmark for Chronicle: one model per target, trained
and scored like Chronicle (same patients, features, scaling, labels, losses and metrics), plus
external validation on another cohort. Standalone: nothing is imported from Chronicle.

```text
src/chronos/train_os.py          train on the Chronicle splits, evaluate on test
src/chronos/validate_os.py       external validation of a trained model
src/chronos/dataset_os.py        Chronicle long format -> one timestep per observed day
src/chronos/chronicle_compat.py  Chronicle code copied verbatim (losses, metrics, scalers)
src/chronos/model.py             + CHRONOSSingleTaskModel (CHRONOSModel itself is unchanged)
configs/validate_external.yaml   options of an external validation (template)
slurm/{train,validate}.sbatch    training and external validation as SLURM jobs (section 4)
```

## Install

```bash
pip install numpy pandas torch scikit-learn lifelines pyarrow joblib pyyaml
```

`pyyaml` is only needed for `validate_os.py --config`. Chronicle itself does not need to be
installed.

Run every command from the repository root.

## 1. Data: Chronicle's cohort directory, as it is

CHRONOS neither splits nor scales anything for training. `--data` is the cohort directory that
Chronicle trains on (Chronicle's `data_path`), as Chronicle's preprocessing left it:

```text
<cohort_dir>/
  <prefix>_train.csv  <prefix>_val.csv  <prefix>_test.csv   Chronicle's splits, already scaled
  scalers/                                                  Chronicle's fitted scalers
```

`train_os.py` reads these files unchanged. It therefore trains, selects the epoch and tests on
exactly Chronicle's patients and values. The scalers are only read: to report regression
predictions in original units, as Chronicle's evaluation does, and to record them in the
checkpoint for the external validation. Nothing is written into the cohort directory. A 0/1
target must have been left unscaled by Chronicle's preprocessing (`--binary-label-cols`); a scaled
one is refused.

## 2. Train

sbatch --export=ALL,DATA=$DATA,GLOBAL_FORMAT=$SCHEMA,TARGET=OS slurm/train.sbatch

```bash
python src/chronos/train_os.py \
  --data <cohort_dir> \
  --global_format <schema.csv> \
  --target OS \
  --out_dir results_os/OS
```

`--target` is any target of the cohort, one per run. The type follows Chronicle's rule
(`Config.finalize_losses`):

| target | type | loss (as Chronicle) | metric (as Chronicle) | label columns |
|---|---|---|---|---|
| listed in `--binary_label_cols` | binary | `bce_logits` on every labelled day | ROC-AUC | `<target>` |
| starts with `label` | regression | `mse` (`--regression_loss`) on every labelled day | Pearson r (scaled values) | `<target>` |
| anything else (`OS`, `TTNT`) | survival | `cox_loss` on one random labelled day per patient, batch = risk set | C-index, mean ± std over `--c_index_samples` draws | `<target>_TIME`, `<target>_EVENT` |

Survival targets in `--recurrent_survival_targets` (default `TTNT`) may restart per episode;
every other survival target must satisfy `day + <target>_TIME = const` per patient, otherwise the
run is refused. Days without a label (NaN, e.g. TTNT before the first therapy) drop out of loss
and metric, as in Chronicle.

### Options

| option | default | meaning |
|---|---|---|
| `--data` | (required) | Chronicle's cohort directory: `<prefix>_{train,val,test}.csv` and `scalers/` (section 1) |
| `--global_format` | (required) | Chronicle feature schema |
| `--target` | `OS` | the one target of this run |
| `--out_dir` | `results_os` | output directory |
| `--scaler_dir` | `<data>/scalers` | scalers of the cohort: inverse transform of regression predictions, recorded for external validation |
| `--binary_label_cols` | `label_supportive_care_transfusion_7_days,label_febrile_neutropenia_21d` | as `binary_label_cols` in Chronicle's `configs/train.yaml` |
| `--recurrent_survival_targets` | `TTNT` | as `recurrent_survival_targets` in `train.yaml` |
| `--regression_loss` | `mse` | `mse`, `l1` or `huber` (Chronicle's `regression_loss`) |
| `--loss_mode` | `static` | `static` = the plain task loss. `uncertainty` = Chronicle's learned uncertainty weighting (`MultiTaskLoss`), for one task |
| `--logvar_prior`, `--logvar_bounds` | `1.0`, `-8 8` | only for `--loss_mode uncertainty`, as in `train.yaml` |
| `--survival_sampling` | `random` | `random` or `last` labelled day per patient (Chronicle's `survival_sampling`) |
| `--static_features` | `age_at_diagnosis,gender_female` | features that go to the static MLP instead of the GRU |
| `--input_features` | empty = all | restrict the vocabulary, like Chronicle's `input_features` |
| `--epochs` | `10` | CHRONOS default |
| `--batch_size` | `1024` | CHRONOS default. For survival also the Cox risk set. Must not exceed the train split, because incomplete batches are dropped |
| `--eval_batch_size` | `0` = `batch_size` | val / test / external batch size. Lower it if long patients run out of memory |
| `--latent_dim`, `--rnn_layers`, `--dropout` | `64`, `1`, `0.1` | CHRONOS defaults |
| `--lr`, `--weight_decay` | `1e-3`, `1e-4` | CHRONOS defaults (AdamW, gradient clipping 1.0) |
| `--num_workers`, `--seed` | `0`, `42` | CHRONOS defaults |
| `--c_index_samples` | `10` | as in `train.yaml` |
| `--selection_metric` | `auto` | `auto` = the target's metric on val (C-index / Pearson r / AUC), `val_loss` = validation loss |
| `--early_stopping_patience` | `0` = off | the best val epoch is evaluated either way |

### Outputs (`--out_dir`)

| file | content |
|---|---|
| `metrics_test.json` | Chronicle's format: `split_size, num_batches, loss, c_index, c_index_std, correlation, auc`. A metric that cannot be computed is omitted |
| `predictions_test/<target>.parquet` | one row per (patient, day). Survival: `patient_id, day, prediction, time, event` (higher prediction = higher hazard). Regression / binary: `patient_id, day, prediction, ground_truth`, unlabelled days dropped. Regression in original units, binary as logit |
| `checkpoints/best_model.pt`, `last_model.pt` | weights, input layout, target, training arguments, path and sha256 of the scalers |
| `history.csv`, `training_summary.json` | per-epoch losses and val metric; best epoch and the evaluated checkpoint |
| `config.json`, `features.csv` | run configuration; one row per input column (feature, value / observed / present / time) |

The predictions join to Chronicle's (`predictions_test/<target>.parquet`) and to the test
timepoints of Chronicle's baselines on `(patient_id, day)`.

## 3. External validation

```bash
python src/chronos/validate_os.py \
  --checkpoint results_os/OS/checkpoints/best_model.pt \
  --data <site_b_dir_or_csv> \
  --out_dir results_os/OS_site_b
```

1. Loads the checkpoint: weights, input layout, target, loss and the scaler directory of the
   training cohort.
2. Loads those scalers and checks their sha256 against the checkpoint. Other scalers are refused
   unless `--allow_other_scalers` is given.
3. Scales the external cohort with them (the only scaling CHRONOS does), exactly as
   `chronicle-preprocess --pretrained-scaler-dir` does: continuous features and regression targets are transformed. Categoricals, survival times
   and 0/1 targets stay unchanged. Nothing is fitted on the external cohort.
4. Builds the daily sequences with the training layout and runs inference.

The external cohort is the **raw, unscaled** CSV, prepared like the training cohort before
preprocessing (outliers removed, labels created with the same rules), with the target's
column(s). By default every patient is evaluated. No split is needed.

The options can also come from a YAML file, keyed by the option names without `--`
(template: [`configs/validate_external.yaml`](configs/validate_external.yaml)):

```bash
python src/chronos/validate_os.py --config configs/validate_external.yaml
python src/chronos/validate_os.py --config configs/validate_external.yaml --seed 1   # command line overrides the file
```

In the YAML, `null` keeps an option's default, `~` and `$VARS` are expanded, and relative paths
resolve from the working directory (as on the command line). Unknown keys and values of the wrong
type are refused.

| option | default | meaning |
|---|---|---|
| `--config` | none | YAML file with any of the options below |
| `--checkpoint` | (required) | `best_model.pt` or `last_model.pt` of `train_os.py` |
| `--data` | (required) | raw cohort: the CSV, or its directory (`<dir>/<dir>.csv` or its single un-split CSV) |
| `--out_dir` | (required) | output directory |
| `--scaler_dir` | from the checkpoint | the training cohort's `scalers/` |
| `--allow_other_scalers` / `--no-allow_other_scalers` | off | accept scalers whose sha256 differs from the training ones |
| `--global_format` | from the checkpoint | feature schema |
| `--patient_ids` | all patients | CSV with a `patient_id` column, e.g. `<site_b>/splits/test_patients.csv` |
| `--eval_batch_size`, `--c_index_samples`, `--seed` | from the training run | |

Outputs: `metrics_external.json`, `predictions_external/<target>.parquet` (same formats as the
test outputs) and `validation_config.json`. The log lists the features of the external cohort
that the model has no input for and the model features the cohort lacks. The days are those
Chronicle would predict on: every day with an observation of a schema feature. A day whose only
observations are features unknown to the model still gets a prediction, from the history so far
and the time features.

## 4. On SLURM

Two job scripts in `slurm/`, submitted from the repository root. The SLURM log goes to
`logs/`, which must exist before submitting. Settings are passed as environment variables with
`--export=ALL,...`; further options of the underlying script go after the job script and are
passed on unchanged. Every script prints the host and commit, checks that `PYTHON` imports what
it needs, and warns when no GPU is visible.

```bash
mkdir -p logs

# train (1 GPU) on Chronicle's cohort directory, one job per target
sbatch --export=ALL,DATA=<cohort_dir>,GLOBAL_FORMAT=<schema.csv>,TARGET=OS slurm/train.sbatch --epochs 20

# external validation (1 GPU) on the raw external cohort, options from the YAML
cp configs/validate_external.yaml configs/OS_site_b.yaml   # then set checkpoint, data, out_dir
sbatch --export=ALL,CONFIG=configs/OS_site_b.yaml slurm/validate.sbatch
```

| script | variables (default) | runs |
|---|---|---|
| `train.sbatch` | `DATA`, `GLOBAL_FORMAT` (required), `TARGET` (OS), `OUT_DIR` (`results_os/$TARGET`), `BINARY_LABEL_COLS` (unset = `train_os.py`'s default), `OVERWRITE` (0), `PYTHON` | `src/chronos/train_os.py` |
| `validate.sbatch` | `CONFIG` (`configs/validate_external.yaml`), `PYTHON` | `src/chronos/validate_os.py --config $CONFIG` |

All scripts take `PYTHON` (default `python`) and `REPO` (default: the directory `sbatch` was called
from). `train.sbatch` refuses an `OUT_DIR` that already holds `checkpoints/best_model.pt`,
unless `OVERWRITE=1`.

The `#SBATCH` resources (partition `gpu_nodes` as in Chronicle's job scripts, 1 GPU, 6 CPUs per
GPU, 64 GB, 24 h for training, 4 h for validation) are defaults; override
them on the command line, e.g. `sbatch --time=48:00:00 --partition=<p> ...`.

## 5. Design changes compared with the original CHRONOS

Unchanged files: `dataset.py`, `train.py`, `run_all_trials.py`, `experiment_suite.py`,
`baseline_compare.py`, `cox_baseline.py`, `run_all_cox.py`. In `model.py`, `CHRONOSModel`, the
MINE network and the contrastive loss are unchanged. `CHRONOSSingleTaskModel` is added as a
subclass. The causal pipeline runs and reproduces as before.

| | original CHRONOS (`train.py`) | Chronicle benchmark (`train_os.py`) |
|---|---|---|
| question | causal effect A → B of one disease edge | prediction of one Chronicle target per (patient, day) |
| input data | event CSV: `CODICE_FISCALE_ASSISTITO, DATA_PRESCRIZIONE` (calendar date), `ICD9_CM` mapped to 16 macro categories, ATC as `DRUG:`, `ANNO_NASCITA, SESSO, FRAILTY_SUM, DATA_REVOCA` | Chronicle's own split CSVs, read unchanged (long format): `patient_id, feature_name, feature_value, time_since_initial_diagnosis` plus label columns. Values scaled with train-fitted scalers |
| cohort | target-trial emulation: first A after ≥180 days of history, contact in the last 60 days, washouts (A 180 d, B 30 d), follow-up not cut by `DATA_REVOCA`, 1:3 controls with the same t0, ≥50 treated | no emulation: every patient of Chronicle's train / val / test split |
| time axis | visit days in [t0−180, t0], at most 60 steps, plus one appended follow-up step | every observed day since diagnosis (a day with an observation of a vocabulary feature), no cap |
| static MLP input (`static_dim`) | 3: birth year min-max normalised, `FRAILTY_SUM/10`, sex (M 1 / F 0 / unknown 0.5) | 4: `[value, observed]` of `age_at_diagnosis` and `gender_female` (their first observed value, scaled as in Chronicle) |
| GRU input per step | count of B in the baseline / 20 and visits in the last 30 days / 10 (both computed at t0, identical at every step), plus A of the previous step | per vocabulary feature the last value of the day and an observed mask (one presence column for categoricals that only occur as 1), plus years since diagnosis and years since the previous observed day. The treatment input is empty |
| missing values | n/a | 0 with mask 0, nothing imputed; earlier values are carried by the GRU state |
| encoder | static MLP + GRU + fusion + LayerNorm | unchanged (same layers and sizes) |
| heads | `mu0`, `mu1`, propensity | removed. One head with the structure of `mu0` (Linear–GELU–Dropout–Linear) and one output per step: log-risk / value / logit |
| loss | BCE of the factual outcome + `lambda_prop`·propensity BCE + `alpha`·effect-guided contrastive + `beta`·MI(z, T) − `gamma`·MI(z, Y) (MINE) | Chronicle's loss of the target only (table in section 2); optionally Chronicle's uncertainty weighting |
| sampling | `--weighted_sampling` balances treated / controls | plain shuffle, `drop_last=True` (as Chronicle) |
| estimation | 5-fold cross-fitting, AIPW, ATE ± SE, p, BH q-value, placebo, ESS | train on train, select the best val epoch by Chronicle's metric, evaluate once on test |
| output | one CSV row per edge | `metrics_<split>.json` and `predictions_<split>/<target>.parquet` in Chronicle's format |
| external validation | none | `validate_os.py` |

The CHRONOS defaults are kept: `epochs 10`, `batch_size 1024`, `latent_dim 64`, one GRU layer,
`dropout 0.1`, AdamW `lr 1e-3`, `weight_decay 1e-4`, gradient clipping 1.0, `seed 42`. These
arguments have no counterpart without an exposure and are gone: `--treatment_codes`,
`--outcome_codes`, `--t0_codes`, all windows / washouts, `--control_ratio`, `--min_treated`,
`--n_folds`, `--alpha`, `--beta`, `--gamma`, `--lambda_prop`, `--perc`, `--margin`, `--max_pairs`,
`--use_mine`, `--mine_interval`, `--critic_steps`, `--weighted_sampling`, `--ps_eps`,
`--add_qvalues`. `--max_seq_len` (60) is not used, so the whole history is seen.

## 6. What is taken from Chronicle, and what still differs

Copied verbatim into `chronicle_compat.py`, with the source in the module docstring:
- `cox_loss`, the loss registry (`mse`, `l1`, `huber`, `bce_logits`) and the target-type rule.
- `MultiTaskLoss`.
- `sample_one_token_per_patient` (draws only labelled days).
- C-index, Pearson r and AUC with their skip rules.
- `TimeSeriesStandardizer` load / transform / inverse transform.

Also taken over:
- the label of a day is the label of its first row (what Chronicle's CLS token clones);
- the `_TIME` invariants and the binary 0/1 check;
- patient ids reduced to integers as in Chronicle.

Differences to Chronicle that remain on purpose (model and training, not data or metric):
- **Sequence model:** a GRU over days, not a causal Transformer over single observations. Several
  values of one feature on the same day become the last one.
- **No augmentation:** Chronicle drops features / observations during training
  (`mask_augmentation`). CHRONOS has no augmentation.
- **No sequence cap:** Chronicle trains on at most 4000 observations (random subsample) and
  evaluates the most recent 30000. CHRONOS always sees the whole history. For a patient with more
  than 30000 observations, Chronicle has no prediction on the earliest days. A join on
  `(patient_id, day)` drops those days.
- **Training setup:** Chronicle trains 10 targets jointly with uncertainty weighting, Adam
  (`weight_decay 1e-6`), cosine annealing, AMP, 100 epochs with patience 20, batch 128. CHRONOS
  trains one target with CHRONOS' optimiser settings.
- **Survival training shortcut:** each patient is cut at its drawn day before the forward pass.
  The GRU is causal, so the loss equals a full-sequence pass (checked numerically).
- **Unknown features at an external site:** Chronicle still embeds them by `global_id` (with
  embeddings never trained on them if the training cohort did not have them). CHRONOS has no
  column for them; they only create the day.
- **Not supported:** `exclude_features`.
- **Inverse transform:** regression predictions are inverse-transformed with Yeo-Johnson as in
  Chronicle. A prediction outside the invertible range becomes NaN in original units, in both
  repositories alike.
