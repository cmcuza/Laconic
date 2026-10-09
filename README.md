# Laconic

Laconic jointly optimizes lossy time-series compression with a downstream
analytics task (classification, clustering, regression or forecasting). Given a
compressor with a mixed discrete/continuous parameter space, an optimizer searches for the
parameters that maximize a trade-off optimization objective between compression ratioa and accuracy of the downstream analytics.
Each run searches per cross-validation fold, then a consensus step selects one
pipeline across folds and scores it on the held-out test set.

## Setup

```bash
git lfs install && git lfs pull                  # datasets under data/ are stored in LFS
conda env create -f environment.yaml && conda activate laconic
bash compression/lib/batch_serfxor/build.sh      # only for the serfxor/adaedge backends
```

The `sz` backend needs SZ3's C library (`export SZ3_LIB_PATH=/path/to/libSZ3c.so`),
and `mixpiece` needs a Java runtime.

## Running

One experiment (all datasets of a group, one model):

```bash
python run_experiments.py --analytics classification --compression laconic \
    --optimizer genetic --model proximity_forest --dataset ucr_small \
    --random_state 32 --alpha 0.75 --budget 100
```

- `--analytics`: `classification`, `clustering`, `regression`, `forecasting`
- `--compression`: `laconic`, `sz`, `mixpiece`, `serfxor`, `adaedge`
- `--optimizer`: `genetic` (TerseTS), `bosmp` (single-parameter baselines), `adaedge`
- models, dataset groups, and search spaces are YAML files under `cfg/`

A whole sweep from a manifest:

```bash
python run_suite.py cfg/suites/main_comparison.yaml --dry-run   # preview
python run_suite.py cfg/suites/main_comparison.yaml --no-visualize
```

Results go to `results/<task>/<compressor>/<optimizer>/<model>/budget_<N>/alpha_<a>/<dataset>.csv`,
with the consensus-selected pipeline under `results/<task>/<compressor>/rank_agg_b1_mean_fitness/`.
Optimizer traces go to `.logs/`, and runs are tracked in MLflow
(`mlflow ui --backend-store-uri sqlite:///mlflow.db`; disable with `--no_mlflow`).

## Tests

```bash
python -m pytest tests
python scripts/verify_optimizer_budgets.py
```
