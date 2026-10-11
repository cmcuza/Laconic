# Laconic

Laconic jointly optimizes lossy time-series compression with a downstream
analytics task (classification, clustering, regression, forecasting or anomaly
detection). Given a compressor with a mixed discrete/continuous parameter space, an optimizer searches for the
parameters that maximize a trade-off optimization objective between compression ratioa and accuracy of the downstream analytics.
For the main tasks, each run searches per cross-validation fold, then a consensus step
selects one pipeline across folds and scores it on the held-out test set. Anomaly
detection searches one pipeline per series (see below).

## Setup

```bash
git lfs install && git lfs pull                  # datasets under data/ are stored in LFS
conda env create -f environment.yaml && conda activate laconic
bash compression/lib/batch_serfxor/build.sh      # only for the serfxor/adaedge backends
```

The `sz` backend needs SZ3's C library (`export SZ3_LIB_PATH=/path/to/libSZ3c.so`),
and `mixpiece` needs a Java runtime.

## Running

### Main tasks

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

### Anomaly detection

Anomaly detection runs the TSB-AD detector Sub_PCA on the 32 TSB-AD-U series in
`data/anomaly/TSB-AD-U/` (Git LFS).

One experiment:

```bash
python run_experiments.py --analytics anomaly --compression laconic \
    --optimizer genetic --model sub_pca --dataset multievent32 \
    --random_state 32 --alpha 0.75 --budget 100
```

All five methods at alpha 0.75 and 0.95:

```bash
python run_suite.py cfg/suites/anomaly_detection.yaml
```

Results go to `results/anomaly/<compressor>/<optimizer>/Sub_PCA/budget_<N>/alpha_<a>/multievent32.csv`,
one row per series plus their mean (fold 0). The objective's quality term is VUS-PR on
the validation range (`q_measure` in `cfg/analytics/anomaly/sub_pca.yaml`).

The corpus list and the split are committed under `cfg/datasets/anomaly/`;
`scripts/anomaly/build_corpus.py` and `scripts/anomaly/build_split.py` rebuild them.
The paper's tables and figures come from, in order: `scripts/anomaly/replay_exemplar.py`
(curves of the example), `analysis/anomaly_paper_tables.py`,
`analysis/anomaly_tradeoff_figure.py` and `analysis/anomaly_example_figure.py`.

## Tests

```bash
python -m pytest tests
python scripts/verify_optimizer_budgets.py
```
