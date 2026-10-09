"""The one model every fold's TEST columns are scored with.

Why this exists
---------------
The CV folds are there to *select* a pipeline, not to produce the number that
gets reported. Fitting a model per fold and then scoring the test split with
that fold's model means two methods whose ``best_val`` landed on different
folds are compared through two different models - the test *data* is identical,
but the mapping from data to prediction is not. Measured across the existing
results, the baseline metric moved across folds within a single
``(dataset, random_state)`` in 71 of 73 cells, by a median of 4-8% and by up to
34% (classification ``Wine``) and 49% (clustering ``Coffee``) - and those two
datasets carry nearly every anomaly flag their tasks produce.

So every fold's test columns come from ONE model per
``(dataset, random_state, model)``, fit on **all** the training data - the union
of every fold's train and validation split, which is what a deployed model
would be trained on. Folds still differ: each searches on its own train/val
split, so ``val_*``, ``best_params`` and the optimizer traces stay per fold
exactly as before. Only the test side is shared, which is what makes methods
comparable.

``experiments/forecasting_runner.py`` applies the same rule with the one
difference its data demands: the splits are temporal, so "all the training
data" is the final fold's train+val prefix rather than a union of shuffled
folds, and the test window is that fold's own. See its own comment.

Cache key
---------
``fold=DEPLOYMENT_FOLD`` (0) is a free sentinel: fold 0 already means "the mean
row" in the results CSVs and is never a real CV fold, so it cannot collide with
a per-fold cache entry.

**A cache caveat that has bitten once already**: ``fit_cached`` keys on dataset,
kwargs hash, train-indices hash, splitter and library versions - *not* on the
estimator's own code. Editing what a model class does internally (as the
``StandardScaler`` fix in ``models/regression.py`` did) does not invalidate the
cache, so the stale model is silently reloaded and the fix appears to do
nothing. Clear ``.logs/_model_cache/<dataset>/<model>/`` when changing a model.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

# Sentinel fold for the deployment fit. 0 is never a real CV fold.
DEPLOYMENT_FOLD = 0


def fit_deployment_model(
    build_model: Callable[[str, Dict[str, Any]], Any],
    model_name: str,
    model_kwargs: Dict[str, Any],
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    *,
    dataset: str,
    random_state: int,
    splitter_name: str,
    logs_dir: str,
):
    """Fit (or load from cache) the single model that scores every fold's test
    columns. ``y_train=None`` dispatches to the unsupervised ``fit(X)`` path,
    exactly as ``fit_cached`` already does per fold."""
    model = build_model(model_name, model_kwargs)
    model.fit_cached(
        X_train,
        *(() if y_train is None else (y_train,)),
        dataset=dataset,
        random_state=random_state,
        fold=DEPLOYMENT_FOLD,
        splitter_name=splitter_name,
        train_indices=np.arange(len(X_train)),
        out_dir=logs_dir,
    )
    return model
