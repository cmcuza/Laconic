"""The one model every fold's TEST columns are scored with."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

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
    """Fit (or load from cache) the single model that scores every fold's test columns."""
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
