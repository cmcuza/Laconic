from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict, Iterator, Iterable
import os
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import convert_tsf_to_dataframe, load_from_tsfile_to_dataframe
from aeon.datasets import load_from_tsv_file


@dataclass
class UCRLoader:
    def __init__(self, root: str = "data/classification/UCRArchive_2018"):
        self.ucr_path = root

    def load_train(self, dataset_name):
        dataset_dir = os.path.join(self.ucr_path, dataset_name)
        train_file = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.tsv")
        return load_from_tsv_file(train_file)

    def load_test(self, dataset_name):
        dataset_dir = os.path.join(self.ucr_path, dataset_name)
        train_file = os.path.join(dataset_dir, f"{dataset_name}_TEST.tsv")
        return load_from_tsv_file(train_file)


@dataclass
class MonashLoader:
    def __init__(self, root: str = "data/forecasting/Monash"):
        root = Path(root)
        files = list(root.rglob("*.tsf"))
        if not files:
            raise FileNotFoundError(f"No .tsf files found under {root}")
        self.data_map: Dict[str, Path] = {f.name.split("_")[0]: f for f in files}
        for f in files:
            alias = f.name.removesuffix(".tsf").removesuffix("_dataset")
            self.data_map.setdefault(alias, f)
        self.values: np.ndarray | None = None
        self.frc_h: int | None = None
        self.freq: str | None = None
        self.has_missing: bool | None = None
        self.start_date: pd.Timestamp | None = None

    def _to_array(self, v) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        if isinstance(v, (list, pd.Series, tuple)):
            return np.asarray(v, dtype=float)
        if isinstance(v, str) and v.strip().startswith("[") and v.strip().endswith("]"):
            import ast
            return np.asarray(ast.literal_eval(v), dtype=float)
        raise ValueError(f"Unsupported series_value cell type: {type(v)}")

    def pick_larges_ts(self, df: pd.DataFrame) -> Tuple[pd.Timestamp, np.ndarray]:
        required = {"series_name", "start_timestamp", "series_value"}
        miss = required - set(df.columns)
        if miss:
            if list(miss)[0] == "start_timestamp":
                df["start_timestamp"] = pd.Timestamp("1991-11-09")
            else:
                raise ValueError(f"Missing required columns: {miss}")

        lengths = df["series_value"].apply(lambda x: len(self._to_array(x))).astype(int)
        max_len = int(lengths.max())

        cand = df.loc[lengths == max_len, ["series_name", "start_timestamp"]].copy()
        cand["__row"] = cand.index
        cand["__start"] = pd.to_datetime(cand["start_timestamp"], errors="coerce").fillna(pd.Timestamp.min)
        cand = cand.sort_values(by=["series_name", "__start", "__row"], kind="mergesort")
        idx = int(cand["__row"].iloc[0])

        values = self._to_array(df.at[idx, "series_value"])
        start  = pd.to_datetime(df.at[idx, "start_timestamp"], errors="coerce")
        return start, values.astype(float, copy=False)

    def _simple_impute(self, y: np.ndarray) -> np.ndarray:
        if not np.isnan(y).any():
            return y
        s = pd.Series(y)
        s = s.ffill().bfill()
        if s.isna().any():
            s = s.fillna(0.0)
        return s.to_numpy(dtype=float)

    def _ensure_loaded(self, dataset_name: str):
        if self.values is not None:
            return
        if dataset_name not in self.data_map:
            raise KeyError(f"Unknown dataset '{dataset_name}'. Known: {sorted(self.data_map.keys())[:10]} ...")

        df, self.freq, self.frc_h, self.has_missing, _ = convert_tsf_to_dataframe(self.data_map[dataset_name])
        self.start_date, vals = self.pick_larges_ts(df)

        if self.has_missing or np.isnan(vals).any():
            print(f"[MonashLoader] Dataset {dataset_name} contains NaNs — applying simple ffill/bfill.")
            vals = self._simple_impute(vals)

        self.values = vals

    def load_train(self, dataset_name: str) -> np.ndarray:
        self._ensure_loaded(dataset_name)
        return self.train

    def load_test(self, dataset_name: str) -> np.ndarray:
        self._ensure_loaded(dataset_name)
        return self.test

    def forward_chain(
        self,
        dataset_name: Optional[str] = None,
        train_fracs: Iterable[float] = (0.30, 0.40, 0.50, 0.60, 0.70),
        val_frac: float = 0.10,
        test_frac: float = 0.20,
        drop_short: bool = True,
        copy_arrays: bool = True,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield forward-chaining train/val/test splits, one per train fraction."""
        if self.values is None:
            if dataset_name is None:
                raise RuntimeError("Call forward_chain with dataset_name or load a dataset first.")
            self._ensure_loaded(dataset_name)

        y = self.values
        N = int(len(y))
        if N <= 0:
            return

        if val_frac < 0 or test_frac < 0:
            raise ValueError("val_frac and test_frac must be non-negative.")
        if max(train_fracs) + val_frac + test_frac > 1.0000001:
            raise ValueError("Train/val/test fractions exceed 1.0; adjust inputs.")

        val_len  = int(round(val_frac  * N))
        test_len = int(round(test_frac * N))

        for t in train_fracs:
            tr_end   = int(round(t * N))
            val_end  = min(N, tr_end + val_len)
            test_end = min(N, val_end + test_len)

            if tr_end <= 0 or val_end <= tr_end or test_end <= val_end:
                continue

            tr = y[:tr_end]
            va = y[tr_end:val_end]
            te = y[val_end:test_end]

            if self.frc_h is not None:
                ok = (len(tr) >= 2 * self.frc_h) and (len(va) >= self.frc_h) and (len(te) >= self.frc_h)
                if drop_short and not ok:
                    continue

            if copy_arrays:
                yield tr.copy(), va.copy(), te.copy()
            else:
                yield tr, va, te

    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """Returns (train, test, frc_h) for convenience."""
        self._ensure_loaded(dataset_name)
        return self.train, self.test, (self.frc_h or 1)

@dataclass
class MonashUCRLoader:
    def __init__(self, root: str = "data/regression/Monash-UCR"):
        self._path = root
        self.dimnesions = None

    def load_train(self, dataset_name):
        dataset_dir = os.path.join(self._path, dataset_name)
        train_file = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.ts")
        X_train, y_train = load_from_tsfile_to_dataframe(train_file)
        self.dimensions = X_train.columns
        ts_size = X_train.at[0, self.dimensions[0]].shape[0]
        X_train_in_full = np.empty((X_train.shape[0], ts_size, len(self.dimensions)), dtype=np.float64)

        for j, d in enumerate(self.dimensions):
            X_train_per_dim = np.empty((X_train.shape[0], ts_size), dtype=np.float64)
            for i in range(X_train.shape[0]):
                X_train_per_dim[i, :] = X_train.at[i, d].values.astype(np.float64)

            X_train_in_full[:, :, j] = X_train_per_dim

        return X_train_in_full, y_train

    def load_test(self, dataset_name):
        dataset_dir = os.path.join(self._path, dataset_name)
        train_file = os.path.join(dataset_dir, f"{dataset_name}_TEST.ts")
        X_test, y_test = load_from_tsfile_to_dataframe(train_file)
        self.dimensions = X_test.columns
        ts_size = X_test.at[0, self.dimensions[0]].shape[0]
        X_test_in_full = np.empty((X_test.shape[0], ts_size, len(self.dimensions)), dtype=np.float64)

        for j, d in enumerate(self.dimensions):
            X_test_per_dim = np.empty((X_test.shape[0], ts_size), dtype=np.float64)
            for i in range(X_test.shape[0]):
                X_test_per_dim[i, :] = X_test.at[i, d].values.astype(np.float64)

            X_test_in_full[:, :, j] = X_test_per_dim

        return X_test_in_full, y_test


def build_loaders(name: str, **kwargs) -> Any:
    key = name.strip().lower()

    if key in ("ucrloader", "ucr", "ucr_dataset"):
        return UCRLoader(**kwargs)

    if key in ("monash", "forecasting", "monashloader"):
        return MonashLoader(**kwargs)

    if key in ("monashucr", "regression", "monashucrloader"):
        return MonashUCRLoader(**kwargs)

    raise ValueError(f"Unknown loader '{name}'. Supported: UCRLoader, MonashLoader, MonashUCRLoader")


def build_splitter(split_cfg: Dict[str, Any]):
    typ = split_cfg["type"]
    kw  = {k: v for k, v in split_cfg.items() if k != "type"}
    if typ == "StratifiedShuffleSplit":
        from sklearn.model_selection import StratifiedShuffleSplit
        return StratifiedShuffleSplit(**kw)
    if typ == "ShuffleSplit":
        from sklearn.model_selection import ShuffleSplit
        return ShuffleSplit(**kw)
    if typ == "KFold":
        from sklearn.model_selection import KFold
        return KFold(**kw)
    raise ValueError(f"Unknown split type: {typ}")
