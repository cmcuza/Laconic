from __future__ import annotations

import numpy as np
from .base import Forecasting
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch import optim


from dataclasses import dataclass
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor


def _make_supervised(
    y: np.ndarray,
    prediction_len: int,
    sequence_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised samples with configurable input and target window sizes.
    X[i] contains `sequence_len` historical points and Y[i] contains the next
    `prediction_len` points. When `sequence_len` is omitted, use the same
    signature as XGBoostModel: input length == prediction length.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    sequence_len = int(prediction_len if sequence_len is None else sequence_len)
    prediction_len = int(prediction_len)
    T = len(y)
    min_required = sequence_len + prediction_len
    if T < min_required:
        raise ValueError(
            f"Need at least sequence_len + prediction_len points "
            f"(got T={T}, sequence_len={sequence_len}, prediction_len={prediction_len})."
        )

    rows = T - sequence_len - prediction_len + 1
    X = np.empty((rows, sequence_len), dtype=float)
    Y = np.empty((rows, prediction_len), dtype=float)

    for i in range(rows):
        x_start = i
        x_end = i + sequence_len
        y_end = x_end + prediction_len
        X[i, :] = y[x_start:x_end]
        Y[i, :] = y[x_end:y_end]

    return X, Y

def _clean_xgb_params(p: dict) -> dict:
    int_keys  = {'n_estimators','max_depth','min_child_weight','n_jobs','random_state','max_leaves'}
    float_keys= {'learning_rate','subsample','colsample_bytree','reg_alpha','reg_lambda','gamma'}
    out = {}
    for k, v in (p or {}).items():
        # unwrap 1-length containers
        if isinstance(v, (list, tuple, np.ndarray)) and np.size(v) == 1:
            v = v[0]
        # cast to the expected type
        if k in int_keys:
            v = int(v)
        elif k in float_keys:
            v = float(v)
        out[k] = v
    return out

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearModel(nn.Module):
    """
    DLinear
    """
    def __init__(
        self,
        sequence_len,
        prediction_len,
        *,
        kernel_size=25,
        hidden_dim=0,
        num_layers=1,
        dropout=0.0,
        activation="gelu",
    ):
        super(DLinearModel, self).__init__()
        self.seq_len = sequence_len
        self.pred_len = prediction_len

        self.decompsition = series_decomp(kernel_size)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.Linear_Seasonal = self._build_head(activation)
        self.Linear_Trend = self._build_head(activation)

    def _build_head(self, activation: str) -> nn.Module:
        if self.num_layers <= 1 or self.hidden_dim <= 0:
            layer = nn.Linear(self.seq_len, self.pred_len)
            layer.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            return layer

        activation_layer = self._build_activation(activation)
        layers = [nn.Linear(self.seq_len, self.hidden_dim), activation_layer]
        if self.dropout > 0.0:
            layers.append(nn.Dropout(self.dropout))

        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self._build_activation(activation))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(self.hidden_dim, self.pred_len))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        n = str(name).lower()
        if n == "relu":
            return nn.ReLU()
        if n == "gelu":
            return nn.GELU()
        if n == "silu":
            return nn.SiLU()
        if n == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported DLinear activation: {name}")

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


@dataclass
class DLinear(Forecasting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "DLinear"
        self.horizon = int(kwargs["horizon"])
        self.seq_len = int(kwargs.get("sequence_len", self.horizon*3))

        self.learning_rate = float(kwargs.get("learning_rate", 1e-3))
        self.train_epochs = int(kwargs.get("train_epochs", kwargs.get("epochs", 100)))
        self.batch_size = int(kwargs.get("batch_size", 32))
        self.kernel_size = int(kwargs.get("kernel_size", 25))
        self.hidden_dim = int(kwargs.get("hidden_dim", 0))
        self.num_layers = int(kwargs.get("num_layers", 1))
        self.dropout = float(kwargs.get("dropout", 0.0))
        self.activation = str(kwargs.get("activation", "gelu"))
        self.random_state = int(kwargs["random_state"])  # injected by run_experiments; a silent 32 here would hide a broken injection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.seq_len <= 0 or self.horizon <= 0:
            raise ValueError("`sequence_len` and `horizon` must both be positive integers.")
        if self.batch_size <= 0 or self.train_epochs < 0:
            raise ValueError("`batch_size` must be positive and `train_epochs` must be non-negative.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("`kernel_size` must be a positive odd integer for DLinear.")
        if self.hidden_dim < 0 or self.num_layers <= 0:
            raise ValueError("`hidden_dim` must be non-negative and `num_layers` must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("`dropout` must be in the range [0, 1).")

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.model = DLinearModel(
            sequence_len=self.seq_len,
            prediction_len=self.horizon,
            kernel_size=self.kernel_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.model.to(self.device)
        self._mu = 0.0
        self._sigma = 1.0

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    @staticmethod
    def _to_tensor_windows(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x_t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.as_tensor(y, dtype=torch.float32).unsqueeze(-1)
        return x_t, y_t

    def train(self, x: np.ndarray, y: np.ndarray):
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        x_t, y_t = self._to_tensor_windows(x, y)

        n_samples = x_t.shape[0]
        if n_samples == 0:
            raise ValueError("DLinear received no supervised training windows.")

        self.model.train()
        
        for epoch in range(self.train_epochs):
            permutation = torch.randperm(n_samples)
            epoch_losses = []

            self.model.train()
            for start in range(0, n_samples, self.batch_size):
                idx = permutation[start:start + self.batch_size]
                batch_x = x_t[idx].to(self.device)
                batch_y = y_t[idx].to(self.device)

                model_optim.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()

                epoch_losses.append(loss.item())

            print(f"Epoch {epoch + 1}/{self.train_epochs} avg_loss={np.mean(epoch_losses):.6f}")


    def fit(self, x_train: np.ndarray) -> None:
        x_train = np.asarray(x_train, dtype=float).reshape(-1)

        x_train = np.asarray(x_train, dtype=float).reshape(-1)
        self._mu = float(x_train.mean())
        self._sigma = float(x_train.std() + 1e-8)
        x_train_norm = (x_train - self._mu) / self._sigma
        x, y = _make_supervised(x_train_norm, self.horizon, self.seq_len)
        self.train(x, y)

    def predict(self, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_val = np.asarray(x_val, dtype=float).reshape(-1)
        x_val_norm = (x_val - self._mu) / self._sigma

        x, y = _make_supervised(x_val_norm, self.horizon, self.seq_len)
        x_t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(-1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_hat = self.model(x_t).squeeze(-1).cpu().numpy()

        y = y * self._sigma + self._mu
        y_hat = y_hat * self._sigma + self._mu
        return y, y_hat

        
        

@dataclass
class XGBoostModel(Forecasting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "XGBoost"
        self.n_estimators=int(kwargs["n_estimators"])
        self.learning_rate=float(kwargs["learning_rate"])
        self.max_depth=int(kwargs["max_depth"])
        self.subsample=float(kwargs["subsample"])
        self.colsample_bytree=float(kwargs["colsample_bytree"])
        self.horizon = int(kwargs["horizon"])
        self.n_jobs=int(kwargs["n_jobs"])
        self.random_state = int(kwargs["random_state"])
        self.model = MultiOutputRegressor(XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
        ), n_jobs=self.n_jobs)

    def fit(self, x_train: np.ndarray) -> None:
        x_train = np.asarray(x_train, dtype=float).reshape(-1)        
        x, y = _make_supervised(x_train, self.horizon)
        self.model.fit(x, y)

    def predict(self, x_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = _make_supervised(x_val, self.horizon)
        y_hat = self.model.predict(x)
        return y, y_hat


def build_model(name: str, kwargs: Dict[str, Any]):
    n = name.lower()
    if n in ("xgboost","boost","xgb"): return XGBoostModel(**kwargs)
    if n in ("dlinear", "d-linear"): return DLinear(**kwargs)
    
    raise ValueError(f"Unknown forecasting model: {name}")
