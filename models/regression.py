# Angus Dempster, Francois Petitjean, Geoff Webb

# Dempster A, Petitjean F, Webb GI (2019) ROCKET: Exceptionally fast and
# accurate time series classification using random convolutional kernels.
# arXiv:1910.13051
import numpy as np
from numba import njit, prange
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict
import numba as nb
from .base import Regressor

# Rocket's apply_kernels is numba parallel=True; NUMBA_NUM_THREADS is the single
# knob governing it, and numba already initialises the pool to that value - so
# there is deliberately no set_num_threads() call here. Callers set the env var:
# process-parallel optimizers (genetic's worker pool) export NUMBA_NUM_THREADS=1
# so each forked worker stays single-threaded and doesn't oversubscribe the CPU
# or re-fork an already-spun-up parallel threadpool; a sequential optimizer
# (bosmp, adaedge) can be given the whole machine.
#
# This used to read `nb.set_num_threads(min(8, nb.config.NUMBA_NUM_THREADS))`,
# which contradicted its own comment: it imposed a hard ceiling of 8 that no env
# var could lift. Measured on IEEEPPG's validation split (354x1000x5, 10k
# kernels), that ceiling cost ~3.5x - 15.8s/evaluation at 8 threads against
# 5.2s at 32 - and 13h vs 45min for one (compressor, alpha) cell.
#
# Raising it cannot change results: apply_kernels' prange is over examples, each
# writing only its own _X[i] slice with no cross-thread reduction. Verified
# bitwise - kernels and output hash identically at 1, 8 and 32 threads, given
# _seed_numba() first (numba keeps a per-thread np.random state separate from
# numpy's, so an unseeded comparison measures only that).

@njit
def _seed_numba(seed):
    np.random.seed(seed)

@njit(fastmath=True)
def generate_kernels(input_length, num_kernels, num_channels=1):
    candidate_lengths = np.array((7, 9, 11), dtype=np.int64)      # use one int type
    candidate_lengths = candidate_lengths[candidate_lengths < input_length]
    lengths = np.random.choice(candidate_lengths, num_kernels).astype(np.int64)

    num_channel_indices = (2 ** np.random.uniform(0.0, np.log2(num_channels + 1.0), num_kernels)).astype(np.int64)

    weights = np.zeros((int(num_channels), int(lengths.sum())), dtype=np.float32)
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int64)
    paddings  = np.zeros(num_kernels, dtype=np.int64)

    channel_indices = np.zeros(int(num_channel_indices.sum()), dtype=np.int64)

    for i in range(num_kernels):
        ln = int(lengths[i])  # ensure same int type in tuple
        _weights = np.random.normal(0.0, 1.0, (int(num_channels), ln)).astype(np.float32)

        a = int(lengths[:i].sum()); b = a + ln
        for j in range(num_channels):
            _weights[j] = _weights[j] - _weights[j].mean()
        weights[:, a:b] = _weights

        a1 = int(num_channel_indices[:i].sum()); b1 = a1 + int(num_channel_indices[i])
        # If this line causes another Numba error (replace=False sometimes unsupported),
        # switch to the fallback shown below.
        channel_indices[a1:b1] = np.random.choice(np.arange(0, num_channels), int(num_channel_indices[i]), replace=False)

        biases[i] = np.random.uniform(-1.0, 1.0)

        # keep everything in float -> int at the end
        max_log = np.log2((input_length - 1.0) / (ln - 1.0))
        dilation = int(2.0 ** np.random.uniform(0.0, max_log))
        dilations[i] = dilation

        paddings[i] = ((ln - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0

    return weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices


@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices, stride):
    # zero padding
    if padding > 0:
        _input_length, _num_channels = X.shape
        _X = np.zeros((_input_length + (2 * padding), _num_channels))
        _X[padding:(padding + _input_length), :] = X
        X = _X

    input_length, num_channels = X.shape

    output_length = input_length - ((length - 1) * dilation)

    _ppv = 0
    _max = -np.inf

    for i in range(0, output_length, stride):
        _sum = bias

        for j in range(length):
            for k in range(num_channel_indices):
                _sum += weights[channel_indices[k], j] * X[i + (j * dilation), channel_indices[k]]

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


@njit(parallel=True, fastmath=True)
def apply_kernels(X, kernels, stride=1):
    weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices = kernels

    num_examples = len(X)
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float32)  # 2 features per kernel

    for i in prange(num_examples):
        a = 0
        a1 = 0
        for j in range(num_kernels):
            b = a + lengths[j]
            b1 = a1 + num_channel_indices[j]

            _X[i, (j * 2):((j * 2) + 2)] = \
                apply_kernel(X[i], weights[:, a:b], lengths[j], biases[j], dilations[j], paddings[j],
                             num_channel_indices[j], channel_indices[a1:b1], stride)

            a = b
            a1 = b1

    return _X


class RocketRegressor(Regressor):
    # Bumped whenever the fitted object changes shape or meaning while the
    # package versions stay put - the cache contract cannot see inside this
    # class, so without this a stale model is silently reloaded and the change
    # appears to do nothing. "+scaler" marks the StandardScaler fix below.
    _cache_implementation_id = "rocket.ridgecv+scaler"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "Rocket"
        self.n_kernels = kwargs.get("n_kernels", 10000)
        self.kernels = None
        # StandardScaler is NOT optional, and its absence is not a subtle bug.
        # Rocket emits two features per kernel: PPV, bounded in [0, 1], and MAX,
        # unbounded. Their standard deviations differ by 6-10 orders of
        # magnitude, so one shared ridge penalty regularises PPV into oblivion
        # while barely touching MAX. Measured on the test split, fitting RidgeCV
        # on raw features scored R^2 = -1094 on Covid3Month and -357 on
        # LiveFuelMoistureContent; with the scaler, +0.15 and -0.31. The damage
        # tracks the feature scale ratio exactly - FloodModeling1, the only
        # dataset with a small ratio (5e3 against >=1e6 elsewhere), was also the
        # only one that fitted correctly without it (R^2 0.986 either way).
        # This matches aeon's reference RocketRegressor, which is what makes
        # Rocket a top-ranked method on the Monash TSER benchmark.
        # tests/test_rocket_feature_scaling.py guards it.
        self.regressor = make_pipeline(
            StandardScaler(with_mean=False),
            RidgeCV(alphas=np.logspace(-3, 3, 10)),
        )
        self.random_state = int(kwargs["random_state"])  # injected by run_experiments

    def fit(self, x_train: np.array, y_train: np.array):
        if self.random_state is not None:
            _seed_numba(int(self.random_state))   # seed Numba RNG deterministically

        self.kernels = generate_kernels(
            input_length=x_train.shape[1],
            num_kernels=self.n_kernels,
            num_channels=x_train.shape[2],
        )
        x_training_transform = apply_kernels(X=x_train, kernels=self.kernels)
        self.regressor.fit(x_training_transform, y_train)
        self.model = {
            "kernels": self.kernels,
            "regressor": self.regressor,
        }


    def predict(self, x: np.array):
        x_test_transform = apply_kernels(x, self.kernels)
        y_pred = self.regressor.predict(x_test_transform)

        return y_pred

    def _get_cache_payload(self):
        return {
            "kernels": self.kernels,
            "regressor": self.regressor,
        }

    def _load_cache_payload(self, payload):
        self.kernels = payload["kernels"]
        self.regressor = payload["regressor"]
        self.model = payload

def build_model(name: str, kwargs: Dict[str, Any]):
    n = name.lower()
    if n in ("rocket","roket","classirocket","rocketregressor"):
        return RocketRegressor(**kwargs)
    raise ValueError(f"Unknown classification model: {name}")
