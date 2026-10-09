from pathlib import Path
import sys

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import ExperimentConfig, resolve_experiment_alpha
from run_suite import _build_command, iter_combos


OPTIMIZER_CONFIGS = sorted((PROJECT_ROOT / "cfg/optimizer").glob("*.yaml"))


@pytest.mark.parametrize("optimizer_path", OPTIMIZER_CONFIGS, ids=lambda path: path.stem)
def test_experiment_alpha_overrides_every_optimizer_config(optimizer_path: Path) -> None:
    optimizer_config = yaml.safe_load(optimizer_path.read_text())

    assert optimizer_config["kwargs"]["alpha"] != 0.123
    assert resolve_experiment_alpha(0.123, optimizer_config["kwargs"]) == 0.123


def test_optimizer_alpha_is_only_a_fallback() -> None:
    assert resolve_experiment_alpha(None, {"alpha": 0.75}) == 0.75


def test_experiment_config_keeps_optimizer_alpha_in_sync() -> None:
    optimizer_kwargs = {"alpha": 0.9, "n_iter": 10}

    cfg = ExperimentConfig(
        task="classification",
        model_name="model",
        model_kwargs={},
        metrics={},
        dataset="dataset",
        loader_name="loader",
        loader_kwargs={},
        split={},
        compressor="compressor",
        compressor_bounds={},
        compressor_space={},
        compressor_methods={},
        optimizer="random",
        optimizer_kwargs=optimizer_kwargs,
        alpha=0.2,
        random_state=32,
        out_dir="results",
        logs_dir=".logs",
    )

    assert cfg.alpha == 0.2
    assert cfg.optimizer_kwargs["alpha"] == 0.2
    assert optimizer_kwargs["alpha"] == 0.9


def test_suite_alpha_is_forwarded_to_experiment_cli() -> None:
    manifest = {
        "random_states": [32],
        "alpha": 0.4,
        "tasks": [{"analytics": "classification", "dataset": "ucr_small"}],
        "compressors": [{"compression": "laconic", "optimizers": ["random"]}],
    }

    combo = next(iter_combos(manifest))
    command = _build_command(combo, "python")

    assert command[command.index("--alpha") + 1] == "0.4"


def test_suite_requires_budget_for_genetic_percentages() -> None:
    manifest = {
        "random_states": [32],
        "alpha": 0.4,
        "tasks": [{"analytics": "classification", "dataset": "ucr_small"}],
        "compressors": [
            {"compression": "laconic", "optimizers": ["genetic"]}
        ],
    }

    with pytest.raises(ValueError, match="genetic must define budget"):
        list(iter_combos(manifest))
