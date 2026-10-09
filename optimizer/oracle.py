from evals.metrics import combine_fitness
import numpy as np
from typing import Any, Dict, Tuple

class DeterministicPreferenceOracle:
    """
    A deterministic preference oracle that compares two pipelines based on their fitness scores.
    """

    def __init__(self, hidden_weight: float = 0.5, reliability: float = 1.0, beta: float = 300.0):
        self.hidden_weight = hidden_weight
        self.reliability = reliability
        self.beta = beta

    def __call__(self, pipeline_a: Tuple[float, float], pipeline_b: Tuple[float, float]) -> bool:
        """
        Compare two pipelines based on their fitness scores.

        Args:
            pipeline_a (Tuple[float, float]): The first pipeline's fitness scores (fitness1, fitness2).
            pipeline_b (Tuple[float, float]): The second pipeline's fitness scores (fitness1, fitness2).

        Returns:
            bool: True if pipeline_a is preferred, False if pipeline_b is preferred, None if they are equal.
        """
        fitness_a = combine_fitness(pipeline_a[0], pipeline_a[1], self.hidden_weight)
        fitness_b = combine_fitness(pipeline_b[0], pipeline_b[1], self.hidden_weight)

        if fitness_a >= fitness_b:
            return True
        
        return False
        

class StochasticPreferenceOracle:
    def __init__(self, hidden_weight: float = 0.5, reliability: float = 0.9, beta: float = 80.0):
        self.hidden_weight = hidden_weight
        # Potentially modify beta according to reliability, but for now, we keep it fixed.
        self.reliability = reliability
        self.beta = beta

    def __call__(self, pipeline_a: Tuple[float, float], pipeline_b: Tuple[float, float]) -> bool:
        """
        Compare two pipelines based on their fitness scores with stochasticity.

        Args: 
            pipeline_a (Tuple[float, float]): The first pipeline's fitness scores (fitness1, fitness2).
            pipeline_b (Tuple[float, float]): The second pipeline's fitness scores (fitness1, fitness2).
        
        Returns:
            bool: True if pipeline_a is preferred, False if pipeline_b is preferred.
        """
        fitness_a = combine_fitness(pipeline_a[0], pipeline_a[1], self.hidden_weight)
        fitness_b = combine_fitness(pipeline_b[0], pipeline_b[1], self.hidden_weight)

        strength_a = 1/(1+np.exp(-self.beta*(fitness_a-fitness_b)))

        if np.random.rand() < strength_a:
            return True

        return False


def build_oracle(name: str, kwargs: Dict[str, Any]):
    if name == "deterministic_oracle": return DeterministicPreferenceOracle(**kwargs)
    if name == "stochastic_oracle": return StochasticPreferenceOracle(**kwargs)
    raise ValueError(f"Unknown oracle: {name}")