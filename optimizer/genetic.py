from __future__ import annotations
import time
import multiprocessing
from typing import Any, Dict, Optional
import numpy as np
import concurrent.futures as cf
from .space import Space
from .run_logger import RunLogger
from profiling import stage_clock
from profiling.timed_call import SEAM_FIELDS, seam_extra, timed_call


_WORKER_OBJECTIVE = None


def _initialize_worker_objective(objective) -> None:
    global _WORKER_OBJECTIVE
    _WORKER_OBJECTIVE = objective


def _evaluate_worker_candidate(candidate):
    return timed_call(_WORKER_OBJECTIVE, candidate)


def _kill_pool(executor: cf.ProcessPoolExecutor) -> None:
    for worker_process in list(executor._processes.values()):
        worker_process.kill()
    executor.shutdown(wait=False, cancel_futures=True)


class GeneticOptimizer:
    _DUPLICATE_RETRIES = 16

    def __init__(
        self,
        pop_size=24,
        initial_pop_size=None,
        gens=20,
        cx_prob=0.8,
        mut_prob=0.5,
        mut_per_var_prob=0.6,
        alpha=0.75,
        sigma_frac=0.1,
        log_sigma=0.75,
        elitism=2,
        tournament_k=2,
        early_stop_patience=5,
        random_state=2048,
        n_workers=10,
        batch_timeout_sec=3600,
        verbose=1,
        log_subdir="_genetic_opt",
    ):
        self.pop_size = int(pop_size)
        self.initial_pop_size = int(initial_pop_size) if initial_pop_size is not None else self.pop_size
        self.gens = int(gens)
        self.elitism = int(elitism)
        self.tournament_k = int(tournament_k)

        self.cx_prob, self.mut_prob = cx_prob, mut_prob
        self.mut_per_var_prob = mut_per_var_prob
        self.sigma_frac, self.log_sigma = sigma_frac, log_sigma
        self.early_stop_patience = early_stop_patience
        self.alpha = alpha
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.n_workers = n_workers
        self.batch_timeout_sec = batch_timeout_sec
        self.verbose = verbose
        self.log_subdir = log_subdir
        self.offspring_per_generation = self.pop_size - self.elitism
        self.total_budget = self.initial_pop_size + self.gens * self.offspring_per_generation

    def _tournament(self, rng, population_scores, excluded_parent_index=None):
        population_size = len(population_scores)
        excludes_parent = excluded_parent_index is not None and population_size > 1
        eligible_count = population_size - int(excludes_parent)
        contestant_indices = np.asarray(
            rng.choice(
                eligible_count,
                size=min(self.tournament_k, eligible_count),
                replace=False,
            ),
            dtype=int,
        )
        if excludes_parent:
            contestant_indices += contestant_indices >= excluded_parent_index

        return int(
            max(
                contestant_indices,
                key=lambda candidate_index: population_scores[int(candidate_index)],
            )
        )

    @staticmethod
    def _candidate_key(candidate, parameter_names):
        return tuple(candidate[parameter_name] for parameter_name in parameter_names)

    def _sample_unique_population(
        self, parameter_space, parameter_names, population_size
    ):
        population = []
        seen_keys = set()
        consecutive_duplicates = 0

        while (
            len(population) < population_size
            and consecutive_duplicates < self._DUPLICATE_RETRIES
        ):
            candidate = parameter_space.sample(self.rng)
            candidate_key = self._candidate_key(candidate, parameter_names)
            if candidate_key in seen_keys:
                consecutive_duplicates += 1
                continue

            population.append(candidate)
            seen_keys.add(candidate_key)
            consecutive_duplicates = 0

        exhausted = len(population) < population_size
        if exhausted and self.verbose:
            print(
                "[genetic] Initial search space exhausted before reaching "
                f"pop_size={population_size}; evaluating {len(population)} unique candidates."
            )
        return population, seen_keys, exhausted

    def _propose_elite_refinement(
        self, parameter_space, elite_candidate, parameter_names, seen_keys
    ):
        for _ in range(self._DUPLICATE_RETRIES):
            refined_candidate = parameter_space.mutate_elite(
                self.rng,
                elite_candidate,
                sigma_frac=self.sigma_frac,
                log_sigma=self.log_sigma,
            )
            refined_key = self._candidate_key(refined_candidate, parameter_names)
            if refined_key not in seen_keys:
                return refined_candidate, refined_key

        refined_candidate = parameter_space.sample(self.rng)
        refined_key = self._candidate_key(refined_candidate, parameter_names)

        return refined_candidate, refined_key

    def _propose_unique_child(
        self, parameter_space, population, population_scores, parameter_names, seen_keys
    ):
        offspring = None
        offspring_key = None
        for _ in range(self._DUPLICATE_RETRIES):
            use_crossover = self.rng.random() < self.cx_prob
            first_parent_index = self._tournament(self.rng, population_scores)
            first_parent = population[first_parent_index]

            if use_crossover:
                second_parent_index = self._tournament(
                    self.rng,
                    population_scores,
                    excluded_parent_index=first_parent_index,
                )
                offspring = parameter_space.crossover(
                    self.rng, first_parent, population[second_parent_index]
                )
            else:
                offspring = first_parent

            if self.rng.random() < self.mut_prob:
                offspring = parameter_space.mutate(
                    self.rng,
                    offspring,
                    prob=self.mut_per_var_prob,
                    sigma_frac=self.sigma_frac,
                    log_sigma=self.log_sigma,
                )
            offspring_key = self._candidate_key(offspring, parameter_names)
            if offspring_key not in seen_keys:
                return offspring, offspring_key

            offspring = parameter_space.mutate(
                self.rng,
                offspring,
                sigma_frac=self.sigma_frac,
                log_sigma=self.log_sigma,
            )
            offspring_key = self._candidate_key(offspring, parameter_names)
            if offspring_key not in seen_keys:
                return offspring, offspring_key

        for _ in range(self._DUPLICATE_RETRIES):
            offspring = parameter_space.sample(self.rng)
            offspring_key = self._candidate_key(offspring, parameter_names)
            if offspring_key not in seen_keys:
                return offspring, offspring_key

        offspring = parameter_space.sample(self.rng)
        offspring_key = self._candidate_key(offspring, parameter_names)

        return offspring, offspring_key

    def maximize(
        self, objective, search_space, space_definition, log_dir, run_metadata, **kwargs
    ) -> Dict[str, float]:
        parameter_space = Space(search_space, space_definition)
        parameter_names = list(parameter_space.variable_specs.keys())

        logger = RunLogger(
            log_dir=log_dir,
            run_metadata=run_metadata,
            optimizer_name="genetic",
            optimizer_meta={
                "pop_size": self.pop_size,
                "initial_pop_size": self.initial_pop_size,
                "gens": self.gens,
                "cx_prob": self.cx_prob,
                "mut_prob": self.mut_prob,
                "mut_per_var_prob": self.mut_per_var_prob,
                "sigma_frac": self.sigma_frac,
                "log_sigma": self.log_sigma,
                "elitism": self.elitism,
                "tournament_k": self.tournament_k,
                "early_stop_patience": self.early_stop_patience,
                "alpha": self.alpha,
                "random_state": self.random_state,
                "n_workers": self.n_workers,
            },
            search_space=search_space,
            space_definition=space_definition,
            param_names=parameter_names,
            total_budget=self.total_budget,
            log_subdir=self.log_subdir,
            verbose=self.verbose,
            extra_eval_fields=["objective_cpu_sec", "batch_wall_sec", "batch_size"]
            + (SEAM_FIELDS if stage_clock.enabled() else []),
        )
        self.last_run_dir = logger.run_dir
        self.last_logger = logger
        self.last_total_objective_cpu_sec = 0.0
        self.last_stage_totals: Dict[str, list] = {}

        executor = (
            cf.ProcessPoolExecutor(
                max_workers=min(int(self.n_workers), max(self.pop_size, self.initial_pop_size)),
                mp_context=multiprocessing.get_context("forkserver"),
                initializer=_initialize_worker_objective,
                initargs=(objective,),
            )
            if (self.n_workers and self.n_workers > 1)
            else None
        )
        evaluations = 0
        best_candidate: Optional[Dict[str, float]] = None
        best_score = float("-inf")
        pool_killed = False

        def _evaluate_batch(candidate_batch):
            nonlocal pool_killed
            batch_started_at = time.perf_counter()
            if executor is not None:
                try:
                    batch_results = list(
                        executor.map(
                            _evaluate_worker_candidate,
                            candidate_batch,
                            timeout=self.batch_timeout_sec,
                        )
                    )
                except TimeoutError as timeout_error:
                    pool_killed = True
                    _kill_pool(executor)
                    raise TimeoutError(
                        f"genetic: a batch of {len(candidate_batch)} evaluations "
                        f"exceeded batch_timeout_sec={self.batch_timeout_sec}s; "
                        f"the worker pool stopped making progress after "
                        f"{evaluations} evaluation(s). Raise batch_timeout_sec in "
                        "cfg/optimizer/genetic.yaml if evaluations are "
                        "legitimately this slow, or set it to null to wait "
                        "indefinitely."
                    ) from timeout_error
            else:
                batch_results = [timed_call(objective, candidate) for candidate in candidate_batch]
            batch_wall_sec = time.perf_counter() - batch_started_at
            batch_scores = [score for score, _record in batch_results]
            batch_records = [record for _score, record in batch_results]
            return batch_scores, batch_records, batch_wall_sec

        def _log_batch(
            candidate_batch,
            candidate_scores,
            candidate_records,
            batch_wall_sec,
            phase,
            proposal_source,
        ):
            nonlocal evaluations, best_candidate, best_score
            if isinstance(proposal_source, str):
                proposal_source = [proposal_source] * len(candidate_batch)
            batch_size = len(candidate_batch)
            for candidate, score, record, candidate_source in zip(
                candidate_batch, candidate_scores, candidate_records, proposal_source
            ):
                evaluations += 1
                global_best_before = best_score
                is_new_best = score > best_score
                if is_new_best:
                    best_score = score
                    best_candidate = candidate
                self.last_total_objective_cpu_sec += record["cpu_sec"]
                for stage_name, (stage_total, stage_count) in record["stages"].items():
                    entry = self.last_stage_totals.setdefault(stage_name, [0.0, 0])
                    entry[0] += stage_total
                    entry[1] += stage_count
                logger.log_evaluation(
                    evaluation=evaluations,
                    params_raw=candidate,
                    params_model=candidate,
                    reward=float(score),
                    elapsed_sec=record["wall_sec"],
                    global_best_before=global_best_before,
                    global_best_after=best_score,
                    is_new_global_best=is_new_best,
                    phase=phase,
                    proposal_source=candidate_source,
                    extra={
                        "objective_cpu_sec": record["cpu_sec"],
                        "batch_wall_sec": batch_wall_sec,
                        "batch_size": batch_size,
                        **seam_extra(record),
                    },
                )

        try:
            population, seen_keys, search_space_exhausted = (
                self._sample_unique_population(
                    parameter_space, parameter_names, self.initial_pop_size
                )
            )
            population_scores, population_records, batch_wall_sec = _evaluate_batch(population)
            _log_batch(
                population,
                population_scores,
                population_records,
                batch_wall_sec,
                "init",
                "unique_random_init",
            )

            patience = 0
            generation = 0

            while generation < self.gens and not search_space_exhausted:
                generation += 1
                ranked_elites = sorted(
                    zip(population, population_scores),
                    key=lambda candidate_score_pair: candidate_score_pair[1],
                    reverse=True,
                )[: self.elitism]
                elite_population = [candidate for candidate, _ in ranked_elites]
                elite_scores = [score for _, score in ranked_elites]

                offspring_batch = []
                proposal_sources = []

                for elite_candidate in elite_population:
                    refined_candidate, refined_key = self._propose_elite_refinement(parameter_space, elite_candidate, parameter_names, seen_keys)

                    offspring_batch.append(refined_candidate)
                    proposal_sources.append("elite_error_refinement")
                    seen_keys.add(refined_key)

                while len(offspring_batch) < self.offspring_per_generation:
                    offspring, offspring_key = self._propose_unique_child(
                        parameter_space,
                        population,
                        population_scores,
                        parameter_names,
                        seen_keys,
                    )
                    
                    offspring_batch.append(offspring)
                    proposal_sources.append("tournament_distinct_parents_unique")
                    seen_keys.add(offspring_key)

                offspring_scores, offspring_records, batch_wall_sec = _evaluate_batch(
                    offspring_batch
                )
                previous_best_score = best_score
                _log_batch(
                    offspring_batch,
                    offspring_scores,
                    offspring_records,
                    batch_wall_sec,
                    "generation",
                    proposal_sources,
                )

                population = elite_population + offspring_batch
                population_scores = elite_scores + offspring_scores

                if best_score > previous_best_score:
                    patience = 0
                    if self.verbose:
                        print(f"New best found: {round(best_score, 4)}")
                else:
                    patience += 1
                    if self.verbose:
                        print(
                            f"No improvement. Patience: {patience}/"
                            f"{self.early_stop_patience}"
                        )
                    if patience >= self.early_stop_patience:
                        if self.verbose:
                            print("Early stopping triggered.")
                        break
        finally:
            if executor is not None and not pool_killed:
                executor.shutdown(wait=True)

        assert (
            best_candidate is not None
        )
        if self.verbose:
            print(f"Optimization finished. Best score: {round(best_score, 4)}")
            print(
                "Best candidate:",
                {
                    parameter_name: round(parameter_value, 4)
                    for parameter_name, parameter_value in best_candidate.items()
                },
            )

        logger.write_summary(
            evaluations=evaluations,
            total_budget=self.total_budget,
            best_reward=best_score,
            best_params=best_candidate,
            status="completed",
        )
        return best_candidate
