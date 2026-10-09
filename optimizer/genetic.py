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
    """Store one objective per worker instead of serializing it per candidate."""
    global _WORKER_OBJECTIVE
    _WORKER_OBJECTIVE = objective


def _evaluate_worker_candidate(candidate):
    # ProcessPoolExecutor runs the initializer before dispatching any task to a
    # worker, and a failing initializer breaks the pool rather than yielding a
    # worker with _WORKER_OBJECTIVE still None - so there is no unset case to
    # guard against here.
    #
    # Returns (score, TimingRecord) - see profiling/timed_call.py. Before this
    # (docs/EXECUTION_TIME_STUDY_PLAN.md §5.4), this returned a bare float, so
    # a child's wall/CPU time was lost the moment it crossed the pool boundary
    # and every wall-clock comparison against this optimizer had to be an "up
    # to" bound derived from the batch schedule instead of a measurement. The
    # score itself is untouched: timed_call calls the objective exactly once
    # and passes its return value through.
    return timed_call(_WORKER_OBJECTIVE, candidate)


def _kill_pool(executor: cf.ProcessPoolExecutor) -> None:
    """Force-kill a pool whose workers stopped responding, without waiting.

    ``shutdown(wait=True)`` joins the worker processes, so it is exactly the
    wrong tool for a pool whose workers are what stopped responding - it would
    inherit the hang it is meant to clean up. ``_processes`` is private, but it
    is the only handle ProcessPoolExecutor gives on the children; it is read
    directly rather than via ``getattr(..., {})`` so that a future rename fails
    loudly instead of silently leaving the workers orphaned.
    """
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
        # Only the initial random population's size, so a budget that doesn't
        # divide evenly can be repaired without perturbing the steady-state
        # generation cost - see resolve_genetic_percentages_for_budget().
        # Defaults to pop_size for direct/legacy construction that never
        # passes it.
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
        # Wall-clock ceiling for one batch of parallel evaluations. This is a
        # deadlock net, not a performance budget: it exists so a worker pool that
        # stops responding aborts the run instead of parking it forever, so the
        # default is deliberately far above any legitimate batch. `None` is the
        # explicit opt-out (wait indefinitely, the pre-timeout behaviour).
        self.batch_timeout_sec = batch_timeout_sec
        self.verbose = verbose
        self.log_subdir = log_subdir
        # Total objective evaluations this config will spend (no early stopping):
        # initial_pop_size initial + gens generations of (pop_size - elitism)
        # new offspring. The constructor checks above guarantee the offspring
        # count is at least 1.
        self.offspring_per_generation = self.pop_size - self.elitism
        self.total_budget = self.initial_pop_size + self.gens * self.offspring_per_generation

    def _tournament(self, rng, population_scores, excluded_parent_index=None):
        """Return a tournament winner's index.

        Contestants are unique within a tournament. The excluded index is
        mapped out of a compact integer range, avoiding a new population-sized
        index array for every parent selection.
        """
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
        """Sample an initial population without scheduling duplicate work."""
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
        """Return an unseen local neighbour of one elite, or completly new sample.

        The neighbour keeps the elite's pipeline (its integer method indices) and
        differs only in the continuous error bounds - the exploitation move for a
        space that factorizes into a discrete arm times continuous errors.

        If the elite's neighbourhood is exhausted, return a fresh sample from the
        search space.
        """
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
        """Return an unseen offspring and its key, or ``None`` if exhausted.

        Bounded retries keep finite search spaces from hanging without spending
        an objective evaluation on a known duplicate.
        """
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
                # mutate() never changes its input, so no defensive parent copy
                # is needed on the no-crossover path.
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

            # The normal operators produced a previously evaluated candidate.
            # Force mutation before spending an evaluation on an exact clone.
            offspring = parameter_space.mutate(
                self.rng,
                offspring,
                sigma_frac=self.sigma_frac,
                log_sigma=self.log_sigma,
            )
            offspring_key = self._candidate_key(offspring, parameter_names)
            if offspring_key not in seen_keys:
                return offspring, offspring_key

        # Mutation can be ineffective in small integer spaces. Try fresh
        # samples before concluding that no unseen offspring can be found.
        for _ in range(self._DUPLICATE_RETRIES):
            offspring = parameter_space.sample(self.rng)
            offspring_key = self._candidate_key(offspring, parameter_names)
            if offspring_key not in seen_keys:
                return offspring, offspring_key

        # The search space is exhausted: all candidates have been evaluated
        # Extremenly unlikely, but possible in small spaces with large populations.
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
            # See docs/EXECUTION_TIME_STUDY_PLAN.md §5.4: objective_elapsed_sec
            # below is now the candidate's own exact wall time (not a batch
            # average), and these three ride along through RunLogger's existing
            # extra_eval_fields mechanism rather than widening the base schema.
            # Per-seam columns only when the stage clock is on - see the same
            # note in optimizer/adaedge.py.
            extra_eval_fields=["objective_cpu_sec", "batch_wall_sec", "batch_size"]
            + (SEAM_FIELDS if stage_clock.enabled() else []),
        )
        self.last_run_dir = logger.run_dir
        # The RunLogger itself, so a caller can read `last_logger.total_logging_sec`
        # - the trace-logging tax (T_log) measured in the SAME pass as the search,
        # which is what lets the execution-time study report the harness tax from
        # one `as_shipped` run instead of a second `quiet` run of the whole matrix.
        self.last_logger = logger
        # Sum of every logged candidate's objective_cpu_sec this call, kept
        # regardless of whether log_dir enables file logging - _log_batch
        # already iterates every (candidate, record) pair unconditionally, so
        # this costs one float addition per candidate. Exists for
        # scripts/benchmark_execution_time.py's `search_cpu_sec` (§7.2): under
        # the default `quiet` condition (log_dir=None) there is no trace file
        # to recover this from afterward, and time.process_time() around the
        # whole maximize() call would only see the PARENT's CPU time, missing
        # every forkserver child's work entirely.
        self.last_total_objective_cpu_sec = 0.0
        # {stage_name: [total_seconds, count]}, summed across every candidate's
        # TimingRecord.stages this call - the worker-side half of §7.2's "the
        # drained stage_clock totals... attributed to search... by summing the
        # returned dicts". Empty when profiling.stage_clock is disabled in the
        # worker processes (each record's "stages" dict is then {} - see
        # profiling/stage_clock.py::drain), never missing the key itself.
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
                    # map()'s timeout covers the batch as a whole, measured from
                    # this call - so it bounds the wait even when the very first
                    # batch is the one that wedges (the case observed in
                    # practice, which logged zero evaluations).
                    batch_results = list(
                        executor.map(
                            _evaluate_worker_candidate,
                            candidate_batch,
                            timeout=self.batch_timeout_sec,
                        )
                    )
                except TimeoutError as timeout_error:
                    # Kill the workers before re-raising: the finally below
                    # would otherwise block on the same unresponsive pool, and
                    # a hang that cannot even be interrupted is what left
                    # orphaned worker processes behind.
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
                # Same timed_call() helper as the worker path (see
                # _evaluate_worker_candidate), so n_workers=1 and n_workers>1
                # produce the exact same per-candidate timing columns - see
                # docs/EXECUTION_TIME_STUDY_PLAN.md §5.4.
                batch_results = [timed_call(objective, candidate) for candidate in candidate_batch]
            batch_wall_sec = time.perf_counter() - batch_started_at
            batch_scores = [score for score, _record in batch_results]
            batch_records = [record for _score, record in batch_results]
            # Batches are never empty: the initial population always holds at
            # least the first sample, and an empty offspring batch ends the run
            # before it reaches here.
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
            # A generation mixes elite refinements with tournament offspring, so
            # the source is per candidate; a bare string covers uniform batches.
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
                    # The candidate's own exact wall time (TimingRecord.wall_sec),
                    # not batch_elapsed / len(batch) - see §5.4/§2.4: that average
                    # was the root cause of every "up to" bound in the plan.
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

            # One flag, checked in the loop header. Exhaustion discovered while
            # breeding still lets this generation's partial batch be evaluated
            # and installed below; the header then ends the run.
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

                # One local refinement per elite, taken OUT of this generation's
                # offspring share rather than added to it - the batch size, and
                # so total_budget, is unchanged. An elite whose neighbourhood is
                # exhausted simply yields its slot to an ordinary offspring.
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
                    # Add the already-computed key immediately so later
                    # offspring in this batch cannot duplicate scheduled work.
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
            # _kill_pool() has already shut the executor down without waiting;
            # joining here would re-block on the workers it just killed.
            if executor is not None and not pool_killed:
                executor.shutdown(wait=True)

        assert (
            best_candidate is not None
        )  # a failed evaluation raises out of _evaluate_batch
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
