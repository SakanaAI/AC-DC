import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_USE_V1"] = "0"

import hydra
import logging
import numpy as np
import time
import wandb
import json  # Add json import for pool logging
import shutil  # Add shutil for rmtree
import re  # Add re for parsing archive filenames
from celery import Celery
from collections import deque
from wandb import AlertLevel  # type: ignore

logging.basicConfig(level=logging.DEBUG)

# import sys # Unused
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional, Union, Dict, Any, Tuple, Set

from datatypes import ACDCMergeResult, ACDCSolution, ACDCArchiveData
from dns.dns_utils import (
    create_ac_dc_solution,
    update_dns_archive,
    save_ac_dc_archive,
    load_ac_dc_archive,
    convert_acdc_to_dns_solution,  # Import the conversion function
)
from dns.metrics import (
    compute_acdc_coverage_metrics,
)  # Import metrics functions
from tasks.acdc_task import ACDCTask
from utils.celery_utils import setup_celery
from utils.helpers import (
    get_latest_generation,
    delete_models_not_in_archive,
)
from workers.ac_dc_worker import ACDCWorker as Worker

# Add imports for ACDC Task Pool integration
from tasks.task_generation import ACDCTaskPool


def setup_optimization_directories(
    cfg: DictConfig, output_dir: Optional[str] = None
) -> Dict[str, str]:
    """Set up directories for optimization run and return directory paths."""
    if not output_dir:
        if cfg.get("restart_dir"):
            # Use the restart directory as the output directory
            output_dir = cfg.restart_dir
        else:
            output_dir = (
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore
            )

    # Ensure output_dir is not None
    if not output_dir:
        raise ValueError("Output directory cannot be determined")

    model_dir = os.path.join(output_dir, "models")
    archive_dir = os.path.join(output_dir, "archives")
    image_dir = os.path.join(output_dir, "images")
    # Define and create the generated tasks directory
    generated_tasks_dir = os.path.join(output_dir, "generated_tasks", "pool")
    vector_db_dir = os.path.join(output_dir, "vector_db")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(
        generated_tasks_dir, exist_ok=True
    )  # Create generated tasks dir

    return {
        "output_dir": output_dir,
        "model_dir": model_dir,
        "archive_dir": archive_dir,
        "image_dir": image_dir,
        "generated_tasks_dir": generated_tasks_dir,  # Add to returned dict
        "vector_db_dir": vector_db_dir,
    }


def handle_merge_result(
    result: Optional[ACDCMergeResult],
) -> Optional[ACDCSolution]:
    """Process merge result and create a DNSSolution."""
    if not result:
        return None

    # Create DNSSolution using the updated function signature
    solution = create_ac_dc_solution(
        model_path=result.save_path,
        task_metrics=result.task_metrics,
        acdc_skill_vector=result.acdc_skill_vector,
        avg_acdc_quality=result.avg_acdc_quality,
        acdc_eval_details=getattr(
            result, "acdc_eval_details", None
        ),  # Pass eval details if present
        is_gibberish=getattr(result, "is_gibberish", False),
    )
    return solution


def create_merge_task(
    cfg: DictConfig,
    call_fn: Any,
    gen: int,
    model_index: int,
    archive_data: ACDCArchiveData,
    np_random: np.random.RandomState,
    task_info: Union[Dict[str, DictConfig], List[str]],  # Add task_info
) -> Any:
    """Create appropriate merge task based on optimization mode."""
    model_dir = archive_data["dirs"]["model_dir"]

    dns_archive = archive_data["dns_archive"]
    if len(dns_archive) >= 2:
        parent_indices = np_random.choice(
            len(dns_archive), size=2, replace=False
        )
        parent1 = dns_archive[int(parent_indices[0])]
        parent2 = dns_archive[int(parent_indices[1])]
        parent1_path = parent1.model_path
        parent2_path = parent2.model_path
    else:
        parent_indices = np_random.choice(
            len(cfg.seed_model_paths), size=2, replace=False
        )
        parent1_path = cfg.seed_model_paths[int(parent_indices[0])]
        parent2_path = cfg.seed_model_paths[int(parent_indices[1])]

    # Check if mutation is disabled
    do_mutate = not cfg.dns.get("disable_mutation", False)

    return call_fn.delay(
        "merge_models",
        parent_paths=[parent1_path, parent2_path],
        save_path=f"{model_dir}/gen_{gen}_ind_{model_index}",
        task_info=task_info,  # Pass task_info to worker
        do_mutate=do_mutate,  # Pass mutation flag to worker
    )


def create_merge_only_task(
    cfg: DictConfig,
    call_fn: Any,
    gen: int,
    model_index: int,
    archive_data: ACDCArchiveData,
    np_random: np.random.RandomState,
) -> Any:
    """Create merge-only task without evaluation."""
    model_dir = archive_data["dirs"]["model_dir"]

    dns_archive = archive_data["dns_archive"]
    if len(dns_archive) >= 2:
        parent_indices = np_random.choice(
            len(dns_archive), size=2, replace=False
        )
        parent1 = dns_archive[int(parent_indices[0])]
        parent2 = dns_archive[int(parent_indices[1])]
        parent1_path = parent1.model_path
        parent2_path = parent2.model_path
    else:
        parent_indices = np_random.choice(
            len(cfg.seed_model_paths), size=2, replace=False
        )
        parent1_path = cfg.seed_model_paths[int(parent_indices[0])]
        parent2_path = cfg.seed_model_paths[int(parent_indices[1])]

    # Check if mutation is disabled
    do_mutate = not cfg.dns.get("disable_mutation", False)

    return call_fn.delay(
        "merge_models_only",
        parent_paths=[parent1_path, parent2_path],
        save_path=f"{model_dir}/gen_{gen}_ind_{model_index}",
        do_mutate=do_mutate,  # Pass mutation flag to worker
    )


def create_eval_only_task(
    call_fn: Any,
    model_path: str,
    task_info: Union[Dict[str, DictConfig], List[str]],
    data_split: str = "train",
    task_name: Optional[str] = None,
) -> Any:
    """Create evaluation-only task for a pre-saved model."""
    return call_fn.delay(
        "eval_model_only",
        model_path=model_path,
        task_info=task_info,
        data_split=data_split,
        task_name=task_name,
    )


def wait_for_promises(promises: List[Any], timeout: float) -> List[Any]:
    """Wait for all promises to complete and return results."""
    results = []
    promise_q = deque(promises)

    while promise_q:
        promise = promise_q.popleft()
        if promise.ready():
            results.append(promise.get(timeout=timeout))
        else:
            promise_q.append(promise)
            time.sleep(0.05)

    return results


def cleanup_old_models(
    gen: int,
    archive_data: ACDCArchiveData,
    skip_interval: Optional[int] = None,
) -> List[str]:
    """Clean up old model files based on optimization mode, with optional skip interval for deletion."""
    model_dir = archive_data["dirs"]["model_dir"]
    model_paths = [sol.model_path for sol in archive_data["dns_archive"]]
    return delete_models_not_in_archive(
        model_dir=model_dir,
        keep_model_paths=model_paths,
        threshold=gen,
        skip_interval=skip_interval,
    )


def calculate_fitness_from_skill_vector(
    acdc_skill_vector: Optional[Dict[str, float]],
) -> float:
    """Calculates fitness as the average score across all tasks in the skill vector."""
    if not acdc_skill_vector:
        return 0.0

    scores = list(acdc_skill_vector.values())
    if not scores:
        return 0.0

    return sum(scores) / len(scores)


class ACDCOptimizer:
    """
    Manages the AC/DC (Assessment Coevolving with Diverse Capabilities)
    optimization process.

    Handles environment setup, worker coordination, population initialization,
    evolutionary loop (merging, evaluation, archive update, task adaptation),
    validation, and cleanup.
    """

    def __init__(self, celery: Celery, cfg: DictConfig):
        """Initializes the ACDCOptimizer."""
        self.celery: Celery = celery
        self.cfg: DictConfig = cfg
        self.call_fn = celery.tasks["call"]
        self.validation_tasks_names: List[str] = getattr(
            cfg, "validation_tasks", []
        )
        self.np_random: np.random.RandomState = np.random.RandomState(
            cfg.seed if cfg.seed > 0 else 42
        )
        self.logger = logging.getLogger("ACDCOptimizer")
        self.tasks: List[ACDCTask] = (
            []
        )  # Populated in setup. TODO: disambiguate self.tasks (objects) and self.task_pool.tasks (paths)
        self.task_pool: Optional[ACDCTaskPool] = None
        self.dirs: Optional[Dict[str, str]] = None
        self.gen: int = 1
        self.gibberish_models_counter: int = 0
        self.gen_0_seed_model_names: Set[str] = set(
            [
                f"gen_0_ind_{model_name.split('/')[-1]}"
                for model_name in self.cfg.seed_model_paths
            ]
        )

    def _load_or_generate_tasks(self) -> None:
        """Loads or generates tasks based on configuration (AC/DC or standard)."""
        if self.cfg.get("use_ac_dc", False) and self.cfg.get("acdc"):
            self.logger.info("Using AC/DC Task Generation.")
            try:
                if self.dirs is None or "generated_tasks_dir" not in self.dirs:
                    raise ValueError(
                        "Directories not set up before task pool initialization."
                    )
                # Pass the dynamically determined path to the constructor
                self.task_pool = ACDCTaskPool(
                    self.cfg,
                    self.dirs["generated_tasks_dir"],
                    self.dirs["vector_db_dir"],
                )

                # Load or generate initial tasks
                if self.cfg.get("restart_dir"):
                    self.logger.info("Loading existing AC/DC tasks...")
                    self.task_pool.load_existing_tasks()
                    self.logger.info(
                        f"Loaded {len(self.task_pool.tasks)} existing AC/DC tasks."
                    )
                else:
                    self.logger.info("Generating initial AC/DC task pool...")
                    self.task_pool.initialize_pool()
                    self.logger.info(
                        f"Generated {len(self.task_pool.tasks)} initial AC/DC tasks."
                    )
                    # For task pool logging, temporarily change gen number to 0, then revert back
                    self.gen = 0
                    self._log_active_task_pool_state()
                    self.gen = 1

                # Get task objects from the pool
                self.tasks = self.task_pool.get_tasks()
                if not self.tasks and not self.cfg.get("restart_dir"):
                    self.logger.warning(
                        "AC/DC Task Pool generated/loaded zero tasks. Check configuration and generation process."
                    )
                    # Decide how to handle this - maybe fall back or raise error?
                    # For now, let it proceed, but downstream might fail.

            except Exception as e:
                self.logger.exception(
                    f"Failed to initialize or load AC/DC Task Pool: {e}"
                )
                raise  # Re-raise exception to halt execution if task pool fails critically
        else:
            raise NotImplementedError(
                "Standard task loading (non-AC/DC) is not implemented yet."
            )
        self.logger.info(f"Loaded {len(self.tasks)} tasks for coordination.")

    def setup_environment(self) -> None:
        """Initializes directories, archives, tasks, and generation counter."""
        self.dirs = setup_optimization_directories(self.cfg)
        self.archive_data: ACDCArchiveData = {
            # "archive_map": None,
            "dns_archive": [],
            "dirs": self.dirs,
        }

        # Set generation counter
        if self.dirs is None:
            raise ValueError(
                "Directories not initialized in setup_environment."
            )
        model_dir = self.dirs.get("model_dir")
        if model_dir is None:
            raise ValueError(
                "Model directory not configured in setup_environment."
            )
        self.gen = (
            get_latest_generation(model_dir) + 1
            if self.cfg.get("restart_dir")
            else 1
        )

        # Load existing archives if restarting
        if self.cfg.get("restart_dir"):
            archive_dir = self.dirs.get("archive_dir")
            if archive_dir is None:
                raise ValueError(
                    "Archive directory not configured in setup_environment."
                )

            # Find the best available archive file for the latest generation
            latest_gen = self.gen - 1
            archive_path = None

            # Priority 1: Try post_adapt_filtered archive first
            post_adapt_path = f"{archive_dir}/gen{latest_gen}_dns_archive_post_adapt_filtered.json"
            if os.path.exists(post_adapt_path):
                archive_path = post_adapt_path
                self.logger.info(
                    f"Found post-adaptation filtered archive: {archive_path}"
                )
            else:
                # Priority 2: Fall back to regular archive
                regular_path = f"{archive_dir}/gen{latest_gen}_dns_archive.json"
                if os.path.exists(regular_path):
                    archive_path = regular_path
                    self.logger.info(f"Found regular archive: {archive_path}")

            if archive_path:
                try:
                    # Use the AC/DC-specific load function
                    loaded_archive = load_ac_dc_archive(archive_path)
                    if self.archive_data is not None:  # Type check
                        self.archive_data["dns_archive"] = loaded_archive
                        self.logger.info(
                            f"Successfully loaded {len(loaded_archive)} solutions from {archive_path}"
                        )
                    else:
                        self.logger.error(
                            "Archive data dictionary is None after initialization."
                        )
                        raise RuntimeError(
                            "Archive data not properly initialized"
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error loading archive {archive_path}: {e}"
                    )
                    raise
            else:
                self.logger.warning(
                    f"Restart specified, but no archive files found for generation {latest_gen}. Starting fresh."
                )
                if self.archive_data is not None:
                    self.archive_data["dns_archive"] = []

    def setup_workers(self) -> None:
        """Set up worker logging."""
        if self.dirs is None:
            raise ValueError(
                "Directories not initialized before setting up workers."
            )
        self.logger.info(
            f"Setting up worker logging with output directory: {self.dirs['output_dir']}"
        )
        setup_promises = []
        for _ in range(self.cfg.celery.num_workers):
            promise = self.call_fn.delay(
                "setup_worker", output_dir=self.dirs["output_dir"]
            )
            time.sleep(30)
            setup_promises.append(promise)

        wait_for_promises(setup_promises, self.cfg.celery.timeout)

    def _determine_task_info_for_initialization(
        self,
    ) -> Union[List[str], Dict[str, DictConfig]]:
        """Determines the task information needed for initializing models."""
        if self.cfg.get("use_ac_dc", False) and self.task_pool:
            task_info = self.task_pool.tasks  # List of AC/DC task directories
            self.logger.info(
                f"Passing {len(task_info)} AC/DC task directories for initialization."
            )
            return task_info
        else:
            # task_info = self.task_configs # Dict of standard task configs
            # self.logger.info(f"Passing {len(task_info)} standard task configs for initialization.")
            raise NotImplementedError(
                "Standard task initialization (non-AC/DC) is not implemented yet."
            )

    def _create_initialization_promises(
        self,
        task_info: Union[List[str], Dict[str, DictConfig]],
        n_promises_to_create: int,
        used_worker_indices: List[int] = [],
    ) -> List[Any]:
        """Creates Celery promises for initializing the population."""
        if self.dirs is None or "model_dir" not in self.dirs:
            raise ValueError(
                "Model directory not set before creating init promises."
            )
        model_dir = self.dirs["model_dir"]
        init_promises = []

        # Check if mutation is disabled
        do_mutate = not self.cfg.dns.get("disable_mutation", False)

        for worker_idx in range(n_promises_to_create):
            while worker_idx in used_worker_indices:
                worker_idx += 1
            used_worker_indices.append(worker_idx)
            model_path = f"{model_dir}/gen_0_ind_{worker_idx}"

            # Select 2 different seed models for crossover
            if len(self.cfg.seed_model_paths) < 2:
                self.logger.info(
                    f"Only {len(self.cfg.seed_model_paths)} seed models provided, using the same model for crossover"
                )
                seed_model_paths = [self.cfg.seed_model_paths[0]] * 2
            else:
                seed_indices = self.np_random.choice(
                    len(self.cfg.seed_model_paths), size=2, replace=False
                )
                seed_model_paths = [
                    self.cfg.seed_model_paths[int(idx)] for idx in seed_indices
                ]

            init_promises.append(
                self.call_fn.delay(
                    "initialize_model",
                    seed_model_paths=seed_model_paths,
                    save_path=model_path,
                    seed=self.np_random.randint(100000),
                    task_info=task_info,
                    do_mutate=do_mutate,  # Pass mutation flag
                )
            )
        return init_promises

    def _create_initialization_promises_with_seed_models(
        self,
        task_info: Union[List[str], Dict[str, DictConfig]],
    ) -> List[Any]:
        """Creates Celery promises for initializing the population."""
        if self.dirs is None or "model_dir" not in self.dirs:
            raise ValueError(
                "Model directory not set before creating init promises."
            )
        model_dir = self.dirs["model_dir"]
        init_promises = []
        for seed_model_path in self.cfg.seed_model_paths:
            model_name = seed_model_path.split("/")[-1]
            save_path = f"{model_dir}/gen_0_ind_{model_name}"

            init_promises.append(
                self.call_fn.delay(
                    "initialize_model",
                    seed_model_paths=[seed_model_path],
                    save_path=save_path,
                    seed=self.np_random.randint(100000),
                    task_info=task_info,
                    do_mutate=False,
                )
            )
        return init_promises

    def _process_initialization_results(
        self, init_promises: List
    ) -> List[ACDCSolution]:
        """Waits for initialization promises and processes the results."""
        init_results = wait_for_promises(init_promises, self.cfg.celery.timeout)
        initial_acdc_solutions = []
        for result in init_results:
            if result:
                if (
                    self.cfg.dns.run_gibberish_check
                    and result.is_gibberish
                    and not self.is_seed_model(result.save_path)
                ):
                    self.logger.info(
                        f"Model {result.save_path} returns gibberish, skipping..."
                    )
                    self.gibberish_models_counter += 1
                    continue
                # Create ACDCSolution using the updated function signature
                solution = create_ac_dc_solution(
                    model_path=result.save_path,
                    task_metrics=result.task_metrics,
                    acdc_skill_vector=result.acdc_skill_vector,
                    avg_acdc_quality=result.avg_acdc_quality,
                    acdc_eval_details=getattr(
                        result, "acdc_eval_details", None
                    ),  # Pass eval details if present
                )
                initial_acdc_solutions.append(solution)
            else:
                self.logger.error(
                    f"Initialization result {result} is not a valid model, skipping..."
                )
        return initial_acdc_solutions

    def _convert_and_update_initial_archive(
        self, initial_acdc_solutions: List[ACDCSolution]
    ) -> List[ACDCSolution]:
        """Converts initial solutions and updates the DNS archive."""
        if not initial_acdc_solutions:
            return []

        if not self.task_pool:
            raise RuntimeError(
                "Task pool not initialized before population initialization."
            )
        ordered_task_ids = self.task_pool.get_ordered_task_ids()
        threshold = self.cfg.dns.acdc_skill_threshold

        # Keep original AC/DC solutions mapped by path
        acdc_solution_map = {s.model_path: s for s in initial_acdc_solutions}

        # Convert to DNSSolution for update_dns_archive
        converted_initial_solutions = [
            convert_acdc_to_dns_solution(sol, ordered_task_ids, threshold)
            for sol in initial_acdc_solutions
        ]

        # Call the refactored update_dns_archive with the dns sub-config
        surviving_dns_solutions = update_dns_archive(
            [], converted_initial_solutions, self.cfg.dns
        )

        # Convert back to ACDCSolution
        updated_archive = [
            acdc_solution_map[dns_sol.model_path]
            for dns_sol in surviving_dns_solutions
            if dns_sol.model_path in acdc_solution_map
        ]
        return updated_archive

    def _save_initial_archive(self, archive: List[ACDCSolution]) -> None:
        """Saves the initial state of the DNS archive."""
        if not self.dirs or "archive_dir" not in self.dirs:
            raise ValueError(
                "Archive directory not set before saving initial archive."
            )
        archive_dir = self.dirs["archive_dir"]
        archive_path = f"{archive_dir}/gen0_dns_archive.json"
        save_ac_dc_archive(
            archive,
            archive_path,
            max_details_to_log=self.cfg.dns.max_details_to_log,
        )
        self.logger.info(
            f"Saved initial DNS archive with {len(archive)} models to {archive_path}"
        )

    def _create_init_only_task(
        self,
        seed_model_paths: List[str],
        save_path: str,
        seed: int,
        do_mutate: Optional[bool] = None,
    ):
        """Create initialization-only task without evaluation."""
        # Check if mutation is disabled via config if not explicitly set
        if do_mutate is None:
            do_mutate = not self.cfg.dns.get("disable_mutation", False)

        return self.call_fn.delay(
            "initialize_model_only",
            seed_model_paths=seed_model_paths,
            save_path=save_path,
            seed=seed,
            do_mutate=do_mutate,
        )

    def _process_initialization_in_batches(
        self, num_models: int, batch_size: int = 2, delay: int = 50
    ) -> List[str]:
        """Create and process initialization tasks in batches to limit concurrent disk I/O."""
        self.logger.info(
            f"Gen 0: Processing {num_models} initializations in batches of {batch_size}..."
        )
        all_saved_paths = []
        used_worker_indices = []
        if self.dirs is None:
            raise ValueError(
                "Directories not initialized before creating merge batches."
            )
        model_dir = self.dirs["model_dir"]

        # Process initializations in batches
        for i in range(0, num_models, batch_size):
            batch_size_actual = min(batch_size, num_models - i)
            self.logger.info(
                f"Processing initialization batch {i//batch_size + 1}: {batch_size_actual} models"
            )

            # Create tasks for this batch
            batch_promises = []
            for j in range(batch_size_actual):
                worker_idx = i + j
                while worker_idx in used_worker_indices:
                    worker_idx += 1
                used_worker_indices.append(worker_idx)
                model_path = f"{model_dir}/gen_0_ind_{worker_idx}"

                # Select seed models for crossover
                if len(self.cfg.seed_model_paths) < 2:
                    seed_model_paths = [self.cfg.seed_model_paths[0]] * 2
                else:
                    seed_indices = self.np_random.choice(
                        len(self.cfg.seed_model_paths), size=2, replace=False
                    )
                    seed_model_paths = [
                        self.cfg.seed_model_paths[int(idx)]
                        for idx in seed_indices
                    ]

                # Check if mutation is disabled
                do_mutate = not self.cfg.dns.get("disable_mutation", False)

                promise = self._create_init_only_task(
                    seed_model_paths=seed_model_paths,
                    save_path=model_path,
                    seed=self.np_random.randint(100000),
                    do_mutate=do_mutate,
                )
                batch_promises.append(promise)
                self.logger.debug(
                    f"Submitted initialization for model index {worker_idx}"
                )
                # Add staggered sleep timer to ensure disk I/O load doesn't happen for all workers at same time
                time.sleep(delay)

            # Wait for this batch to complete before starting next batch
            batch_results = wait_for_promises(
                batch_promises, self.cfg.celery.timeout
            )

            # Process batch results
            for result in batch_results:
                if result:  # result is the save_path string
                    all_saved_paths.append(result)
                    self.logger.info(f"Initialization complete: {result}")
                else:
                    self.logger.error(
                        "Initialization operation failed, no model path returned"
                    )

            self.logger.info(
                f"Batch {i//batch_size + 1} complete. Total initializations done: {len(all_saved_paths)}/{num_models}"
            )

        return all_saved_paths

    def is_seed_model(self, model_path: str) -> bool:
        """Check if a model path is a seed model."""
        return model_path.split("/")[-1] in self.gen_0_seed_model_names

    def initialize_population_phase1_async(self) -> Tuple[List[Any], List[str]]:
        """Phase 1: Start async initialization of models without waiting.

        Returns:
            Tuple of (promises, expected_paths) where:
            - promises: List of Celery promises for model initialization
            - expected_paths: List of expected save paths for the models
        """
        if self.archive_data is None:
            raise RuntimeError(
                "Archive data not initialized before population initialization."
            )

        # Check if archive is already loaded (from restart)
        if self.cfg.get("restart_dir") and self.archive_data.get("dns_archive"):
            self.logger.info(
                f"Restarting run, loaded {len(self.archive_data['dns_archive'])} models from archive."
            )
            return [], []  # Skip initialization

        self.logger.info(
            f"Initializing archive with seed models: {self.cfg.seed_model_paths}"
        )
        self.logger.info(
            "Generation 0: Phase 1 - Starting async model initialization..."
        )

        all_promises = []
        expected_paths = []

        if self.dirs is None:
            raise ValueError("Directories not initialized.")
        model_dir = self.dirs["model_dir"]

        # First process seed models without mutation
        if self.cfg.dns.init_population_with_seed_models:
            self.logger.info(
                f"Submitting {len(self.cfg.seed_model_paths)} seed models for initialization without mutation..."
            )
            for seed_model_path in self.cfg.seed_model_paths:
                model_name = seed_model_path.split("/")[-1]
                save_path = f"{model_dir}/gen_0_ind_{model_name}"

                promise = self._create_init_only_task(
                    seed_model_paths=[seed_model_path],
                    save_path=save_path,
                    seed=self.np_random.randint(100000),
                    do_mutate=False,
                )
                all_promises.append(promise)
                expected_paths.append(save_path)

        # Calculate how many more models we need
        remaining_models = self.cfg.dns.init_population_size - len(all_promises)

        # Submit additional models with crossover/mutation
        if remaining_models > 0:
            # Create extra models to account for potential failures
            extra_models = max(
                remaining_models,
                self.cfg.dns.get("n_min_init_pop_promises", 5),
            )

            self.logger.info(
                f"Submitting {extra_models} additional models for initialization with crossover/mutation..."
            )

            for i in range(extra_models):
                worker_idx = len(expected_paths)
                model_path = f"{model_dir}/gen_0_ind_{worker_idx}"

                # Select seed models for crossover
                if len(self.cfg.seed_model_paths) < 2:
                    seed_model_paths = [self.cfg.seed_model_paths[0]] * 2
                else:
                    seed_indices = self.np_random.choice(
                        len(self.cfg.seed_model_paths), size=2, replace=False
                    )
                    seed_model_paths = [
                        self.cfg.seed_model_paths[int(idx)]
                        for idx in seed_indices
                    ]

                # Check if mutation is disabled
                do_mutate = not self.cfg.dns.get("disable_mutation", False)

                promise = self._create_init_only_task(
                    seed_model_paths=seed_model_paths,
                    save_path=model_path,
                    seed=self.np_random.randint(100000),
                    do_mutate=do_mutate,
                )
                all_promises.append(promise)
                expected_paths.append(model_path)

                # Add small delay between submissions to avoid overwhelming the queue
                if i < extra_models - 1:
                    time.sleep(120)

        self.logger.info(
            f"Submitted {len(all_promises)} model initialization tasks asynchronously."
        )
        return all_promises, expected_paths

    def wait_for_phase1_and_get_paths(
        self, promises: List[Any], expected_paths: List[str]
    ) -> List[str]:
        """Wait for Phase 1 promises to complete and return successful model paths.

        Args:
            promises: List of Celery promises from Phase 1
            expected_paths: List of expected model paths

        Returns:
            List of successfully saved model paths
        """
        if not promises:
            return []  # Restart case

        self.logger.info(
            f"Waiting for {len(promises)} Phase 1 initialization tasks to complete..."
        )

        # Wait for all promises with extended timeout
        results = wait_for_promises(promises, self.cfg.celery.timeout * 2)

        # Collect successful paths
        successful_paths = []
        for i, result in enumerate(results):
            if result:  # result is the save_path string
                successful_paths.append(result)
                self.logger.info(f"Model initialized successfully: {result}")
            else:
                self.logger.error(
                    f"Failed to initialize model at {expected_paths[i]}"
                )

        # Ensure we have enough models
        if len(successful_paths) < self.cfg.dns.init_population_size:
            self.logger.warning(
                f"Only {len(successful_paths)}/{self.cfg.dns.init_population_size} models initialized successfully. "
                f"Proceeding with available models."
            )

        # Return up to the desired population size
        return successful_paths[: self.cfg.dns.init_population_size]

    def initialize_population_phase2(self, saved_paths: List[str]) -> None:
        """Phase 2: Evaluate saved models and update archive.

        Args:
            saved_paths: List of model paths to evaluate.
        """
        if not saved_paths:
            return  # Nothing to evaluate (restart case)

        if not hasattr(self, "tasks") or not self.tasks:
            raise RuntimeError(
                "Tasks not loaded before Phase 2 evaluation. Ensure task loading is complete."
            )

        task_info = self._determine_task_info_for_initialization()
        start_time = time.time()

        # Phase 2: Evaluate saved models with staggered loading
        self.logger.info(
            f"Generation 0: Phase 2 - Evaluating {len(saved_paths)} initialized models..."
        )
        initial_acdc_solutions = self._evaluate_saved_models_staggered(
            model_paths=saved_paths,
            task_info=task_info,
            batch_size=2,  # Max 2 concurrent evaluations
            stagger_delay=30.0,  # 30 seconds between evaluation batches
        )

        # Filter out None results and gibberish models
        valid_solutions = []
        for solution in initial_acdc_solutions:
            if solution:
                if (
                    self.cfg.dns.run_gibberish_check
                    and solution.is_gibberish
                    and not self.is_seed_model(solution.model_path)
                ):
                    self.logger.info(
                        f"Model {solution.model_path} returns gibberish, skipping..."
                    )
                    continue
                valid_solutions.append(solution)

        end_time = time.time()
        self.logger.info(
            f"Time taken to evaluate {len(valid_solutions)} valid models: {round((end_time - start_time) / 60, 2)} minutes"
        )

        updated_archive = self._convert_and_update_initial_archive(
            valid_solutions
        )

        self.archive_data["dns_archive"] = updated_archive
        self._save_initial_archive(updated_archive)

    def process_merge_results(
        self,
        merge_promises: List,
        task_info: Union[List[str], Dict[str, DictConfig]],
    ) -> List[Optional[ACDCSolution]]:
        """
        Processes merge results from Celery promises, handling retries for failures.

        Args:
            merge_promises: A list of Celery AsyncResult objects for merge tasks.
            task_info: Task information (list of paths or dict of configs) used for retries.

        Returns:
            A list of ACDCSolution objects (or None for failures after retries).
        """
        # Use a dictionary to track original index for logging/retry purposes if needed
        promise_map = {p.id: (p, i) for i, p in enumerate(merge_promises)}
        completed_merges = 0
        models_per_gen = self.cfg.dns.get("num_model_per_gen")
        total_merges = len(merge_promises)
        new_solutions: List[Optional[ACDCSolution]] = [
            None
        ] * models_per_gen  # Pre-allocate list
        wandb_alerted = False
        retries_attempted: Dict[str, int] = (
            {}
        )  # Track retries per original promise ID

        self.logger.info(
            f"Waiting for {total_merges} merge operations to complete..."
        )

        while promise_map:
            processed_ids = []
            retries_to_add = []  # Collect retries to add after iteration
            for promise_id, (promise, original_index) in promise_map.items():
                try:
                    if promise.ready():
                        processed_ids.append(promise_id)
                        result = promise.get(timeout=self.cfg.celery.timeout)
                        if result and completed_merges < models_per_gen:
                            solution = handle_merge_result(result)
                            new_solutions[original_index] = (
                                solution  # Place in correct slot
                            )
                            completed_merges += 1
                            self.logger.info(
                                f"Merge {completed_merges}/{total_merges} (Original Index: {original_index}, Task ID: {promise.id}) complete."
                            )
                        elif completed_merges < models_per_gen:
                            # Handle failure, potentially retry
                            wandb_alerted, retry_entry = self._handle_failed_merge(
                                promise,
                                original_index,
                                promise_map,  # Pass the map for reference
                                retries_attempted,
                                wandb_alerted,
                                task_info,
                            )
                            if retry_entry:
                                retries_to_add.append(retry_entry)
                        # If we have generated enough models, remove all promises
                        elif completed_merges >= models_per_gen:
                            for pid, (
                                promise,
                                original_index,
                            ) in promise_map.items():
                                # Remaining promise_id due for deletion
                                processed_ids.append(promise_id)
                    elif completed_merges >= models_per_gen:
                        for pid, (
                            promise,
                            original_index,
                        ) in promise_map.items():
                            # Remaining promise_id due for deletion
                            processed_ids.append(promise_id)

                # If not ready, do nothing, will check again in next loop iteration
                except Exception as e:
                    self.logger.error(
                        f"Error processing merge promise {promise.id} (Original Index: {original_index}): {e}"
                    )
                    # Decide on error handling: retry, mark as failed, etc.
                    # For now, let's retry similar to a failed result
                    processed_ids.append(promise_id)  # Remove from map to retry
                    wandb_alerted, retry_entry = self._handle_failed_merge(
                        promise,
                        original_index,
                        promise_map,
                        retries_attempted,
                        wandb_alerted,
                        task_info,
                        is_exception=True,  # Indicate it was an exception
                    )
                    if retry_entry:
                        retries_to_add.append(retry_entry)

            # Remove processed promises from the map
            for pid in processed_ids:
                if promise_map and pid in promise_map:
                    del promise_map[pid]

            # Add retries after iteration completes to avoid dict mutation during iteration
            for retry_id, retry_data in retries_to_add:
                promise_map[retry_id] = retry_data

            if promise_map:  # Only sleep if there are still pending promises
                time.sleep(0.1)  # Slightly longer sleep

        self.logger.info(
            f"Finished processing {completed_merges}/{total_merges} merge operations to get {len(new_solutions)} models."
        )
        return new_solutions

    def _handle_failed_merge(
        self,
        failed_promise,
        original_index: int,
        promise_map: Dict[str, tuple],
        retries_attempted: Dict[str, int],
        wandb_alerted: bool,
        task_info: Union[List[str], Dict[str, DictConfig]],
        is_exception: bool = False,
        max_retries: int = 1,  # Configure max retries if needed
    ) -> Tuple[bool, Optional[tuple]]:
        """Handles a failed merge operation, logs, alerts, and potentially retries.

        Returns:
            Tuple of (wandb_alerted, retry_entry) where retry_entry is (promise_id, (promise, original_index))
            to be added to promise_map after iteration, or None if no retry.
        """
        retry_count = retries_attempted.get(failed_promise.id, 0)
        log_prefix = (
            "Exception occurred in" if is_exception else "Failed merge task"
        )

        if retry_count < max_retries:
            retries_attempted[failed_promise.id] = retry_count + 1
            self.logger.error(
                f"{log_prefix} {failed_promise.id} (Original Index: {original_index}). Attempting retry {retry_count + 1}/{max_retries}..."
            )

            # Create a new merge task for retry
            # Note: Using original_index for the *new* model path might overwrite if not careful.
            # Consider a different naming scheme for retries or ensure cleanup handles it.
            # For now, assume overwriting is acceptable or handled elsewhere.
            new_promise = create_merge_task(
                self.cfg,
                self.call_fn,
                self.gen,
                original_index,  # Use original index for consistency? Or a new one?
                self.archive_data,
                self.np_random,
                task_info=task_info,
            )
            # Return the promise to be added after iteration completes
            self.logger.info(
                f"Submitted retry task {new_promise.id} for original index {original_index}."
            )

            if not wandb_alerted:
                wandb.alert(  # type: ignore
                    title="Merge Task Failed (Retrying)",
                    text=f"{log_prefix} {failed_promise.id} (Original Index: {original_index}). Retrying ({retry_count + 1}/{max_retries})...",
                    level=AlertLevel.WARN,
                )
                wandb_alerted = (
                    True  # Alert only once per generation cycle perhaps
                )
            time.sleep(10)  # Shorter sleep for retry submission
            return wandb_alerted, (new_promise.id, (new_promise, original_index))
        else:
            self.logger.error(
                f"{log_prefix} {failed_promise.id} (Original Index: {original_index}). Max retries ({max_retries}) reached. Marking as failed."
            )
            # Solution at new_solutions[original_index] will remain None
            if (
                not wandb_alerted
            ):  # Alert if max retries hit and no previous alert sent
                wandb.alert(  # type: ignore
                    title="Merge Task Failed (Max Retries)",
                    text=f"{log_prefix} {failed_promise.id} (Original Index: {original_index}). Max retries reached. Giving up.",
                    level=AlertLevel.ERROR,
                )
                wandb_alerted = True

        return wandb_alerted, None

    def _evaluate_saved_models_staggered(
        self,
        model_paths: List[str],
        task_info: Union[List[str], Dict[str, DictConfig]],
        batch_size: int = 2,
        stagger_delay: float = 30.0,
    ) -> List[Optional[ACDCSolution]]:
        """Evaluate saved models with staggered job submission and return results."""
        self.logger.info(
            f"Gen {self.gen}: Evaluating {len(model_paths)} models with batch size {batch_size} and {stagger_delay}s delay..."
        )
        all_eval_promises = []

        # Submit evaluations in batches with delays between batches
        for i in range(0, len(model_paths), batch_size):
            if i > 0:
                self.logger.info(
                    f"Waiting {stagger_delay}s before submitting next evaluation batch..."
                )
                time.sleep(stagger_delay)

            batch_paths = model_paths[i : i + batch_size]
            self.logger.info(
                f"Submitting evaluation batch {i//batch_size + 1}: {len(batch_paths)} evaluations"
            )

            # Create tasks for this batch
            for model_path in batch_paths:
                promise = create_eval_only_task(
                    self.call_fn,
                    model_path,
                    task_info,
                )
                all_eval_promises.append(promise)
                self.logger.debug(f"Submitted evaluation for {model_path}")

        # Wait for all evaluations to complete
        self.logger.info("Waiting for all evaluation tasks to complete...")
        eval_results = wait_for_promises(
            all_eval_promises, self.cfg.celery.timeout
        )

        # Process results
        valid_results = []
        for i, result in enumerate(eval_results):
            if result:
                valid_results.append(handle_merge_result(result))
                self.logger.info(
                    f"Evaluation {i+1}/{len(eval_results)} complete"
                )
            else:
                valid_results.append(None)
                self.logger.error(
                    f"Evaluation {i+1}/{len(eval_results)} failed"
                )

        self.logger.info(
            f"Completed {len([r for r in valid_results if r])}/{len(model_paths)} evaluations successfully"
        )
        return valid_results

    def _cleanup_generation_resources(self) -> None:
        """Performs cleanup operations like removing old models and stopping containers."""
        # --- Disk Cleaning (New Logic) ---
        if self.gen > 0 and self.gen % self.cfg.disk_cleaning_interval == 0:
            self.logger.info(
                f"Generation {self.gen}: Cleaning up old models..."
            )
            deleted_models_count = 0
            try:
                if (
                    not self.dirs
                    or "model_dir" not in self.dirs
                    or "archive_dir" not in self.dirs
                ):
                    raise ValueError("Model or Archive directory not set.")
                if (
                    self.archive_data is None
                    or "dns_archive" not in self.archive_data
                ):
                    raise ValueError("Archive data not available for cleanup.")

                model_dir = self.dirs["model_dir"]
                archive_dir = self.dirs["archive_dir"]
                skip_interval = getattr(
                    self.cfg, "model_cleanup_skip_interval", None
                )

                # 1. Get model paths from the current archive
                keep_model_paths = set(
                    sol.model_path
                    for sol in self.archive_data["dns_archive"]
                    if sol and sol.model_path
                )
                self.logger.debug(
                    f"Keeping {len(keep_model_paths)} models from current archive."
                )

                # 2. Get model paths from interval archives
                if skip_interval and skip_interval > 0:
                    self.logger.info(
                        f"Additionally keeping models from archives at intervals of {skip_interval} generations."
                    )
                    archive_file_pattern = re.compile(
                        r"gen(\d+)_dns_archive.*\.json"
                    )  # Allow variations like post_adapt
                    try:
                        for filename in os.listdir(archive_dir):
                            match = archive_file_pattern.match(filename)
                            if match:
                                archive_gen = int(match.group(1))
                                if (
                                    archive_gen > 0
                                    or (
                                        self.cfg.save_init_gen_models
                                        and archive_gen == 0
                                    )
                                ) and archive_gen % skip_interval == 0:
                                    archive_path = os.path.join(
                                        archive_dir, filename
                                    )
                                    self.logger.debug(
                                        f"Loading interval archive: {archive_path}"
                                    )
                                    try:
                                        interval_archive = load_ac_dc_archive(
                                            archive_path
                                        )
                                        interval_paths = set(
                                            sol.model_path
                                            for sol in interval_archive
                                            if sol and sol.model_path
                                        )
                                        new_paths_found = len(
                                            interval_paths - keep_model_paths
                                        )
                                        if new_paths_found > 0:
                                            self.logger.debug(
                                                f"Adding {new_paths_found} paths from {filename} to keep list."
                                            )
                                        keep_model_paths.update(interval_paths)
                                    except FileNotFoundError:
                                        self.logger.warning(
                                            f"Interval archive file not found during cleanup: {archive_path}"
                                        )
                                    except Exception as load_err:
                                        self.logger.error(
                                            f"Error loading interval archive {archive_path}: {load_err}"
                                        )
                    except FileNotFoundError:
                        self.logger.warning(
                            f"Archive directory not found during interval check: {archive_dir}"
                        )
                    except Exception as list_err:
                        self.logger.error(
                            f"Error listing archive directory {archive_dir}: {list_err}"
                        )

                self.logger.info(
                    f"Total unique model paths to keep: {len(keep_model_paths)}"
                )

                # 3. Iterate through model directory and delete models not in the keep set
                if not os.path.exists(model_dir):
                    self.logger.warning(
                        f"Model directory {model_dir} does not exist. Skipping deletion."
                    )
                else:
                    for item_name in os.listdir(model_dir):
                        item_path = os.path.join(model_dir, item_name)
                        if os.path.isdir(
                            item_path
                        ):  # Check if it's a directory
                            if item_path not in keep_model_paths:
                                try:
                                    shutil.rmtree(item_path)
                                    self.logger.info(
                                        f"Deleted model directory: {item_path}"
                                    )
                                    deleted_models_count += 1
                                except Exception as delete_err:
                                    self.logger.error(
                                        f"Error deleting directory {item_path}: {delete_err}"
                                    )

                self.logger.info(
                    f"Disk cleanup complete for generation {self.gen}. Deleted {deleted_models_count} model directories."
                )

            except Exception as e:
                self.logger.exception(  # Use exception to log traceback
                    f"Error during disk cleanup for generation {self.gen}: {e}"
                )

        # --- Docker Cleanup (AgentBench OS specific - Kept as is) ---
        # Consider making this conditional or configurable if not always needed
        self.logger.info(
            f"Generation {self.gen}: Cleaning up old Docker containers..."
        )
        try:
            # Simpler os.system call (less robust error handling)
            os.system(
                "docker stop $(docker ps -q --filter ancestor=local-os/default --format '{{.ID}} {{.RunningFor}}' | grep -E '2 hours|[3-9] hours|[0-9]+ days' | awk '{print $1}') > /dev/null 2>&1"
            )
            self.logger.info(
                f"Docker cleanup command executed for generation {self.gen}."
            )

        except Exception as e:
            self.logger.error(
                f"Error during Docker cleanup for generation {self.gen}: {e}"
            )

    def _determine_task_info_for_generation(
        self,
    ) -> Union[List[str], Dict[str, DictConfig]]:
        """Determines the task information for the current generation's merges."""
        if self.cfg.get("use_ac_dc", False) and self.task_pool:
            # TODO: Implement task adaptation logic here if needed for selection
            # For now, use the current tasks in the pool
            task_info = self.task_pool.tasks  # Pass list of dirs directly
            self.logger.info(
                f"Gen {self.gen}: Using {len(task_info)} AC/DC tasks for merges."
            )
            return task_info
        else:
            # task_info = self.task_configs  # Dict of standard task configs
            # self.logger.info(f"Gen {self.gen}: Using {len(task_info)} standard tasks.")
            raise NotImplementedError(
                "Standard task merging (non-AC/DC) is not implemented yet."
            )

    def _process_merges_in_batches(
        self, num_models: int, batch_size: int = 2, delay: int = 50
    ) -> List[str]:
        """Create and process merge tasks in batches to limit concurrent disk I/O."""
        self.logger.info(
            f"Gen {self.gen}: Processing {num_models} merges in batches of {batch_size}..."
        )
        all_saved_paths = []

        # Process merges in batches
        for i in range(0, num_models, batch_size):
            batch_size_actual = min(batch_size, num_models - i)
            self.logger.info(
                f"Processing merge batch {i//batch_size + 1}: {batch_size_actual} merges"
            )

            # Create tasks for this batch
            batch_promises = []
            for j in range(batch_size_actual):
                model_index = i + j
                promise = create_merge_only_task(
                    self.cfg,
                    self.call_fn,
                    self.gen,
                    model_index,
                    self.archive_data,
                    self.np_random,
                )
                batch_promises.append(promise)
                self.logger.debug(
                    f"Submitted merge for model index {model_index}"
                )
                # Add staggered sleep timer to ensure disk I/O load doesn't happen for all workers at same time
                time.sleep(delay)

            # Wait for this batch to complete before starting next batch
            batch_results = wait_for_promises(
                batch_promises, self.cfg.celery.timeout
            )

            # Process batch results
            for result in batch_results:
                if result:  # result is the save_path string
                    all_saved_paths.append(result)
                    self.logger.info(f"Merge complete: {result}")
                else:
                    self.logger.error(
                        "Merge operation failed, no model path returned"
                    )

            self.logger.info(
                f"Batch {i//batch_size + 1} complete. Total merges done: {len(all_saved_paths)}/{num_models}"
            )

        return all_saved_paths

    def _create_merge_tasks(
        self, task_info: Union[List[str], Dict[str, DictConfig]]
    ) -> List:
        """Creates Celery promises for merge operations."""
        self.logger.info(
            f"Gen {self.gen}: Creating {self.cfg.celery.num_workers} merge tasks..."
        )
        merge_promises = [
            create_merge_task(
                self.cfg,
                self.call_fn,
                self.gen,
                i,
                self.archive_data,
                self.np_random,
                task_info=task_info,
            )
            for i in range(self.cfg.celery.num_workers)
        ]
        return merge_promises

    def _update_archive_after_merge(
        self, new_solutions: List[Optional[ACDCSolution]]
    ) -> None:
        """Converts new solutions, updates the archive, and converts back."""
        if not self.task_pool:
            raise RuntimeError(
                "Task pool not initialized before archive update."
            )
        if self.archive_data is None or "dns_archive" not in self.archive_data:
            raise RuntimeError("Archive data not properly initialized.")

        self.logger.info(
            f"Gen {self.gen}: Updating archive with {len([s for s in new_solutions if s])} new solutions..."
        )

        ordered_task_ids = self.task_pool.get_ordered_task_ids()
        threshold = self.cfg.dns.acdc_skill_threshold
        current_acdc_archive = list(
            self.archive_data["dns_archive"]
        )  # Get current archive

        # Filter out None results from new_solutions before creating map
        valid_new_solutions = [s for s in new_solutions if s]

        # Filter archive by looking at model responses and discarding gibberish models
        if self.cfg.dns.run_gibberish_check:
            new_valid_new_solutions = []
            for sol in valid_new_solutions:
                if self.is_seed_model(sol.model_path):
                    continue
                if sol.is_gibberish:
                    self.gibberish_models_counter += 1
                    continue
                new_valid_new_solutions.append(sol)
            valid_new_solutions = new_valid_new_solutions

        # Keep original AC/DC solutions mapped by path (current + valid new ones)
        acdc_solution_map = {
            s.model_path: s for s in current_acdc_archive + valid_new_solutions
        }

        # Convert current archive and new solutions to DNSSolution format for update_dns_archive
        converted_archive = [
            convert_acdc_to_dns_solution(sol, ordered_task_ids, threshold)
            for sol in current_acdc_archive
            if sol  # Ensure sol is not None
        ]
        converted_new = [
            convert_acdc_to_dns_solution(sol, ordered_task_ids, threshold)
            for sol in valid_new_solutions  # Already filtered Nones
        ]

        # Call the standard update_dns_archive function
        surviving_dns_solutions = update_dns_archive(
            converted_archive,
            converted_new,
            self.cfg.dns,  # Pass dns sub-config
        )

        # Convert back to ACDCSolution format and update the instance's archive
        self.archive_data["dns_archive"] = [
            acdc_solution_map[dns_sol.model_path]
            for dns_sol in surviving_dns_solutions
            if dns_sol.model_path in acdc_solution_map  # Ensure path exists
        ]
        self.logger.info(
            f"Gen {self.gen}: Archive updated. Size: {len(self.archive_data['dns_archive'])}"
        )

    def _save_archive_state(self) -> None:
        """Saves the current state of the DNS archive."""
        if not self.dirs or "archive_dir" not in self.dirs:
            raise ValueError(
                "Archive directory not set before saving archive state."
            )
        if self.archive_data is None or "dns_archive" not in self.archive_data:
            raise RuntimeError(
                "Archive data not properly initialized before saving."
            )

        archive_dir = self.dirs["archive_dir"]
        archive_path = f"{archive_dir}/gen{self.gen}_dns_archive.json"
        save_ac_dc_archive(
            self.archive_data["dns_archive"],
            archive_path,
            max_details_to_log=self.cfg.dns.max_details_to_log,
        )
        self.logger.info(
            f"Saved DNS archive for gen {self.gen} to {archive_path}"
        )

    def _reevaluate_archive_on_new_tasks(
        self, newly_added_task_paths: List[str]
    ) -> None:
        """Evaluates existing archive solutions on newly added tasks and updates them."""
        if not newly_added_task_paths:
            return
        if self.archive_data is None or "dns_archive" not in self.archive_data:
            raise RuntimeError(
                "Archive data not properly initialized before re-evaluation."
            )

        self.logger.info(
            f"Gen {self.gen}: Evaluating {len(self.archive_data['dns_archive'])} archive models on {len(newly_added_task_paths)} new tasks..."
        )
        task_info_for_reeval = newly_added_task_paths  # Evaluate only new tasks
        reeval_promises = []
        current_archive_solutions = list(
            self.archive_data["dns_archive"]
        )  # Copy list

        for solution in current_archive_solutions:
            if not solution:  # Skip if somehow None got into the archive
                continue
            reeval_promises.append(
                self.call_fn.delay(
                    "eval_model",
                    model_path=solution.model_path,
                    save_path=solution.model_path,
                    data_split="train",  # Assuming train split for skill vector update
                    task_info=task_info_for_reeval,
                )
            )

        # Use original timeout as we evaluate fewer tasks per model
        reeval_timeout = self.cfg.celery.timeout
        reeval_results = wait_for_promises(reeval_promises, reeval_timeout)

        # Update archive solutions in-place with new skill vector entries and recalculated fitness
        results_map: Dict[str, ACDCMergeResult] = {
            res.save_path: res for res in reeval_results if res
        }
        updated_count = 0
        for i, solution in enumerate(self.archive_data["dns_archive"]):
            if not solution:
                self.logger.warning(
                    f"Skipping update for invalid solution at archive index {i}"
                )
                continue

            if solution.model_path in results_map:
                result = results_map[solution.model_path]

                # Initialize skill vector and eval details if None
                if solution.acdc_skill_vector is None:
                    solution.acdc_skill_vector = {}
                if (
                    not hasattr(solution, "acdc_eval_details")
                    or solution.acdc_eval_details is None
                ):
                    # Initialize if attribute doesn't exist or is None
                    solution.acdc_eval_details = []

                # Update existing skill vector and append new eval details
                if result:
                    if result.acdc_skill_vector:
                        solution.acdc_skill_vector.update(
                            result.acdc_skill_vector
                        )
                    # Check if the result has the details and they are not None
                    if (
                        hasattr(result, "acdc_eval_details")
                        and result.acdc_eval_details
                    ):
                        # Append new details from this re-evaluation
                        solution.acdc_eval_details.extend(
                            result.acdc_eval_details
                        )
                        # Truncation to 5 happens during saving in save_ac_dc_archive

                # Recalculate fitness based on the *complete updated* skill vector
                new_fitness = calculate_fitness_from_skill_vector(
                    solution.acdc_skill_vector
                )
                solution.fitness = new_fitness
                updated_count += 1
            else:
                # If a model failed evaluation on new tasks, its skill vector remains unchanged for those tasks
                # We still need to recalculate its fitness based on its current vector
                current_fitness = calculate_fitness_from_skill_vector(
                    solution.acdc_skill_vector
                )
                solution.fitness = current_fitness
                self.logger.warning(
                    f"No re-evaluation result found for archive model: {solution.model_path}. Fitness recalculated based on existing skills."
                )

        self.logger.info(
            f"Incorporated re-evaluation results for {updated_count}/{len(self.archive_data['dns_archive'])} archive models."
        )

    def _filter_archive_skill_vectors_to_active_pool(self) -> None:
        """Filters skill vectors in the archive to only include tasks currently in the active pool."""
        if not self.task_pool:
            self.logger.warning(
                "Cannot filter skill vectors: Task pool not available."
            )
            return
        if self.archive_data is None or "dns_archive" not in self.archive_data:
            raise RuntimeError(
                "Archive data not properly initialized before filtering."
            )

        self.logger.info(
            f"Gen {self.gen}: Filtering archive skill vectors to match active task pool..."
        )
        active_task_ids = set(self.task_pool.get_ordered_task_ids())
        filtered_count = 0
        for solution in self.archive_data["dns_archive"]:
            if solution and solution.acdc_skill_vector:
                original_size = len(solution.acdc_skill_vector)
                filtered_vector = {
                    task_id: score
                    for task_id, score in solution.acdc_skill_vector.items()
                    if task_id in active_task_ids
                }
                if len(filtered_vector) < original_size:
                    filtered_count += 1
                    # self.logger.debug(f"Filtered skill vector for {solution.model_path} from {original_size} to {len(filtered_vector)} tasks.")
                solution.acdc_skill_vector = filtered_vector
                # Recalculate fitness based *only* on active tasks
                solution.fitness = calculate_fitness_from_skill_vector(
                    solution.acdc_skill_vector
                )

        self.logger.info(
            f"Filtered skill vectors for {filtered_count} solutions. Recalculated fitness for all solutions based on active tasks."
        )

    def _save_post_adaptation_archive(self) -> None:
        """Saves the archive state after task adaptation and filtering."""
        if not self.dirs or "archive_dir" not in self.dirs:
            raise ValueError(
                "Archive directory not set before saving post-adaptation archive."
            )
        if self.archive_data is None or "dns_archive" not in self.archive_data:
            raise RuntimeError(
                "Archive data not properly initialized before saving."
            )

        archive_dir = self.dirs["archive_dir"]
        archive_path = (
            f"{archive_dir}/gen{self.gen}_dns_archive_post_adapt_filtered.json"
        )
        save_ac_dc_archive(
            self.archive_data["dns_archive"],
            archive_path,
            max_details_to_log=self.cfg.dns.max_details_to_log,
        )
        self.logger.info(
            f"Saved post-adaptation/filtered DNS archive for gen {self.gen} to {archive_path}"
        )

    def _log_active_task_pool_state(self) -> None:
        """Logs the current list of active task paths and active limbo map to JSON files."""
        if not self.task_pool:
            self.logger.warning(
                "Cannot log active task pool state: Task pool not available."
            )
            return
        if not self.dirs:
            raise ValueError(
                "Directories not set before logging task pool state."
            )

        active_task_paths = self.task_pool.tasks
        # Ensure generated_tasks_dir exists (it should from setup)
        pool_log_dir = self.dirs.get(
            "generated_tasks_dir",
            os.path.join(self.dirs["output_dir"], "generated_tasks", "pool"),
        )
        os.makedirs(pool_log_dir, exist_ok=True)  # Ensure directory exists

        # Save active task pool
        pool_log_path = os.path.join(
            pool_log_dir, f"active_pool_gen_{self.gen}.json"
        )
        try:
            with open(pool_log_path, "w") as f:
                json.dump(active_task_paths, f, indent=2)
            self.logger.info(
                f"Saved active task pool state ({len(active_task_paths)} tasks) for gen {self.gen} to {pool_log_path}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to save active task pool state for gen {self.gen}: {e}"
            )

        # Save active limbo map
        limbo_log_path = os.path.join(
            pool_log_dir, f"active_limbo_map_gen_{self.gen}.json"
        )
        try:
            with open(limbo_log_path, "w") as f:
                json.dump(self.task_pool.active_limbo_map, f, indent=2)
            self.logger.info(
                f"Saved active limbo map ({len(self.task_pool.active_limbo_map)} entries) for gen {self.gen} to {limbo_log_path}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to save active limbo map for gen {self.gen}: {e}"
            )

    def _adapt_task_pool_and_reevaluate_archive(self) -> None:
        """Synchronize the archive skill vectors with the active task pool after every generation, and adapt the pool at the configured interval."""
        if not (self.cfg.get("use_ac_dc", False) and self.task_pool):
            return  # Adaptation not enabled or configured

        newly_added_task_paths = []
        # 1. Adapt the task pool if at adaptation interval
        if (
            self.cfg.acdc.get("task_generation_interval", 0) > 0
            and self.gen > 0
            and self.gen % self.cfg.acdc.task_generation_interval == 0
        ):
            self.logger.info(f"Generation {self.gen}: Adapting task pool...")
            try:
                newly_added_task_paths = self.task_pool.adapt_task_pool(
                    self.archive_data, self.gen
                )
                self.logger.info(
                    f"Task pool adaptation complete. Added {len(newly_added_task_paths)} new tasks."
                )
            except Exception as e:
                self.logger.exception(f"Error during task pool adaptation: {e}")

        # 2. Always synchronize skill vectors for all models in the archive
        if not self.archive_data or "dns_archive" not in self.archive_data:
            self.logger.warning(
                "Archive data not available for skill vector synchronization."
            )
            return
        if not self.task_pool:
            self.logger.warning(
                "Task pool not available for skill vector synchronization."
            )
            return

        active_task_ids = set(self.task_pool.get_ordered_task_ids())
        self.logger.info(
            f"Synchronizing skill vectors to {len(active_task_ids)} active tasks."
        )
        missing_tasks_per_model = []
        for i, model in enumerate(self.archive_data["dns_archive"]):
            if (
                not model
                or not hasattr(model, "acdc_skill_vector")
                or model.acdc_skill_vector is None
            ):
                model.acdc_skill_vector = {}
            # Remove tasks not in active pool
            before = set(model.acdc_skill_vector.keys())
            model.acdc_skill_vector = {
                k: v
                for k, v in model.acdc_skill_vector.items()
                if k in active_task_ids
            }
            after = set(model.acdc_skill_vector.keys())
            removed = before - after
            if removed:
                self.logger.debug(
                    f"Model {getattr(model, 'model_path', i)}: Removed {len(removed)} tasks from skill vector not in active pool."
                )
            # Find missing tasks
            missing_tasks = active_task_ids - set(model.acdc_skill_vector.keys())
            # Given the missing_tasks partial strings, get the paths from self.task_pool.tasks
            missing_tasks_paths = []
            for missing_task in missing_tasks:
                for task_path in self.task_pool.tasks:
                    if missing_task in task_path:
                        missing_tasks_paths.append(task_path)
                        break
            missing_tasks_per_model.append(missing_tasks_paths)

        # Assert all missing_tasks sets are the same
        if missing_tasks_per_model:
            first_missing = missing_tasks_per_model[0]
            for idx, missing in enumerate(missing_tasks_per_model):
                assert (
                    missing == first_missing
                ), f"Mismatch in missing tasks across models in archive! Model {idx} missing: {missing}, expected: {first_missing}"
            if first_missing:
                self.logger.info(
                    f"All models missing {len(first_missing)} tasks from skill vectors. Reevaluating on these tasks."
                )
                self._reevaluate_archive_on_new_tasks(list(first_missing))
            else:
                self.logger.info(
                    "No missing tasks in skill vectors for any model."
                )
        else:
            self.logger.info(
                "No models in archive to synchronize skill vectors."
            )

        # 3. Save post-adaptation archive every generation
        self._save_post_adaptation_archive()

        # 4. Log Active Pool State (always log if pool exists)
        if self.task_pool:
            self._log_active_task_pool_state()

    def _log_generation_metrics(self, prev_log_time: float) -> None:
        """Logs key metrics for the current generation to W&B."""
        if self.archive_data is None or not self.archive_data.get(
            "dns_archive"
        ):
            self.logger.warning(
                f"Gen {self.gen}: Skipping metrics logging, archive is empty."
            )
            return

        try:
            # Find the solution with the highest fitness
            # Handle potential None values in archive if retries failed permanently
            valid_solutions = [
                s
                for s in self.archive_data["dns_archive"]
                if s and hasattr(s, "fitness")
            ]
            if not valid_solutions:
                self.logger.warning(
                    f"Gen {self.gen}: Skipping metrics logging, no valid solutions with fitness found in archive."
                )
                return

            top_solution = max(valid_solutions, key=lambda x: x.fitness)

            # Compute coverage metrics using the AC/DC skill vector format
            coverage_metrics = compute_acdc_coverage_metrics(
                self.archive_data,
                self.tasks,
                threshold=self.cfg.dns.acdc_skill_threshold,
                validation_tasks=getattr(self.cfg, "validation_tasks", None),
            )

            log_data = {
                "dns/best_fitness": top_solution.fitness,
                "dns/archive_size": len(
                    self.archive_data["dns_archive"]
                ),  # Log total size including potential Nones
                "dns/valid_archive_size": len(
                    valid_solutions
                ),  # Log count of valid solutions
                **coverage_metrics,  # Add coverage metrics
                "base_info/generation": self.gen,
                "base_info/gpu_num": self.cfg.celery.num_workers,
                "base_info/log_interval_seconds": time.time() - prev_log_time,
            }

            # Add skill vector length if available
            if top_solution.acdc_skill_vector:
                log_data["dns/skill_vector_length"] = len(
                    top_solution.acdc_skill_vector
                )

            if self.cfg.dns.run_gibberish_check:
                log_data["dns/gibberish_models_counter"] = (
                    self.gibberish_models_counter
                )
                self.gibberish_models_counter = 0

            wandb.log(
                log_data, step=self.gen, commit=False
            )  # Commit is handled later or by validation
            self.logger.info(
                f"Gen {self.gen}: Logged metrics. Best Fitness: {top_solution.fitness:.4f}, Archive Size: {len(valid_solutions)}, Coverage Metrics: {len(coverage_metrics)} keys"
            )

        except Exception as e:
            self.logger.exception(
                f"Error during metrics logging for generation {self.gen}: {e}"
            )

    def _cleanup_and_advance_generation(self) -> float:
        """Performs end-of-generation cleanup and increments generation counter."""
        self._cleanup_generation_resources()
        self.gen += 1
        return time.time()

    def run(self) -> None:
        """Runs the main AC/DC optimization loop."""
        self.logger.info(
            f"Starting AC/DC optimization for {self.cfg.dns.num_generations} generations..."
        )
        self.logger.info(f"Seed models: {self.cfg.seed_model_paths}")

        # Step 1: Setup dirs, worker log file names, etc.
        self.setup_environment()
        self.setup_workers()

        # Step 2: Start Phase 1 model initialization asynchronously (don't wait)
        self.logger.info("Starting Phase 1 model initialization (async)...")
        phase1_promises, expected_paths = (
            self.initialize_population_phase1_async()
        )

        # Step 3: Run setup_environment (task generation) while Phase 1 runs in background
        self.logger.info("Running task generation while models initialize...")
        self._load_or_generate_tasks()

        # Step 4: Wait for Phase 1 to complete and get successful paths
        self.logger.info(
            "Waiting for Phase 1 model initialization to complete..."
        )
        saved_model_paths = self.wait_for_phase1_and_get_paths(
            phase1_promises, expected_paths
        )

        # Step 5: Run Phase 2 evaluation with loaded tasks and saved models
        self.initialize_population_phase2(saved_model_paths)

        self.logger.info(
            f"Output directory: {self.dirs['output_dir'] if self.dirs else 'N/A'}"
        )

        if self.archive_data is None:  # Final check after setup/init
            raise RuntimeError(
                "Optimizer archive_data is still None after setup and initialization."
            )

        # --- Main Loop ---
        prev_log_time = time.time()
        while self.gen <= self.cfg.dns.num_generations:
            self.logger.info(f"--- Generation {self.gen} ---")

            # 1. Determine Tasks for this generation
            task_info = self._determine_task_info_for_generation()

            # 2. Phase 1: Merge models and save to disk (limited concurrent operations)
            self.logger.info(
                f"Generation {self.gen}: Phase 1 - Merging models..."
            )
            saved_model_paths = self._process_merges_in_batches(
                num_models=self.cfg.dns.num_model_per_gen,
                batch_size=8,  # Max 2 concurrent merge operations
            )

            # 3. Phase 2: Evaluate saved models with staggered loading
            self.logger.info(
                f"Generation {self.gen}: Phase 2 - Evaluating models..."
            )
            new_solutions = self._evaluate_saved_models_staggered(
                model_paths=saved_model_paths,
                task_info=task_info,
                batch_size=2,  # Max 2 concurrent evaluations
                stagger_delay=30.0,  # 30 seconds between evaluation batches
            )

            # 4. Update Archive with New Solutions
            self._update_archive_after_merge(new_solutions)

            # 4. Save Current Archive State
            self._save_archive_state()

            # 5. Adapt Task Pool & Re-evaluate Archive (If Enabled and Interval Met)
            self._adapt_task_pool_and_reevaluate_archive()

            # 6. Log Metrics
            self._log_generation_metrics(prev_log_time)

            # 7. Cleanup and Advance Generation
            prev_log_time = self._cleanup_and_advance_generation()

            # Commit W&B logs at the end of the generation loop
            wandb.log(
                {}, step=self.gen - 1, commit=True
            )  # Commit logs from this generation (gen-1 because gen was incremented)

        # --- Finalization ---
        self.logger.info("Optimization loop complete. Performing final save.")
        self._save_final_archive()

    def _save_final_archive(self) -> None:
        """Saves the final state of the DNS archive."""
        if not self.dirs or "archive_dir" not in self.dirs:
            self.logger.error(
                "Cannot save final archive: Archive directory not set."
            )
            return
        if self.archive_data is None or "dns_archive" not in self.archive_data:
            self.logger.error(
                "Cannot save final archive: Archive data not available."
            )
            return

        archive_dir = self.dirs["archive_dir"]
        final_archive_path = f"{archive_dir}/final_dns_archive.json"
        try:
            save_ac_dc_archive(
                self.archive_data["dns_archive"],
                final_archive_path,
                max_details_to_log=self.cfg.dns.max_details_to_log,
            )
            self.logger.info(
                f"Saved final DNS archive ({len(self.archive_data['dns_archive'])} solutions) to {final_archive_path}"
            )
        except Exception as e:
            self.logger.exception(
                f"Error saving final archive to {final_archive_path}: {e}"
            )


def run_optimization(celery: Celery, cfg: DictConfig) -> None:
    """Run DNS optimization."""
    optimizer = ACDCOptimizer(celery, cfg)
    optimizer.run()


@hydra.main(version_base=None, config_path="configs", config_name="ac_dc")
def main(cfg: DictConfig):
    """Main entry point for the optimization process.

    The celery mode can be overridden via command line:
    python main_sandbox.py celery.mode=worker

    Available modes:
    - main: Run as the main coordinator
    - worker: Run as a worker node
    - solo: Run in single-process mode
    """
    print(OmegaConf.to_yaml(cfg))

    def get_worker_cls(cfg):
        def init_func():
            return Worker(cfg)

        return init_func

    celery = setup_celery(
        name=cfg.celery.name,
        mode=cfg.celery.mode,
        worker_cls=get_worker_cls(cfg),
        celery_broker=cfg.celery.broker,
        celery_backend=cfg.celery.backend,
    )

    if cfg.wandb_resume_id:
        resume = "must"
    else:
        resume = None

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
            resume=resume,
            id=cfg.wandb_resume_id,
        )
        wandb.run.log_code(  # type: ignore
            ".",
            include_fn=lambda path: path.endswith(".py")
            or path.endswith(".ipynb")
            or path.endswith(".yaml"),
            exclude_fn=lambda path, root: any(
                os.path.relpath(path, root).startswith(dir)
                for dir in [
                    "outputs/",
                    "multirun/",
                    "cache/",
                    "wandb/",
                    "evaluation/",
                    "evaluation_results/",
                    "AC-DC-eval_harness/",
                ]
            ),
        )

    run_optimization(celery, cfg)


if __name__ == "__main__":
    main()
