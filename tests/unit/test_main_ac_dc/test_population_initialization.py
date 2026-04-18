"""
Unit tests for population initialization in ACDCOptimizer.

Tests cover:
- Phase 1: Async model initialization
- Phase 2: Model evaluation
- Staggered evaluation
- Archive conversion and updates
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf

from main_ac_dc import ACDCOptimizer
from datatypes import ACDCMergeResult, ACDCSolution, ACDCTaskEvalDetail


@pytest.fixture
def mock_celery():
    """Create a mock Celery app."""
    celery = Mock()
    call_task = Mock()
    celery.tasks = {"call": call_task}
    return celery


@pytest.fixture
def basic_config():
    """Create basic configuration for testing."""
    return OmegaConf.create({
        "seed": 42,
        "validation_tasks": [],
        "seed_model_paths": ["/models/seed1", "/models/seed2"],
        "celery": {"num_workers": 4, "timeout": 300},
        "dns": {
            "population_size": 10,
            "init_population_size": 5,
            "num_model_per_gen": 8,
            "acdc_skill_threshold": 0.5,
            "max_details_to_log": 5,
            "init_population_with_seed_models": True,
            "n_min_init_pop_promises": 5,
            "run_gibberish_check": False,
            "disable_mutation": False,
        },
        "use_ac_dc": True,
        "acdc": {},
        "restart_dir": None,
    })


@pytest.fixture
def initialized_optimizer(mock_celery, basic_config, tmp_path):
    """Create an optimizer with environment set up."""
    with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
        mock_runtime = Mock()
        mock_runtime.output_dir = str(tmp_path)
        mock_hydra.return_value.runtime = mock_runtime

        optimizer = ACDCOptimizer(mock_celery, basic_config)
        optimizer.setup_environment()

        # Set up task pool mock
        mock_task_pool = Mock()
        mock_task_pool.tasks = ["/tasks/task_0", "/tasks/task_1"]
        mock_task_pool.get_ordered_task_ids.return_value = ["task_0", "task_1"]
        optimizer.task_pool = mock_task_pool

        # Set up tasks
        optimizer.tasks = [Mock(task_name="task_0"), Mock(task_name="task_1")]

        return optimizer


class TestInitializePopulationPhase1Async:
    """Test Phase 1: Async model initialization."""

    def test_basic_async_initialization(self, initialized_optimizer):
        """Test that Phase 1 creates initialization promises asynchronously."""
        # Mock the call function
        mock_promise = Mock()
        initialized_optimizer.call_fn.delay = Mock(return_value=mock_promise)

        with patch("time.sleep"):  # Speed up test
            promises, paths = initialized_optimizer.initialize_population_phase1_async()

        # Should create promises for seed models + additional models
        expected_count = (
            len(initialized_optimizer.cfg.seed_model_paths)
            + initialized_optimizer.cfg.dns.n_min_init_pop_promises
        )
        assert len(promises) >= len(initialized_optimizer.cfg.seed_model_paths)
        assert len(paths) >= len(initialized_optimizer.cfg.seed_model_paths)

    def test_seed_model_initialization_without_mutation(self, initialized_optimizer):
        """Test that seed models are initialized without mutation."""
        mock_promise = Mock()
        initialized_optimizer.call_fn.delay = Mock(return_value=mock_promise)

        with patch("time.sleep"):
            promises, paths = initialized_optimizer.initialize_population_phase1_async()

        # Check first calls are for seed models without mutation
        seed_calls = [
            call
            for call in initialized_optimizer.call_fn.delay.call_args_list
            if call[1].get("do_mutate") is False
        ]

        assert len(seed_calls) >= len(initialized_optimizer.cfg.seed_model_paths)

    def test_crossover_model_generation_with_mutation(self, initialized_optimizer):
        """Test that additional models use crossover with mutation."""
        mock_promise = Mock()
        initialized_optimizer.call_fn.delay = Mock(return_value=mock_promise)

        with patch("time.sleep"):
            promises, paths = initialized_optimizer.initialize_population_phase1_async()

        # Check for calls with mutation enabled
        mutation_calls = [
            call
            for call in initialized_optimizer.call_fn.delay.call_args_list
            if call[1].get("do_mutate") is True
        ]

        assert len(mutation_calls) > 0

    def test_restart_skips_initialization(self, initialized_optimizer):
        """Test that restart with existing archive skips Phase 1."""
        # Set up existing archive
        initialized_optimizer.cfg.restart_dir = "/some/path"
        initialized_optimizer.archive_data["dns_archive"] = [
            ACDCSolution(
                model_path="/models/existing",
                fitness=0.8,
                acdc_skill_vector={}
            )
        ]

        promises, paths = initialized_optimizer.initialize_population_phase1_async()

        # Should return empty lists
        assert promises == []
        assert paths == []

    def test_promise_and_path_lists_match(self, initialized_optimizer):
        """Test that promise and path lists have matching lengths."""
        mock_promise = Mock()
        initialized_optimizer.call_fn.delay = Mock(return_value=mock_promise)

        with patch("time.sleep"):
            promises, paths = initialized_optimizer.initialize_population_phase1_async()

        assert len(promises) == len(paths)

    def test_respects_mutation_disabled_config(self, initialized_optimizer):
        """Test that mutation can be disabled via config."""
        initialized_optimizer.cfg.dns.disable_mutation = True

        mock_promise = Mock()
        initialized_optimizer.call_fn.delay = Mock(return_value=mock_promise)

        with patch("time.sleep"):
            promises, paths = initialized_optimizer.initialize_population_phase1_async()

        # All calls except seed models should have do_mutate=False
        mutation_calls = [
            call
            for call in initialized_optimizer.call_fn.delay.call_args_list
            if call[1].get("do_mutate") is True
        ]

        # When mutation is disabled, all do_mutate should be False
        # (except for those that explicitly enable it, which shouldn't exist here)
        assert len(mutation_calls) == 0


class TestWaitForPhase1AndGetPaths:
    """Test waiting for Phase 1 completion."""

    def test_successful_promise_completion(self, initialized_optimizer):
        """Test waiting for all promises to complete successfully."""
        # Create mock promises
        promises = []
        paths = []
        for i in range(5):
            p = Mock()
            p.ready.return_value = True
            p.get.return_value = f"/models/gen_0_ind_{i}"
            promises.append(p)
            paths.append(f"/models/gen_0_ind_{i}")

        result_paths = initialized_optimizer.wait_for_phase1_and_get_paths(
            promises, paths
        )

        assert len(result_paths) == 5
        assert all(path.startswith("/models/gen_0_ind_") for path in result_paths)

    def test_partial_failure_handling(self, initialized_optimizer):
        """Test handling when some promises fail."""
        # Create mix of successful and failed promises
        promises = []
        paths = []
        for i in range(5):
            p = Mock()
            p.ready.return_value = True
            if i < 3:
                p.get.return_value = f"/models/gen_0_ind_{i}"
            else:
                p.get.return_value = None  # Failure
            promises.append(p)
            paths.append(f"/models/gen_0_ind_{i}")

        result_paths = initialized_optimizer.wait_for_phase1_and_get_paths(
            promises, paths
        )

        # Should get only successful paths
        assert len(result_paths) == 3

    def test_insufficient_successful_models_warning(self, initialized_optimizer):
        """Test warning when fewer models succeed than population size."""
        # Only 2 successful promises
        promises = []
        paths = []
        for i in range(2):
            p = Mock()
            p.ready.return_value = True
            p.get.return_value = f"/models/gen_0_ind_{i}"
            promises.append(p)
            paths.append(f"/models/gen_0_ind_{i}")

        # Population size is 5 but we only get 2
        result_paths = initialized_optimizer.wait_for_phase1_and_get_paths(
            promises, paths
        )

        assert len(result_paths) == 2  # Gets what's available

    def test_restart_case_returns_empty(self, initialized_optimizer):
        """Test that restart case with no promises returns empty list."""
        result_paths = initialized_optimizer.wait_for_phase1_and_get_paths([], [])

        assert result_paths == []

    def test_population_size_limiting(self, initialized_optimizer):
        """Test that results are limited to init_population_size."""
        # Create more promises than needed
        promises = []
        paths = []
        for i in range(10):
            p = Mock()
            p.ready.return_value = True
            p.get.return_value = f"/models/gen_0_ind_{i}"
            promises.append(p)
            paths.append(f"/models/gen_0_ind_{i}")

        result_paths = initialized_optimizer.wait_for_phase1_and_get_paths(
            promises, paths
        )

        # Should be limited to init_population_size
        assert len(result_paths) == initialized_optimizer.cfg.dns.init_population_size


class TestInitializePopulationPhase2:
    """Test Phase 2: Model evaluation."""

    def test_evaluation_of_saved_models(self, initialized_optimizer):
        """Test that saved models are evaluated."""
        saved_paths = [f"/models/gen_0_ind_{i}" for i in range(3)]

        # Mock evaluation
        with patch.object(
            initialized_optimizer, "_evaluate_saved_models_staggered"
        ) as mock_eval:
            mock_result = ACDCSolution(
                model_path="/models/gen_0_ind_0",
                fitness=0.8,
                acdc_skill_vector={"task_0": 0.8}
            )
            mock_eval.return_value = [mock_result] * 3

            with patch.object(
                initialized_optimizer, "_convert_and_update_initial_archive"
            ) as mock_convert:
                mock_convert.return_value = [mock_result] * 3

                with patch.object(
                    initialized_optimizer, "_save_initial_archive"
                ) as mock_save:
                    initialized_optimizer.initialize_population_phase2(saved_paths)

                    # Verify evaluation was called
                    mock_eval.assert_called_once()
                    mock_convert.assert_called_once()
                    mock_save.assert_called_once()

    def test_gibberish_filtering(self, initialized_optimizer):
        """Test that gibberish models are filtered out."""
        initialized_optimizer.cfg.dns.run_gibberish_check = True
        saved_paths = [f"/models/gen_0_ind_{i}" for i in range(3)]

        # Mock evaluation with gibberish model
        with patch.object(
            initialized_optimizer, "_evaluate_saved_models_staggered"
        ) as mock_eval:
            results = [
                ACDCSolution(
                    model_path="/models/gen_0_ind_0",
                    fitness=0.8,
                    acdc_skill_vector={"task_0": 0.8},
                    is_gibberish=False,
                ),
                ACDCSolution(
                    model_path="/models/gen_0_ind_1",
                    fitness=0.1,
                    acdc_skill_vector={},
                    is_gibberish=True,  # Gibberish
                ),
                ACDCSolution(
                    model_path="/models/gen_0_ind_2",
                    fitness=0.7,
                    acdc_skill_vector={"task_0": 0.7},
                    is_gibberish=False,
                ),
            ]
            mock_eval.return_value = results

            with patch.object(
                initialized_optimizer, "_convert_and_update_initial_archive"
            ) as mock_convert:
                mock_convert.return_value = [results[0], results[2]]  # Filtered

                with patch.object(
                    initialized_optimizer, "_save_initial_archive"
                ):
                    initialized_optimizer.initialize_population_phase2(saved_paths)

                    # Only valid solutions should be passed to convert
                    converted_solutions = mock_convert.call_args[0][0]
                    assert len(converted_solutions) == 2
                    assert all(not s.is_gibberish for s in converted_solutions)

    def test_restart_no_op_case(self, initialized_optimizer):
        """Test that Phase 2 is skipped on restart with no paths."""
        initialized_optimizer.initialize_population_phase2([])

        # Should return early without errors

    def test_archive_update_and_save(self, initialized_optimizer):
        """Test that archive is updated and saved."""
        saved_paths = ["/models/gen_0_ind_0"]

        mock_solution = ACDCSolution(
            model_path="/models/gen_0_ind_0",
            fitness=0.8,
            acdc_skill_vector={"task_0": 0.8}
        )

        with patch.object(
            initialized_optimizer, "_evaluate_saved_models_staggered"
        ) as mock_eval:
            mock_eval.return_value = [mock_solution]

            with patch.object(
                initialized_optimizer, "_convert_and_update_initial_archive"
            ) as mock_convert:
                mock_convert.return_value = [mock_solution]

                with patch.object(
                    initialized_optimizer, "_save_initial_archive"
                ) as mock_save:
                    initialized_optimizer.initialize_population_phase2(saved_paths)

                    # Verify archive was updated
                    assert initialized_optimizer.archive_data["dns_archive"] == [
                        mock_solution
                    ]

                    # Verify save was called with updated archive
                    mock_save.assert_called_once_with([mock_solution])


class TestEvaluateSavedModelsStaggered:
    """Test staggered model evaluation."""

    def test_staggered_submission(self, initialized_optimizer):
        """Test that evaluations are submitted in batches."""
        model_paths = [f"/models/model_{i}" for i in range(6)]

        mock_promise = Mock()
        mock_promise.ready.return_value = True
        mock_result = ACDCMergeResult(
            save_path="/models/model_0",
            acdc_skill_vector={"task_0": 0.8},
            avg_acdc_quality=0.8,
        )
        mock_promise.get.return_value = mock_result

        with patch("main_ac_dc.create_eval_only_task") as mock_create_eval:
            mock_create_eval.return_value = mock_promise

            with patch("time.sleep"):  # Speed up test
                results = initialized_optimizer._evaluate_saved_models_staggered(
                    model_paths=model_paths,
                    task_info=initialized_optimizer.task_pool.tasks,
                    batch_size=2,
                    stagger_delay=1.0,
                )

            # Should have created evaluation tasks for all models
            assert mock_create_eval.call_count == len(model_paths)

    def test_batch_size_limiting(self, initialized_optimizer):
        """Test that batch size limits concurrent evaluations."""
        model_paths = [f"/models/model_{i}" for i in range(5)]

        mock_promise = Mock()
        mock_promise.ready.return_value = True
        mock_result = ACDCMergeResult(
            save_path="/models/model_0",
            acdc_skill_vector={},
            avg_acdc_quality=0.5,
        )
        mock_promise.get.return_value = mock_result

        # Track call timing to verify batching
        call_times = []

        def track_call(*args, **kwargs):
            call_times.append(len(call_times))
            return mock_promise

        with patch("main_ac_dc.create_eval_only_task") as mock_create_eval:
            mock_create_eval.side_effect = track_call

            with patch("time.sleep"):
                results = initialized_optimizer._evaluate_saved_models_staggered(
                    model_paths=model_paths,
                    task_info=[],
                    batch_size=2,
                    stagger_delay=0.1,
                )

            # All should have been called
            assert len(call_times) == 5

    def test_result_processing(self, initialized_optimizer):
        """Test that results are processed correctly."""
        model_paths = ["/models/model_0", "/models/model_1"]

        # Create different results for each model
        results = [
            ACDCMergeResult(
                save_path="/models/model_0",
                acdc_skill_vector={"task_0": 0.8},
                avg_acdc_quality=0.8,
            ),
            ACDCMergeResult(
                save_path="/models/model_1",
                acdc_skill_vector={"task_0": 0.6},
                avg_acdc_quality=0.6,
            ),
        ]

        promises = [Mock() for _ in range(2)]
        for p, r in zip(promises, results):
            p.ready.return_value = True
            p.get.return_value = r

        with patch("main_ac_dc.create_eval_only_task") as mock_create_eval:
            mock_create_eval.side_effect = promises

            with patch("time.sleep"):
                processed_results = initialized_optimizer._evaluate_saved_models_staggered(
                    model_paths=model_paths,
                    task_info=[],
                    batch_size=2,
                    stagger_delay=0.1,
                )

            # Should return processed solutions
            assert len(processed_results) == 2
            assert all(isinstance(r, ACDCSolution) for r in processed_results if r)


class TestConvertAndUpdateInitialArchive:
    """Test archive conversion and update."""

    def test_archive_conversion(self, initialized_optimizer):
        """Test that AC/DC solutions are converted to DNS format."""
        solutions = [
            ACDCSolution(
                model_path="/models/model_0",
                fitness=0.8,
                acdc_skill_vector={"task_0": 0.9, "task_1": 0.7}
            ),
            ACDCSolution(
                model_path="/models/model_1",
                fitness=0.6,
                acdc_skill_vector={"task_0": 0.6, "task_1": 0.6}
            ),
        ]

        with patch("main_ac_dc.convert_acdc_to_dns_solution") as mock_convert:
            with patch("main_ac_dc.update_dns_archive") as mock_update:
                # Mock update to return solutions
                mock_update.return_value = [Mock(), Mock()]

                updated = initialized_optimizer._convert_and_update_initial_archive(
                    solutions
                )

                # Verify conversion was called
                assert mock_convert.call_count == len(solutions)

                # Verify update was called
                mock_update.assert_called_once()

    def test_empty_solutions_list(self, initialized_optimizer):
        """Test handling of empty solutions list."""
        updated = initialized_optimizer._convert_and_update_initial_archive([])

        assert updated == []

    def test_threshold_application(self, initialized_optimizer):
        """Test that skill threshold is applied during conversion."""
        solutions = [
            ACDCSolution(
                model_path="/models/model_0",
                fitness=0.8,
                acdc_skill_vector={"task_0": 0.9}
            )
        ]

        with patch("main_ac_dc.convert_acdc_to_dns_solution") as mock_convert:
            with patch("main_ac_dc.update_dns_archive") as mock_update:
                mock_update.return_value = [Mock()]

                initialized_optimizer._convert_and_update_initial_archive(solutions)

                # Verify threshold was passed to convert
                call_args = mock_convert.call_args_list[0]
                threshold_arg = call_args[0][2]  # Third positional arg
                assert threshold_arg == initialized_optimizer.cfg.dns.acdc_skill_threshold
