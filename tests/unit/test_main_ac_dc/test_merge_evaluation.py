"""
Unit tests for merge and evaluation workflows in ACDCOptimizer.

Tests cover:
- Merge result processing
- Retry logic for failed merges
- Batch merge operations
- Archive updates after merges
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf

from main_ac_dc import ACDCOptimizer
from datatypes import ACDCMergeResult, ACDCSolution


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
        "seed_model_paths": ["/models/seed1", "/models/seed2"],
        "celery": {"num_workers": 4, "timeout": 300},
        "dns": {
            "population_size": 10,
            "num_model_per_gen": 8,
            "acdc_skill_threshold": 0.5,
            "run_gibberish_check": False,
            "disable_mutation": False,
        },
        "use_ac_dc": True,
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

        # Set up task pool
        mock_task_pool = Mock()
        mock_task_pool.tasks = ["/tasks/task_0"]
        mock_task_pool.get_ordered_task_ids.return_value = ["task_0"]
        optimizer.task_pool = mock_task_pool

        return optimizer


class TestProcessMergeResults:
    """Test merge result processing with retries."""

    def test_successful_merge_processing(self, initialized_optimizer):
        """Test processing all successful merge results."""
        # Set num_model_per_gen to match the number of promises
        initialized_optimizer.cfg.dns.num_model_per_gen = 3

        # Create successful merge results
        promises = []
        for i in range(3):
            p = Mock()
            p.ready.return_value = True
            p.id = f"task_{i}"
            result = ACDCMergeResult(
                save_path=f"/models/gen_1_ind_{i}",
                acdc_skill_vector={"task_0": 0.8},
                avg_acdc_quality=0.8,
            )
            p.get.return_value = result
            promises.append(p)

        task_info = ["/tasks/task_0"]

        solutions = initialized_optimizer.process_merge_results(
            promises, task_info
        )

        # Should have processed all results
        assert len(solutions) == 3
        assert all(isinstance(s, ACDCSolution) for s in solutions if s)

    def test_retry_logic_for_failed_merge(self, initialized_optimizer):
        """Test that failed merges trigger retries."""
        # Set num_model_per_gen to 1 since we're only testing one promise
        initialized_optimizer.cfg.dns.num_model_per_gen = 1

        # Create a failed promise followed by retry
        failed_promise = Mock()
        failed_promise.ready.return_value = True
        failed_promise.get.return_value = None  # Failure
        failed_promise.id = "failed_task"

        retry_promise = Mock()
        retry_promise.ready.return_value = True
        retry_result = ACDCMergeResult(
            save_path="/models/gen_1_ind_0",
            acdc_skill_vector={"task_0": 0.8},
            avg_acdc_quality=0.8,
        )
        retry_promise.get.return_value = retry_result
        retry_promise.id = "retry_task"

        # Mock create_merge_task to return retry promise
        with patch("main_ac_dc.create_merge_task") as mock_create:
            mock_create.return_value = retry_promise

            with patch("main_ac_dc.wandb.alert"):
                with patch("time.sleep"):
                    solutions = initialized_optimizer.process_merge_results(
                        [failed_promise], ["/tasks/task_0"]
                    )

                    # Should have attempted retry
                    mock_create.assert_called_once()

    def test_max_retries_exceeded(self, initialized_optimizer):
        """Test behavior when max retries is exceeded."""
        # Set num_model_per_gen to 1
        initialized_optimizer.cfg.dns.num_model_per_gen = 1

        # Create promises that always fail
        failed_promise = Mock()
        failed_promise.ready.return_value = True
        failed_promise.get.return_value = None  # Always fails
        failed_promise.id = "failed_task"

        with patch("main_ac_dc.create_merge_task") as mock_create:
            # Mock retry also fails
            retry_promise = Mock()
            retry_promise.ready.return_value = True
            retry_promise.get.return_value = None
            retry_promise.id = "retry_failed"
            mock_create.return_value = retry_promise

            with patch("main_ac_dc.wandb.alert"):
                with patch("time.sleep"):
                    solutions = initialized_optimizer.process_merge_results(
                        [failed_promise], ["/tasks/task_0"]
                    )

                    # Should have None for failed merges
                    assert None in solutions

    def test_promise_completion_tracking(self, initialized_optimizer):
        """Test that promise completion is tracked correctly."""
        # Set num_model_per_gen to 3
        initialized_optimizer.cfg.dns.num_model_per_gen = 3

        # Create promises that complete at different times
        promises = []
        for i in range(3):
            p = Mock()
            # First check not ready, second check ready
            p.ready.side_effect = [False, True]
            result = ACDCMergeResult(
                save_path=f"/models/gen_1_ind_{i}",
                acdc_skill_vector={},
                avg_acdc_quality=0.5,
            )
            p.get.return_value = result
            p.id = f"task_{i}"
            promises.append(p)

        with patch("time.sleep"):
            solutions = initialized_optimizer.process_merge_results(
                promises, ["/tasks/task_0"]
            )

            # All should eventually complete
            assert len(solutions) == 3

    def test_model_index_preservation(self, initialized_optimizer):
        """Test that model indices are preserved through processing."""
        # Set num_model_per_gen to 3
        initialized_optimizer.cfg.dns.num_model_per_gen = 3

        # Create promises with specific indices
        promises = []
        for i in [0, 1, 2]:  # Use sequential indices matching num_model_per_gen
            p = Mock()
            p.ready.return_value = True
            result = ACDCMergeResult(
                save_path=f"/models/gen_1_ind_{i}",
                acdc_skill_vector={},
                avg_acdc_quality=0.5,
            )
            p.get.return_value = result
            p.id = f"task_{i}"
            promises.append(p)

        solutions = initialized_optimizer.process_merge_results(
            promises, ["/tasks/task_0"]
        )

        # Should maintain index mapping
        assert len(solutions) == 3
        # Verify solutions are at correct indices
        for i in range(3):
            assert solutions[i] is not None
            assert solutions[i].model_path == f"/models/gen_1_ind_{i}"


class TestHandleFailedMerge:
    """Test failed merge handling logic."""

    def test_retry_creation(self, initialized_optimizer):
        """Test that retry creates new merge task."""
        failed_promise = Mock()
        failed_promise.id = "failed_task"

        promise_map = {}
        retries_attempted = {}
        task_info = ["/tasks/task_0"]

        with patch("main_ac_dc.create_merge_task") as mock_create:
            retry_promise = Mock()
            retry_promise.id = "retry_task"
            mock_create.return_value = retry_promise

            with patch("main_ac_dc.wandb.alert"):
                wandb_alerted, retry_entry = initialized_optimizer._handle_failed_merge(
                    failed_promise,
                    original_index=0,
                    promise_map=promise_map,
                    retries_attempted=retries_attempted,
                    wandb_alerted=False,
                    task_info=task_info,
                )

                # Should have created retry
                mock_create.assert_called_once()

                # The method returns the retry entry, caller should add it to promise_map
                assert retry_entry is not None
                promise_id, (promise, original_index) = retry_entry
                promise_map[promise_id] = (promise, original_index)

                assert retry_promise.id in promise_map
                assert retries_attempted[failed_promise.id] == 1

    def test_max_retries_reached(self, initialized_optimizer):
        """Test behavior when max retries is reached."""
        failed_promise = Mock()
        failed_promise.id = "failed_task"

        promise_map = {}
        retries_attempted = {failed_promise.id: 1}  # Already at max
        task_info = ["/tasks/task_0"]

        with patch("main_ac_dc.create_merge_task") as mock_create:
            with patch("main_ac_dc.wandb.alert"):
                wandb_alerted = initialized_optimizer._handle_failed_merge(
                    failed_promise,
                    original_index=0,
                    promise_map=promise_map,
                    retries_attempted=retries_attempted,
                    wandb_alerted=False,
                    task_info=task_info,
                    max_retries=1,
                )

                # Should NOT create retry
                mock_create.assert_not_called()

    def test_wandb_alerts(self, initialized_optimizer):
        """Test that wandb alerts are sent."""
        failed_promise = Mock()
        failed_promise.id = "failed_task"

        with patch("main_ac_dc.create_merge_task"):
            with patch("main_ac_dc.wandb.alert") as mock_alert:
                initialized_optimizer._handle_failed_merge(
                    failed_promise,
                    original_index=0,
                    promise_map={},
                    retries_attempted={},
                    wandb_alerted=False,
                    task_info=[],
                )

                # Should have sent alert
                mock_alert.assert_called_once()

    def test_exception_vs_failure_handling(self, initialized_optimizer):
        """Test different handling for exceptions vs failures."""
        failed_promise = Mock()
        failed_promise.id = "failed_task"

        with patch("main_ac_dc.create_merge_task"):
            with patch("main_ac_dc.wandb.alert"):
                # Test with exception flag
                initialized_optimizer._handle_failed_merge(
                    failed_promise,
                    original_index=0,
                    promise_map={},
                    retries_attempted={},
                    wandb_alerted=False,
                    task_info=[],
                    is_exception=True,
                )

                # Should handle same as failure (creates retry)


class TestProcessMergesInBatches:
    """Test batch merge processing."""

    def test_batch_processing(self, initialized_optimizer):
        """Test that merges are processed in batches."""
        num_models = 6
        batch_size = 2

        mock_promise = Mock()
        mock_promise.ready.return_value = True
        mock_promise.get.return_value = "/models/saved_path"

        with patch("main_ac_dc.create_merge_only_task") as mock_create:
            mock_create.return_value = mock_promise

            with patch("time.sleep"):
                paths = initialized_optimizer._process_merges_in_batches(
                    num_models=num_models,
                    batch_size=batch_size,
                    delay=0.1,
                )

            # Should have created all tasks
            assert mock_create.call_count == num_models
            assert len(paths) == num_models

    def test_batch_size_limiting(self, initialized_optimizer):
        """Test that batch size limits concurrent operations."""
        num_models = 5
        batch_size = 2

        call_count = 0

        def track_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            p = Mock()
            p.ready.return_value = True
            p.get.return_value = f"/models/model_{call_count}"
            return p

        with patch("main_ac_dc.create_merge_only_task") as mock_create:
            mock_create.side_effect = track_calls

            with patch("time.sleep"):
                paths = initialized_optimizer._process_merges_in_batches(
                    num_models=num_models,
                    batch_size=batch_size,
                    delay=0.1,
                )

            assert call_count == num_models

    def test_result_collection(self, initialized_optimizer):
        """Test that results are collected correctly."""
        num_models = 3

        results = [f"/models/gen_1_ind_{i}" for i in range(num_models)]

        def create_promise(idx):
            p = Mock()
            p.ready.return_value = True
            p.get.return_value = results[idx]
            return p

        call_idx = [0]

        def track_create(*args, **kwargs):
            p = create_promise(call_idx[0])
            call_idx[0] += 1
            return p

        with patch("main_ac_dc.create_merge_only_task") as mock_create:
            mock_create.side_effect = track_create

            with patch("time.sleep"):
                paths = initialized_optimizer._process_merges_in_batches(
                    num_models=num_models,
                    batch_size=2,
                    delay=0.1,
                )

            assert len(paths) == num_models
            assert all(p in results for p in paths)


class TestCreateMergeTasks:
    """Test merge task creation."""

    def test_task_creation(self, initialized_optimizer):
        """Test that merge tasks are created correctly."""
        task_info = ["/tasks/task_0"]

        with patch("main_ac_dc.create_merge_task") as mock_create:
            mock_create.return_value = Mock()

            promises = initialized_optimizer._create_merge_tasks(task_info)

            # Should create tasks for all workers
            assert mock_create.call_count == initialized_optimizer.cfg.celery.num_workers
            assert len(promises) == initialized_optimizer.cfg.celery.num_workers

    def test_worker_count_matching(self, initialized_optimizer):
        """Test that number of tasks matches worker count."""
        task_info = []

        with patch("main_ac_dc.create_merge_task") as mock_create:
            mock_create.return_value = Mock()

            promises = initialized_optimizer._create_merge_tasks(task_info)

            assert len(promises) == initialized_optimizer.cfg.celery.num_workers

    def test_task_info_propagation(self, initialized_optimizer):
        """Test that task_info is passed to merge tasks."""
        task_info = ["/tasks/task_0", "/tasks/task_1"]

        with patch("main_ac_dc.create_merge_task") as mock_create:
            mock_create.return_value = Mock()

            initialized_optimizer._create_merge_tasks(task_info)

            # Verify task_info was passed to all calls
            for call in mock_create.call_args_list:
                assert call[1]["task_info"] == task_info


class TestUpdateArchiveAfterMerge:
    """Test archive update after merging."""

    def test_archive_update_with_new_solutions(self, initialized_optimizer):
        """Test updating archive with new solutions."""
        new_solutions = [
            ACDCSolution(
                model_path=f"/models/gen_1_ind_{i}",
                fitness=0.7 + i * 0.05,
                acdc_skill_vector={"task_0": 0.8},
            )
            for i in range(3)
        ]

        # Set existing archive
        initialized_optimizer.archive_data["dns_archive"] = [
            ACDCSolution(
                model_path="/models/gen_0_ind_0",
                fitness=0.6,
                acdc_skill_vector={"task_0": 0.6},
            )
        ]

        with patch("main_ac_dc.convert_acdc_to_dns_solution") as mock_convert:
            with patch("main_ac_dc.update_dns_archive") as mock_update:
                mock_update.return_value = [Mock(model_path=s.model_path) for s in new_solutions]

                initialized_optimizer._update_archive_after_merge(new_solutions)

                # Should have called update
                mock_update.assert_called_once()

    def test_gibberish_filtering(self, initialized_optimizer):
        """Test that gibberish models are filtered."""
        initialized_optimizer.cfg.dns.run_gibberish_check = True

        new_solutions = [
            ACDCSolution(
                model_path="/models/good",
                fitness=0.8,
                acdc_skill_vector={},
                is_gibberish=False,
            ),
            ACDCSolution(
                model_path="/models/gibberish",
                fitness=0.1,
                acdc_skill_vector={},
                is_gibberish=True,
            ),
        ]

        with patch("main_ac_dc.convert_acdc_to_dns_solution"):
            with patch("main_ac_dc.update_dns_archive") as mock_update:
                mock_update.return_value = []

                initialized_optimizer._update_archive_after_merge(new_solutions)

                # Should have incremented gibberish counter
                assert initialized_optimizer.gibberish_models_counter == 1

    def test_conversion_to_dns_and_back(self, initialized_optimizer):
        """Test AC/DC solution conversion to DNS and back."""
        new_solutions = [
            ACDCSolution(
                model_path="/models/test",
                fitness=0.8,
                acdc_skill_vector={"task_0": 0.8},
            )
        ]

        with patch("main_ac_dc.convert_acdc_to_dns_solution") as mock_convert:
            dns_solution = Mock(model_path="/models/test")
            mock_convert.return_value = dns_solution

            with patch("main_ac_dc.update_dns_archive") as mock_update:
                mock_update.return_value = [dns_solution]

                initialized_optimizer._update_archive_after_merge(new_solutions)

                # Verify conversion happened
                mock_convert.assert_called()

    def test_archive_size_management(self, initialized_optimizer):
        """Test that archive respects size limits."""
        # Create many new solutions
        new_solutions = [
            ACDCSolution(
                model_path=f"/models/gen_1_ind_{i}",
                fitness=0.8,
                acdc_skill_vector={},
            )
            for i in range(20)
        ]

        with patch("main_ac_dc.convert_acdc_to_dns_solution"):
            with patch("main_ac_dc.update_dns_archive") as mock_update:
                # Mock returns only population_size solutions with matching model_paths
                # Use the first N model paths from new_solutions to match acdc_solution_map
                kept_solutions = [
                    Mock(model_path=f"/models/gen_1_ind_{i}")
                    for i in range(initialized_optimizer.cfg.dns.population_size)
                ]
                mock_update.return_value = kept_solutions

                initialized_optimizer._update_archive_after_merge(new_solutions)

                # Archive should respect population size
                assert len(initialized_optimizer.archive_data["dns_archive"]) == \
                       initialized_optimizer.cfg.dns.population_size
