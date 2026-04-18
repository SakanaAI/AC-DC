"""
Unit tests for task pool adaptation in ACDCOptimizer.

Tests cover:
- Task pool adaptation at intervals
- Skill vector synchronization
- Re-evaluation on new tasks
- Archive filtering
- Active pool state logging
"""

import pytest
import json
from pathlib import Path
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
        "seed_model_paths": ["/models/seed1"],
        "celery": {"num_workers": 4, "timeout": 300},
        "dns": {
            "population_size": 10,
            "acdc_skill_threshold": 0.5,
        },
        "use_ac_dc": True,
        "acdc": {
            "task_generation_interval": 5,
        },
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
        mock_task_pool.tasks = ["/tasks/task_0", "/tasks/task_1"]
        mock_task_pool.get_ordered_task_ids.return_value = ["task_0", "task_1"]
        mock_task_pool.active_limbo_map = {}
        optimizer.task_pool = mock_task_pool

        # Set up archive with some solutions
        optimizer.archive_data["dns_archive"] = [
            ACDCSolution(
                model_path="/models/gen_1_ind_0",
                fitness=0.8,
                acdc_skill_vector={"task_0": 0.8, "task_1": 0.7},
            ),
            ACDCSolution(
                model_path="/models/gen_1_ind_1",
                fitness=0.6,
                acdc_skill_vector={"task_0": 0.6, "task_1": 0.6},
            ),
        ]

        return optimizer


class TestAdaptTaskPoolAndReevaluateArchive:
    """Test task pool adaptation workflow."""

    def test_task_pool_adaptation_at_interval(self, initialized_optimizer):
        """Test that task pool is adapted at the configured interval."""
        initialized_optimizer.gen = 5  # At interval

        # Set up task pool to return task_2 as the new active task
        initialized_optimizer.task_pool.get_ordered_task_ids.return_value = ["task_0", "task_1", "task_2"]
        initialized_optimizer.task_pool.tasks = ["/tasks/task_0", "/tasks/task_1", "/tasks/task_2"]

        # Archive models only have task_0 and task_1, missing task_2
        # This simulates the state after new tasks are added to the pool

        with patch.object(
            initialized_optimizer.task_pool, "adapt_task_pool"
        ) as mock_adapt:
            mock_adapt.return_value = ["/tasks/task_2"]  # New task added

            with patch.object(
                initialized_optimizer, "_reevaluate_archive_on_new_tasks"
            ) as mock_reeval:
                with patch.object(
                    initialized_optimizer, "_save_post_adaptation_archive"
                ):
                    with patch.object(
                        initialized_optimizer, "_log_active_task_pool_state"
                    ):
                        initialized_optimizer._adapt_task_pool_and_reevaluate_archive()

                        # Should have called adapt
                        mock_adapt.assert_called_once()
                        # Should have re-evaluated on the missing task (task_2)
                        # The call will include the full path
                        mock_reeval.assert_called_once()
                        # Verify the task_2 path is in the call
                        call_args = mock_reeval.call_args[0][0]
                        assert any("task_2" in path for path in call_args)

    def test_no_adaptation_between_intervals(self, initialized_optimizer):
        """Test that adaptation is skipped between intervals."""
        initialized_optimizer.gen = 3  # Not at interval (interval is 5)

        with patch.object(
            initialized_optimizer.task_pool, "adapt_task_pool"
        ) as mock_adapt:
            with patch.object(
                initialized_optimizer, "_save_post_adaptation_archive"
            ):
                with patch.object(
                    initialized_optimizer, "_log_active_task_pool_state"
                ):
                    initialized_optimizer._adapt_task_pool_and_reevaluate_archive()

                    # Should NOT have called adapt
                    mock_adapt.assert_not_called()

    def test_skill_vector_synchronization(self, initialized_optimizer):
        """Test that skill vectors are synchronized to active pool."""
        # Add task_2 to skill vectors but not in active pool
        for solution in initialized_optimizer.archive_data["dns_archive"]:
            solution.acdc_skill_vector["task_2"] = 0.5  # Not in active pool

        initialized_optimizer.gen = 5

        with patch.object(
            initialized_optimizer.task_pool, "adapt_task_pool"
        ) as mock_adapt:
            mock_adapt.return_value = []  # No new tasks

            with patch.object(
                initialized_optimizer, "_save_post_adaptation_archive"
            ):
                with patch.object(
                    initialized_optimizer, "_log_active_task_pool_state"
                ):
                    initialized_optimizer._adapt_task_pool_and_reevaluate_archive()

                    # task_2 should be removed from skill vectors
                    for solution in initialized_optimizer.archive_data["dns_archive"]:
                        assert "task_2" not in solution.acdc_skill_vector

    def test_post_adaptation_archive_saving(self, initialized_optimizer):
        """Test that archive is saved after adaptation."""
        initialized_optimizer.gen = 5

        with patch.object(initialized_optimizer.task_pool, "adapt_task_pool"):
            with patch.object(
                initialized_optimizer, "_save_post_adaptation_archive"
            ) as mock_save:
                with patch.object(
                    initialized_optimizer, "_log_active_task_pool_state"
                ):
                    initialized_optimizer._adapt_task_pool_and_reevaluate_archive()

                    # Should save archive
                    mock_save.assert_called_once()

    def test_active_pool_state_logging(self, initialized_optimizer):
        """Test that active pool state is logged."""
        initialized_optimizer.gen = 5

        with patch.object(initialized_optimizer.task_pool, "adapt_task_pool"):
            with patch.object(
                initialized_optimizer, "_save_post_adaptation_archive"
            ):
                with patch.object(
                    initialized_optimizer, "_log_active_task_pool_state"
                ) as mock_log:
                    initialized_optimizer._adapt_task_pool_and_reevaluate_archive()

                    # Should log pool state
                    mock_log.assert_called_once()


class TestReevaluateArchiveOnNewTasks:
    """Test re-evaluation of archive on new tasks."""

    def test_reevaluation_promise_creation(self, initialized_optimizer):
        """Test that re-evaluation promises are created for all archive models."""
        new_tasks = ["/tasks/task_2"]

        mock_promise = Mock()
        mock_promise.ready.return_value = True
        mock_result = ACDCMergeResult(
            save_path="/models/gen_1_ind_0",
            acdc_skill_vector={"task_2": 0.7},
            avg_acdc_quality=0.7,
        )
        mock_promise.get.return_value = mock_result

        initialized_optimizer.call_fn.delay = Mock(return_value=mock_promise)

        initialized_optimizer._reevaluate_archive_on_new_tasks(new_tasks)

        # Should create promises for all models in archive
        assert initialized_optimizer.call_fn.delay.call_count == len(
            initialized_optimizer.archive_data["dns_archive"]
        )

    def test_skill_vector_updates(self, initialized_optimizer):
        """Test that skill vectors are updated with new task results."""
        new_tasks = ["/tasks/task_2"]

        # Mock evaluation results
        results = [
            ACDCMergeResult(
                save_path="/models/gen_1_ind_0",
                acdc_skill_vector={"task_2": 0.9},
                avg_acdc_quality=0.9,
            ),
            ACDCMergeResult(
                save_path="/models/gen_1_ind_1",
                acdc_skill_vector={"task_2": 0.5},
                avg_acdc_quality=0.5,
            ),
        ]

        promises = [Mock() for _ in results]
        for p, r in zip(promises, results):
            p.ready.return_value = True
            p.get.return_value = r

        initialized_optimizer.call_fn.delay = Mock(side_effect=promises)

        initialized_optimizer._reevaluate_archive_on_new_tasks(new_tasks)

        # Skill vectors should be updated
        assert "task_2" in initialized_optimizer.archive_data["dns_archive"][0].acdc_skill_vector
        assert "task_2" in initialized_optimizer.archive_data["dns_archive"][1].acdc_skill_vector

    def test_fitness_recalculation(self, initialized_optimizer):
        """Test that fitness is recalculated after skill vector update."""
        new_tasks = ["/tasks/task_2"]

        original_fitness = [
            s.fitness for s in initialized_optimizer.archive_data["dns_archive"]
        ]

        mock_result = ACDCMergeResult(
            save_path="/models/gen_1_ind_0",
            acdc_skill_vector={"task_2": 0.9},
            avg_acdc_quality=0.9,
        )

        mock_promise = Mock()
        mock_promise.ready.return_value = True
        mock_promise.get.return_value = mock_result

        initialized_optimizer.call_fn.delay = Mock(return_value=mock_promise)

        initialized_optimizer._reevaluate_archive_on_new_tasks(new_tasks)

        # Fitness should be recalculated (different from original)
        new_fitness = [
            s.fitness for s in initialized_optimizer.archive_data["dns_archive"]
        ]

        # At least one should have changed
        assert new_fitness != original_fitness

    def test_missing_evaluation_results_handling(self, initialized_optimizer):
        """Test handling when some evaluations fail."""
        new_tasks = ["/tasks/task_2"]

        # First succeeds, second fails
        promises = [Mock(), Mock()]
        promises[0].ready.return_value = True
        promises[0].get.return_value = ACDCMergeResult(
            save_path="/models/gen_1_ind_0",
            acdc_skill_vector={"task_2": 0.8},
            avg_acdc_quality=0.8,
        )
        promises[1].ready.return_value = True
        promises[1].get.return_value = None  # Failed

        initialized_optimizer.call_fn.delay = Mock(side_effect=promises)

        # Should handle gracefully
        initialized_optimizer._reevaluate_archive_on_new_tasks(new_tasks)

        # First should have task_2, second should not
        assert "task_2" in initialized_optimizer.archive_data["dns_archive"][0].acdc_skill_vector
        # Second model's fitness should still be recalculated based on existing tasks


class TestFilterArchiveSkillVectorsToActivePool:
    """Test skill vector filtering."""

    def test_skill_vector_filtering(self, initialized_optimizer):
        """Test that skill vectors are filtered to active tasks."""
        # Add extra tasks to skill vectors
        for solution in initialized_optimizer.archive_data["dns_archive"]:
            solution.acdc_skill_vector["task_removed"] = 0.5

        initialized_optimizer._filter_archive_skill_vectors_to_active_pool()

        # task_removed should be gone
        for solution in initialized_optimizer.archive_data["dns_archive"]:
            assert "task_removed" not in solution.acdc_skill_vector
            # Original tasks should remain
            assert "task_0" in solution.acdc_skill_vector

    def test_fitness_recalculation_after_filtering(self, initialized_optimizer):
        """Test that fitness is recalculated after filtering."""
        # Add tasks with high scores to existing skill vectors
        for solution in initialized_optimizer.archive_data["dns_archive"]:
            solution.acdc_skill_vector["high_score_task"] = 1.0
            # Recalculate fitness to include the new high score task
            # Average will be higher with the 1.0 score included
            from main_ac_dc import calculate_fitness_from_skill_vector
            solution.fitness = calculate_fitness_from_skill_vector(solution.acdc_skill_vector)

        original_fitness = [
            s.fitness for s in initialized_optimizer.archive_data["dns_archive"]
        ]

        # Filter (removes high_score_task) - active pool only has task_0 and task_1
        initialized_optimizer._filter_archive_skill_vectors_to_active_pool()

        new_fitness = [
            s.fitness for s in initialized_optimizer.archive_data["dns_archive"]
        ]

        # Fitness should be lower now (removed high score task)
        for orig, new in zip(original_fitness, new_fitness):
            assert new < orig

    def test_with_empty_active_pool(self, initialized_optimizer):
        """Test behavior with empty active pool (edge case)."""
        initialized_optimizer.task_pool.get_ordered_task_ids.return_value = []

        initialized_optimizer._filter_archive_skill_vectors_to_active_pool()

        # All skill vectors should be empty
        for solution in initialized_optimizer.archive_data["dns_archive"]:
            assert len(solution.acdc_skill_vector) == 0


class TestLogActiveTaskPoolState:
    """Test active pool state logging."""

    def test_active_pool_logging(self, initialized_optimizer):
        """Test that active pool is logged to JSON."""
        initialized_optimizer.gen = 5

        initialized_optimizer._log_active_task_pool_state()

        # Check that file was created
        pool_log_path = (
            Path(initialized_optimizer.dirs["generated_tasks_dir"])
            / f"active_pool_gen_{initialized_optimizer.gen}.json"
        )

        assert pool_log_path.exists()

        # Verify content
        with open(pool_log_path) as f:
            logged_tasks = json.load(f)

        assert logged_tasks == initialized_optimizer.task_pool.tasks

    def test_limbo_map_logging(self, initialized_optimizer):
        """Test that limbo map is logged to JSON."""
        initialized_optimizer.gen = 5
        initialized_optimizer.task_pool.active_limbo_map = {
            "limbo_task_1": 3,
            "limbo_task_2": 1,
        }

        initialized_optimizer._log_active_task_pool_state()

        # Check that limbo file was created
        limbo_log_path = (
            Path(initialized_optimizer.dirs["generated_tasks_dir"])
            / f"active_limbo_map_gen_{initialized_optimizer.gen}.json"
        )

        assert limbo_log_path.exists()

        # Verify content
        with open(limbo_log_path) as f:
            logged_limbo = json.load(f)

        assert logged_limbo == initialized_optimizer.task_pool.active_limbo_map

    def test_file_creation(self, initialized_optimizer):
        """Test that log files are created in correct directory."""
        initialized_optimizer.gen = 10

        initialized_optimizer._log_active_task_pool_state()

        pool_dir = Path(initialized_optimizer.dirs["generated_tasks_dir"])

        # Check both files exist
        assert (pool_dir / "active_pool_gen_10.json").exists()
        assert (pool_dir / "active_limbo_map_gen_10.json").exists()


class TestTaskPoolAdaptationIntegration:
    """Integration tests for task pool adaptation."""

    def test_full_adaptation_cycle(self, initialized_optimizer):
        """Test complete adaptation cycle: adapt → save → log (and reeval if missing tasks)."""
        initialized_optimizer.gen = 5  # At interval

        # Set up task pool to include a new task that archive models don't have
        initialized_optimizer.task_pool.get_ordered_task_ids.return_value = ["task_0", "task_1", "task_new"]
        initialized_optimizer.task_pool.tasks = ["/tasks/task_0", "/tasks/task_1", "/tasks/task_new"]

        # Track all operations
        operations = []

        def track_adapt(*args, **kwargs):
            operations.append("adapt")
            return ["/tasks/task_new"]

        def track_reeval(*args, **kwargs):
            operations.append("reeval")

        def track_save():
            operations.append("save")

        def track_log():
            operations.append("log")

        with patch.object(
            initialized_optimizer.task_pool, "adapt_task_pool", side_effect=track_adapt
        ):
            with patch.object(
                initialized_optimizer,
                "_reevaluate_archive_on_new_tasks",
                side_effect=track_reeval,
            ):
                with patch.object(
                    initialized_optimizer,
                    "_save_post_adaptation_archive",
                    side_effect=track_save,
                ):
                    with patch.object(
                        initialized_optimizer,
                        "_log_active_task_pool_state",
                        side_effect=track_log,
                    ):
                        initialized_optimizer._adapt_task_pool_and_reevaluate_archive()

        # Verify order of operations - reeval happens because models are missing task_new
        assert operations == ["adapt", "reeval", "save", "log"]

    def test_synchronization_with_missing_tasks(self, initialized_optimizer):
        """Test synchronization when all models are missing the same active tasks."""
        # Remove task_1 from ALL models' skill vectors (simulating new task added to pool)
        for model in initialized_optimizer.archive_data["dns_archive"]:
            model.acdc_skill_vector = {"task_0": 0.8}

        initialized_optimizer.gen = 5

        # Mock re-evaluation results for all models
        results = []
        for i, model in enumerate(initialized_optimizer.archive_data["dns_archive"]):
            mock_result = ACDCMergeResult(
                save_path=model.model_path,
                acdc_skill_vector={"task_1": 0.7},
                avg_acdc_quality=0.7,
            )
            mock_promise = Mock()
            mock_promise.ready.return_value = True
            mock_promise.get.return_value = mock_result
            results.append(mock_promise)

        initialized_optimizer.call_fn.delay = Mock(side_effect=results)

        with patch.object(initialized_optimizer.task_pool, "adapt_task_pool"):
            with patch.object(
                initialized_optimizer, "_save_post_adaptation_archive"
            ):
                with patch.object(
                    initialized_optimizer, "_log_active_task_pool_state"
                ):
                    initialized_optimizer._adapt_task_pool_and_reevaluate_archive()

        # All models should now have task_1
        for model in initialized_optimizer.archive_data["dns_archive"]:
            assert "task_1" in model.acdc_skill_vector
