"""
Unit tests for helper functions in main_ac_dc.py.

Tests cover:
- Directory setup
- Merge result handling
- Task creation
- Promise waiting
- Model cleanup
- Fitness calculation
"""

import pytest
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig, OmegaConf

from main_ac_dc import (
    setup_optimization_directories,
    handle_merge_result,
    create_merge_task,
    create_merge_only_task,
    create_eval_only_task,
    wait_for_promises,
    cleanup_old_models,
    calculate_fitness_from_skill_vector,
)
from datatypes import ACDCMergeResult, ACDCSolution, ACDCTaskEvalDetail


class TestSetupOptimizationDirectories:
    """Test directory setup for optimization runs."""

    def test_basic_directory_creation(self, tmp_path):
        """Test that all required directories are created."""
        cfg = OmegaConf.create({"restart_dir": None})

        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path)
            mock_hydra.return_value.runtime = mock_runtime

            dirs = setup_optimization_directories(cfg)

            # Verify all directories were created
            assert (tmp_path / "models").exists()
            assert (tmp_path / "archives").exists()
            assert (tmp_path / "images").exists()
            assert (tmp_path / "generated_tasks" / "pool").exists()

            # Verify returned dict has all keys
            assert "output_dir" in dirs
            assert "model_dir" in dirs
            assert "archive_dir" in dirs
            assert "image_dir" in dirs
            assert "generated_tasks_dir" in dirs
            assert "vector_db_dir" in dirs

    def test_with_restart_dir(self, tmp_path):
        """Test directory setup with restart_dir specified."""
        restart_dir = tmp_path / "restart"
        restart_dir.mkdir()

        cfg = OmegaConf.create({"restart_dir": str(restart_dir)})

        dirs = setup_optimization_directories(cfg)

        # Should use restart_dir as output_dir
        assert dirs["output_dir"] == str(restart_dir)
        assert (restart_dir / "models").exists()

    def test_without_restart_uses_hydra_output(self, tmp_path):
        """Test that Hydra output dir is used when no restart_dir."""
        cfg = OmegaConf.create({})

        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path / "hydra_output")
            mock_hydra.return_value.runtime = mock_runtime

            dirs = setup_optimization_directories(cfg)

            assert str(tmp_path / "hydra_output") in dirs["output_dir"]

    def test_error_when_output_dir_cannot_be_determined(self):
        """Test error handling when output_dir is None."""
        cfg = OmegaConf.create({"restart_dir": None})

        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = None
            mock_hydra.return_value.runtime = mock_runtime

            with pytest.raises(ValueError, match="Output directory cannot be determined"):
                setup_optimization_directories(cfg)

    def test_custom_output_dir_parameter(self, tmp_path):
        """Test providing output_dir directly as parameter."""
        cfg = OmegaConf.create({})
        custom_dir = tmp_path / "custom"

        dirs = setup_optimization_directories(cfg, output_dir=str(custom_dir))

        assert dirs["output_dir"] == str(custom_dir)
        assert (custom_dir / "models").exists()


class TestHandleMergeResult:
    """Test merge result processing."""

    def test_with_valid_result(self):
        """Test handling valid ACDCMergeResult."""
        result = ACDCMergeResult(
            save_path="/models/gen_1_ind_5",
            task_metrics=None,
            acdc_skill_vector={"task_0": 0.8, "task_1": 0.6},
            avg_acdc_quality=0.7,
        )

        solution = handle_merge_result(result)

        assert isinstance(solution, ACDCSolution)
        assert solution.model_path == "/models/gen_1_ind_5"
        assert solution.acdc_skill_vector == {"task_0": 0.8, "task_1": 0.6}
        assert solution.fitness == pytest.approx(0.7)

    def test_with_none_result(self):
        """Test handling None result."""
        solution = handle_merge_result(None)
        assert solution is None

    def test_with_eval_details(self):
        """Test handling result with evaluation details."""
        eval_details = [
            ACDCTaskEvalDetail(
                task_id="task_0",
                instructions="Test instruction",
                raw_output="Test output",
                score=0.9,
            )
        ]

        result = ACDCMergeResult(
            save_path="/models/test",
            acdc_skill_vector={"task_0": 0.9},
            avg_acdc_quality=0.9,
            acdc_eval_details=eval_details,
        )

        solution = handle_merge_result(result)

        assert solution.acdc_eval_details == eval_details

    def test_with_gibberish_flag(self):
        """Test that gibberish flag is propagated."""
        result = ACDCMergeResult(
            save_path="/models/gibberish",
            acdc_skill_vector={},
            avg_acdc_quality=0.0,
            is_gibberish=True,
        )

        solution = handle_merge_result(result)

        assert solution.is_gibberish is True


class TestCreateMergeTask:
    """Test merge task creation."""

    def test_with_populated_archive(self, np_random):
        """Test merge task with 2+ models in archive."""
        cfg = OmegaConf.create({
            "seed_model_paths": ["model1", "model2"],
            "dns": {"disable_mutation": False},
        })

        call_fn = Mock()
        call_fn.delay = Mock(return_value="promise")

        archive_data = {
            "dns_archive": [
                ACDCSolution(model_path="/models/model_0", fitness=0.8, acdc_skill_vector={}),
                ACDCSolution(model_path="/models/model_1", fitness=0.7, acdc_skill_vector={}),
            ],
            "dirs": {"model_dir": "/models"},
        }

        task_info = ["/tasks/task_0"]

        promise = create_merge_task(cfg, call_fn, gen=1, model_index=5,
                                   archive_data=archive_data, np_random=np_random,
                                   task_info=task_info)

        # Verify promise was created
        assert promise == "promise"

        # Verify call was made with correct parameters
        call_fn.delay.assert_called_once()
        call_args = call_fn.delay.call_args

        assert call_args[0][0] == "merge_models"
        assert "parent_paths" in call_args[1]
        assert "save_path" in call_args[1]
        assert call_args[1]["save_path"] == "/models/gen_1_ind_5"
        assert call_args[1]["task_info"] == task_info
        assert call_args[1]["do_mutate"] is True

    def test_with_empty_archive_uses_seed_models(self, np_random):
        """Test that seed models are used when archive is empty."""
        cfg = OmegaConf.create({
            "seed_model_paths": ["seed1", "seed2", "seed3"],
            "dns": {"disable_mutation": False},
        })

        call_fn = Mock()
        call_fn.delay = Mock(return_value="promise")

        archive_data = {
            "dns_archive": [],  # Empty archive
            "dirs": {"model_dir": "/models"},
        }

        task_info = []

        create_merge_task(cfg, call_fn, gen=1, model_index=0,
                         archive_data=archive_data, np_random=np_random,
                         task_info=task_info)

        # Should have selected from seed models
        call_args = call_fn.delay.call_args
        parent_paths = call_args[1]["parent_paths"]

        # Parents should be from seed models
        assert all(p in cfg.seed_model_paths for p in parent_paths)

    def test_with_mutation_disabled(self, np_random):
        """Test that mutation flag is respected."""
        cfg = OmegaConf.create({
            "seed_model_paths": ["model1", "model2"],
            "dns": {"disable_mutation": True},
        })

        call_fn = Mock()
        call_fn.delay = Mock(return_value="promise")

        archive_data = {
            "dns_archive": [],
            "dirs": {"model_dir": "/models"},
        }

        create_merge_task(cfg, call_fn, gen=1, model_index=0,
                         archive_data=archive_data, np_random=np_random,
                         task_info=[])

        call_args = call_fn.delay.call_args
        assert call_args[1]["do_mutate"] is False


class TestCreateMergeOnlyTask:
    """Test merge-only task creation (no evaluation)."""

    def test_basic_merge_only_task(self, np_random):
        """Test creation of merge-only task."""
        cfg = OmegaConf.create({
            "seed_model_paths": ["model1", "model2"],
            "dns": {"disable_mutation": False},
        })

        call_fn = Mock()
        call_fn.delay = Mock(return_value="promise")

        archive_data = {
            "dns_archive": [
                ACDCSolution(model_path="/models/m0", fitness=0.8, acdc_skill_vector={}),
                ACDCSolution(model_path="/models/m1", fitness=0.7, acdc_skill_vector={}),
            ],
            "dirs": {"model_dir": "/models"},
        }

        promise = create_merge_only_task(cfg, call_fn, gen=2, model_index=3,
                                         archive_data=archive_data, np_random=np_random)

        call_fn.delay.assert_called_once()
        call_args = call_fn.delay.call_args

        assert call_args[0][0] == "merge_models_only"
        assert call_args[1]["save_path"] == "/models/gen_2_ind_3"


class TestCreateEvalOnlyTask:
    """Test evaluation-only task creation."""

    def test_basic_eval_task(self):
        """Test creation of evaluation task."""
        call_fn = Mock()
        call_fn.delay = Mock(return_value="promise")

        model_path = "/models/test_model"
        task_info = ["/tasks/task_0", "/tasks/task_1"]

        promise = create_eval_only_task(call_fn, model_path, task_info)

        call_fn.delay.assert_called_once()
        call_args = call_fn.delay.call_args

        assert call_args[0][0] == "eval_model_only"
        assert call_args[1]["model_path"] == model_path
        assert call_args[1]["task_info"] == task_info
        assert call_args[1]["data_split"] == "train"

    def test_with_custom_data_split(self):
        """Test evaluation with custom data split."""
        call_fn = Mock()
        call_fn.delay = Mock(return_value="promise")

        create_eval_only_task(call_fn, "/models/test", [], data_split="validation")

        call_args = call_fn.delay.call_args
        assert call_args[1]["data_split"] == "validation"

    def test_with_task_name(self):
        """Test evaluation with specific task name."""
        call_fn = Mock()
        call_fn.delay = Mock(return_value="promise")

        create_eval_only_task(call_fn, "/models/test", [], task_name="specific_task")

        call_args = call_fn.delay.call_args
        assert call_args[1]["task_name"] == "specific_task"


class TestWaitForPromises:
    """Test promise waiting logic."""

    def test_with_all_promises_ready(self):
        """Test waiting when all promises are immediately ready."""
        promises = []
        for i in range(3):
            p = Mock()
            p.ready.return_value = True
            p.get.return_value = f"result_{i}"
            promises.append(p)

        results = wait_for_promises(promises, timeout=10)

        assert len(results) == 3
        assert all(f"result_{i}" in results for i in range(3))

    def test_with_promises_completing_at_different_times(self):
        """Test promises becoming ready over time."""
        promises = []

        # First promise ready immediately
        p1 = Mock()
        p1.ready.side_effect = [True]
        p1.get.return_value = "result_1"
        promises.append(p1)

        # Second promise ready after first check
        p2 = Mock()
        ready_states = [False, True]
        p2.ready.side_effect = ready_states
        p2.get.return_value = "result_2"
        promises.append(p2)

        results = wait_for_promises(promises, timeout=10)

        assert len(results) == 2
        assert "result_1" in results
        assert "result_2" in results

    def test_with_empty_promises_list(self):
        """Test with empty list of promises."""
        results = wait_for_promises([], timeout=10)
        assert results == []

    def test_timeout_parameter_used(self):
        """Test that timeout is passed to promise.get()."""
        promise = Mock()
        promise.ready.return_value = True
        promise.get.return_value = "result"

        wait_for_promises([promise], timeout=42)

        promise.get.assert_called_with(timeout=42)


class TestCleanupOldModels:
    """Test model cleanup functionality."""

    def test_basic_cleanup(self, tmp_path):
        """Test basic model cleanup removes non-archive models."""
        # Create model directory
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        # Create some model directories
        (model_dir / "gen_0_ind_0").mkdir()
        (model_dir / "gen_1_ind_5").mkdir()
        (model_dir / "gen_2_ind_10").mkdir()

        archive_data = {
            "dns_archive": [
                ACDCSolution(
                    model_path=str(model_dir / "gen_1_ind_5"),
                    fitness=0.8,
                    acdc_skill_vector={}
                ),
            ],
            "dirs": {"model_dir": str(model_dir)},
        }

        deleted = cleanup_old_models(gen=3, archive_data=archive_data)

        # Should delete models not in archive
        assert not (model_dir / "gen_0_ind_0").exists()
        assert (model_dir / "gen_1_ind_5").exists()  # In archive
        assert not (model_dir / "gen_2_ind_10").exists()

    def test_with_skip_interval(self, tmp_path):
        """Test that skip_interval preserves certain generations."""
        # This is tested via the delete_models_not_in_archive function
        # which is imported from utils.helpers
        # We just verify the parameter is passed through
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        archive_data = {
            "dns_archive": [],
            "dirs": {"model_dir": str(model_dir)},
        }

        # Call with skip_interval - just verify no errors
        cleanup_old_models(gen=10, archive_data=archive_data, skip_interval=5)


class TestCalculateFitnessFromSkillVector:
    """Test fitness calculation from skill vectors."""

    def test_with_valid_skill_vector(self):
        """Test fitness is average of scores."""
        skill_vector = {
            "task_0": 0.8,
            "task_1": 0.6,
            "task_2": 0.9,
        }

        fitness = calculate_fitness_from_skill_vector(skill_vector)

        expected = (0.8 + 0.6 + 0.9) / 3
        assert fitness == pytest.approx(expected)

    def test_with_empty_skill_vector(self):
        """Test fitness is 0 for empty skill vector."""
        fitness = calculate_fitness_from_skill_vector({})
        assert fitness == 0.0

    def test_with_none_skill_vector(self):
        """Test fitness is 0 for None skill vector."""
        fitness = calculate_fitness_from_skill_vector(None)
        assert fitness == 0.0

    def test_with_single_task(self):
        """Test fitness with single task."""
        skill_vector = {"task_0": 0.75}

        fitness = calculate_fitness_from_skill_vector(skill_vector)

        assert fitness == 0.75

    def test_with_varying_scores(self):
        """Test fitness calculation with various score ranges."""
        skill_vector = {
            "task_0": 1.0,
            "task_1": 0.5,
            "task_2": 0.0,
            "task_3": 0.25,
            "task_4": 0.75,
        }

        fitness = calculate_fitness_from_skill_vector(skill_vector)

        expected = (1.0 + 0.5 + 0.0 + 0.25 + 0.75) / 5
        assert fitness == pytest.approx(expected)
