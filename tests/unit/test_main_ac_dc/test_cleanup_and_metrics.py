"""
Unit tests for cleanup and metrics logging in ACDCOptimizer.

Tests cover:
- Model cleanup at intervals
- Metrics calculation and logging
- Archive saving
- Generation advancement
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from omegaconf import OmegaConf

from main_ac_dc import ACDCOptimizer
from datatypes import ACDCSolution


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
            "max_details_to_log": 5,
            "run_gibberish_check": True,
        },
        "validation_tasks": [],
        "disk_cleaning_interval": 10,
        "model_cleanup_skip_interval": 5,
        "save_init_gen_models": True,
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

        # Set up archive with some solutions
        optimizer.archive_data["dns_archive"] = [
            ACDCSolution(
                model_path=str(tmp_path / "models" / "gen_1_ind_0"),
                fitness=0.8,
                acdc_skill_vector={"task_0": 0.8, "task_1": 0.7},
            ),
            ACDCSolution(
                model_path=str(tmp_path / "models" / "gen_1_ind_1"),
                fitness=0.6,
                acdc_skill_vector={"task_0": 0.6, "task_1": 0.6},
            ),
        ]

        # Set up tasks
        optimizer.tasks = [Mock(task_name="task_0"), Mock(task_name="task_1")]

        return optimizer


class TestCleanupGenerationResources:
    """Test generation resource cleanup."""

    def test_model_cleanup_at_interval(self, initialized_optimizer):
        """Test that models are cleaned up at the configured interval."""
        initialized_optimizer.gen = 10  # At cleaning interval

        # Create some model directories
        model_dir = Path(initialized_optimizer.dirs["model_dir"])
        (model_dir / "gen_5_ind_0").mkdir(parents=True)
        (model_dir / "gen_7_ind_0").mkdir(parents=True)

        initialized_optimizer._cleanup_generation_resources()

        # Old models should be deleted (if not in archive or interval)
        # Archive models should be preserved

    def test_interval_archive_preservation(self, initialized_optimizer, tmp_path):
        """Test that models from interval archives are preserved."""
        initialized_optimizer.gen = 10

        # Create archive at interval (gen 5)
        archive_dir = Path(initialized_optimizer.dirs["archive_dir"])
        archive_data = [
            {
                "model_path": str(tmp_path / "models" / "gen_5_ind_0"),
                "fitness": 0.9,
                "acdc_skill_vector": {},
                "rank": None,
                "validation_quality": None,
            }
        ]

        archive_path = archive_dir / "gen5_dns_archive.json"
        with open(archive_path, "w") as f:
            json.dump(archive_data, f)

        # Create the model directory
        model_dir = Path(initialized_optimizer.dirs["model_dir"])
        (model_dir / "gen_5_ind_0").mkdir(parents=True)

        initialized_optimizer._cleanup_generation_resources()

        # Interval model should be preserved
        assert (model_dir / "gen_5_ind_0").exists()

    def test_docker_cleanup_command(self, initialized_optimizer):
        """Test that Docker cleanup command is executed."""
        initialized_optimizer.gen = 10

        with patch("os.system") as mock_system:
            initialized_optimizer._cleanup_generation_resources()

            # Should have called Docker cleanup
            mock_system.assert_called_once()
            call_arg = mock_system.call_args[0][0]
            assert "docker" in call_arg

    def test_error_handling_in_cleanup(self, initialized_optimizer):
        """Test that cleanup errors are handled gracefully."""
        initialized_optimizer.gen = 10

        # Make os.system raise an error
        with patch("os.system", side_effect=Exception("Cleanup failed")):
            # Should not raise, just log error
            initialized_optimizer._cleanup_generation_resources()


class TestLogGenerationMetrics:
    """Test metrics calculation and logging."""

    def test_metrics_calculation(self, initialized_optimizer):
        """Test that metrics are calculated correctly."""
        prev_log_time = 100.0

        with patch("main_ac_dc.compute_acdc_coverage_metrics") as mock_coverage:
            mock_coverage.return_value = {
                "coverage/total_tasks_covered": 2,
                "coverage/coverage_ratio": 1.0,
            }

            with patch("main_ac_dc.wandb.log") as mock_log:
                with patch("time.time", return_value=200.0):
                    initialized_optimizer._log_generation_metrics(prev_log_time)

                    # Should have logged metrics
                    mock_log.assert_called_once()

                    log_data = mock_log.call_args[1]
                    assert "step" in log_data
                    assert log_data["step"] == initialized_optimizer.gen

    def test_wandb_logging(self, initialized_optimizer):
        """Test that metrics are logged to wandb."""
        with patch("main_ac_dc.compute_acdc_coverage_metrics") as mock_coverage:
            mock_coverage.return_value = {}

            with patch("main_ac_dc.wandb.log") as mock_log:
                initialized_optimizer._log_generation_metrics(100.0)

                # Verify wandb.log was called
                mock_log.assert_called_once()

                # Get logged data
                log_call = mock_log.call_args
                log_data = log_call[0][0]

                # Should include basic metrics
                assert "dns/best_fitness" in log_data
                assert "dns/archive_size" in log_data

    def test_coverage_metrics_integration(self, initialized_optimizer):
        """Test that coverage metrics are included."""
        with patch("main_ac_dc.compute_acdc_coverage_metrics") as mock_coverage:
            coverage_data = {
                "coverage/total_tasks_covered": 2,
                "coverage/coverage_ratio": 1.0,
                "coverage/mean_coverage": 0.9,
            }
            mock_coverage.return_value = coverage_data

            with patch("main_ac_dc.wandb.log") as mock_log:
                initialized_optimizer._log_generation_metrics(100.0)

                log_data = mock_log.call_args[0][0]

                # Coverage metrics should be included
                for key in coverage_data:
                    assert key in log_data

    def test_with_empty_archive(self, initialized_optimizer):
        """Test logging with empty archive."""
        initialized_optimizer.archive_data["dns_archive"] = []

        with patch("main_ac_dc.wandb.log") as mock_log:
            initialized_optimizer._log_generation_metrics(100.0)

            # Should not log when archive is empty (returns early)
            mock_log.assert_not_called()

    def test_gibberish_counter_logging(self, initialized_optimizer):
        """Test that gibberish counter is logged and reset."""
        initialized_optimizer.gibberish_models_counter = 5

        with patch("main_ac_dc.compute_acdc_coverage_metrics"):
            with patch("main_ac_dc.wandb.log") as mock_log:
                initialized_optimizer._log_generation_metrics(100.0)

                log_data = mock_log.call_args[0][0]

                # Should include gibberish counter
                assert log_data["dns/gibberish_models_counter"] == 5

        # Counter should be reset
        assert initialized_optimizer.gibberish_models_counter == 0


class TestCleanupAndAdvanceGeneration:
    """Test generation advancement."""

    def test_generation_increment(self, initialized_optimizer):
        """Test that generation counter is incremented."""
        initial_gen = initialized_optimizer.gen

        initialized_optimizer._cleanup_and_advance_generation()

        assert initialized_optimizer.gen == initial_gen + 1

    def test_cleanup_invocation(self, initialized_optimizer):
        """Test that cleanup is called."""
        with patch.object(
            initialized_optimizer, "_cleanup_generation_resources"
        ) as mock_cleanup:
            initialized_optimizer._cleanup_and_advance_generation()

            mock_cleanup.assert_called_once()

    def test_returns_timestamp(self, initialized_optimizer):
        """Test that current time is returned."""
        with patch("time.time", return_value=12345.67):
            timestamp = initialized_optimizer._cleanup_and_advance_generation()

            assert timestamp == 12345.67


class TestSaveArchiveState:
    """Test archive state saving."""

    def test_archive_saving(self, initialized_optimizer):
        """Test that archive is saved to correct location."""
        initialized_optimizer.gen = 5

        with patch("main_ac_dc.save_ac_dc_archive") as mock_save:
            initialized_optimizer._save_archive_state()

            # Should have called save
            mock_save.assert_called_once()

            # Verify path
            call_args = mock_save.call_args
            save_path = call_args[0][1]
            assert "gen5_dns_archive.json" in save_path

    def test_file_path_creation(self, initialized_optimizer):
        """Test that file path is constructed correctly."""
        initialized_optimizer.gen = 10

        with patch("main_ac_dc.save_ac_dc_archive") as mock_save:
            initialized_optimizer._save_archive_state()

            save_path = mock_save.call_args[0][1]
            assert initialized_optimizer.dirs["archive_dir"] in save_path
            assert "gen10_dns_archive.json" in save_path


class TestSavePostAdaptationArchive:
    """Test post-adaptation archive saving."""

    def test_post_adaptation_archive_saving(self, initialized_optimizer):
        """Test that post-adaptation archive is saved."""
        initialized_optimizer.gen = 5

        with patch("main_ac_dc.save_ac_dc_archive") as mock_save:
            initialized_optimizer._save_post_adaptation_archive()

            mock_save.assert_called_once()

            # Verify filename includes post_adapt_filtered
            save_path = mock_save.call_args[0][1]
            assert "post_adapt_filtered" in save_path
            assert "gen5" in save_path


class TestSaveFinalArchive:
    """Test final archive saving."""

    def test_final_archive_saving(self, initialized_optimizer):
        """Test that final archive is saved."""
        with patch("main_ac_dc.save_ac_dc_archive") as mock_save:
            initialized_optimizer._save_final_archive()

            mock_save.assert_called_once()

            # Verify path
            save_path = mock_save.call_args[0][1]
            assert "final_dns_archive.json" in save_path

    def test_error_handling(self, initialized_optimizer):
        """Test error handling during final save."""
        with patch(
            "main_ac_dc.save_ac_dc_archive",
            side_effect=Exception("Save failed"),
        ):
            # Should handle error gracefully
            initialized_optimizer._save_final_archive()
            # Should not raise exception


class TestDetermineTaskInfoForGeneration:
    """Test task info determination."""

    def test_task_info_from_task_pool(self, initialized_optimizer):
        """Test that task info is retrieved from task pool."""
        mock_task_pool = Mock()
        mock_task_pool.tasks = ["/tasks/task_0", "/tasks/task_1"]
        initialized_optimizer.task_pool = mock_task_pool

        task_info = initialized_optimizer._determine_task_info_for_generation()

        assert task_info == mock_task_pool.tasks

    def test_non_acdc_mode_raises_error(self, initialized_optimizer):
        """Test that non-AC/DC mode raises NotImplementedError."""
        initialized_optimizer.cfg.use_ac_dc = False

        with pytest.raises(NotImplementedError, match="Standard task merging"):
            initialized_optimizer._determine_task_info_for_generation()


class TestMetricsLoggingIntegration:
    """Integration tests for metrics logging."""

    def test_complete_metrics_logging_cycle(self, initialized_optimizer):
        """Test complete metrics calculation and logging."""
        initialized_optimizer.gen = 5
        initialized_optimizer.gibberish_models_counter = 3

        coverage_metrics = {
            "coverage/total_tasks": 2,
            "coverage/coverage_ratio": 1.0,
        }

        with patch("main_ac_dc.compute_acdc_coverage_metrics") as mock_coverage:
            mock_coverage.return_value = coverage_metrics

            with patch("main_ac_dc.wandb.log") as mock_log:
                with patch("time.time", return_value=1000.0):
                    initialized_optimizer._log_generation_metrics(900.0)

                    log_data = mock_log.call_args[0][0]

                    # Verify all expected metrics
                    assert "dns/best_fitness" in log_data
                    assert "dns/archive_size" in log_data
                    assert "dns/gibberish_models_counter" in log_data
                    assert "base_info/generation" in log_data
                    assert "base_info/log_interval_seconds" in log_data

                    # Verify coverage metrics included
                    for key in coverage_metrics:
                        assert key in log_data

    def test_best_fitness_calculation(self, initialized_optimizer):
        """Test that best fitness is correctly identified."""
        # Set different fitness values
        initialized_optimizer.archive_data["dns_archive"][0].fitness = 0.9
        initialized_optimizer.archive_data["dns_archive"][1].fitness = 0.7

        with patch("main_ac_dc.compute_acdc_coverage_metrics"):
            with patch("main_ac_dc.wandb.log") as mock_log:
                initialized_optimizer._log_generation_metrics(100.0)

                log_data = mock_log.call_args[0][0]

                # Best fitness should be 0.9
                assert log_data["dns/best_fitness"] == 0.9

    def test_skill_vector_length_logging(self, initialized_optimizer):
        """Test that skill vector length is logged."""
        with patch("main_ac_dc.compute_acdc_coverage_metrics"):
            with patch("main_ac_dc.wandb.log") as mock_log:
                initialized_optimizer._log_generation_metrics(100.0)

                log_data = mock_log.call_args[0][0]

                # Should include skill vector length
                assert "dns/skill_vector_length" in log_data
                # Length should match number of tasks in best solution
                best_solution = max(
                    initialized_optimizer.archive_data["dns_archive"],
                    key=lambda x: x.fitness,
                )
                assert log_data["dns/skill_vector_length"] == len(
                    best_solution.acdc_skill_vector
                )
