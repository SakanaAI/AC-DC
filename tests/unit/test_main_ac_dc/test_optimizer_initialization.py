"""
Unit tests for ACDCOptimizer initialization and setup.

Tests cover:
- __init__ method
- setup_environment method
- setup_workers method
- _load_or_generate_tasks method
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from omegaconf import DictConfig, OmegaConf
import numpy as np

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
        "validation_tasks": ["val_task_1", "val_task_2"],
        "seed_model_paths": [
            "/models/seed_model_1",
            "/models/seed_model_2",
        ],
        "celery": {
            "num_workers": 4,
            "timeout": 300,
        },
        "dns": {
            "population_size": 10,
            "init_population_size": 5,
            "num_model_per_gen": 8,
            "acdc_skill_threshold": 0.5,
            "max_details_to_log": 5,
        },
        "use_ac_dc": True,
        "acdc": {
            "task_generation_interval": 5,
        },
        "restart_dir": None,
        "disk_cleaning_interval": 10,
    })


class TestACDCOptimizerInit:
    """Test ACDCOptimizer initialization."""

    def test_basic_initialization(self, mock_celery, basic_config):
        """Test basic optimizer initialization."""
        optimizer = ACDCOptimizer(mock_celery, basic_config)

        assert optimizer.celery == mock_celery
        assert optimizer.cfg == basic_config
        assert optimizer.call_fn == mock_celery.tasks["call"]
        assert optimizer.validation_tasks_names == ["val_task_1", "val_task_2"]
        assert optimizer.gen == 1
        assert optimizer.tasks == []
        assert optimizer.task_pool is None
        assert optimizer.dirs is None
        assert optimizer.gibberish_models_counter == 0

    def test_random_state_initialization(self, mock_celery, basic_config):
        """Test that random state is initialized with seed."""
        optimizer = ACDCOptimizer(mock_celery, basic_config)

        assert isinstance(optimizer.np_random, np.random.RandomState)

        # Verify seed is used (same seed = same random values)
        val1 = optimizer.np_random.randint(1000)
        optimizer2 = ACDCOptimizer(mock_celery, basic_config)
        val2 = optimizer2.np_random.randint(1000)

        assert val1 == val2  # Same seed should give same random values

    def test_negative_seed_uses_default(self, mock_celery, basic_config):
        """Test that negative seed uses default value."""
        basic_config.seed = -1
        optimizer = ACDCOptimizer(mock_celery, basic_config)

        # Should use seed 42 as default
        assert optimizer.np_random is not None

    def test_gen_0_seed_model_names_creation(self, mock_celery, basic_config):
        """Test that seed model names are extracted correctly."""
        optimizer = ACDCOptimizer(mock_celery, basic_config)

        expected_names = {
            "gen_0_ind_seed_model_1",
            "gen_0_ind_seed_model_2",
        }

        assert optimizer.gen_0_seed_model_names == expected_names

    def test_validation_tasks_defaults_to_empty(self, mock_celery, basic_config):
        """Test that validation_tasks defaults to empty list if not specified."""
        del basic_config["validation_tasks"]
        optimizer = ACDCOptimizer(mock_celery, basic_config)

        assert optimizer.validation_tasks_names == []


class TestSetupEnvironment:
    """Test environment setup."""

    def test_directory_creation(self, mock_celery, basic_config, tmp_path):
        """Test that directories are created properly."""
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path)
            mock_hydra.return_value.runtime = mock_runtime

            optimizer = ACDCOptimizer(mock_celery, basic_config)
            optimizer.setup_environment()

            # Verify directories were created
            assert optimizer.dirs is not None
            assert "output_dir" in optimizer.dirs
            assert "model_dir" in optimizer.dirs
            assert "archive_dir" in optimizer.dirs

            # Verify directories exist
            assert Path(optimizer.dirs["model_dir"]).exists()
            assert Path(optimizer.dirs["archive_dir"]).exists()

    def test_generation_counter_initialization(self, mock_celery, basic_config, tmp_path):
        """Test that generation counter starts at 1 for new runs."""
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path)
            mock_hydra.return_value.runtime = mock_runtime

            optimizer = ACDCOptimizer(mock_celery, basic_config)
            optimizer.setup_environment()

            assert optimizer.gen == 1

    def test_archive_data_initialization(self, mock_celery, basic_config, tmp_path):
        """Test that archive_data is initialized correctly."""
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path)
            mock_hydra.return_value.runtime = mock_runtime

            optimizer = ACDCOptimizer(mock_celery, basic_config)
            optimizer.setup_environment()

            assert optimizer.archive_data is not None
            assert "dns_archive" in optimizer.archive_data
            assert "dirs" in optimizer.archive_data
            assert optimizer.archive_data["dns_archive"] == []

    def test_restart_loads_existing_archive(self, mock_celery, basic_config, tmp_path):
        """Test that restart loads existing archive."""
        # Setup existing archive
        model_dir = tmp_path / "models"
        archive_dir = tmp_path / "archives"
        model_dir.mkdir(parents=True)
        archive_dir.mkdir(parents=True)

        # Create a model directory for gen_1
        (model_dir / "gen_1_ind_0").mkdir()

        # Create archive file
        archive_data = [
            {
                "model_path": str(model_dir / "gen_1_ind_0"),
                "fitness": 0.8,
                "acdc_skill_vector": {"task_0": 0.8},
                "rank": None,
                "validation_quality": None,
            }
        ]

        archive_path = archive_dir / "gen1_dns_archive.json"
        with open(archive_path, "w") as f:
            json.dump(archive_data, f)

        # Configure restart
        basic_config.restart_dir = str(tmp_path)

        optimizer = ACDCOptimizer(mock_celery, basic_config)
        optimizer.setup_environment()

        # Verify archive was loaded
        assert len(optimizer.archive_data["dns_archive"]) == 1
        assert optimizer.archive_data["dns_archive"][0].fitness == 0.8

        # Verify generation counter is incremented
        assert optimizer.gen == 2

    def test_restart_prefers_post_adapt_filtered_archive(
        self, mock_celery, basic_config, tmp_path
    ):
        """Test that post_adapt_filtered archive is preferred over regular."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir(parents=True)

        # Create both archive types
        regular_archive = [
            {
                "model_path": "/models/model_0",
                "fitness": 0.5,
                "acdc_skill_vector": {},
                "rank": None,
                "validation_quality": None,
            }
        ]

        filtered_archive = [
            {
                "model_path": "/models/model_1",
                "fitness": 0.9,
                "acdc_skill_vector": {},
                "rank": None,
                "validation_quality": None,
            }
        ]

        with open(archive_dir / "gen1_dns_archive.json", "w") as f:
            json.dump(regular_archive, f)

        with open(archive_dir / "gen1_dns_archive_post_adapt_filtered.json", "w") as f:
            json.dump(filtered_archive, f)

        basic_config.restart_dir = str(tmp_path)

        # Also need model_dir for generation detection
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "gen_1_ind_0").mkdir()

        optimizer = ACDCOptimizer(mock_celery, basic_config)
        optimizer.setup_environment()

        # Should have loaded the filtered archive (fitness = 0.9)
        assert optimizer.archive_data["dns_archive"][0].fitness == 0.9

    def test_restart_with_missing_archive_starts_fresh(
        self, mock_celery, basic_config, tmp_path
    ):
        """Test that missing archive results in fresh start with warning."""
        archive_dir = tmp_path / "archives"
        model_dir = tmp_path / "models"
        archive_dir.mkdir(parents=True)
        model_dir.mkdir(parents=True)

        # Create a gen_1 model directory to indicate we're at gen 2
        (model_dir / "gen_1_ind_0").mkdir()

        basic_config.restart_dir = str(tmp_path)

        optimizer = ACDCOptimizer(mock_celery, basic_config)
        optimizer.setup_environment()

        # Should start with empty archive
        assert optimizer.archive_data["dns_archive"] == []
        assert optimizer.gen == 2  # Still increments generation


class TestSetupWorkers:
    """Test worker setup."""

    def test_worker_logging_setup(self, mock_celery, basic_config, tmp_path):
        """Test that worker logging is set up correctly."""
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path)
            mock_hydra.return_value.runtime = mock_runtime

            optimizer = ACDCOptimizer(mock_celery, basic_config)
            optimizer.setup_environment()

            # Mock the call_fn to track setup calls
            mock_promise = Mock()
            mock_promise.ready.return_value = True
            mock_promise.get.return_value = None
            optimizer.call_fn.delay = Mock(return_value=mock_promise)

            # Mock time.sleep to speed up test
            with patch("time.sleep"):
                optimizer.setup_workers()

            # Verify setup was called for each worker
            assert optimizer.call_fn.delay.call_count == basic_config.celery.num_workers

            # Verify each call was for setup_worker
            for call_args in optimizer.call_fn.delay.call_args_list:
                assert call_args[0][0] == "setup_worker"
                assert call_args[1]["output_dir"] == str(tmp_path)

    def test_setup_workers_waits_for_completion(self, mock_celery, basic_config, tmp_path):
        """Test that setup waits for all workers to complete."""
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path)
            mock_hydra.return_value.runtime = mock_runtime

            optimizer = ACDCOptimizer(mock_celery, basic_config)
            optimizer.setup_environment()

            # Create promises that become ready
            promises = []
            for _ in range(basic_config.celery.num_workers):
                p = Mock()
                p.ready.return_value = True
                p.get.return_value = None
                promises.append(p)

            optimizer.call_fn.delay = Mock(side_effect=promises)

            with patch("time.sleep"):
                optimizer.setup_workers()

            # All promises should have been waited on
            for p in promises:
                p.get.assert_called_once()


class TestLoadOrGenerateTasks:
    """Test task loading and generation."""

    def test_acdc_task_pool_initialization(self, mock_celery, basic_config, tmp_path):
        """Test AC/DC task pool initialization."""
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path)
            mock_hydra.return_value.runtime = mock_runtime

            optimizer = ACDCOptimizer(mock_celery, basic_config)
            optimizer.setup_environment()

            # Mock ACDCTaskPool
            with patch("main_ac_dc.ACDCTaskPool") as mock_pool_class:
                mock_pool = Mock()
                mock_pool.tasks = ["/tasks/task_0", "/tasks/task_1"]
                mock_pool.get_tasks.return_value = [
                    Mock(task_name="task_0"),
                    Mock(task_name="task_1"),
                ]
                mock_pool_class.return_value = mock_pool

                optimizer._load_or_generate_tasks()

                # Verify task pool was created
                mock_pool_class.assert_called_once()

                # Verify initialize_pool was called
                mock_pool.initialize_pool.assert_called_once()

                # Verify tasks were loaded
                assert len(optimizer.tasks) == 2

    def test_acdc_task_pool_loading_on_restart(self, mock_celery, basic_config, tmp_path):
        """Test that existing tasks are loaded on restart."""
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path)
            mock_hydra.return_value.runtime = mock_runtime

            basic_config.restart_dir = str(tmp_path)

            # Create model directory with at least one generation to avoid ValueError
            model_dir = tmp_path / "models"
            model_dir.mkdir(parents=True)
            (model_dir / "gen_1_ind_0").mkdir(parents=True)

            optimizer = ACDCOptimizer(mock_celery, basic_config)
            optimizer.setup_environment()

            # Mock ACDCTaskPool
            with patch("main_ac_dc.ACDCTaskPool") as mock_pool_class:
                mock_pool = Mock()
                mock_pool.tasks = ["/tasks/task_0"]
                mock_pool.get_tasks.return_value = [Mock(task_name="task_0")]
                mock_pool_class.return_value = mock_pool

                optimizer._load_or_generate_tasks()

                # Should call load_existing_tasks instead of initialize_pool
                mock_pool.load_existing_tasks.assert_called_once()
                mock_pool.initialize_pool.assert_not_called()

    def test_error_handling_for_failed_task_generation(
        self, mock_celery, basic_config, tmp_path
    ):
        """Test that task generation errors are handled."""
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path)
            mock_hydra.return_value.runtime = mock_runtime

            optimizer = ACDCOptimizer(mock_celery, basic_config)
            optimizer.setup_environment()

            # Mock ACDCTaskPool to raise error
            with patch("main_ac_dc.ACDCTaskPool") as mock_pool_class:
                mock_pool_class.side_effect = Exception("Task generation failed")

                with pytest.raises(Exception, match="Task generation failed"):
                    optimizer._load_or_generate_tasks()

    def test_warning_for_zero_tasks_generated(self, mock_celery, basic_config, tmp_path):
        """Test warning when zero tasks are generated."""
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path)
            mock_hydra.return_value.runtime = mock_runtime

            optimizer = ACDCOptimizer(mock_celery, basic_config)
            optimizer.setup_environment()

            # Mock ACDCTaskPool with no tasks
            with patch("main_ac_dc.ACDCTaskPool") as mock_pool_class:
                mock_pool = Mock()
                mock_pool.tasks = []
                mock_pool.get_tasks.return_value = []
                mock_pool_class.return_value = mock_pool

                # Should not raise error but will log warning
                optimizer._load_or_generate_tasks()

                assert optimizer.tasks == []

    def test_non_acdc_mode_raises_not_implemented(
        self, mock_celery, basic_config, tmp_path
    ):
        """Test that non-AC/DC mode raises NotImplementedError."""
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_hydra:
            mock_runtime = Mock()
            mock_runtime.output_dir = str(tmp_path)
            mock_hydra.return_value.runtime = mock_runtime

            basic_config.use_ac_dc = False

            optimizer = ACDCOptimizer(mock_celery, basic_config)
            optimizer.setup_environment()

            with pytest.raises(NotImplementedError, match="Standard task loading"):
                optimizer._load_or_generate_tasks()
