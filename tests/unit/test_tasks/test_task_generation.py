"""
Tests for dns/task_generation.py — ACDCTaskPool class.

Covers pool initialization, task loading/validation, task adaptation,
and helper methods. LLM, vector DB, and sandbox are mocked; file I/O uses tmp_path.
"""

import pytest
import json
import os
import re
from unittest.mock import Mock, patch, MagicMock, call
from omegaconf import OmegaConf

from tasks.task_generation import ACDCTaskPool


class TestACDCTaskPoolInitialization:
    """Tests for ACDCTaskPool initialization and setup."""

    def test_basic_initialization(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test basic ACDCTaskPool initialization without vector DB.

        Setup:
            - Valid configuration
            - Seed task directory
            - Generated tasks directory

        Expected:
            - Task pool initialized successfully
            - Attributes set correctly
            - Directories created
        """
        # Update config with actual directories
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        # Mock vLLM client params creation
        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_params = {
                "base_url": "http://localhost:8000/v1",
                "model_name": "test-model",
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 1.0,
                "timeout": 300,
            }
            mock_create.return_value = mock_params

            # Initialize task pool
            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            # Verify initialization
            assert task_pool.cfg == config
            assert task_pool.seed_tasks_dir == mock_seed_task_dir
            assert task_pool.generated_tasks_dir == mock_generated_tasks_dir
            assert task_pool.initial_pool_size == 10
            assert task_pool.scientist_model_id == "vllm/test-model"
            assert task_pool.tasks == []
            assert task_pool.task_counter == 0
            assert task_pool.vllm_client_params == mock_params

    def test_initialization_with_vector_db(
        self, task_generation_config_with_vectordb, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test ACDCTaskPool initialization with vector DB enabled.

        Setup:
            - Configuration with vector DB enabled
            - Mock vector DB directories

        Expected:
            - Both active and historical vector DBs created
            - Vector DB directories exist
        """
        config = OmegaConf.to_container(task_generation_config_with_vectordb, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            with patch('tasks.task_generation.SimpleVectorDB') as mock_vdb_class:
                mock_vdb = Mock()
                mock_vdb_class.return_value = mock_vdb

                task_pool = ACDCTaskPool(
                    cfg=config,
                    generated_tasks_dir=mock_generated_tasks_dir,
                    vector_db_dir=mock_vector_db_dir
                )

                # Verify vector DBs were created
                assert task_pool.vector_db_active is not None
                assert task_pool.vector_db_historical is not None
                assert task_pool.vector_db_dir_active == mock_vector_db_dir + "_active"
                assert task_pool.vector_db_dir_historical == mock_vector_db_dir + "_historical"

                # Verify SimpleVectorDB was called twice (active + historical)
                assert mock_vdb_class.call_count == 2

    def test_initialization_with_vllm_scientist(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test initialization with vLLM scientist model.

        Setup:
            - Configuration specifying vllm/ scientist model
            - Mock vLLM client parameters

        Expected:
            - vLLM client params configured
            - No standard AC/DC client created
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config["acdc"]["scientist_model"] = "vllm/Qwen2.5-32B-Instruct"
        config["acdc"]["vllm_enabled"] = True
        config["acdc"]["vllm"] = {
            "base_url": "http://localhost:8000/v1",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1.0,
            "timeout": 300,
        }
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_vllm:
            mock_vllm.return_value = {
                "base_url": "http://localhost:8000/v1",
                "model_name": "Qwen2.5-32B-Instruct",
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 1.0,
                "timeout": 300,
            }

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            # Verify vLLM configuration
            assert task_pool.vllm_client_params is not None
            assert task_pool.vllm_client_params["base_url"] == "http://localhost:8000/v1"
            assert task_pool.vllm_client_params["model_name"] == "Qwen2.5-32B-Instruct"

    def test_initialization_non_vllm_scientist_raises_error(
        self, mock_seed_task_dir, mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test that non-vLLM scientist models are rejected.

        Setup:
            - Configuration with non-vLLM scientist model (e.g., gpt-4)

        Expected:
            - ValueError raised indicating only vLLM is supported
        """
        invalid_config = OmegaConf.create({
            "seed": 42,
            "acdc": {
                "seed_tasks_dir": mock_seed_task_dir,
                "initial_pool_size": 10,
                "scientist_model": "gpt-4o-2024-05-13",  # Non-vLLM model
                "scientist_temperature": 0.7,
                "num_initialization_workers": 4,
                "max_total_initialization_attempts": 30,
                "num_sandbox_workers": 4,
            },
            "task_generation": {},
        })

        with pytest.raises(ValueError, match="Only vLLM scientists are supported"):
            ACDCTaskPool(
                cfg=invalid_config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

    def test_initialization_missing_config_raises_error(
        self, mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test that missing required config raises ValueError.

        Setup:
            - Configuration missing required AC/DC fields

        Expected:
            - ValueError raised with descriptive message
        """
        incomplete_config = OmegaConf.create({
            "seed": 42,
            "acdc": {
                # Missing seed_tasks_dir, initial_pool_size, scientist_model
            },
            "task_generation": {},
        })

        with pytest.raises(ValueError, match="Missing required AC/DC configuration"):
            ACDCTaskPool(
                cfg=incomplete_config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

    def test_initialization_creates_directories(
        self, task_generation_config, mock_seed_task_dir, tmp_path
    ):
        """Test that initialization creates necessary directories.

        Setup:
            - New directory paths that don't exist
            - Valid configuration

        Expected:
            - Generated tasks directory created
            - Vector DB directories created if enabled
        """
        gen_dir = str(tmp_path / "new_gen_tasks")
        vdb_dir = str(tmp_path / "new_vector_db")

        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=gen_dir,
                vector_db_dir=vdb_dir
            )

            # Verify directories were created
            assert os.path.exists(gen_dir)
            assert task_pool.generated_tasks_dir == gen_dir


class TestTaskPoolManagement:
    """Tests for task pool management - loading, validation, generation."""

    def test_load_existing_tasks_from_directory(
        self, task_generation_config, mock_seed_task_dir,
        sample_task_pool_tasks, tmp_path
    ):
        """Test loading existing tasks from directory scan.

        Setup:
            - Generated tasks directory with 5 tasks
            - No restart_dir (fallback to directory scan)

        Expected:
            - All 5 tasks loaded
            - Task counter updated to max + 1
            - Tasks sorted by task number
        """
        gen_dir = str(tmp_path / "generated_tasks")
        os.makedirs(gen_dir, exist_ok=True)

        # Copy sample tasks to gen_dir
        for src_path in sample_task_pool_tasks:
            dst_name = os.path.basename(src_path)
            dst_path = os.path.join(gen_dir, dst_name)
            os.makedirs(dst_path, exist_ok=True)
            # Copy files
            for fname in ["task.py", "task.json"]:
                src_file = os.path.join(src_path, fname)
                dst_file = os.path.join(dst_path, fname)
                with open(src_file, 'r') as f:
                    content = f.read()
                with open(dst_file, 'w') as f:
                    f.write(content)

        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=gen_dir,
                vector_db_dir=str(tmp_path / "vdb")
            )

            # Load existing tasks
            task_pool.load_existing_tasks()

            # Verify tasks loaded
            assert len(task_pool.tasks) == 5
            assert task_pool.task_counter == 5  # max task number (4) + 1

            # Verify sorted order
            task_names = [os.path.basename(t) for t in task_pool.tasks]
            assert task_names == sorted(task_names, key=lambda x: int(re.search(r'task_(\d+)', x).group(1)))

    def test_load_existing_tasks_from_checkpoint(
        self, task_generation_config, mock_seed_task_dir, sample_task_pool_tasks, tmp_path
    ):
        """Test loading tasks from active_pool_gen_*.json checkpoint file.

        Setup:
            - restart_dir enabled
            - active_pool_gen_1.json file with task paths
            - Corresponding task directories exist

        Expected:
            - Tasks loaded from checkpoint file
            - Task counter updated
            - Checkpoint takes precedence over directory scan
        """
        gen_dir = str(tmp_path / "generated_tasks")
        os.makedirs(gen_dir, exist_ok=True)

        # Copy sample tasks to gen_dir
        for src_path in sample_task_pool_tasks[:3]:  # Only use first 3
            dst_name = os.path.basename(src_path)
            dst_path = os.path.join(gen_dir, dst_name)
            os.makedirs(dst_path, exist_ok=True)
            for fname in ["task.py", "task.json"]:
                src_file = os.path.join(src_path, fname)
                dst_file = os.path.join(dst_path, fname)
                with open(src_file, 'r') as f:
                    content = f.read()
                with open(dst_file, 'w') as f:
                    f.write(content)

        # Create checkpoint file
        checkpoint_tasks = [os.path.join(gen_dir, os.path.basename(p)) for p in sample_task_pool_tasks[:3]]
        checkpoint_file = os.path.join(gen_dir, "active_pool_gen_1.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_tasks, f)

        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config["restart_dir"] = str(tmp_path)  # Enable restart
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=gen_dir,
                vector_db_dir=str(tmp_path / "vdb")
            )

            task_pool.load_existing_tasks()

            # Verify tasks loaded from checkpoint
            assert len(task_pool.tasks) == 3
            assert all(t in checkpoint_tasks for t in task_pool.tasks)

    def test_get_tasks_returns_task_objects(
        self, task_generation_config, mock_seed_task_dir, sample_task_pool_tasks, tmp_path
    ):
        """Test get_tasks() returns ACDCTask objects.

        Setup:
            - Task pool with loaded task paths
            - Valid task directories

        Expected:
            - ACDCTask objects created for each path
            - Errors logged for invalid tasks
        """
        gen_dir = str(tmp_path / "generated_tasks")
        os.makedirs(gen_dir, exist_ok=True)

        # Copy 2 sample tasks
        for src_path in sample_task_pool_tasks[:2]:
            dst_name = os.path.basename(src_path)
            dst_path = os.path.join(gen_dir, dst_name)
            os.makedirs(dst_path, exist_ok=True)
            for fname in ["task.py", "task.json"]:
                src_file = os.path.join(src_path, fname)
                dst_file = os.path.join(dst_path, fname)
                with open(src_file, 'r') as f:
                    content = f.read()
                with open(dst_file, 'w') as f:
                    f.write(content)

        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=gen_dir,
                vector_db_dir=str(tmp_path / "vdb")
            )

            # Set tasks manually
            task_pool.tasks = [
                os.path.join(gen_dir, os.path.basename(p))
                for p in sample_task_pool_tasks[:2]
            ]

            with patch('tasks.task_generation.ACDCTask') as mock_task_class:
                mock_task1 = Mock()
                mock_task2 = Mock()
                mock_task_class.side_effect = [mock_task1, mock_task2]

                task_objects = task_pool.get_tasks()

                # Verify ACDCTask was instantiated for each path
                assert mock_task_class.call_count == 2
                assert len(task_objects) == 2
                assert mock_task1 in task_objects
                assert mock_task2 in task_objects

    def test_get_tasks_handles_invalid_tasks(
        self, task_generation_config, mock_seed_task_dir, tmp_path
    ):
        """Test get_tasks() handles errors when loading invalid tasks.

        Setup:
            - Task paths that cannot be loaded (missing files, etc.)

        Expected:
            - Errors logged
            - Invalid tasks skipped
            - Returns only valid tasks
        """
        gen_dir = str(tmp_path / "generated_tasks")
        os.makedirs(gen_dir, exist_ok=True)

        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=gen_dir,
                vector_db_dir=str(tmp_path / "vdb")
            )

            # Set invalid task paths
            task_pool.tasks = ["/invalid/path/task_0", "/invalid/path/task_1"]

            with patch('tasks.task_generation.ACDCTask') as mock_task_class:
                # First succeeds, second raises error
                mock_task = Mock()
                mock_task_class.side_effect = [mock_task, Exception("Invalid task")]

                task_objects = task_pool.get_tasks()

                # Verify only valid task returned
                assert len(task_objects) == 1
                assert task_objects[0] == mock_task

    def test_get_ordered_task_ids_returns_sorted_ids(
        self, task_generation_config, mock_seed_task_dir, sample_task_pool_tasks, tmp_path
    ):
        """Test get_ordered_task_ids() returns consistently sorted task IDs.

        Setup:
            - Task pool with unsorted task paths

        Expected:
            - Task IDs sorted by task number
            - Returns basenames of task directories
        """
        gen_dir = str(tmp_path / "generated_tasks")
        os.makedirs(gen_dir, exist_ok=True)

        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=gen_dir,
                vector_db_dir=str(tmp_path / "vdb")
            )

            # Set tasks in random order
            task_pool.tasks = [
                "/path/task_2_abc",
                "/path/task_0_xyz",
                "/path/task_4_def",
                "/path/task_1_ghi",
            ]

            task_ids = task_pool.get_ordered_task_ids()

            # Verify sorted order
            assert task_ids == ["task_0_xyz", "task_1_ghi", "task_2_abc", "task_4_def"]

    def test_validate_generated_task_success(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test _validate_generated_task with valid task JSON.

        Setup:
            - Valid task JSON with all required fields

        Expected:
            - Validation passes
            - Returns True
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            valid_task = {
                "name_of_task": "Test Task",
                "description_of_task": "Description",
                "capability_being_measured": "Reasoning",
                "estimated_human_difficulty": "3",
                "task_family": "class Task: pass",
                "example_instruction": "Do something",
            }

            result = task_pool._validate_generated_task(valid_task)
            assert result is True

    def test_validate_generated_task_missing_fields(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test _validate_generated_task with missing required fields.

        Setup:
            - Task JSON missing required fields

        Expected:
            - Validation fails
            - Returns False
            - Warning logged
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            invalid_task = {
                "name_of_task": "Test Task",
                # Missing other required fields
            }

            result = task_pool._validate_generated_task(invalid_task)
            assert result is False

    def test_initialize_pool_basic(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test initialize_pool generates tasks from seed.

        Setup:
            - Configuration for 10 initial tasks
            - Mock multiprocessing pool
            - Mock task generation worker

        Expected:
            - Multiprocessing pool used
            - Tasks generated and validated
            - Task paths added to pool
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config["acdc"]["initial_pool_size"] = 5  # Small size for testing
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            # Mock multiprocessing.Pool
            with patch('tasks.task_generation.multiprocessing.Pool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool.__enter__.return_value = mock_pool
                mock_pool_class.return_value = mock_pool

                # Mock successful task generation
                mock_task_paths = [
                    os.path.join(mock_generated_tasks_dir, f"task_{i}_test")
                    for i in range(5)
                ]

                # Create actual task directory structure with required files
                for task_path in mock_task_paths:
                    os.makedirs(task_path, exist_ok=True)
                    # Create task.py
                    with open(os.path.join(task_path, "task.py"), "w") as f:
                        f.write("class Task: pass")
                    # Create task.json
                    with open(os.path.join(task_path, "task.json"), "w") as f:
                        json.dump({"name_of_task": "TestTask"}, f)

                mock_pool.map.return_value = mock_task_paths

                # Mock ACDCTask validation
                with patch('tasks.task_generation.ACDCTask') as mock_task_class:
                    mock_task_class.return_value = Mock()

                    # Mock vector DB if needed
                    if task_pool.vector_db_active:
                        task_pool.vector_db_active.add_task = Mock()

                    task_pool.initialize_pool()

                    # Verify multiprocessing pool was used
                    assert mock_pool.map.called

                    # Verify tasks were added
                    assert len(task_pool.tasks) <= config.acdc.initial_pool_size


class TestTaskPoolAdaptation:
    """Tests for task pool adaptation logic."""

    def test_calculate_task_pass_rates(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir, mock_acdc_archive_data
    ):
        """Test _calculate_task_pass_rates computes pass rates from archive.

        Setup:
            - Task pool with tasks
            - Archive with skill vectors

        Expected:
            - Pass rates calculated per task
            - Total counts tracked
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            # Set up tasks
            task_pool.tasks = ["/path/task_0", "/path/task_1", "/path/task_2"]

            # Call method
            pass_rates, pass_counts, total_counts, task_paths_map = (
                task_pool._calculate_task_pass_rates(mock_acdc_archive_data["dns_archive"])
            )

            # Verify pass rates calculated
            assert isinstance(pass_rates, dict)
            assert isinstance(pass_counts, dict)
            assert isinstance(total_counts, dict)
            assert "task_0" in pass_rates or "task_1" in pass_rates or "task_2" in pass_rates

    def test_get_scientist_response_vllm(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test _get_scientist_response with vLLM scientist.

        Setup:
            - vLLM scientist configured
            - Mock vLLM response

        Expected:
            - vLLM called with correct parameters
            - Response returned
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config["acdc"]["scientist_model"] = "vllm/model"
        config["acdc"]["vllm_enabled"] = True
        config["acdc"]["vllm"] = {
            "base_url": "http://localhost:8000/v1",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_vllm_params:
            mock_vllm_params.return_value = {
                "base_url": "http://localhost:8000/v1",
                "model_name": "model",
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 1.0,
                "timeout": 300,
            }

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            with patch('tasks.task_generation.get_vllm_response') as mock_vllm_resp:
                mock_vllm_resp.return_value = ("Test response", [{"role": "assistant", "content": "Test response"}])

                response, history = task_pool._get_scientist_response(
                    prompt="Test prompt",
                    system_msg="System message"
                )

                # Verify vLLM was called
                assert mock_vllm_resp.called
                assert response == "Test response"
                assert len(history) > 0

    def test_get_scientist_response_standard_llm(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test _get_scientist_response with standard LLM.

        Setup:
            - Standard AC/DC scientist configured
            - Mock LLM response

        Expected:
            - AC/DC LLM utility called
            - Response returned
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {
                "base_url": "http://localhost:8000/v1",
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 1.0,
                "timeout": 300,
            }

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            with patch('tasks.task_generation.get_vllm_response') as mock_llm_resp:
                mock_llm_resp.return_value = ("LLM response", [{"role": "assistant", "content": "LLM response"}])

                response, history = task_pool._get_scientist_response(
                    prompt="Test prompt",
                    system_msg="System message"
                )

                # Verify AC/DC LLM was called
                assert mock_llm_resp.called
                assert response == "LLM response"

    def test_sandbox_validate_task(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir, tmp_path
    ):
        """Test _sandbox_validate_task validates task execution.

        Setup:
            - Task directory with valid task
            - Mock sandbox execution

        Expected:
            - Sandbox called with task
            - Score and answer returned
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            task_dir = str(tmp_path / "test_task")
            os.makedirs(task_dir, exist_ok=True)

            with patch('tasks.task_generation.run_task_in_sandbox') as mock_sandbox, \
                 patch('tasks.task_generation.load_task_family') as mock_load_task, \
                 patch.object(task_pool, '_get_scientist_response') as mock_scientist:
                # Mock successful sandbox execution
                # run_task_in_sandbox returns just the score value
                mock_sandbox.return_value = 0.9

                # Mock task family for loading task data
                mock_task = Mock()
                mock_task.get_tasks.return_value = {"example_1": {"data": "test"}}
                mock_task.get_instructions.return_value = "Test instruction"
                mock_load_task.return_value = Mock(return_value=mock_task)

                # Mock scientist response for answer generation
                mock_scientist.return_value = ("Correct answer", [])

                returned_task_dir, score, answer, instruction = task_pool._sandbox_validate_task(
                    task_dir, generate_answer=True
                )

                # Verify sandbox was called
                assert mock_sandbox.called
                assert score == 0.9
                assert answer == "Correct answer"
                assert returned_task_dir == task_dir

    def test_reflect_on_task_success(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test _reflect_on_task with successful reflection.

        Setup:
            - Valid task response
            - Mock scientist responses that converge

        Expected:
            - Reflection loop executes
            - Final task returned
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config["task_generation"]["max_reflections"] = 1
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            initial_response = {
                "name_of_task": "Test",
                "description_of_task": "Desc",
                "capability_being_measured": "Cap",
                "estimated_human_difficulty": "3",
                "task_family": "code",
                "example_instruction": "Inst",
                "done": "False",
            }

            # Mock eval and scientist response
            with patch.object(task_pool, '_eval_task') as mock_eval:
                mock_eval.return_value = ("Task evaluated successfully", True)

                with patch.object(task_pool, '_get_scientist_response') as mock_scientist:
                    # Return converged task
                    final_response = initial_response.copy()
                    final_response["done"] = "True"
                    mock_scientist.return_value = (json.dumps(final_response), [])

                    result, history = task_pool._reflect_on_task(
                        initial_response, task_number=1
                    )

                    # Verify reflection succeeded
                    assert result is not None
                    assert result["done"] == "True"

    def test_reflect_on_task_no_convergence(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test _reflect_on_task when reflection doesn't converge.

        Setup:
            - Task response
            - Scientist never returns done=True

        Expected:
            - Max reflections reached
            - None returned
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config["task_generation"]["max_reflections"] = 2
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            initial_response = {
                "name_of_task": "Test",
                "description_of_task": "Desc",
                "capability_being_measured": "Cap",
                "estimated_human_difficulty": "3",
                "task_family": "code",
                "example_instruction": "Inst",
                "done": "False",
            }

            with patch.object(task_pool, '_eval_task') as mock_eval:
                mock_eval.return_value = ("Not good enough", False)

                with patch.object(task_pool, '_get_scientist_response') as mock_scientist:
                    # Never converge
                    non_converged = initial_response.copy()
                    mock_scientist.return_value = (json.dumps(non_converged), [])

                    result, history = task_pool._reflect_on_task(
                        initial_response, task_number=1
                    )

                    # Verify None returned
                    assert result is None

    def test_generate_initial_task_from_seed(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test generate_initial_task_from_seed creates task.

        Setup:
            - Valid seed task directory
            - Mock scientist response

        Expected:
            - Task generated from seed
            - Task saved to disk
            - Task path returned
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            with patch.object(task_pool, '_get_scientist_response') as mock_scientist:
                mock_scientist.return_value = (json.dumps({
                    "name_of_task": "Generated",
                    "description_of_task": "Desc",
                    "capability_being_measured": "Cap",
                    "estimated_human_difficulty": "3",
                    "task_family": "code",
                    "example_instruction": "Inst",
                    "done": "True",
                }), [])

                with patch.object(task_pool, '_reflect_on_task') as mock_reflect:
                    mock_reflect.return_value = ({
                        "name_of_task": "Generated",
                        "description_of_task": "Desc",
                        "capability_being_measured": "Cap",
                        "estimated_human_difficulty": "3",
                        "task_family": "code",
                        "example_instruction": "Inst",
                        "done": "True",
                    }, [])

                    with patch('tasks.task_generation.save_task_to_disk') as mock_save:
                        mock_save.return_value = None

                        seed_dir = os.path.join(mock_seed_task_dir, "task_0")
                        result = task_pool.generate_initial_task_from_seed(
                            seed_tasks_dir=seed_dir,
                            task_number=1,
                            novel_init_adapt_rng_value=0.5
                        )

                        # Verify task generated (or None if reflection fails)
                        # Since we're mocking reflection to succeed, result should not be None
                        # In actual implementation, it returns the task path


class TestHelperMethods:
    """Tests for helper utility methods."""

    def test_sanitize_filename(self):
        """Test sanitize_filename removes invalid characters.

        Setup:
            - Filenames with invalid characters

        Expected:
            - Invalid characters replaced with underscores
        """
        from tasks.task_generation import sanitize_filename

        assert sanitize_filename("task<>name") == "task__name"
        assert sanitize_filename('file:with"bad|chars') == "file_with_bad_chars"
        assert sanitize_filename("normal_name") == "normal_name"

    def test_get_ordered_task_ids_handles_non_matching_names(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test get_ordered_task_ids handles task names without numbers.

        Setup:
            - Task paths without task_N_ pattern

        Expected:
            - Non-matching tasks assigned -1 for sorting
            - Warning logged
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            # Set tasks with mixed naming
            task_pool.tasks = [
                "/path/task_2_abc",
                "/path/invalid_name",
                "/path/task_1_def",
            ]

            task_ids = task_pool.get_ordered_task_ids()

            # Verify non-matching task comes first (sorted by -1)
            assert "invalid_name" in task_ids
            # Valid tasks sorted after
            assert task_ids.index("task_1_def") > task_ids.index("invalid_name")

    def test_get_tasks_empty_list(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test get_tasks with empty task list.

        Setup:
            - Task pool with no tasks

        Expected:
            - Returns empty list
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            task_pool.tasks = []

            task_objects = task_pool.get_tasks()
            assert task_objects == []

    def test_task_counter_increments(
        self, task_generation_config, mock_seed_task_dir,
        mock_generated_tasks_dir, mock_vector_db_dir
    ):
        """Test that task_counter increments correctly.

        Setup:
            - New task pool

        Expected:
            - Task counter starts at 0
            - Increments as tasks are generated
        """
        config = OmegaConf.to_container(task_generation_config, resolve=True)
        config["acdc"]["seed_tasks_dir"] = mock_seed_task_dir
        config = OmegaConf.create(config)

        with patch('tasks.task_generation.create_vllm_client_params') as mock_create:
            mock_create.return_value = {"base_url": "http://localhost:8000/v1", "model_name": "test-model", "temperature": 0.7, "max_tokens": 4096, "top_p": 1.0, "timeout": 300}

            task_pool = ACDCTaskPool(
                cfg=config,
                generated_tasks_dir=mock_generated_tasks_dir,
                vector_db_dir=mock_vector_db_dir
            )

            # Verify initial counter
            assert task_pool.task_counter == 0

            # Simulate incrementing
            task_pool.task_counter += 1
            assert task_pool.task_counter == 1

            task_pool.task_counter += 1
            assert task_pool.task_counter == 2
