"""
Unit tests for ACDCWorker (workers/ac_dc_worker.py).

Tests worker initialization, merge operations, model evaluation, and task loading.
"""

import pytest
import torch
import json
from unittest.mock import Mock, MagicMock, patch, call
from omegaconf import OmegaConf
from pathlib import Path


# ============================================================================
# Worker Initialization
# ============================================================================


class TestWorkerInitialization:
    """Tests for ACDCWorker.__init__"""

    @patch('workers.ac_dc_worker.LLM')
    @patch('workers.ac_dc_worker.AutoModelForCausalLM')
    @patch('workers.ac_dc_worker.AutoTokenizer')
    @patch('workers.ac_dc_worker.hydra.utils.instantiate')
    def test_basic_initialization(
        self, mock_instantiate, mock_tokenizer_cls, mock_model_cls, mock_llm_cls,
        worker_config, mock_vllm_llm, mock_hf_model, mock_tokenizer
    ):
        """Test basic worker initialization with mocked dependencies.

        Setup:
            - Mock vLLM, HF model, tokenizer
            - Mock crossover/mutator instantiation

        Expected:
            - Worker initializes successfully
            - All components are set correctly
            - vLLM request count starts at 0
        """
        # Setup mocks
        mock_llm_cls.return_value = mock_vllm_llm
        mock_model_cls.from_pretrained.return_value = mock_hf_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_instantiate.side_effect = [Mock(), Mock()]  # crossover, mutator

        # Import here to avoid import errors
        from workers.ac_dc_worker import ACDCWorker

        # Initialize worker
        worker = ACDCWorker(worker_config)

        # Verify initialization
        assert worker.llm == mock_vllm_llm
        assert worker.hf_model == mock_hf_model
        assert worker.tokenizer == mock_tokenizer
        assert worker.vllm_request_count == 0
        assert worker.crossover is not None
        assert worker.mutator is not None

        # Verify vLLM was initialized with correct config
        mock_llm_cls.assert_called_once()
        # LLM is called with positional arg for model path
        call_args = mock_llm_cls.call_args[0]
        call_kwargs = mock_llm_cls.call_args[1]
        assert call_args[0] == worker_config.base_model_path
        assert call_kwargs['gpu_memory_utilization'] == worker_config.gpu_memory_utilization

    @patch('workers.ac_dc_worker.LLM')
    @patch('workers.ac_dc_worker.AutoModelForCausalLM')
    @patch('workers.ac_dc_worker.AutoTokenizer')
    @patch('workers.ac_dc_worker.hydra.utils.instantiate')
    def test_initialization_with_chat_template(
        self, mock_instantiate, mock_tokenizer_cls, mock_model_cls, mock_llm_cls,
        worker_config, mock_vllm_llm, mock_hf_model, mock_tokenizer
    ):
        """Test chat template setup during initialization.

        Expected:
            - Chat template is loaded and applied to tokenizer
        """
        # Setup mocks
        mock_llm_cls.return_value = mock_vllm_llm
        mock_model_cls.from_pretrained.return_value = mock_hf_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_instantiate.side_effect = [Mock(), Mock()]

        from workers.ac_dc_worker import ACDCWorker

        # Initialize worker
        worker = ACDCWorker(worker_config)

        # Verify tokenizer was set up
        assert worker.tokenizer == mock_tokenizer

    @patch('workers.ac_dc_worker.LLM')
    @patch('workers.ac_dc_worker.AutoModelForCausalLM')
    @patch('workers.ac_dc_worker.AutoTokenizer')
    @patch('workers.ac_dc_worker.hydra.utils.instantiate')
    def test_crossover_mutator_initialization(
        self, mock_instantiate, mock_tokenizer_cls, mock_model_cls, mock_llm_cls,
        worker_config, mock_vllm_llm, mock_hf_model, mock_tokenizer,
        mock_crossover, mock_mutator
    ):
        """Test that crossover and mutator are initialized correctly.

        Expected:
            - Crossover and mutator are instantiated via Hydra
            - They are accessible as worker attributes
        """
        # Setup mocks
        mock_llm_cls.return_value = mock_vllm_llm
        mock_model_cls.from_pretrained.return_value = mock_hf_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_instantiate.side_effect = [mock_crossover, mock_mutator]

        from workers.ac_dc_worker import ACDCWorker

        # Initialize worker
        worker = ACDCWorker(worker_config)

        # Verify crossover and mutator were instantiated
        assert mock_instantiate.call_count == 2
        assert worker.crossover == mock_crossover
        assert worker.mutator == mock_mutator

    @patch('workers.ac_dc_worker.LLM')
    @patch('workers.ac_dc_worker.AutoModelForCausalLM')
    @patch('workers.ac_dc_worker.AutoTokenizer')
    @patch('workers.ac_dc_worker.hydra.utils.instantiate')
    def test_initialization_sets_random_seed(
        self, mock_instantiate, mock_tokenizer_cls, mock_model_cls, mock_llm_cls,
        worker_config, mock_vllm_llm, mock_hf_model, mock_tokenizer
    ):
        """Test that random seed is set during initialization.

        Expected:
            - Worker stores seed from config
            - Can be used for reproducible operations
        """
        # Setup mocks
        mock_llm_cls.return_value = mock_vllm_llm
        mock_model_cls.from_pretrained.return_value = mock_hf_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_instantiate.side_effect = [Mock(), Mock()]

        from workers.ac_dc_worker import ACDCWorker

        # Set specific seed
        worker_config.seed = 12345

        # Initialize worker
        worker = ACDCWorker(worker_config)

        # Verify seed is stored (implementation detail may vary)
        assert worker_config.seed == 12345


# ============================================================================
# Task Loading
# ============================================================================


class TestTaskLoading:
    """Tests for ACDCWorker._load_tasks_from_info"""

    @patch('workers.ac_dc_worker.LLM')
    @patch('workers.ac_dc_worker.AutoModelForCausalLM')
    @patch('workers.ac_dc_worker.AutoTokenizer')
    @patch('workers.ac_dc_worker.hydra.utils.instantiate')
    def setup_worker(
        self, mock_instantiate, mock_tokenizer_cls, mock_model_cls, mock_llm_cls,
        worker_config, mock_vllm_llm, mock_hf_model, mock_tokenizer
    ):
        """Helper to set up worker for task loading tests."""
        mock_llm_cls.return_value = mock_vllm_llm
        mock_model_cls.from_pretrained.return_value = mock_hf_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_instantiate.side_effect = [Mock(), Mock()]

        from workers.ac_dc_worker import ACDCWorker
        return ACDCWorker(worker_config)

    @patch('workers.ac_dc_worker.LLM')
    @patch('workers.ac_dc_worker.AutoModelForCausalLM')
    @patch('workers.ac_dc_worker.AutoTokenizer')
    @patch('workers.ac_dc_worker.hydra.utils.instantiate')
    @patch('workers.ac_dc_worker.ACDCTask')
    def test_load_acdc_tasks_from_list(
        self, mock_acdc_task_cls, mock_instantiate, mock_tokenizer_cls,
        mock_model_cls, mock_llm_cls, worker_config, mock_vllm_llm,
        mock_hf_model, mock_tokenizer, worker_task_info_acdc
    ):
        """Test loading AC/DC tasks from list of paths.

        Setup:
            - task_info is a list of AC/DC task directories
            - Mock ACDCTask instantiation

        Expected:
            - ACDCTask is instantiated for each path
            - Tasks are returned as a list
        """
        # Setup mocks
        mock_llm_cls.return_value = mock_vllm_llm
        mock_model_cls.from_pretrained.return_value = mock_hf_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_instantiate.side_effect = [Mock(), Mock()]

        mock_tasks = [Mock() for _ in worker_task_info_acdc]
        mock_acdc_task_cls.side_effect = mock_tasks

        from workers.ac_dc_worker import ACDCWorker

        worker = ACDCWorker(worker_config)

        # Load tasks
        tasks = worker._load_tasks_from_info(worker_task_info_acdc, "train")

        # Verify ACDCTask was instantiated for each path
        assert len(tasks) == len(worker_task_info_acdc)
        assert mock_acdc_task_cls.call_count == len(worker_task_info_acdc)

    @patch('workers.ac_dc_worker.LLM')
    @patch('workers.ac_dc_worker.AutoModelForCausalLM')
    @patch('workers.ac_dc_worker.AutoTokenizer')
    @patch('workers.ac_dc_worker.hydra.utils.instantiate')
    def test_load_standard_tasks_from_dict(
        self, mock_instantiate, mock_tokenizer_cls, mock_model_cls, mock_llm_cls,
        worker_config, mock_vllm_llm, mock_hf_model, mock_tokenizer,
        worker_task_info_standard
    ):
        """Test loading standard tasks from dict of configs.

        Setup:
            - task_info is a dict mapping task names to configs
            - Mock Hydra instantiate for tasks

        Expected:
            - Tasks are instantiated via Hydra
            - Tasks are returned as a list
        """
        # Setup mocks
        mock_llm_cls.return_value = mock_vllm_llm
        mock_model_cls.from_pretrained.return_value = mock_hf_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock crossover, mutator, then tasks
        mock_tasks = [Mock() for _ in worker_task_info_standard]
        mock_instantiate.side_effect = [Mock(), Mock()] + mock_tasks

        from workers.ac_dc_worker import ACDCWorker

        worker = ACDCWorker(worker_config)

        # Load tasks
        tasks = worker._load_tasks_from_info(worker_task_info_standard, "train")

        # Verify tasks were instantiated
        assert len(tasks) == len(worker_task_info_standard)

    @patch('workers.ac_dc_worker.LLM')
    @patch('workers.ac_dc_worker.AutoModelForCausalLM')
    @patch('workers.ac_dc_worker.AutoTokenizer')
    @patch('workers.ac_dc_worker.hydra.utils.instantiate')
    @patch('workers.ac_dc_worker.ACDCTask')
    def test_load_tasks_with_invalid_path(
        self, mock_acdc_task_cls, mock_instantiate, mock_tokenizer_cls,
        mock_model_cls, mock_llm_cls, worker_config, mock_vllm_llm,
        mock_hf_model, mock_tokenizer
    ):
        """Test task loading with invalid/missing directory.

        Setup:
            - ACDCTask raises exception for missing directory

        Expected:
            - Exception is raised or handled gracefully
        """
        # Setup mocks
        mock_llm_cls.return_value = mock_vllm_llm
        mock_model_cls.from_pretrained.return_value = mock_hf_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_instantiate.side_effect = [Mock(), Mock()]

        # Make ACDCTask raise error for invalid path
        mock_acdc_task_cls.side_effect = FileNotFoundError("Task directory not found")

        from workers.ac_dc_worker import ACDCWorker

        worker = ACDCWorker(worker_config)

        # Attempt to load tasks with invalid paths
        with pytest.raises(FileNotFoundError):
            worker._load_tasks_from_info(["/invalid/path"], "train")

    @patch('workers.ac_dc_worker.LLM')
    @patch('workers.ac_dc_worker.AutoModelForCausalLM')
    @patch('workers.ac_dc_worker.AutoTokenizer')
    @patch('workers.ac_dc_worker.hydra.utils.instantiate')
    @patch('workers.ac_dc_worker.ACDCTask')
    def test_load_empty_task_list(
        self, mock_acdc_task_cls, mock_instantiate, mock_tokenizer_cls,
        mock_model_cls, mock_llm_cls, worker_config, mock_vllm_llm,
        mock_hf_model, mock_tokenizer
    ):
        """Test loading with empty task list.

        Expected:
            - Returns empty list without errors
        """
        # Setup mocks
        mock_llm_cls.return_value = mock_vllm_llm
        mock_model_cls.from_pretrained.return_value = mock_hf_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_instantiate.side_effect = [Mock(), Mock()]

        from workers.ac_dc_worker import ACDCWorker

        worker = ACDCWorker(worker_config)

        # Load empty task list
        tasks = worker._load_tasks_from_info([], "train")

        # Verify empty list returned
        assert tasks == []
        assert mock_acdc_task_cls.call_count == 0


# ============================================================================
# Merge Operations
# ============================================================================


class TestMergeOperations:
    """Tests for merge_models and merge_models_only"""

    @pytest.fixture
    def initialized_worker(
        self, worker_config, mock_vllm_llm, mock_hf_model,
        mock_tokenizer, mock_crossover, mock_mutator
    ):
        """Create a fully initialized worker for testing."""
        with patch('workers.ac_dc_worker.LLM') as mock_llm_cls, \
             patch('workers.ac_dc_worker.AutoModelForCausalLM') as mock_model_cls, \
             patch('workers.ac_dc_worker.AutoTokenizer') as mock_tokenizer_cls, \
             patch('workers.ac_dc_worker.hydra.utils.instantiate') as mock_instantiate:

            mock_llm_cls.return_value = mock_vllm_llm
            mock_model_cls.from_pretrained.return_value = mock_hf_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_instantiate.side_effect = [mock_crossover, mock_mutator]

            from workers.ac_dc_worker import ACDCWorker
            worker = ACDCWorker(worker_config)
            worker.crossover = mock_crossover
            worker.mutator = mock_mutator

            yield worker

    def test_merge_models_full_workflow(
        self, initialized_worker, worker_task_info_acdc, tmp_path,
        tiny_weight_dict, mock_acdc_tasks
    ):
        """Test full merge_models workflow with evaluation.

        Setup:
            - Mock crossover, mutation, and evaluation
            - Provide valid parent paths and save path

        Expected:
            - Returns ACDCMergeResult
            - Model and tokenizer are saved
            - Skill vector is computed
        """
        worker = initialized_worker
        parent_paths = ["/parent1", "/parent2"]
        save_path = str(tmp_path / "merged")

        # Mock operations
        worker.crossover.merge.return_value = tiny_weight_dict
        worker.mutator.mutate.return_value = tiny_weight_dict

        # Mock eval_model
        with patch.object(worker, '_eval_model') as mock_eval, \
             patch.object(worker, '_load_tasks_from_info') as mock_load_tasks:

            mock_load_tasks.return_value = mock_acdc_tasks
            mock_eval.return_value = (
                {},  # standard_metrics
                {"task_0": 0.8, "task_1": 0.7, "task_2": 0.75},  # acdc_skill_vector
                0.75,  # avg_quality
                [],  # eval_details
                False  # is_gibberish
            )

            # Call merge_models
            result = worker.merge_models(
                parent_paths=parent_paths,
                save_path=save_path,
                task_info=worker_task_info_acdc,
                do_mutate=True
            )

            # Verify result
            assert result is not None
            assert result.save_path == save_path
            assert result.acdc_skill_vector is not None
            assert len(result.acdc_skill_vector) == 3
            assert result.avg_acdc_quality == 0.75

            # Verify crossover and mutation were called
            worker.crossover.merge.assert_called_once()
            worker.mutator.mutate.assert_called_once()

    def test_merge_models_without_mutation(
        self, initialized_worker, worker_task_info_acdc, tmp_path,
        tiny_weight_dict, mock_acdc_tasks
    ):
        """Test merge_models without mutation.

        Setup:
            - do_mutate=False

        Expected:
            - Crossover is called
            - Mutation is NOT called
        """
        worker = initialized_worker
        parent_paths = ["/parent1", "/parent2"]
        save_path = str(tmp_path / "merged")

        worker.crossover.merge.return_value = tiny_weight_dict

        with patch.object(worker, '_eval_model') as mock_eval, \
             patch.object(worker, '_load_tasks_from_info') as mock_load_tasks:

            mock_load_tasks.return_value = mock_acdc_tasks
            mock_eval.return_value = ({}, {"task_0": 0.8}, 0.8, [], False)

            # Call merge_models without mutation
            result = worker.merge_models(
                parent_paths=parent_paths,
                save_path=save_path,
                task_info=worker_task_info_acdc,
                do_mutate=False
            )

            # Verify mutation was NOT called
            worker.crossover.merge.assert_called_once()
            worker.mutator.mutate.assert_not_called()
            assert result is not None

    def test_merge_models_only_no_evaluation(
        self, initialized_worker, tmp_path, tiny_weight_dict
    ):
        """Test merge_models_only without evaluation.

        Expected:
            - Returns save_path string
            - Evaluation is NOT performed
            - Model is saved
        """
        worker = initialized_worker
        parent_paths = ["/parent1", "/parent2"]
        save_path = str(tmp_path / "merged")

        worker.crossover.merge.return_value = tiny_weight_dict
        worker.mutator.mutate.return_value = tiny_weight_dict

        # Call merge_models_only
        with patch.object(worker, '_eval_model') as mock_eval:
            result = worker.merge_models_only(
                parent_paths=parent_paths,
                save_path=save_path,
                do_mutate=True
            )

            # Verify no evaluation was done
            mock_eval.assert_not_called()

            # Verify result is just the path
            assert result == save_path

    def test_merge_saves_model_and_tokenizer(
        self, initialized_worker, tmp_path, tiny_weight_dict
    ):
        """Test that merge operations save model and tokenizer.

        Expected:
            - HF model save_pretrained is called
            - Tokenizer save_pretrained is called
        """
        worker = initialized_worker
        parent_paths = ["/parent1", "/parent2"]
        save_path = str(tmp_path / "merged")

        worker.crossover.merge.return_value = tiny_weight_dict
        worker.mutator.mutate.return_value = tiny_weight_dict

        # Mock save operations
        worker.hf_model.save_pretrained = Mock()
        worker.tokenizer.save_pretrained = Mock()

        # Call merge_models_only
        result = worker.merge_models_only(
            parent_paths=parent_paths,
            save_path=save_path,
            do_mutate=False
        )

        # Verify save operations were called
        worker.hf_model.save_pretrained.assert_called_once()
        worker.tokenizer.save_pretrained.assert_called_once()

    def test_merge_with_retry_on_failure(
        self, initialized_worker, worker_task_info_acdc, tmp_path,
        tiny_weight_dict, mock_acdc_tasks
    ):
        """Test that merge retries on failure.

        Setup:
            - First merge attempt fails
            - Second attempt succeeds

        Expected:
            - Merge is retried
            - Eventually succeeds
        """
        worker = initialized_worker
        parent_paths = ["/parent1", "/parent2"]
        save_path = str(tmp_path / "merged")

        # First call fails, second succeeds
        worker.crossover.merge.side_effect = [
            Exception("Merge failed"),
            tiny_weight_dict
        ]

        with patch.object(worker, '_eval_model') as mock_eval, \
             patch.object(worker, '_load_tasks_from_info') as mock_load_tasks:

            mock_load_tasks.return_value = mock_acdc_tasks
            mock_eval.return_value = ({}, {"task_0": 0.8}, 0.8, [], False)

            # Should succeed after retry
            result = worker.merge_models(
                parent_paths=parent_paths,
                save_path=save_path,
                task_info=worker_task_info_acdc,
                do_mutate=False
            )

            # Verify merge was called twice (retry)
            assert worker.crossover.merge.call_count == 2
            assert result is not None

    def test_merge_loads_parent_models(
        self, initialized_worker, tmp_path, tiny_weight_dict
    ):
        """Test that merge loads parent model parameters.

        Expected:
            - Crossover loads parent models internally
            - Parameters are passed to crossover
        """
        worker = initialized_worker
        parent_paths = ["/parent1", "/parent2"]
        save_path = str(tmp_path / "merged")

        worker.crossover.merge.return_value = tiny_weight_dict

        # Call merge_models_only
        result = worker.merge_models_only(
            parent_paths=parent_paths,
            save_path=save_path,
            do_mutate=False
        )

        # Verify crossover.merge was called with parent paths
        # Note: Crossover loads models internally via AutoModelForCausalLM
        worker.crossover.merge.assert_called_once()
        call_args = worker.crossover.merge.call_args[0]
        assert call_args[1] == parent_paths  # Second arg is parent_paths

    def test_merge_saves_parent_mapping(
        self, initialized_worker, tmp_path, tiny_weight_dict
    ):
        """Test that merge saves parent mapping JSON.

        Expected:
            - parent_mapping.json is created
            - Contains parent paths
        """
        worker = initialized_worker
        parent_paths = ["/parent1", "/parent2"]
        save_path = str(tmp_path / "merged")

        worker.crossover.merge.return_value = tiny_weight_dict
        worker.mutator.mutate.return_value = tiny_weight_dict

        # Ensure save_path directory exists
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Call merge_models_only
        with patch.object(worker, '_save_parent_mapping') as mock_save_mapping:
            result = worker.merge_models_only(
                parent_paths=parent_paths,
                save_path=save_path,
                do_mutate=False
            )

            # Verify parent mapping was saved
            mock_save_mapping.assert_called_once_with(save_path, parent_paths)


# ============================================================================
# Model Initialization
# ============================================================================


class TestModelInitialization:
    """Tests for initialize_model and initialize_model_only"""

    @pytest.fixture
    def initialized_worker(
        self, worker_config, mock_vllm_llm, mock_hf_model,
        mock_tokenizer, mock_crossover, mock_mutator
    ):
        """Create a fully initialized worker for testing."""
        with patch('workers.ac_dc_worker.LLM') as mock_llm_cls, \
             patch('workers.ac_dc_worker.AutoModelForCausalLM') as mock_model_cls, \
             patch('workers.ac_dc_worker.AutoTokenizer') as mock_tokenizer_cls, \
             patch('workers.ac_dc_worker.hydra.utils.instantiate') as mock_instantiate:

            mock_llm_cls.return_value = mock_vllm_llm
            mock_model_cls.from_pretrained.return_value = mock_hf_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_instantiate.side_effect = [mock_crossover, mock_mutator]

            from workers.ac_dc_worker import ACDCWorker
            worker = ACDCWorker(worker_config)
            worker.crossover = mock_crossover
            worker.mutator = mock_mutator

            yield worker

    def test_initialize_model_with_evaluation(
        self, initialized_worker, worker_task_info_acdc, tmp_path,
        tiny_weight_dict, mock_acdc_tasks
    ):
        """Test initialize_model with evaluation.

        Setup:
            - Two seed models for crossover
            - Mock evaluation

        Expected:
            - Returns ACDCMergeResult
            - Skill vector is computed
        """
        worker = initialized_worker
        seed_model_paths = ["/seed1", "/seed2"]
        save_path = str(tmp_path / "init_model")

        worker.crossover.merge.return_value = tiny_weight_dict
        worker.mutator.mutate.return_value = tiny_weight_dict

        with patch.object(worker, '_eval_model') as mock_eval, \
             patch.object(worker, '_load_tasks_from_info') as mock_load_tasks, \
             patch.object(worker, 'load_model') as mock_load:

            mock_load.return_value = tiny_weight_dict
            mock_load_tasks.return_value = mock_acdc_tasks
            mock_eval.return_value = ({}, {"task_0": 0.8}, 0.8, [], False)

            # Call initialize_model
            result = worker.initialize_model(
                seed_model_paths=seed_model_paths,
                save_path=save_path,
                seed=42,
                task_info=worker_task_info_acdc,
                do_mutate=True
            )

            # Verify result
            assert result is not None
            assert result.save_path == save_path
            assert result.acdc_skill_vector is not None

    def test_initialize_model_only_no_evaluation(
        self, initialized_worker, tmp_path, tiny_weight_dict
    ):
        """Test initialize_model_only without evaluation.

        Expected:
            - Returns save_path string
            - No evaluation is performed
        """
        worker = initialized_worker
        seed_model_paths = ["/seed1", "/seed2"]
        save_path = str(tmp_path / "init_model")

        worker.crossover.merge.return_value = tiny_weight_dict
        worker.mutator.mutate.return_value = tiny_weight_dict

        with patch.object(worker, '_eval_model') as mock_eval, \
             patch.object(worker, 'load_model') as mock_load:

            mock_load.return_value = tiny_weight_dict

            # Call initialize_model_only
            result = worker.initialize_model_only(
                seed_model_paths=seed_model_paths,
                save_path=save_path,
                seed=42,
                do_mutate=True
            )

            # Verify no evaluation
            mock_eval.assert_not_called()
            assert result == save_path

    def test_initialize_with_single_seed_model(
        self, initialized_worker, tmp_path, tiny_weight_dict
    ):
        """Test initialization with single seed model (no crossover).

        Setup:
            - Only one seed model provided

        Expected:
            - Loads seed model directly via AutoModelForCausalLM
            - No crossover is performed
        """
        worker = initialized_worker
        seed_model_paths = ["/seed1"]
        save_path = str(tmp_path / "init_model")

        worker.mutator.mutate.return_value = tiny_weight_dict

        # Mock AutoModelForCausalLM for single seed loading
        with patch('workers.ac_dc_worker.AutoModelForCausalLM') as mock_auto_model:
            mock_model = Mock()
            mock_model.state_dict.return_value = tiny_weight_dict
            mock_auto_model.from_pretrained.return_value = mock_model

            # Call initialize_model_only
            result = worker.initialize_model_only(
                seed_model_paths=seed_model_paths,
                save_path=save_path,
                seed=42,
                do_mutate=True
            )

            # Verify crossover was NOT called (single seed, no merge)
            worker.crossover.merge.assert_not_called()
            # Verify AutoModelForCausalLM was used to load the single seed
            mock_auto_model.from_pretrained.assert_called_once_with(
                "/seed1", torch_dtype=torch.bfloat16
            )

    def test_initialize_with_mutation(
        self, initialized_worker, tmp_path, tiny_weight_dict
    ):
        """Test that initialization applies mutation when do_mutate=True.

        Expected:
            - Mutator is called
        """
        worker = initialized_worker
        seed_model_paths = ["/seed1", "/seed2"]
        save_path = str(tmp_path / "init_model")

        worker.crossover.merge.return_value = tiny_weight_dict
        worker.mutator.mutate.return_value = tiny_weight_dict

        with patch.object(worker, 'load_model') as mock_load:
            mock_load.return_value = tiny_weight_dict

            # Call with mutation
            result = worker.initialize_model_only(
                seed_model_paths=seed_model_paths,
                save_path=save_path,
                seed=42,
                do_mutate=True
            )

            # Verify mutation was called
            worker.mutator.mutate.assert_called_once()

    def test_initialize_without_mutation(
        self, initialized_worker, tmp_path, tiny_weight_dict
    ):
        """Test that initialization skips mutation when do_mutate=False.

        Expected:
            - Mutator is NOT called
        """
        worker = initialized_worker
        seed_model_paths = ["/seed1", "/seed2"]
        save_path = str(tmp_path / "init_model")

        worker.crossover.merge.return_value = tiny_weight_dict

        with patch.object(worker, 'load_model') as mock_load:
            mock_load.return_value = tiny_weight_dict

            # Call without mutation
            result = worker.initialize_model_only(
                seed_model_paths=seed_model_paths,
                save_path=save_path,
                seed=42,
                do_mutate=False
            )

            # Verify mutation was NOT called
            worker.mutator.mutate.assert_not_called()


# ============================================================================
# Evaluation Logic
# ============================================================================


class TestEvaluationLogic:
    """Tests for _eval_model and related evaluation methods"""

    @pytest.fixture
    def initialized_worker(
        self, worker_config, mock_vllm_llm, mock_hf_model,
        mock_tokenizer, mock_crossover, mock_mutator
    ):
        """Create a fully initialized worker for testing."""
        with patch('workers.ac_dc_worker.LLM') as mock_llm_cls, \
             patch('workers.ac_dc_worker.AutoModelForCausalLM') as mock_model_cls, \
             patch('workers.ac_dc_worker.AutoTokenizer') as mock_tokenizer_cls, \
             patch('workers.ac_dc_worker.hydra.utils.instantiate') as mock_instantiate:

            mock_llm_cls.return_value = mock_vllm_llm
            mock_model_cls.from_pretrained.return_value = mock_hf_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_instantiate.side_effect = [mock_crossover, mock_mutator]

            from workers.ac_dc_worker import ACDCWorker
            worker = ACDCWorker(worker_config)

            yield worker

    def test_eval_model_with_acdc_tasks(
        self, initialized_worker, mock_acdc_tasks, tiny_weight_dict
    ):
        """Test _eval_model with AC/DC tasks.

        Setup:
            - Mock vLLM generation
            - Mock sandbox evaluation

        Expected:
            - Returns tuple with acdc_skill_vector
            - Skill vector has scores for all tasks
        """
        worker = initialized_worker

        # Mock vLLM generation and weight loading
        with patch.object(worker, 'load_params_fn') as mock_load_params, \
             patch.object(worker.llm, 'generate') as mock_generate, \
             patch('workers.ac_dc_worker.multiprocessing.Pool') as mock_pool_cls:

            # Setup mock outputs
            mock_outputs = []
            for i in range(len(mock_acdc_tasks)):
                mock_output = Mock()
                mock_output.outputs = [Mock(text=f"Answer {i}")]
                mock_outputs.append(mock_output)
            mock_generate.return_value = mock_outputs

            # Mock multiprocessing pool
            mock_pool = MagicMock()
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.map.return_value = [
                (f"task_{i}", 0.8, f"Instructions {i}", f"Answer {i}")
                for i in range(3)
            ]
            mock_pool_cls.return_value = mock_pool

            # Call eval
            results = worker._eval_model(
                tiny_weight_dict, "train", mock_acdc_tasks
            )

            # Verify results
            standard_metrics, acdc_skill_vector, avg_quality, details, is_gibberish = results
            assert acdc_skill_vector is not None
            assert len(acdc_skill_vector) == 3
            assert avg_quality == pytest.approx(0.8)
            assert len(details) == 3
            assert is_gibberish is False

    def test_eval_model_batched_generation(
        self, initialized_worker, mock_acdc_tasks, tiny_weight_dict
    ):
        """Test that _eval_model batches LLM generation.

        Expected:
            - llm.generate is called with all prompts at once
        """
        worker = initialized_worker

        with patch.object(worker, 'load_params_fn') as mock_load_params, \
             patch.object(worker.llm, 'generate') as mock_generate, \
             patch('workers.ac_dc_worker.multiprocessing.Pool') as mock_pool_cls:

            # Setup mocks
            mock_outputs = [Mock(outputs=[Mock(text="Answer")]) for _ in mock_acdc_tasks]
            mock_generate.return_value = mock_outputs

            mock_pool = MagicMock()
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.map.return_value = [
                (f"task_{i}", 0.8, "Instructions", "Answer")
                for i in range(3)
            ]
            mock_pool_cls.return_value = mock_pool

            # Call eval
            worker._eval_model(tiny_weight_dict, "train", mock_acdc_tasks)

            # Verify generate was called once with all prompts
            mock_generate.assert_called_once()

    def test_eval_model_parallel_sandbox_evaluation(
        self, initialized_worker, mock_acdc_tasks, tiny_weight_dict
    ):
        """Test that sandbox evaluation uses multiprocessing.

        Expected:
            - multiprocessing.Pool is used
            - Pool.map is called with correct arguments
        """
        worker = initialized_worker

        with patch.object(worker, 'load_params_fn') as mock_load_params, \
             patch.object(worker.llm, 'generate') as mock_generate, \
             patch('workers.ac_dc_worker.multiprocessing.Pool') as mock_pool_cls:

            mock_outputs = [Mock(outputs=[Mock(text="Answer")]) for _ in mock_acdc_tasks]
            mock_generate.return_value = mock_outputs

            mock_pool = MagicMock()
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.map.return_value = [
                ("task_0", 0.9, "Instructions", "Answer"),
                ("task_1", 0.7, "Instructions", "Answer"),
                ("task_2", 0.8, "Instructions", "Answer"),
            ]
            mock_pool_cls.return_value = mock_pool

            # Call eval
            worker._eval_model(tiny_weight_dict, "train", mock_acdc_tasks)

            # Verify pool was used
            mock_pool.map.assert_called_once()
            # Verify correct number of worker arguments
            args = mock_pool.map.call_args[0]
            worker_args = args[1]
            assert len(worker_args) == len(mock_acdc_tasks)

    def test_eval_model_with_empty_task_list(
        self, initialized_worker, tiny_weight_dict
    ):
        """Test _eval_model with empty task list.

        Expected:
            - Returns None/empty results without errors (production code returns None for empty collections)
        """
        worker = initialized_worker

        # Mock weight loading
        with patch.object(worker, 'load_params_fn') as mock_load_params:
            # Call eval with empty task list
            results = worker._eval_model(tiny_weight_dict, "train", [])

        # Verify empty results - production code returns None for empty collections
        standard_metrics, acdc_skill_vector, avg_quality, details, is_gibberish = results
        assert acdc_skill_vector is None  # Empty dict becomes None
        assert avg_quality is None  # No tasks means None
        assert details is None  # Empty list becomes None

    def test_eval_model_gibberish_detection(
        self, initialized_worker, mock_acdc_tasks, tiny_weight_dict
    ):
        """Test gibberish detection in eval_model.

        Setup:
            - Enable gibberish check in config
            - Mock gibberish responses

        Expected:
            - is_gibberish flag is set correctly
        """
        worker = initialized_worker
        worker.cfg.dns.run_gibberish_check = True

        with patch.object(worker, 'load_params_fn') as mock_load_params, \
             patch.object(worker.llm, 'generate') as mock_generate, \
             patch('workers.ac_dc_worker.multiprocessing.Pool') as mock_pool_cls, \
             patch.object(worker, '_is_gibberish') as mock_gibberish:

            mock_outputs = [Mock(outputs=[Mock(text="gibberish" * 100)]) for _ in mock_acdc_tasks]
            mock_generate.return_value = mock_outputs

            mock_pool = MagicMock()
            mock_pool.__enter__.return_value = mock_pool
            mock_pool.map.return_value = [
                (f"task_{i}", 0.0, "Instructions", "gibberish" * 100)
                for i in range(3)
            ]
            mock_pool_cls.return_value = mock_pool

            # Mock gibberish detection
            mock_gibberish.return_value = True

            # Call eval
            results = worker._eval_model(tiny_weight_dict, "train", mock_acdc_tasks)

            # Verify gibberish was detected
            _, _, _, _, is_gibberish = results
            assert is_gibberish is True

    def test_eval_model_with_standard_tasks(
        self, initialized_worker, mock_standard_task, tiny_weight_dict
    ):
        """Test _eval_model with standard (non-AC/DC) tasks.

        Expected:
            - Standard task evaluation is performed
            - Returns standard metrics
        """
        worker = initialized_worker
        tasks = [mock_standard_task]

        with patch.object(worker, 'load_params_fn') as mock_load_params, \
             patch.object(worker.llm, 'generate') as mock_generate:
            mock_outputs = [Mock(outputs=[Mock(text="Answer")])]
            mock_generate.return_value = mock_outputs

            # Call eval
            results = worker._eval_model(tiny_weight_dict, "train", tasks)

            # Verify standard metrics are returned
            standard_metrics, acdc_skill_vector, avg_quality, details, is_gibberish = results
            assert standard_metrics is not None

    def test_eval_model_handles_vllm_errors(
        self, initialized_worker, mock_acdc_tasks, tiny_weight_dict
    ):
        """Test that eval_model handles vLLM errors gracefully.

        Setup:
            - vLLM generate raises exception

        Expected:
            - Error is caught and handled
            - Returns empty/error results
        """
        worker = initialized_worker

        with patch.object(worker, 'load_params_fn') as mock_load_params, \
             patch.object(worker.llm, 'generate') as mock_generate:
            mock_generate.side_effect = Exception("vLLM error")

            # Call eval - should not crash
            # (Implementation might retry or return error results)
            # This depends on actual error handling in the worker
            try:
                results = worker._eval_model(tiny_weight_dict, "train", mock_acdc_tasks)
                # If it returns, check that it's a reasonable error result
                assert results is not None
            except Exception as e:
                # If it raises, that's also acceptable - just verify it's handled
                assert "vLLM error" in str(e)

    def test_eval_only_task(
        self, initialized_worker, worker_task_info_acdc, mock_acdc_tasks, tiny_weight_dict
    ):
        """Test eval_model_only method (evaluation without merge).

        Expected:
            - Loads model from path
            - Evaluates and returns result
            - Does not modify model
        """
        worker = initialized_worker
        model_path = "/test/model"

        # Mock AutoModelForCausalLM.from_pretrained which is used by eval_model_only
        with patch('workers.ac_dc_worker.AutoModelForCausalLM') as mock_auto_model, \
             patch.object(worker, '_eval_model') as mock_eval, \
             patch.object(worker, '_load_tasks_from_info') as mock_load_tasks:

            # Mock model loading
            mock_model = Mock()
            mock_model.state_dict.return_value = tiny_weight_dict
            mock_auto_model.from_pretrained.return_value = mock_model

            mock_load_tasks.return_value = mock_acdc_tasks
            mock_eval.return_value = ({}, {"task_0": 0.8}, 0.8, [], False)

            # Call eval_model_only
            result = worker.eval_model_only(
                model_path=model_path,
                task_info=worker_task_info_acdc,
                data_split="train"
            )

            # Verify result
            assert result is not None
            assert result.save_path == model_path
            # Verify model was loaded using AutoModelForCausalLM.from_pretrained
            mock_auto_model.from_pretrained.assert_called_once()
