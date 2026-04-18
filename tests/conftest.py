"""
Shared pytest fixtures for AC/DC test suite.

This module provides reusable fixtures for testing various components of the AC/DC framework.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import Mock, MagicMock, patch
from omegaconf import OmegaConf


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_addoption(parser):
    """Add custom pytest command-line options."""
    parser.addoption(
        "--run-docker",
        action="store_true",
        default=False,
        help="Run Docker integration tests (requires Docker daemon and acdc-sandbox image)"
    )


# ============================================================================
# Basic Fixtures
# ============================================================================


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def np_random(random_seed):
    """NumPy random state with fixed seed."""
    return np.random.RandomState(random_seed)


@pytest.fixture
def torch_device():
    """Device for torch operations (CPU for testing)."""
    return torch.device("cpu")


# ============================================================================
# Model Weight Fixtures
# ============================================================================


@pytest.fixture
def tiny_weight_dict(torch_device):
    """
    Create a minimal weight dictionary for testing.
    Simulates a tiny model with just a few layers.
    """
    return {
        "model.embed_tokens.weight": torch.randn(
            100, 64, dtype=torch.bfloat16, device=torch_device
        ),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(
            64, 64, dtype=torch.bfloat16, device=torch_device
        ),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(
            64, 64, dtype=torch.bfloat16, device=torch_device
        ),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(
            64, 64, dtype=torch.bfloat16, device=torch_device
        ),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(
            128, 64, dtype=torch.bfloat16, device=torch_device
        ),
        "model.layers.0.mlp.up_proj.weight": torch.randn(
            128, 64, dtype=torch.bfloat16, device=torch_device
        ),
        "model.layers.0.mlp.down_proj.weight": torch.randn(
            64, 128, dtype=torch.bfloat16, device=torch_device
        ),
        "lm_head.weight": torch.randn(
            100, 64, dtype=torch.bfloat16, device=torch_device
        ),
    }


@pytest.fixture
def tiny_weight_dict_with_norms(tiny_weight_dict, torch_device):
    """
    Weight dictionary including normalization layers.
    Used for testing mutation operators that skip normalization layers.
    """
    weights = tiny_weight_dict.copy()
    weights.update(
        {
            "model.layers.0.input_layernorm.weight": torch.ones(
                64, dtype=torch.bfloat16, device=torch_device
            ),
            "model.layers.0.post_attention_layernorm.weight": torch.ones(
                64, dtype=torch.bfloat16, device=torch_device
            ),
            "model.norm.weight": torch.ones(
                64, dtype=torch.bfloat16, device=torch_device
            ),
        }
    )
    return weights


@pytest.fixture
def base_model_weights(tiny_weight_dict):
    """Base model weights (before fine-tuning)."""
    return tiny_weight_dict


@pytest.fixture
def finetuned_model_weights_1(base_model_weights):
    """
    First fine-tuned model weights.
    Adds a small perturbation to base weights.
    """
    return {
        k: v + torch.randn_like(v) * 0.01 for k, v in base_model_weights.items()
    }


@pytest.fixture
def finetuned_model_weights_2(base_model_weights):
    """
    Second fine-tuned model weights.
    Adds a different perturbation to base weights.
    """
    return {
        k: v + torch.randn_like(v) * 0.01 for k, v in base_model_weights.items()
    }


@pytest.fixture
def task_vector_1(base_model_weights, finetuned_model_weights_1):
    """Task vector (difference between fine-tuned and base model)."""
    return {
        k: finetuned_model_weights_1[k] - base_model_weights[k]
        for k in base_model_weights
    }


@pytest.fixture
def task_vector_2(base_model_weights, finetuned_model_weights_2):
    """Second task vector."""
    return {
        k: finetuned_model_weights_2[k] - base_model_weights[k]
        for k in base_model_weights
    }


# ============================================================================
# SVD Fixtures (for SVD-based mutation)
# ============================================================================


@pytest.fixture
def mock_svd_dict(task_vector_1, tmp_path):
    """
    Create a mock SVD dictionary for testing SVD-based mutators.
    Saves it to a temporary file and returns the path.
    """
    svd_dict = {}
    q_name = "test_task"

    for key, value in task_vector_1.items():
        if "norm" in key:
            continue

        # Compute SVD
        matrix = value.float()
        if len(matrix.shape) == 1:
            # Skip 1D tensors
            continue

        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

        # Store SVD components
        svd_dict[f"{key}.U"] = U.bfloat16()
        svd_dict[f"{key}.S"] = S.bfloat16()
        svd_dict[f"{key}.V"] = (
            Vh.T.bfloat16()
        )  # Transpose to match expected format

    # Nest under task name
    svd_dict = {q_name: svd_dict}

    # Save to temporary file
    svd_path = tmp_path / "test_svd.pt"
    torch.save(svd_dict, svd_path)

    return str(svd_path)


# ============================================================================
# Skill Vector Fixtures
# ============================================================================


@pytest.fixture
def sample_skill_vectors():
    """
    Sample binary skill vectors for testing coverage algorithms.

    Returns:
        List of skill vectors where each vector is a list of 0s and 1s.
        Represents which tasks each model can solve.
    """
    return [
        [1, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # Model 0: solves tasks 0,1,5,8
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # Model 1: solves tasks 0,2,6,9
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],  # Model 2: solves tasks 1,3,7
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],  # Model 3: solves tasks 4,5,8
        [0, 0, 1, 1, 1, 0, 1, 0, 0, 1],  # Model 4: solves tasks 2,3,4,6,9
    ]


@pytest.fixture
def sample_skill_vectors_continuous():
    """
    Sample continuous skill vectors (scores between 0 and 1).
    Used for testing fitness-based selection.
    """
    return [
        {
            "task_0": 0.9,
            "task_1": 0.8,
            "task_2": 0.2,
            "task_3": 0.1,
            "task_4": 0.5,
        },
        {
            "task_0": 0.7,
            "task_1": 0.3,
            "task_2": 0.9,
            "task_3": 0.4,
            "task_4": 0.6,
        },
        {
            "task_0": 0.2,
            "task_1": 0.9,
            "task_2": 0.1,
            "task_3": 0.8,
            "task_4": 0.3,
        },
        {
            "task_0": 0.5,
            "task_1": 0.4,
            "task_2": 0.7,
            "task_3": 0.6,
            "task_4": 0.9,
        },
        {
            "task_0": 0.1,
            "task_1": 0.2,
            "task_2": 0.5,
            "task_3": 0.9,
            "task_4": 0.8,
        },
    ]


# ============================================================================
# Archive Fixtures
# ============================================================================


@pytest.fixture
def sample_archive_data():
    """Sample archive data for testing."""
    from datatypes import ACDCSolution

    return [
        ACDCSolution(
            model_path="/models/gen_0_ind_0",
            fitness=0.75,
            acdc_skill_vector={"task_0": 0.9, "task_1": 0.8, "task_2": 0.5},
            rank=None,
        ),
        ACDCSolution(
            model_path="/models/gen_1_ind_5",
            fitness=0.82,
            acdc_skill_vector={"task_0": 0.7, "task_1": 0.9, "task_2": 0.8},
            rank=None,
        ),
        ACDCSolution(
            model_path="/models/gen_2_ind_12",
            fitness=0.68,
            acdc_skill_vector={"task_0": 0.6, "task_1": 0.7, "task_2": 0.7},
            rank=None,
        ),
    ]


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_experiment_dir(tmp_path):
    """
    Create a temporary experiment directory structure.
    Useful for integration tests that need file I/O.
    """
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    # Create subdirectories
    (exp_dir / "models").mkdir()
    (exp_dir / "archives").mkdir()
    (exp_dir / "global_skill_vectors").mkdir()
    (exp_dir / "generated_tasks").mkdir()
    (exp_dir / "generated_tasks" / "pool").mkdir()

    return exp_dir


@pytest.fixture
def temp_model_dir(temp_experiment_dir):
    """Temporary directory for storing model checkpoints."""
    return temp_experiment_dir / "models"


@pytest.fixture
def temp_archive_dir(temp_experiment_dir):
    """Temporary directory for storing archives."""
    return temp_experiment_dir / "archives"


# ============================================================================
# Model Path Fixtures
# ============================================================================


@pytest.fixture
def mock_model_paths(temp_model_dir):
    """
    Create mock model paths for testing.
    Does not actually save models, just returns paths.
    """
    paths = [
        str(temp_model_dir / f"gen_{gen}_ind_{ind}")
        for gen, ind in [(0, 0), (0, 1), (1, 5), (1, 7), (2, 12)]
    ]

    # Create the directories so they exist
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

    return paths


# ============================================================================
# Helper Functions
# ============================================================================


def assert_weight_dicts_close(
    dict1: Dict[str, torch.Tensor],
    dict2: Dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """
    Assert that two weight dictionaries are approximately equal.

    Args:
        dict1: First weight dictionary
        dict2: Second weight dictionary
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    assert (
        dict1.keys() == dict2.keys()
    ), "Weight dictionaries have different keys"

    for key in dict1:
        torch.testing.assert_close(
            dict1[key],
            dict2[key],
            rtol=rtol,
            atol=atol,
            msg=f"Mismatch in key: {key}",
        )


def assert_weight_dict_shapes_match(
    dict1: Dict[str, torch.Tensor], dict2: Dict[str, torch.Tensor]
):
    """
    Assert that two weight dictionaries have matching shapes.

    Args:
        dict1: First weight dictionary
        dict2: Second weight dictionary
    """
    assert (
        dict1.keys() == dict2.keys()
    ), "Weight dictionaries have different keys"

    for key in dict1:
        assert (
            dict1[key].shape == dict2[key].shape
        ), f"Shape mismatch for key {key}: {dict1[key].shape} vs {dict2[key].shape}"


# Export helper functions
pytest.assert_weight_dicts_close = assert_weight_dicts_close
pytest.assert_weight_dict_shapes_match = assert_weight_dict_shapes_match


# ============================================================================
# DNS and VectorDB Fixtures
# ============================================================================


@pytest.fixture
def mock_embedding_client():
    """Mock OpenAI client for embedding generation."""
    from unittest.mock import Mock

    mock_client = Mock()
    mock_embedding_data = Mock()
    mock_embedding_data.embedding = list(np.random.rand(384).astype(np.float32))
    mock_response = Mock()
    mock_response.data = [mock_embedding_data]
    mock_client.embeddings.create.return_value = mock_response

    return mock_client


@pytest.fixture
def vector_db_with_mock_embedding(tmp_path, mock_embedding_client):
    """VectorDB instance with mocked embedding server."""
    from unittest.mock import patch
    from tasks.simple_vectordb import SimpleVectorDB

    with patch("tasks.simple_vectordb.OpenAI") as mock_openai:
        mock_openai.return_value = mock_embedding_client

        db = SimpleVectorDB(
            storage_path=str(tmp_path / "vector_db"),
            embedding_model_name="test-model",
            embedding_vllm_url="http://localhost:8010/v1",
            task_representation_vector_db="metadata",
            dimension=384,
        )

        yield db


@pytest.fixture
def vector_db_with_samples(vector_db_with_mock_embedding):
    """VectorDB pre-populated with sample tasks."""
    db = vector_db_with_mock_embedding

    # Add diverse samples
    samples = [
        {
            "content": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "metadata": {"category": "recursion", "difficulty": "medium"},
        },
        {
            "content": "def bubble_sort(arr): pass",
            "metadata": {"category": "sorting", "difficulty": "easy"},
        },
        {
            "content": "def merge_sort(arr): pass",
            "metadata": {"category": "sorting", "difficulty": "hard"},
        },
    ]

    for sample in samples:
        db.add_sample(content=sample["content"], metadata=sample["metadata"])

    return db


@pytest.fixture
def vector_db_with_diverse_samples(vector_db_with_mock_embedding):
    """VectorDB with diverse samples for similarity testing."""
    db = vector_db_with_mock_embedding

    samples = [
        {
            "content": "def task_logic_1(): pass",
            "metadata": {"category": "logic", "difficulty": "easy"},
        },
        {
            "content": "def task_logic_2(): pass",
            "metadata": {"category": "logic", "difficulty": "medium"},
        },
        {
            "content": "def task_math_1(): pass",
            "metadata": {"category": "math", "difficulty": "hard"},
        },
        {
            "content": "def task_string_1(): pass",
            "metadata": {"category": "string", "difficulty": "easy"},
        },
        {
            "content": "def task_array_1(): pass",
            "metadata": {"category": "array", "difficulty": "medium"},
        },
    ]

    for sample in samples:
        db.add_sample(content=sample["content"], metadata=sample["metadata"])

    return db


@pytest.fixture
def vector_db_with_many_samples(vector_db_with_mock_embedding):
    """VectorDB with many samples for top_n testing."""
    db = vector_db_with_mock_embedding

    for i in range(20):
        db.add_sample(
            content=f"def task_{i}(): pass",
            metadata={"category": "common", "index": i},
        )

    return db


@pytest.fixture
def dns_solutions():
    """Sample DNS solutions for testing."""
    from datatypes import DNSSolution

    # Define skill vectors first, then compute fitness
    skill_vectors = [
        [True, True, False, True, False],  # 3/5 = 0.6
        [True, False, True, False, True],  # 3/5 = 0.6
        [True, True, True, True, False],  # 4/5 = 0.8
    ]

    return [
        DNSSolution(
            model_path=["gen_0_ind_0", "gen_0_ind_1", "gen_1_ind_5"][i],
            fitness=sum(skill_vectors[i]) / len(skill_vectors[i]),
            skill_vector=skill_vectors[i],
        )
        for i in range(3)
    ]


@pytest.fixture
def dns_population():
    """DNS population for difficulty weight testing."""
    from datatypes import DNSSolution

    # Define skill vectors first
    solutions = []
    for i in range(10):
        skill_vector = [True if (i + j) % 2 == 0 else False for j in range(5)]
        solutions.append(
            DNSSolution(
                model_path=f"model_{i}",
                fitness=sum(skill_vector) / len(skill_vector),
                skill_vector=skill_vector,
            )
        )
    return solutions


@pytest.fixture
def ac_dc_solutions():
    """Sample AC/DC solutions for testing."""
    from datatypes import ACDCSolution, ACDCTaskEvalDetail

    return [
        ACDCSolution(
            model_path="gen_1_ind_0",
            fitness=0.75,
            acdc_skill_vector={
                "task_0_example_0": 0.9,
                "task_0_example_1": 0.7,
                "task_1_example_0": 0.6,
            },
            acdc_eval_details=[
                ACDCTaskEvalDetail(
                    task_id="task_0",
                    instructions="Test instructions for task_0",
                    raw_output="correct",
                    score=0.9,
                )
            ],
        ),
        ACDCSolution(
            model_path="gen_1_ind_1",
            fitness=0.65,
            acdc_skill_vector={
                "task_0_example_0": 0.8,
                "task_1_example_0": 0.5,
            },
        ),
    ]


@pytest.fixture
def ac_dc_solution_with_many_details():
    """AC/DC solution with many eval details for truncation testing."""
    from datatypes import ACDCSolution, ACDCTaskEvalDetail

    eval_details = [
        ACDCTaskEvalDetail(
            task_id=f"task_{i}",
            instructions=f"Instructions for task_{i}",
            raw_output="output",
            score=0.8,
        )
        for i in range(10)
    ]

    return ACDCSolution(
        model_path="gen_2_ind_5",
        fitness=0.7,
        acdc_skill_vector={f"task_{i}_example_0": 0.8 for i in range(10)},
        acdc_eval_details=eval_details,
    )


@pytest.fixture
def mock_task_metrics():
    """Mock task metrics for skill vector testing."""
    from datatypes import TaskMetric
    from types import SimpleNamespace

    metrics = {}
    for task_num in range(3):
        task_name = f"task_{task_num}"
        example_results = {}

        for example_num in range(2):
            example_id = f"example_{example_num}"
            # Create mock example result
            result = SimpleNamespace(correct=task_num % 2 == 0)
            example_results[example_id] = result

        metric = SimpleNamespace(
            example_results=example_results, quality=0.7 + task_num * 0.1
        )
        metrics[task_name] = metric

    return metrics


@pytest.fixture
def mock_task_metrics_qd():
    """Mock task metrics for QD mode testing."""
    from types import SimpleNamespace

    metrics = {}
    for task_num in range(3):
        task_name = f"task_{task_num}"
        example_results = {}

        for example_num in range(2):
            example_id = f"example_{example_num}"
            result = SimpleNamespace(correct=(task_num + example_num) % 2 == 0)
            example_results[example_id] = result

        metric = SimpleNamespace(example_results=example_results)
        metrics[task_name] = metric

    return metrics


@pytest.fixture
def mock_tasks():
    """Mock task objects for testing."""
    from types import SimpleNamespace

    return [SimpleNamespace(task_name=f"task_{i}") for i in range(3)]


@pytest.fixture
def dns_archive():
    """DNS archive for update testing."""
    from datatypes import DNSSolution

    # Define skill vectors first
    skill_vectors = [
        [True, False, False, False, False],  # 1/5 = 0.2
        [True, False, True, False, True],  # 3/5 = 0.6
        [True, False, False, True, False],  # 2/5 = 0.4
        [True, False, False, False, True],  # 2/5 = 0.4
        [True, False, False, False, False],  # 1/5 = 0.2
    ]

    return [
        DNSSolution(
            model_path=f"archive_model_{i}",
            fitness=sum(skill_vectors[i]) / len(skill_vectors[i]),
            skill_vector=skill_vectors[i],
        )
        for i in range(5)
    ]


@pytest.fixture
def new_dns_solutions():
    """New DNS solutions to add to archive."""
    from datatypes import DNSSolution

    # Define skill vectors first
    skill_vectors = [
        [False, True, True, True, True],  # 4/5 = 0.8
        [True, False, True, True, True],  # 4/5 = 0.8
        [True, True, False, True, True],  # 4/5 = 0.8
    ]

    return [
        DNSSolution(
            model_path=f"new_model_{i}",
            fitness=sum(skill_vectors[i]) / len(skill_vectors[i]),
            skill_vector=skill_vectors[i],
        )
        for i in range(3)
    ]


@pytest.fixture
def dns_cfg_basic():
    """Basic DNS configuration."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "population_size": 5,
            "use_top_fitness_selection": False,
            "use_difficulty_weights": False,
            "k_neighbors": 3,
            "dominated_score": 999.0,
            "use_skill_ratio": False,
            "skill_ratio_to_full": False,
        }
    )


@pytest.fixture
def dns_cfg_top_fitness():
    """DNS config with top-fitness selection."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "population_size": 5,
            "use_top_fitness_selection": True,
        }
    )


@pytest.fixture
def dns_cfg_top_fitness_small():
    """DNS config with small population size."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "population_size": 3,
            "use_top_fitness_selection": True,
        }
    )


@pytest.fixture
def dns_cfg_novelty():
    """DNS config emphasizing novelty."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "population_size": 5,
            "use_top_fitness_selection": False,
            "use_difficulty_weights": True,
            "k_neighbors": 3,
            "dominated_score": 999.0,
            "use_skill_ratio": True,
            "skill_ratio_to_full": False,
        }
    )


@pytest.fixture
def high_fitness_archive():
    """Archive with high-fitness solutions."""
    from datatypes import DNSSolution

    # All solutions solve all 5 tasks (fitness = 1.0)
    return [
        DNSSolution(
            model_path=f"high_fitness_{i}",
            fitness=1.0,
            skill_vector=[True] * 5,
        )
        for i in range(5)
    ]


@pytest.fixture
def low_fitness_solutions():
    """Solutions with low fitness."""
    from datatypes import DNSSolution

    # Define skill vectors with low fitness
    skill_vectors = [
        [False, True, False, True, False],  # 2/5 = 0.4
        [True, False, False, False, False],  # 1/5 = 0.2
        [False, False, True, False, False],  # 1/5 = 0.2
    ]

    return [
        DNSSolution(
            model_path=f"low_fitness_{i}",
            fitness=sum(skill_vectors[i]) / len(skill_vectors[i]),
            skill_vector=skill_vectors[i],
        )
        for i in range(3)
    ]


@pytest.fixture
def many_dns_solutions():
    """Many DNS solutions for population size testing."""
    from datatypes import DNSSolution

    # Define skill vectors first
    solutions = []
    for i in range(20):
        skill_vector = [True if (i + j) % 3 == 0 else False for j in range(5)]
        solutions.append(
            DNSSolution(
                model_path=f"solution_{i}",
                fitness=sum(skill_vector) / len(skill_vector),
                skill_vector=skill_vector,
            )
        )
    return solutions


@pytest.fixture
def dns_archive_diverse():
    """Diverse DNS archive for novelty testing."""
    from datatypes import DNSSolution

    # Specialists each solve 1/5 tasks, generalist solves 5/5
    return [
        DNSSolution(
            model_path="specialist_0",
            fitness=0.2,  # 1/5
            skill_vector=[True, False, False, False, False],
        ),
        DNSSolution(
            model_path="specialist_1",
            fitness=0.2,  # 1/5
            skill_vector=[False, True, False, False, False],
        ),
        DNSSolution(
            model_path="specialist_2",
            fitness=0.2,  # 1/5
            skill_vector=[False, False, True, False, False],
        ),
        DNSSolution(
            model_path="generalist",
            fitness=1.0,  # 5/5
            skill_vector=[True, True, True, True, True],
        ),
    ]


@pytest.fixture
def new_dns_solutions_novel():
    """Novel DNS solutions for testing novelty selection."""
    from datatypes import DNSSolution

    return [
        DNSSolution(
            model_path="novel_specialist_3",
            fitness=0.2,  # 1/5
            skill_vector=[False, False, False, True, False],
        ),
        DNSSolution(
            model_path="novel_specialist_4",
            fitness=0.2,  # 1/5
            skill_vector=[False, False, False, False, True],
        ),
        DNSSolution(
            model_path="duplicate_generalist",
            fitness=1.0,  # 5/5
            skill_vector=[True, True, True, True, True],  # Similar to existing
        ),
    ]


# ============================================================================
# Worker Module Fixtures (for ac_dc_worker.py testing)
# ============================================================================


@pytest.fixture
def worker_config():
    """Minimal worker configuration for testing."""
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "base_model_path": "test/model",
        "chat_template": "llama3",
        "gpu_memory_utilization": 0.9,
        "seed": 42,
        "evo": {
            "crossover": {"_target_": "crossover.model_linear.LinearCrossover", "merge_lambda": 0.5},
            "mutation": {"_target_": "mutation.gaussian_mutator.GaussianMutator", "std": 0.01}
        },
        "svd_expert_names": ["task1", "task2"],
        "vllm_pop": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "stop_token_ids": [],
            "eval_cot": False
        },
        "dns": {"run_gibberish_check": False},
        "acdc": {"num_sandbox_workers": 4},
        "evaluation": {"max_retries": 3}
    })


@pytest.fixture
def mock_vllm_llm():
    """Mock vLLM LLM instance."""
    from unittest.mock import Mock

    mock_llm = Mock()
    # Mock the nested model executor structure
    mock_model = Mock()
    mock_model.named_parameters.return_value = []
    mock_llm.llm_engine.model_executor.driver_worker.worker.model_runner.model = mock_model

    # Mock generate method
    mock_output = Mock()
    mock_output.outputs = [Mock(text="Test output")]
    mock_llm.generate.return_value = [mock_output]

    return mock_llm


@pytest.fixture
def mock_hf_model(tiny_weight_dict):
    """Mock HuggingFace model."""
    from unittest.mock import Mock

    mock_model = Mock()
    mock_model.state_dict.return_value = tiny_weight_dict
    return mock_model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer."""
    from unittest.mock import Mock

    mock_tok = Mock()
    mock_tok.chat_template = None  # Will be set by worker
    mock_tok.apply_chat_template.return_value = ["Test prompt"]
    return mock_tok


@pytest.fixture
def mock_crossover():
    """Mock crossover operator."""
    from unittest.mock import Mock

    mock_cross = Mock()
    mock_cross.merge.return_value = {"test.weight": torch.randn(10, 10)}
    mock_cross.update_seed = Mock()
    return mock_cross


@pytest.fixture
def mock_mutator():
    """Mock mutation operator."""
    from unittest.mock import Mock

    mock_mut = Mock()
    mock_mut.mutate.return_value = {"test.weight": torch.randn(10, 10)}
    mock_mut.update_seed = Mock()
    return mock_mut


@pytest.fixture
def mock_acdc_task(tmp_path):
    """Mock AC/DC task instance."""
    from unittest.mock import Mock

    task = Mock()
    task.task_id = "task_0"
    # Create actual task directory for tests
    task_dir = tmp_path / "task_0"
    task_dir.mkdir()
    task.task_dir = str(task_dir)
    task.get_evaluation_prompt.return_value = "Solve problem 0"
    task.get_instructions.return_value = "Instructions for task 0"
    task.evaluate_response_sandboxed.return_value = 0.8
    return task


@pytest.fixture
def mock_acdc_tasks(tmp_path):
    """Create multiple mock AC/DC task instances."""
    from unittest.mock import Mock
    from tasks.acdc_task import ACDCTask
    import os

    tasks = []
    for i in range(3):
        # Use spec=ACDCTask to make isinstance() checks work
        task = Mock(spec=ACDCTask)
        task.task_id = f"task_{i}"
        # Create actual task directory for tests
        task_dir = tmp_path / f"task_{i}"
        task_dir.mkdir(exist_ok=True)
        task.task_dir = str(task_dir)
        task.get_evaluation_prompt.return_value = f"Solve problem {i}"
        task.get_instructions.return_value = f"Instructions for task {i}"
        task.evaluate_response_sandboxed.return_value = 0.7 + i * 0.05
        tasks.append(task)
    return tasks


@pytest.fixture
def mock_standard_task():
    """Mock standard (non-AC/DC) task."""
    from unittest.mock import Mock
    from datatypes import TaskMetric
    from tasks.base import BaseTask

    # Use spec=BaseTask to make isinstance() checks work
    task = Mock(spec=BaseTask)
    task.task_name = "standard_task_0"
    task.bc_num_dims = 2
    task.bc_grid_sizes = [5, 5]
    task.get_q_and_bc.return_value = TaskMetric(
        quality=0.75,
        bc_ids=(2, 3),
        example_results={}
    )
    return task


@pytest.fixture
def mock_merge_result():
    """Mock ACDCMergeResult."""
    from datatypes import ACDCMergeResult, TaskMetric, ACDCTaskEvalDetail

    return ACDCMergeResult(
        save_path="/tmp/merged_model",
        task_metrics={"task_0": TaskMetric(quality=0.8, bc_ids=(1, 2), example_results={})},
        acdc_skill_vector={"task_0": 0.8, "task_1": 0.7},
        avg_acdc_quality=0.75,
        acdc_eval_details=[
            ACDCTaskEvalDetail(
                task_id="task_0",
                instructions="Test instructions",
                raw_output="Test output",
                score=0.8
            )
        ],
        is_gibberish=False
    )


@pytest.fixture
def worker_task_info_acdc(tmp_path):
    """Task info in AC/DC format (list of paths)."""
    paths = []
    for i in range(3):
        task_dir = tmp_path / f"task_{i}"
        task_dir.mkdir()
        paths.append(str(task_dir))
    return paths


@pytest.fixture
def worker_task_info_standard():
    """Task info in standard format (dict of configs)."""
    from omegaconf import OmegaConf

    return {
        "task_0": OmegaConf.create({"_target_": "tasks.SomeTask", "param": "value"}),
        "task_1": OmegaConf.create({"_target_": "tasks.AnotherTask", "param": "value"}),
    }


# ============================================================================
# Utils/Helpers Test Fixtures
# ============================================================================


@pytest.fixture
def mock_hf_state_dict_qwen_simple():
    """Mock Hugging Face state dict for Qwen model (simple, no biases)."""
    import torch

    return {
        "model.embed_tokens.weight": torch.randn(1000, 512),
        "model.norm.weight": torch.randn(512),
        "lm_head.weight": torch.randn(1000, 512),
        # Layer 0 - attention
        "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(512, 512),
        # Layer 0 - MLP
        "model.layers.0.mlp.gate_proj.weight": torch.randn(2048, 512),
        "model.layers.0.mlp.up_proj.weight": torch.randn(2048, 512),
        "model.layers.0.mlp.down_proj.weight": torch.randn(512, 2048),
        # Layer 0 - layernorms
        "model.layers.0.input_layernorm.weight": torch.randn(512),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(512),
    }


@pytest.fixture
def mock_hf_state_dict_qwen_with_bias():
    """Mock Qwen state dict with bias terms."""
    import torch

    state_dict = {
        "model.embed_tokens.weight": torch.randn(1000, 512),
        "model.norm.weight": torch.randn(512),
        "lm_head.weight": torch.randn(1000, 512),
        # Layer 0 - attention with biases
        "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(512),
        "model.layers.0.self_attn.k_proj.bias": torch.randn(512),
        "model.layers.0.self_attn.v_proj.bias": torch.randn(512),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(512, 512),
        # Layer 0 - MLP
        "model.layers.0.mlp.gate_proj.weight": torch.randn(2048, 512),
        "model.layers.0.mlp.up_proj.weight": torch.randn(2048, 512),
        "model.layers.0.mlp.down_proj.weight": torch.randn(512, 2048),
        # Layer 0 - layernorms
        "model.layers.0.input_layernorm.weight": torch.randn(512),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(512),
    }
    return state_dict


@pytest.fixture
def mock_hf_state_dict_qwen_no_lm_head():
    """Mock Qwen state dict without lm_head (for word embedding tying test)."""
    import torch

    return {
        "model.embed_tokens.weight": torch.randn(1000, 512),
        "model.norm.weight": torch.randn(512),
        # No lm_head.weight - should be copied from embed_tokens
        # Layer 0 - attention
        "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(512, 512),
        # Layer 0 - MLP
        "model.layers.0.mlp.gate_proj.weight": torch.randn(2048, 512),
        "model.layers.0.mlp.up_proj.weight": torch.randn(2048, 512),
        "model.layers.0.mlp.down_proj.weight": torch.randn(512, 2048),
        # Layer 0 - layernorms
        "model.layers.0.input_layernorm.weight": torch.randn(512),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(512),
    }


@pytest.fixture
def mock_hf_state_dict_llama():
    """Mock Llama state dict (no biases)."""
    import torch

    return {
        "model.embed_tokens.weight": torch.randn(1000, 512),
        "model.norm.weight": torch.randn(512),
        "lm_head.weight": torch.randn(1000, 512),
        # Layer 0 - attention (no biases for Llama)
        "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(512, 512),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(512, 512),
        # Layer 0 - MLP
        "model.layers.0.mlp.gate_proj.weight": torch.randn(2048, 512),
        "model.layers.0.mlp.up_proj.weight": torch.randn(2048, 512),
        "model.layers.0.mlp.down_proj.weight": torch.randn(512, 2048),
        # Layer 0 - layernorms
        "model.layers.0.input_layernorm.weight": torch.randn(512),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(512),
    }


@pytest.fixture
def mock_vllm_for_weight_loading():
    """Mock vLLM model for testing weight loading functions."""
    from unittest.mock import Mock
    import torch

    mock_llm = Mock()
    mock_model = Mock()

    # Mock model config
    mock_model.config.num_hidden_layers = 2

    # Dictionary to store parameters
    mock_params = {}

    def get_param(name):
        """Create mock parameters on-demand with appropriate shapes."""
        if name not in mock_params:
            # Create parameter with torch.nn.Parameter
            if "embed_tokens" in name or "lm_head" in name:
                param = torch.nn.Parameter(torch.randn(1000, 512))
            elif "qkv_proj.weight" in name:
                param = torch.nn.Parameter(torch.randn(512 * 3, 512))
            elif "qkv_proj.bias" in name:
                param = torch.nn.Parameter(torch.randn(512 * 3))
            elif "gate_up_proj" in name:
                param = torch.nn.Parameter(torch.randn(2048 * 2, 512))
            elif "down_proj" in name:
                param = torch.nn.Parameter(torch.randn(512, 2048))
            elif "norm" in name:
                param = torch.nn.Parameter(torch.randn(512))
            else:
                param = torch.nn.Parameter(torch.randn(512, 512))

            # Add mock copy_ method
            param.copy_ = Mock(return_value=param)
            # Note: device and dtype are read-only properties on torch tensors
            # The tensor is already on CPU with default dtype from torch.randn()

            mock_params[name] = param

        return mock_params[name]

    mock_model.get_parameter = Mock(side_effect=get_param)

    # Wire up the nested structure
    mock_llm.llm_engine.model_executor.driver_worker.worker.model_runner.model = (
        mock_model
    )

    # Mock apply_model
    mock_llm.apply_model = Mock()

    return mock_llm, mock_model


@pytest.fixture
def mock_hf_params_full():
    """Full HF parameters for 2-layer model."""
    import torch

    params = {
        "model.embed_tokens.weight": torch.randn(1000, 512),
        "model.norm.weight": torch.randn(512),
        "lm_head.weight": torch.randn(1000, 512),
    }

    # Add parameters for 2 layers
    for i in range(2):
        params.update(
            {
                f"model.layers.{i}.self_attn.q_proj.weight": torch.randn(512, 512),
                f"model.layers.{i}.self_attn.k_proj.weight": torch.randn(512, 512),
                f"model.layers.{i}.self_attn.v_proj.weight": torch.randn(512, 512),
                f"model.layers.{i}.self_attn.o_proj.weight": torch.randn(512, 512),
                f"model.layers.{i}.mlp.gate_proj.weight": torch.randn(2048, 512),
                f"model.layers.{i}.mlp.up_proj.weight": torch.randn(2048, 512),
                f"model.layers.{i}.mlp.down_proj.weight": torch.randn(512, 2048),
                f"model.layers.{i}.input_layernorm.weight": torch.randn(512),
                f"model.layers.{i}.post_attention_layernorm.weight": torch.randn(512),
            }
        )

    return params


@pytest.fixture
def mock_qwen_params_with_bias():
    """Qwen HF parameters with bias terms."""
    import torch

    params = {
        "model.embed_tokens.weight": torch.randn(1000, 512),
        "model.norm.weight": torch.randn(512),
    }

    # Add parameters for 2 layers with biases
    for i in range(2):
        params.update(
            {
                f"model.layers.{i}.self_attn.q_proj.weight": torch.randn(512, 512),
                f"model.layers.{i}.self_attn.k_proj.weight": torch.randn(512, 512),
                f"model.layers.{i}.self_attn.v_proj.weight": torch.randn(512, 512),
                f"model.layers.{i}.self_attn.q_proj.bias": torch.randn(512),
                f"model.layers.{i}.self_attn.k_proj.bias": torch.randn(512),
                f"model.layers.{i}.self_attn.v_proj.bias": torch.randn(512),
                f"model.layers.{i}.self_attn.o_proj.weight": torch.randn(512, 512),
                f"model.layers.{i}.mlp.gate_proj.weight": torch.randn(2048, 512),
                f"model.layers.{i}.mlp.up_proj.weight": torch.randn(2048, 512),
                f"model.layers.{i}.mlp.down_proj.weight": torch.randn(512, 2048),
                f"model.layers.{i}.input_layernorm.weight": torch.randn(512),
                f"model.layers.{i}.post_attention_layernorm.weight": torch.randn(512),
            }
        )

    return params


@pytest.fixture
def mock_archive_data():
    """Mock archive data for save/load tests."""
    from dataclasses import dataclass, asdict

    @dataclass
    class MockArchive:
        model_path: str
        fitness: float

    return {
        "task_0": {
            (0, 1): MockArchive("/models/gen_1_ind_0", 0.85),
            (1, 2): MockArchive("/models/gen_1_ind_1", 0.90),
        },
        "task_1": {
            (2, 3): MockArchive("/models/gen_2_ind_0", 0.75),
        },
    }


# ============================================================================
# Task Generation Fixtures
# ============================================================================


@pytest.fixture
def task_generation_config():
    """Minimal configuration for task generation testing."""
    return OmegaConf.create({
        "seed": 42,
        "acdc": {
            "seed_tasks_dir": "/tmp/seed_tasks",
            "initial_pool_size": 10,
            "scientist_model": "vllm/test-model",
            "scientist_temperature": 0.7,
            "num_initialization_workers": 4,
            "max_total_initialization_attempts": 30,
            "num_sandbox_workers": 4,
            "vllm_enabled": True,
            "vllm": {
                "base_url": "http://localhost:8000/v1",
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 1.0,
                "timeout": 300,
            },
        },
        "dns": {
            "population_size": 10,
            "acdc_skill_threshold": 0.5,
            "num_eval_tasks": 5,
            "num_train_tasks": 10,
        },
        "task_generation": {
            "max_reflections": 2,
            "novel_prompt_probability": 0.2,
            "do_similarity_search": False,
            "use_init_queue_novelty_filtering": False,
            "grow_seed_pool_with_micro_batches": False,
            "micro_batch_size": 10,
            "experimental_conditional_parent_replacement": False,
            "task_creation_prompt_name": "task_creation_system_msg",
            "vector_db": {
                "embedding_model_name": "all-MiniLM-L6-v2",
                "embedding_vllm_url": "http://localhost:8010/v1",
                "max_seq_length": 1024,
                "task_representation_vector_db": "metadata",
            },
        },
        "print_llm_debug": False,
    })


@pytest.fixture
def task_generation_config_with_vectordb(task_generation_config):
    """Configuration with vector DB enabled."""
    config = OmegaConf.to_container(task_generation_config, resolve=True)
    config["task_generation"]["do_similarity_search"] = True
    return OmegaConf.create(config)


@pytest.fixture
def mock_seed_task_dir(tmp_path):
    """Create a mock seed task directory with valid task files."""
    seed_dir = tmp_path / "seed_tasks" / "task_0"
    seed_dir.mkdir(parents=True)

    # Create task.py
    task_py = seed_dir / "task.py"
    task_py.write_text("""
class Task:
    def __init__(self):
        self.name = "test_task"

    def evaluate(self, response):
        return 1.0 if "correct" in response.lower() else 0.0
""")

    # Create task.json
    task_json = seed_dir / "task.json"
    task_json.write_text(json.dumps({
        "name_of_task": "Test Task",
        "description_of_task": "A test task for unit testing",
        "capability_being_measured": "Basic reasoning",
        "estimated_human_difficulty": "3",
        "example_instruction": "Solve this problem",
        "done": "True"
    }, indent=2))

    return str(tmp_path / "seed_tasks")


@pytest.fixture
def mock_generated_tasks_dir(tmp_path):
    """Create mock generated tasks directory."""
    gen_dir = tmp_path / "generated_tasks"
    gen_dir.mkdir(parents=True)
    return str(gen_dir)


@pytest.fixture
def mock_vector_db_dir(tmp_path):
    """Create mock vector DB directory."""
    vdb_dir = tmp_path / "vector_db"
    vdb_dir.mkdir(parents=True)
    return str(vdb_dir)


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for scientist."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock()
    return client


@pytest.fixture
def mock_scientist_response():
    """Mock successful scientist response."""
    return json.dumps({
        "name_of_task": "Generated Task",
        "description_of_task": "An AI-generated task",
        "capability_being_measured": "Advanced reasoning",
        "estimated_human_difficulty": "4",
        "task_family": "class GeneratedTask:\\n    pass",
        "example_instruction": "Complete the following task",
        "done": "True"
    })


@pytest.fixture
def mock_vector_db():
    """Mock SimpleVectorDB for testing."""
    from unittest.mock import Mock
    vdb = Mock()
    vdb.add_task = Mock()
    vdb.query = Mock(return_value=[])
    vdb.get_all_tasks = Mock(return_value=[])
    vdb.save = Mock()
    vdb.load = Mock()
    return vdb


@pytest.fixture
def sample_task_pool_tasks(tmp_path):
    """Create sample task directories for task pool testing."""
    tasks_dir = tmp_path / "generated_tasks"
    tasks_dir.mkdir(parents=True)

    task_paths = []
    for i in range(5):
        task_dir = tasks_dir / f"task_{i}_sample"
        task_dir.mkdir()

        # Create task.py
        (task_dir / "task.py").write_text(f"# Task {i}")

        # Create task.json
        (task_dir / "task.json").write_text(json.dumps({
            "name_of_task": f"Task {i}",
            "description_of_task": f"Description {i}",
            "capability_being_measured": f"Capability {i}",
            "estimated_human_difficulty": str((i % 5) + 1),
            "example_instruction": f"Instruction {i}",
            "done": "True"
        }))

        task_paths.append(str(task_dir))

    return task_paths


@pytest.fixture
def mock_acdc_archive_data():
    """Mock ACDCArchiveData for task adaptation tests."""
    from datatypes import ACDCSolution

    archive = []
    for i in range(3):
        solution = Mock(spec=ACDCSolution)
        solution.model_path = f"/models/gen_1_ind_{i}"
        solution.fitness = 0.7 + i * 0.1
        solution.acdc_skill_vector = {
            "task_0": 0.8,
            "task_1": 0.6,
            "task_2": 0.9,
        }
        solution.avg_acdc_quality = 0.75 + i * 0.05
        archive.append(solution)

    return {
        "dns_archive": archive,
        "dirs": {
            "output_dir": "/tmp/output",
            "model_dir": "/tmp/models",
        }
    }


# ============================================================================
# Fixtures for Metrics Testing
# ============================================================================

@pytest.fixture
def sample_acdc_skill_vectors():
    """Sample AC/DC skill vectors with task_id -> score mapping."""
    return [
        {"task_001": 0.9, "task_002": 0.4, "task_003": 0.8, "task_004": 0.6},
        {"task_001": 0.3, "task_002": 0.7, "task_003": 0.6, "task_004": 0.9},
        {"task_001": 0.8, "task_002": 0.2, "task_003": 0.9, "task_004": 0.5},
        {"task_001": 0.5, "task_002": 0.8, "task_003": 0.3, "task_004": 0.7},
        {"task_001": 0.7, "task_002": 0.5, "task_003": 0.7, "task_004": 0.4},
    ]


@pytest.fixture
def mock_acdc_solutions(sample_acdc_skill_vectors):
    """Mock ACDCSolution objects for metrics testing."""
    solutions = []
    for idx, skill_vec in enumerate(sample_acdc_skill_vectors):
        solution = Mock()
        solution.model_path = f"/models/gen_1_ind_{idx}"
        solution.acdc_skill_vector = skill_vec
        solution.fitness = sum(skill_vec.values()) / len(skill_vec)
        solution.validation_quality = None
        solution.is_gibberish = False
        solutions.append(solution)
    return solutions


@pytest.fixture
def mock_acdc_archive_data_for_metrics(mock_acdc_solutions):
    """Mock archive data structure for AC/DC metrics testing."""
    return {
        "dns_archive": mock_acdc_solutions,
        "dirs": {
            "output_dir": "/tmp/output",
            "model_dir": "/tmp/models",
            "archive_dir": "/tmp/archives",
        }
    }


@pytest.fixture
def sample_combined_archive():
    """Sample combined archive for coverage analysis."""
    return {
        "model_1": {
            "task_1_ex_1": True,
            "task_1_ex_2": False,
            "task_2_ex_1": True,
            "task_3_ex_1": False,
        },
        "model_2": {
            "task_1_ex_1": False,
            "task_1_ex_2": True,
            "task_2_ex_1": True,
            "task_3_ex_1": True,
        },
        "model_3": {
            "task_1_ex_1": True,
            "task_1_ex_2": True,
            "task_2_ex_1": False,
            "task_3_ex_1": False,
        },
    }


@pytest.fixture
def mock_archive_map():
    """Mock archive_map for QD mode testing."""
    from unittest.mock import Mock

    # Structure: Dict[task_name, Dict[Tuple[int], ArchiveData]]
    archive_map = {}
    for task_idx in range(3):
        task_name = f"task_{task_idx}"
        archive_map[task_name] = {}
        for bc_idx in range(5):
            bc_tuple = (bc_idx,)
            archive_data = Mock()
            archive_data.model_path = f"/models/task{task_idx}_model{bc_idx}"
            archive_data.overall_fitness = 0.5 + bc_idx * 0.1
            archive_data.quality = 0.6 + bc_idx * 0.05
            archive_data.validation_quality = 0.55 + bc_idx * 0.08
            archive_data.skill_vector = [True] * (bc_idx + 1) + [False] * (5 - bc_idx - 1)
            archive_map[task_name][bc_tuple] = archive_data
    return archive_map


@pytest.fixture
def mock_tasks_for_metrics():
    """Mock BaseTask objects with get_example_ids method."""
    from unittest.mock import Mock

    tasks = []
    for task_idx in range(3):
        task = Mock()
        task.task_name = f"task_{task_idx}"
        # Return 10 example IDs per task
        task.get_example_ids = Mock(return_value=[f"ex_{i}" for i in range(10)])
        tasks.append(task)
    return tasks


# ============================================================================
# Coverage Evaluation Fixtures
# ============================================================================


@pytest.fixture
def mock_lm_harness_results(tmp_path):
    """Mock lm-harness evaluation results directory structure."""
    eval_dir = tmp_path / "eval" / "lm_harness"
    eval_dir.mkdir(parents=True)

    # Create mock model result directories
    model_names = ["gen_0_ind_0", "gen_0_ind_1", "gen_1_ind_0"]

    for model_name in model_names:
        model_dir = eval_dir / model_name
        model_dir.mkdir()

        # Create results JSON
        results = {
            "results": {
                "gsm8k_llama": {
                    "exact_match,flexible_extract": 0.75,
                    "exact_match,strict_match": 0.70
                },
                "mmlu_cot_llama": {
                    "exact_match,strict_match": 0.80
                }
            },
            "groups": {},
            "group_subtasks": {}
        }

        with open(model_dir / "results_gsm8k.json", "w") as f:
            json.dump(results, f)

    return str(eval_dir)


@pytest.fixture
def mock_benchmark_samples_gsm8k(tmp_path):
    """Mock GSM8K benchmark sample files."""
    eval_dir = tmp_path / "eval_output"
    eval_dir.mkdir()

    samples = []
    for i in range(5):
        sample = {
            "doc_id": i,
            "doc": {
                "question": f"Question {i}?",
                "answer": f"Answer {i}"
            },
            "target": f"Answer {i}",
            "resps": [[f"Response {i}"]],
            "filtered_resps": [f"Answer {i}" if i < 3 else f"Wrong {i}"],
            "filter": "flexible_extract",
            "exact_match": 1.0 if i < 3 else 0.0
        }
        samples.append(sample)

    # Write to JSONL
    with open(eval_dir / "samples_gsm8k_llama_test.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return str(eval_dir)


@pytest.fixture
def mock_archive_with_generations(tmp_path):
    """Mock multi-generation archive structure."""
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()

    # Create archives for multiple generations
    for gen in [5, 10, 15, 20]:
        archive = []
        for ind in range(5):
            solution = {
                "model_path": f"{tmp_path}/models/gen_{gen}_ind_{ind}",
                "fitness": 0.6 + (ind * 0.05),
                "acdc_skill_vector": {
                    "task_0": 0.7 + (ind * 0.05),
                    "task_1": 0.6 + (ind * 0.04),
                    "task_2": 0.8 - (ind * 0.03)
                }
            }
            archive.append(solution)

        archive_file = archive_dir / f"gen{gen}_dns_archive.json"
        with open(archive_file, "w") as f:
            json.dump(archive, f)

    return str(archive_dir)


@pytest.fixture
def mock_global_skill_vectors(tmp_path):
    """Mock global skill vector directory."""
    skill_vector_dir = tmp_path / "global_skill_vectors"
    skill_vector_dir.mkdir()

    # Create skill vectors for multiple models
    for gen in range(3):
        for ind in range(3):
            model_id = f"gen_{gen}_ind_{ind}"
            skill_vector = {
                "task_0": 0.7 + (gen * 0.1),
                "task_1": 0.6 + (ind * 0.1),
                "task_2": 0.8 - (gen * 0.05)
            }

            skill_file = skill_vector_dir / f"{model_id}_skill_vector.json"
            with open(skill_file, "w") as f:
                json.dump(skill_vector, f)

    return str(skill_vector_dir)


@pytest.fixture
def mock_selection_methods_config():
    """Mock selection methods configuration."""
    return [
        {
            "func_name": "get_top_n_models_based_on_fitness_across_entire_archive",
            "kwargs": {
                "relevant_gens": [5, 10, 15, 20]
            },
            "save_name": "fitness_across_archive"
        },
        {
            "func_name": "get_top_n_models_randomly",
            "kwargs": {
                "seed": 42,
                "relevant_gens": [5, 10, 15, 20]
            },
            "save_name": "random_baseline"
        }
    ]


@pytest.fixture
def mock_baseline_models():
    """Mock baseline model configuration."""
    return {
        "big_model": "Meta-Llama-3-70B-Instruct",
        "control": "Meta-Llama-3-8B-Instruct",
        "expert_1": "Expert-Model-1",
        "expert_2": "Expert-Model-2",
        "expert_3": "Expert-Model-3"
    }


@pytest.fixture
def mock_experiment_dir_with_eval(tmp_path):
    """Full mock experiment directory with evaluation results."""
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    # Create subdirectories
    models_dir = exp_dir / "models"
    archives_dir = exp_dir / "archives"
    eval_dir = exp_dir / "eval" / "lm_harness"
    global_skill_dir = exp_dir / "global_skill_vectors"

    models_dir.mkdir(parents=True)
    archives_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    global_skill_dir.mkdir(parents=True)

    # Create model directories and their eval results
    for gen in [0, 1, 2]:
        for ind in range(3):
            model_name = f"gen_{gen}_ind_{ind}"
            model_dir = models_dir / model_name
            model_dir.mkdir()

            # Create eval results
            eval_model_dir = eval_dir / model_name
            eval_model_dir.mkdir()

            results = {
                "results": {
                    "gsm8k_llama": {"exact_match,flexible_extract": 0.7 + (ind * 0.05)},
                    "mmlu_cot_llama": {"exact_match,strict_match": 0.75 + (ind * 0.03)}
                }
            }

            with open(eval_model_dir / "results_test.json", "w") as f:
                json.dump(results, f)

            # Create global skill vector
            skill_vector = {
                "task_0": 0.7 + (gen * 0.05),
                "task_1": 0.6 + (ind * 0.1)
            }
            with open(global_skill_dir / f"{model_name}_skill_vector.json", "w") as f:
                json.dump(skill_vector, f)

    # Create archives
    for gen in [0, 1, 2]:
        archive = []
        for ind in range(3):
            solution = {
                "model_path": f"{models_dir}/gen_{gen}_ind_{ind}",
                "fitness": 0.65 + (ind * 0.05),
                "acdc_skill_vector": {
                    "task_0": 0.7,
                    "task_1": 0.6
                }
            }
            archive.append(solution)

        with open(archives_dir / f"gen{gen}_dns_archive.json", "w") as f:
            json.dump(archive, f)

    return exp_dir
