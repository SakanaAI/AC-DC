# ACDC Test Suite

Tests for the AC/DC (Assessment Coevolving with Diverse Capabilities) framework.

## Directory Structure

```
tests/
├── unit/
│   ├── test_mutation/
│   │   └── test_svd_model_weights_gaussian_mutator.py
│   ├── test_crossover/
│   │   └── test_model_linear.py
│   ├── test_dns/
│   │   ├── test_dns_utils.py
│   │   ├── test_novelty_metrics.py
│   │   ├── test_archive_update.py
│   │   └── test_metrics.py
│   ├── test_tasks/
│   │   ├── test_simple_vectordb.py
│   │   ├── test_vectordb_search.py
│   │   ├── test_task_generation.py
│   │   └── test_docker_sandbox.py
│   ├── test_main_ac_dc/
│   │   ├── test_cleanup_and_metrics.py
│   │   ├── test_helper_functions.py
│   │   ├── test_merge_evaluation.py
│   │   ├── test_optimizer_initialization.py
│   │   ├── test_population_initialization.py
│   │   └── test_task_pool_adaptation.py
│   ├── test_utils/
│   │   └── test_helpers.py
│   └── test_workers/
│       └── test_ac_dc_worker.py
├── integration/
│   ├── test_full_pipeline.py
│   └── test_global_skill_vector.py
├── evaluation/
│   └── test_taskforce_selection.py
└── conftest.py
```

## Operators Tested

### SVDModelWeightsGaussianMutator (`mutation/svd_model_weights_gaussian_mutator.py`)
Computes on-the-fly SVD of model weights, perturbs top-k singular values with Gaussian noise, and reconstructs mutated weights.

- `mutation_rate`: magnitude of singular value perturbations
- `keep_rank`: limits mutation to top-k singular values (e.g., 512)
- `include_bias_mutation`: whether to mutate bias terms (default: False)

Verified: skips normalization layers and 1D tensors, clamps singular values ≥ 0, preserves shapes/dtypes (bfloat16), GPU ops, seed reproducibility.

### ModelwiseLinearMerge (`crossover/model_linear.py`)
Computes task vectors (fine-tuned − base), samples weights from Normal(1 + mean, std), merges as normalized weighted sum, adds to base.

- `std`: standard deviation for weight sampling (default: 0.01)
- `merge_params[0]`: mean offset; `merge_params[1]`: std

Verified: normalized merging (weights sum to 1), N ≥ 2 parents, seed reproducibility, shape/dtype preservation.

## Test Markers

- `unit` / `integration` / `evaluation`: test category
- `mutation` / `crossover`: operator-specific tests
- `slow`: full pipeline tests (~15-30 min)
- `requires_gpu`: SVD mutation tests needing CUDA
- `requires_model`: tests loading actual model checkpoints

## Quick Commands

```bash
# Fast tests only (recommended for development)
pytest -m "not slow and not requires_gpu" tests/

# Unit tests
pytest tests/unit/

# Mutation / crossover only
pytest tests/unit/test_mutation/
pytest tests/unit/test_crossover/

# DNS unit tests
pytest tests/unit/test_dns/ -v

# Task unit tests (vector DB, task generation)
pytest tests/unit/test_tasks/ -v

# Docker sandbox tests (requires Docker daemon + acdc-sandbox:latest image)
pytest tests/unit/test_tasks/test_docker_sandbox.py --run-docker -v

# Integration tests (excluding slow full pipeline)
pytest tests/integration/ -m "not slow"

# Full pipeline integration test (~15-30 min, requires GPU)
pytest tests/integration/test_full_pipeline.py -v -s

# With coverage
pytest --cov=mutation --cov=crossover --cov=evaluation tests/unit/ tests/evaluation/
```

## Shared Fixtures (`conftest.py`)

**Model weights:** `tiny_weight_dict`, `tiny_weight_dict_with_norms`, `base_model_weights`, `finetuned_model_weights_1/2`, `task_vector_1/2`

**Skill vectors:** `sample_skill_vectors` (binary), `sample_skill_vectors_continuous`

**Archive:** `sample_archive_data`

**Directories:** `temp_experiment_dir`, `temp_model_dir`, `temp_archive_dir`

**Utilities:** `random_seed` (42), `np_random`, `torch_device` (CPU)
