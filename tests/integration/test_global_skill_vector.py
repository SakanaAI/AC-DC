"""
Integration tests for global skill vector evaluation.

Tests the end-to-end pipeline for evaluating models against global task pool
and generating skill vectors.
"""

import pytest
import json
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from evaluation.utils import (
    get_relevant_archive_models_and_skill_vectors,
    get_top_n_models_based_on_global_skill_vector,
    is_seed_model,
)


class TestGlobalSkillVectorPipeline:
    """Integration tests for global skill vector evaluation pipeline."""

    def test_skill_vector_directory_structure(self, temp_experiment_dir):
        """Test that skill vector directory structure is correct."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        assert skill_vector_dir.exists()
        assert skill_vector_dir.is_dir()

    def test_create_skill_vector_files(self, temp_experiment_dir):
        """Test creating skill vector files for models."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        # Create mock skill vectors
        model_ids = ["gen_0_ind_0", "gen_1_ind_5", "gen_2_ind_12"]

        for model_id in model_ids:
            skill_vector = {
                "task_0": 0.85,
                "task_1": 0.92,
                "task_2": 0.67,
                "task_3": 0.41,
            }

            skill_vector_file = (
                skill_vector_dir / f"{model_id}_skill_vector.json"
            )
            with open(skill_vector_file, "w") as f:
                json.dump(skill_vector, f)

        # Verify files were created
        skill_vector_files = list(skill_vector_dir.glob("*_skill_vector.json"))
        assert len(skill_vector_files) == 3

    def test_load_skill_vectors_from_directory(
        self, temp_experiment_dir, temp_model_dir
    ):
        """Test loading skill vectors from directory."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        # Create skill vector files
        skill_vectors_data = {
            "gen_0_ind_0": {"task_0": 0.9, "task_1": 0.8, "task_2": 0.5},
            "gen_1_ind_5": {"task_0": 0.7, "task_1": 0.9, "task_2": 0.8},
            "gen_2_ind_12": {"task_0": 0.6, "task_1": 0.7, "task_2": 0.7},
        }

        for model_id, skill_vector in skill_vectors_data.items():
            skill_vector_file = (
                skill_vector_dir / f"{model_id}_skill_vector.json"
            )
            with open(skill_vector_file, "w") as f:
                json.dump(skill_vector, f)

            # Also create model directories
            model_dir = temp_model_dir / model_id
            model_dir.mkdir(exist_ok=True)

        # Load using the utility function
        archive = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_vector_dir), models_dir=str(temp_model_dir)
        )

        assert len(archive) == 3

        # Check structure
        for model_data in archive:
            assert "model_path" in model_data
            assert "acdc_skill_vector" in model_data
            assert "fitness" in model_data

    def test_fitness_calculation_from_skill_vector(
        self, temp_experiment_dir, temp_model_dir
    ):
        """Test that fitness is calculated correctly from skill vectors."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        # Model with perfect skill vector
        skill_vector_perfect = {"task_0": 1.0, "task_1": 1.0, "task_2": 1.0}
        skill_vector_file = skill_vector_dir / "gen_0_ind_0_skill_vector.json"
        with open(skill_vector_file, "w") as f:
            json.dump(skill_vector_perfect, f)

        (temp_model_dir / "gen_0_ind_0").mkdir(exist_ok=True)

        archive = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_vector_dir), models_dir=str(temp_model_dir)
        )

        # Fitness should be average of skill vector values
        expected_fitness = (1.0 + 1.0 + 1.0) / 3
        assert archive[0]["fitness"] == pytest.approx(expected_fitness)

    def test_filter_by_generation(self, temp_experiment_dir, temp_model_dir):
        """Test filtering skill vectors by generation number."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        # Create skill vectors for different generations
        for gen in range(5):
            skill_vector = {"task_0": 0.8, "task_1": 0.7}
            model_id = f"gen_{gen}_ind_0"

            skill_vector_file = (
                skill_vector_dir / f"{model_id}_skill_vector.json"
            )
            with open(skill_vector_file, "w") as f:
                json.dump(skill_vector, f)

            (temp_model_dir / model_id).mkdir(exist_ok=True)

        # Load with max_gen filter
        archive = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_vector_dir),
            models_dir=str(temp_model_dir),
            max_gen=2,
        )

        # Should only get models from gen 0, 1, 2
        assert len(archive) == 3

    def test_is_seed_model_detection(self):
        """Test detection of seed models vs evolved models."""
        # Seed models (named after base models)
        assert is_seed_model("gen_0_ind_Qwen2.5-7B") == True
        assert is_seed_model("gen_0_ind_Qwen2.5-7B-Instruct") == True
        assert is_seed_model("/path/to/gen_0_ind_Llama-3-8B") == True

        # Evolved models (numbered)
        assert is_seed_model("gen_0_ind_0") == False
        assert is_seed_model("gen_1_ind_5") == False
        assert is_seed_model("/path/to/gen_5_ind_123") == False

    def test_exclude_seed_models(self, temp_experiment_dir, temp_model_dir):
        """Test excluding seed models from archive."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        # Create skill vectors for seed and evolved models
        models = {
            "gen_0_ind_Qwen2.5-7B": {"task_0": 0.9, "task_1": 0.8},  # Seed
            "gen_0_ind_0": {"task_0": 0.7, "task_1": 0.6},  # Evolved
            "gen_1_ind_5": {"task_0": 0.8, "task_1": 0.7},  # Evolved
        }

        for model_id, skill_vector in models.items():
            skill_vector_file = (
                skill_vector_dir / f"{model_id}_skill_vector.json"
            )
            with open(skill_vector_file, "w") as f:
                json.dump(skill_vector, f)

            (temp_model_dir / model_id).mkdir(exist_ok=True)

        # Load with include_seed_models=None (exclude all seed models)
        archive = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_vector_dir),
            models_dir=str(temp_model_dir),
            include_seed_models=None,
        )

        # Should only get evolved models
        assert len(archive) == 2

        # Verify no seed models in archive
        for model_data in archive:
            model_name = model_data["model_path"].split("/")[-1]
            assert not is_seed_model(model_name)

    def test_include_specific_seed_models(
        self, temp_experiment_dir, temp_model_dir
    ):
        """Test including specific seed models with infinite fitness."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        # Create skill vectors
        models = {
            "gen_0_ind_Qwen2.5-7B": {"task_0": 0.9, "task_1": 0.8},  # Seed
            "gen_0_ind_0": {"task_0": 0.7, "task_1": 0.6},  # Evolved
        }

        for model_id, skill_vector in models.items():
            skill_vector_file = (
                skill_vector_dir / f"{model_id}_skill_vector.json"
            )
            with open(skill_vector_file, "w") as f:
                json.dump(skill_vector, f)

            (temp_model_dir / model_id).mkdir(exist_ok=True)

        # Include specific seed model
        archive = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_vector_dir),
            models_dir=str(temp_model_dir),
            include_seed_models={"gen_0_ind_Qwen2.5-7B"},
        )

        # Find the seed model
        seed_model = [m for m in archive if "Qwen2.5-7B" in m["model_path"]][0]

        # Should have infinite fitness
        import math

        assert math.isinf(seed_model["fitness"])


class TestGlobalSkillVectorSelection:
    """Tests for selecting models based on global skill vectors."""

    def test_get_top_n_models_coverage_method(
        self, temp_experiment_dir, temp_model_dir
    ):
        """Test selecting top N models by coverage with binary skill vectors."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        # Create binary skill vectors with different coverage
        # Models 2 and 3 together provide perfect coverage (all tasks covered)
        models = {
            "gen_0_ind_0": {
                "task_0": 1,
                "task_1": 0,
                "task_2": 0,
            },  # Covers only task_0
            "gen_0_ind_1": {
                "task_0": 0,
                "task_1": 1,
                "task_2": 0,
            },  # Covers only task_1
            "gen_0_ind_2": {
                "task_0": 0,
                "task_1": 0,
                "task_2": 1,
            },  # Covers only task_2
            "gen_0_ind_3": {
                "task_0": 1,
                "task_1": 1,
                "task_2": 0,
            },  # Covers task_0 and task_1
        }

        for model_id, skill_vector in models.items():
            skill_vector_file = (
                skill_vector_dir / f"{model_id}_skill_vector.json"
            )
            with open(skill_vector_file, "w") as f:
                json.dump(skill_vector, f)

            (temp_model_dir / model_id).mkdir(exist_ok=True)

        # Select top 2 models by coverage using greedy method
        selected = get_top_n_models_based_on_global_skill_vector(
            experiment_path=str(temp_experiment_dir),
            n=2,
            selection_method="coverage",
            coverage_optimization_method="greedy",
        )

        assert len(selected) == 2

        # Verify that models with perfect coverage were selected
        # gen_0_ind_3 (covers task_0, task_1) + gen_0_ind_2 (covers task_2) = perfect coverage
        selected_names = [Path(p).name for p in selected]
        assert (
            "gen_0_ind_2" in selected_names
        ), "gen_0_ind_2 should be selected for task_2 coverage"
        assert (
            "gen_0_ind_3" in selected_names
        ), "gen_0_ind_3 should be selected for task_0 and task_1 coverage"

    def test_get_top_n_models_fitness_method(
        self, temp_experiment_dir, temp_model_dir
    ):
        """Test selecting top N models by fitness."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        # Create skill vectors with different fitness
        models = {
            "gen_0_ind_0": {"task_0": 0.5, "task_1": 0.5},  # fitness=0.5
            "gen_0_ind_1": {"task_0": 0.9, "task_1": 0.9},  # fitness=0.9
            "gen_0_ind_2": {"task_0": 0.7, "task_1": 0.7},  # fitness=0.7
        }

        for model_id, skill_vector in models.items():
            skill_vector_file = (
                skill_vector_dir / f"{model_id}_skill_vector.json"
            )
            with open(skill_vector_file, "w") as f:
                json.dump(skill_vector, f)

            (temp_model_dir / model_id).mkdir(exist_ok=True)

        # Select top 2 models by fitness
        selected = get_top_n_models_based_on_global_skill_vector(
            experiment_path=str(temp_experiment_dir),
            n=2,
            selection_method="fitness",
        )

        assert len(selected) == 2

        # Verify highest fitness models were selected
        # gen_0_ind_1 (0.9) and gen_0_ind_2 (0.7) should be selected
        selected_names = [Path(p).name for p in selected]
        assert "gen_0_ind_1" in selected_names
        assert "gen_0_ind_2" in selected_names


class TestMaxGenTaskFiltering:
    """Tests for do_max_gen_task_filtering argument logic."""

    def test_task_filtering_without_max_gen(
        self, temp_experiment_dir, temp_model_dir
    ):
        """Test that without max_gen, all tasks in skill vectors are preserved."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        # Create skill vector with tasks from different generations
        models = {
            "gen_0_ind_0": {
                "task_gen0_0": 1,
                "task_gen0_1": 1,
                "task_gen1_0": 0,
                "task_gen2_0": 1,
            },
        }

        for model_id, skill_vector in models.items():
            skill_vector_file = (
                skill_vector_dir / f"{model_id}_skill_vector.json"
            )
            with open(skill_vector_file, "w") as f:
                json.dump(skill_vector, f)

            (temp_model_dir / model_id).mkdir(exist_ok=True)

        # Load without task filtering
        archive = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_vector_dir),
            models_dir=str(temp_model_dir),
            do_max_gen_task_filtering=False,
        )

        # All tasks should be present
        assert len(archive) == 1
        assert len(archive[0]["acdc_skill_vector"]) == 4
        assert "task_gen0_0" in archive[0]["acdc_skill_vector"]
        assert "task_gen2_0" in archive[0]["acdc_skill_vector"]

    def test_task_filtering_with_max_gen(
        self, temp_experiment_dir, temp_model_dir
    ):
        """Test that with max_gen and do_max_gen_task_filtering, only active tasks are kept."""
        # Create task pool directory structure
        task_pool_dir = temp_experiment_dir / "generated_tasks" / "pool"
        task_pool_dir.mkdir(parents=True, exist_ok=True)

        # Create active pool files for different generations
        # Gen 0: only task_gen0_0 and task_gen0_1
        active_pool_gen0 = [
            "/path/to/tasks/task_gen0_0",
            "/path/to/tasks/task_gen0_1",
        ]
        with open(task_pool_dir / "active_pool_gen_0.json", "w") as f:
            json.dump(active_pool_gen0, f)

        # Gen 1: adds task_gen1_0
        active_pool_gen1 = [
            "/path/to/tasks/task_gen0_0",
            "/path/to/tasks/task_gen0_1",
            "/path/to/tasks/task_gen1_0",
        ]
        with open(task_pool_dir / "active_pool_gen_1.json", "w") as f:
            json.dump(active_pool_gen1, f)

        # Gen 2: adds task_gen2_0
        active_pool_gen2 = [
            "/path/to/tasks/task_gen0_0",
            "/path/to/tasks/task_gen0_1",
            "/path/to/tasks/task_gen1_0",
            "/path/to/tasks/task_gen2_0",
        ]
        with open(task_pool_dir / "active_pool_gen_2.json", "w") as f:
            json.dump(active_pool_gen2, f)

        # Create skill vectors with all tasks
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"
        models = {
            "gen_0_ind_0": {
                "task_gen0_0": 1,
                "task_gen0_1": 1,
                "task_gen1_0": 0,
                "task_gen2_0": 1,
            },
        }

        for model_id, skill_vector in models.items():
            skill_vector_file = (
                skill_vector_dir / f"{model_id}_skill_vector.json"
            )
            with open(skill_vector_file, "w") as f:
                json.dump(skill_vector, f)

            (temp_model_dir / model_id).mkdir(exist_ok=True)

        # Load with task filtering up to gen 1
        archive = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_vector_dir),
            models_dir=str(temp_model_dir),
            max_gen=1,
            do_max_gen_task_filtering=True,
        )

        # Should only have tasks from gen 0 and 1
        assert len(archive) == 1
        skill_vector = archive[0]["acdc_skill_vector"]
        assert (
            len(skill_vector) == 3
        )  # Only task_gen0_0, task_gen0_1, task_gen1_0
        assert "task_gen0_0" in skill_vector
        assert "task_gen0_1" in skill_vector
        assert "task_gen1_0" in skill_vector
        assert "task_gen2_0" not in skill_vector

        # Verify fitness is recalculated with filtered tasks
        expected_fitness = (1 + 1 + 0) / 3  # Average of remaining tasks
        assert archive[0]["fitness"] == pytest.approx(expected_fitness)
