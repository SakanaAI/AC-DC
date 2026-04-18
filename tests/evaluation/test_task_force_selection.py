"""
Tests for coverage model selection algorithms.

Tests coverage-based and fitness-based selection methods, including
greedy and optimal algorithms.
"""

import pytest
import numpy as np
import json
from pathlib import Path

from evaluation.utils import (
    get_best_n_models_based_on_coverage,
    get_best_n_models_based_on_fitness,
    greedy_model_selection,
    optimal_model_selection,
    is_seed_model,
)


class TestGreedyModelSelection:
    """Tests for greedy coverage-based model selection."""

    def test_greedy_selection_basic(self, sample_skill_vectors):
        """Test basic greedy selection."""
        n = 2
        best_combination, best_coverage = greedy_model_selection(
            skill_vectors=sample_skill_vectors, n=n
        )

        assert len(best_combination) == n
        assert best_coverage > 0
        assert best_coverage <= len(
            sample_skill_vectors[0]
        )  # Can't cover more than total tasks

    def test_greedy_selection_coverage_increases(self, sample_skill_vectors):
        """Test that greedy selection incrementally increases coverage."""
        n = 3
        best_combination, best_coverage = greedy_model_selection(
            skill_vectors=sample_skill_vectors, n=n
        )

        # Verify coverage is non-decreasing as we add models
        coverages = []
        covered = set()

        for idx in best_combination:
            for task_idx, val in enumerate(sample_skill_vectors[idx]):
                if val == 1:
                    covered.add(task_idx)
            coverages.append(len(covered))

        # Each step should maintain or increase coverage
        for i in range(1, len(coverages)):
            assert (
                coverages[i] >= coverages[i - 1]
            ), "Coverage decreased when adding a model"

    def test_greedy_selection_single_model(self):
        """Test selecting single best model with deterministic behavior."""
        skill_vectors = [
            [1, 1, 0, 0],  # Model 0: covers 2 tasks
            [0, 1, 1, 0],  # Model 1: covers 2 tasks
            [0, 0, 1, 1],  # Model 2: covers 2 tasks
        ]

        # Run selection multiple times to ensure deterministic behavior
        results = []
        for _ in range(5):
            best_combination, best_coverage = greedy_model_selection(
                skill_vectors=skill_vectors, n=1
            )
            results.append((best_combination, best_coverage))

        # All runs should produce the same result
        assert len(results) == 5
        for combination, coverage in results:
            assert len(combination) == 1
            assert coverage == 2

        # Should consistently select model 0 (first in list order with max coverage)
        for combination, _ in results:
            assert combination[0] == 0, (
                "Greedy selection should deterministically pick index 0 "
                "when multiple models have equal coverage (list order sensitive)"
            )

    def test_greedy_selection_all_models(self, sample_skill_vectors):
        """Test selecting all models."""
        n = len(sample_skill_vectors)
        best_combination, best_coverage = greedy_model_selection(
            skill_vectors=sample_skill_vectors, n=n
        )

        assert len(best_combination) == n
        # Coverage should be maximum possible
        all_tasks_covered = set()
        for vec in sample_skill_vectors:
            for idx, val in enumerate(vec):
                if val == 1:
                    all_tasks_covered.add(idx)

        assert best_coverage == len(all_tasks_covered)

    def test_greedy_selection_optimal_case(self):
        """Test case where greedy finds optimal solution."""
        # Disjoint skill vectors - greedy should find optimal
        skill_vectors = [
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ]

        best_combination, best_coverage = greedy_model_selection(
            skill_vectors=skill_vectors, n=2
        )

        assert best_coverage == 4  # All tasks covered

    def test_greedy_selection_no_coverage(self):
        """Test when no model covers any tasks."""
        skill_vectors = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        best_combination, best_coverage = greedy_model_selection(
            skill_vectors=skill_vectors, n=2
        )

        assert best_coverage == 0

    def test_greedy_selection_full_coverage(self):
        """Test when all models cover all tasks."""
        skill_vectors = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]

        best_combination, best_coverage = greedy_model_selection(
            skill_vectors=skill_vectors, n=1
        )

        # Any single model covers everything
        assert best_coverage == 3

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_greedy_selection_various_n(self, sample_skill_vectors, n):
        """Test greedy selection with different values of n."""
        best_combination, best_coverage = greedy_model_selection(
            skill_vectors=sample_skill_vectors, n=n
        )

        assert len(best_combination) == n
        assert best_coverage > 0


class TestFitnessBasedSelection:
    """Tests for fitness-based model selection."""

    def test_fitness_selection_basic(self, temp_experiment_dir, temp_model_dir):
        """Test basic fitness-based selection."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        # Create models with different fitness
        models = {
            "gen_0_ind_0": {"task_0": 0.3, "task_1": 0.3},  # fitness=0.3
            "gen_0_ind_1": {"task_0": 0.9, "task_1": 0.9},  # fitness=0.9
            "gen_0_ind_2": {"task_0": 0.7, "task_1": 0.7},  # fitness=0.7
            "gen_0_ind_3": {"task_0": 0.5, "task_1": 0.5},  # fitness=0.5
        }

        for model_id, skill_vector in models.items():
            skill_vector_file = (
                skill_vector_dir / f"{model_id}_skill_vector.json"
            )
            with open(skill_vector_file, "w") as f:
                json.dump(skill_vector, f)

            (temp_model_dir / model_id).mkdir(exist_ok=True)

        # Select top 2 by fitness
        selected = get_best_n_models_based_on_fitness(
            archive_path=str(skill_vector_dir),
            n=2,
            models_dir=str(temp_model_dir),
        )

        assert len(selected) == 2

        # Should select models with highest fitness
        selected_names = [Path(p).name for p in selected]
        assert "gen_0_ind_1" in selected_names  # fitness=0.9
        assert "gen_0_ind_2" in selected_names  # fitness=0.7

    def test_fitness_selection_ordering(
        self, temp_experiment_dir, temp_model_dir
    ):
        """Test that fitness selection returns models in descending fitness order."""
        skill_vector_dir = temp_experiment_dir / "global_skill_vectors"

        models = {
            "gen_0_ind_0": {"task_0": 0.3},
            "gen_0_ind_1": {"task_0": 0.9},
            "gen_0_ind_2": {"task_0": 0.6},
        }

        for model_id, skill_vector in models.items():
            skill_vector_file = (
                skill_vector_dir / f"{model_id}_skill_vector.json"
            )
            with open(skill_vector_file, "w") as f:
                json.dump(skill_vector, f)

            (temp_model_dir / model_id).mkdir(exist_ok=True)

        selected = get_best_n_models_based_on_fitness(
            archive_path=str(skill_vector_dir),
            n=3,
            models_dir=str(temp_model_dir),
        )

        # Check ordering
        selected_names = [Path(p).name for p in selected]
        assert selected_names[0] == "gen_0_ind_1"  # Highest
        assert selected_names[1] == "gen_0_ind_2"  # Middle
        assert selected_names[2] == "gen_0_ind_0"  # Lowest
