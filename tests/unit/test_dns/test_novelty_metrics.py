"""
Unit tests for DNS novelty metrics and difficulty weighting.

Tests dominated novelty score computation with various configurations.
"""

import pytest
import numpy as np

from dns.dns_utils import (
    compute_difficulty_weights,
    compute_dominated_novelty_score,
)
from datatypes import DNSSolution


class TestDifficultyWeights:
    """Test difficulty weight computation."""

    def test_compute_difficulty_weights_basic(self, dns_population):
        """Test difficulty weight computation from population."""
        weights = compute_difficulty_weights(dns_population)

        assert len(weights) > 0
        assert all(0.0 <= w <= 1.0 for w in weights)

    def test_compute_difficulty_weights_all_pass(self):
        """Test difficulty weights when all models pass all tasks."""
        population = [
            DNSSolution(
                model_path=f"model_{i}",
                fitness=1.0,
                skill_vector=[True, True, True],
            )
            for i in range(5)
        ]

        weights = compute_difficulty_weights(population)

        # All tasks passed by all models → all weights should be 0
        assert all(w == 0.0 for w in weights)

    def test_compute_difficulty_weights_all_fail(self):
        """Test difficulty weights when all models fail all tasks."""
        population = [
            DNSSolution(
                model_path=f"model_{i}",
                fitness=0.0,
                skill_vector=[False, False, False],
            )
            for i in range(5)
        ]

        weights = compute_difficulty_weights(population)

        # All tasks failed by all models → all weights should be 1.0
        assert all(w == 1.0 for w in weights)

    def test_compute_difficulty_weights_mixed(self):
        """Test difficulty weights with mixed performance."""
        population = [
            DNSSolution(
                model_path="model_0",
                fitness=0.67,
                skill_vector=[True, True, False],
            ),
            DNSSolution(
                model_path="model_1",
                fitness=0.67,
                skill_vector=[True, False, True],
            ),
            DNSSolution(
                model_path="model_2",
                fitness=0.67,
                skill_vector=[False, True, True],
            ),
        ]

        weights = compute_difficulty_weights(population)

        # Each task failed by 1 out of 3 models → all weights should be 1/3
        assert all(w == pytest.approx(1/3) for w in weights)

    def test_compute_difficulty_weights_empty_population(self):
        """Test difficulty weights with empty population."""
        weights = compute_difficulty_weights([])
        assert weights == []

    def test_compute_difficulty_weights_ordering(self):
        """Test that difficulty weights match skill vector ordering."""
        population = [
            DNSSolution(
                model_path="model_0",
                fitness=0.5,
                skill_vector=[True, False, True, False],
            ),
            DNSSolution(
                model_path="model_1",
                fitness=0.5,
                skill_vector=[False, False, True, True],
            ),
        ]

        weights = compute_difficulty_weights(population)

        # Task 0: 1 failure (model_1) → 0.5
        # Task 1: 2 failures (both) → 1.0
        # Task 2: 0 failures → 0.0
        # Task 3: 1 failure (model_0) → 0.5
        assert weights[0] == pytest.approx(0.5)
        assert weights[1] == pytest.approx(1.0)
        assert weights[2] == pytest.approx(0.0)
        assert weights[3] == pytest.approx(0.5)


class TestDominatedNoveltyBasic:
    """Test basic dominated novelty score computation."""

    def test_novelty_no_fitter_solutions(self):
        """Test that best fitness gets dominated_score."""
        solution = DNSSolution(
            model_path="best_model",
            fitness=1.0,
            skill_vector=[True, True, True, True, True],
        )

        fitter_solutions = []  # No fitter solutions
        k_neighbors = 3
        dominated_score = 999.0

        score, fittest_model_found = compute_dominated_novelty_score(
            solution, fitter_solutions, k_neighbors, dominated_score
        )

        # Should return dominated_score since no fitter solutions exist
        assert score == dominated_score

    def test_novelty_completely_dominated(self):
        """Test solution dominated on all tasks."""
        weak_solution = DNSSolution(
            model_path="weak",
            fitness=0.5,
            skill_vector=[True, False, True, False, False],
        )

        strong_solution = DNSSolution(
            model_path="strong",
            fitness=0.9,
            skill_vector=[True, True, True, True, True],
        )

        fitter_solutions = [strong_solution]
        k_neighbors = 1

        score, fittest_model_found = compute_dominated_novelty_score(
            weak_solution,
            fitter_solutions,
            k_neighbors,
            use_skill_ratio=False,
        )

        # Hamming distance should be low (weak solution dominated)
        # Distance = number of positions where they differ
        # weak: [T, F, T, F, F]
        # strong: [T, T, T, T, T]
        # Differences at positions 1, 3, 4 = 3
        assert score == pytest.approx(3.0)

    def test_novelty_with_unique_skills(self):
        """Test novelty from unique skills."""
        solution = DNSSolution(
            model_path="specialist",
            fitness=0.6,
            skill_vector=[True, True, False, False, True],
        )

        fitter_solution = DNSSolution(
            model_path="generalist",
            fitness=0.8,
            skill_vector=[True, False, True, True, False],
        )

        fitter_solutions = [fitter_solution]
        k_neighbors = 1

        score, fittest_model_found = compute_dominated_novelty_score(
            solution,
            fitter_solutions,
            k_neighbors,
            use_skill_ratio=False,
        )

        # Hamming distance calculation:
        # solution:   [T, T, F, F, T]
        # fitter:     [T, F, T, T, F]
        # Differences at positions 1, 2, 3, 4 = 4
        assert score == pytest.approx(4.0)


class TestDominatedNoveltyWithSkillRatio:
    """Test dominated novelty with skill ratio metric."""

    def test_novelty_skill_ratio_basic(self):
        """Test novelty with skill ratio (no weighting)."""
        weak_solution = DNSSolution(
            model_path="weak",
            fitness=0.4,
            skill_vector=[True, True, False, False, False],
        )

        strong_solution = DNSSolution(
            model_path="strong",
            fitness=0.8,
            skill_vector=[True, False, True, True, False],
        )

        fitter_solutions = [strong_solution]
        k_neighbors = 1

        score, fittest_model_found = compute_dominated_novelty_score(
            weak_solution,
            fitter_solutions,
            k_neighbors,
            use_skill_ratio=True,
            skill_ratio_to_full=False,
        )

        # Unique skills: positions where weak=True and strong=False
        # Position 1: weak=True, strong=False → unique
        # Total unsolved by strong: positions where strong=False
        # Positions 1, 4: strong=False → 2 unsolved
        # Score = 1 / 2 * 100.0 = 50.0
        assert score == pytest.approx(50.0)

    def test_novelty_skill_ratio_to_full(self):
        """Test novelty with skill_ratio_to_full=True."""
        weak_solution = DNSSolution(
            model_path="weak",
            fitness=0.4,
            skill_vector=[True, True, False, False, False],
        )

        strong_solution = DNSSolution(
            model_path="strong",
            fitness=0.8,
            skill_vector=[True, False, True, True, False],
        )

        fitter_solutions = [strong_solution]
        k_neighbors = 1

        score, fittest_model_found = compute_dominated_novelty_score(
            weak_solution,
            fitter_solutions,
            k_neighbors,
            use_skill_ratio=True,
            skill_ratio_to_full=True,
        )

        # Unique skills: 1 (position 1)
        # Total skills: 5
        # Score = 1 / 5 * 100.0 = 20.0
        assert score == pytest.approx(20.0)

    def test_novelty_skill_ratio_no_unique_skills(self):
        """Test skill ratio when no unique skills."""
        weak_solution = DNSSolution(
            model_path="weak",
            fitness=0.4,
            skill_vector=[True, False, True, False, False],
        )

        strong_solution = DNSSolution(
            model_path="strong",
            fitness=0.8,
            skill_vector=[True, True, True, True, True],
        )

        fitter_solutions = [strong_solution]
        k_neighbors = 1

        score, fittest_model_found = compute_dominated_novelty_score(
            weak_solution,
            fitter_solutions,
            k_neighbors,
            use_skill_ratio=True,
        )

        # No unique skills (strong solves everything weak solves and more)
        # Score should be 0
        assert score == pytest.approx(0.0)

    def test_novelty_skill_ratio_strong_solves_everything(self):
        """Test skill ratio when stronger solution solves everything."""
        weak_solution = DNSSolution(
            model_path="weak",
            fitness=0.6,
            skill_vector=[True, True, False, True, False],
        )

        perfect_solution = DNSSolution(
            model_path="perfect",
            fitness=1.0,
            skill_vector=[True, True, True, True, True],
        )

        fitter_solutions = [perfect_solution]
        k_neighbors = 1

        score, fittest_model_found = compute_dominated_novelty_score(
            weak_solution,
            fitter_solutions,
            k_neighbors,
            use_skill_ratio=True,
            skill_ratio_to_full=False,
        )

        # Strong solves everything → total_unsolved_by_stronger = 0
        # Score should be 0.0
        assert score == pytest.approx(0.0)


class TestDominatedNoveltyWithDifficultyWeights:
    """Test dominated novelty with difficulty weighting."""

    def test_novelty_with_difficulty_weights(self):
        """Test novelty with difficulty weighting enabled."""
        weak_solution = DNSSolution(
            model_path="weak",
            fitness=0.4,
            skill_vector=[True, True, False, False, False],
        )

        strong_solution = DNSSolution(
            model_path="strong",
            fitness=0.8,
            skill_vector=[True, False, True, True, False],
        )

        fitter_solutions = [strong_solution]
        k_neighbors = 1

        # Difficulty weights: higher weight = harder task
        difficulty_weights = [0.1, 0.9, 0.3, 0.5, 0.2]

        score, fittest_model_found = compute_dominated_novelty_score(
            weak_solution,
            fitter_solutions,
            k_neighbors,
            use_skill_ratio=True,
            use_difficulty_weights=True,
            difficulty_weights=difficulty_weights,
            skill_ratio_to_full=False,
        )

        # Unique solved (weighted): position 1 (weight=0.9)
        # Total unsolved by stronger: positions 1, 4 (weights=0.9, 0.2)
        # Score = 0.9 / (0.9 + 0.2) * 100.0 ≈ 81.82
        assert score == pytest.approx(81.82, rel=0.01)

    def test_novelty_weighted_to_full(self):
        """Test weighted novelty with skill_ratio_to_full=True."""
        weak_solution = DNSSolution(
            model_path="weak",
            fitness=0.4,
            skill_vector=[True, True, False, False, False],
        )

        strong_solution = DNSSolution(
            model_path="strong",
            fitness=0.8,
            skill_vector=[True, False, True, True, False],
        )

        fitter_solutions = [strong_solution]
        k_neighbors = 1

        difficulty_weights = [0.1, 0.9, 0.3, 0.5, 0.2]

        score, fittest_model_found = compute_dominated_novelty_score(
            weak_solution,
            fitter_solutions,
            k_neighbors,
            use_skill_ratio=True,
            use_difficulty_weights=True,
            difficulty_weights=difficulty_weights,
            skill_ratio_to_full=True,
        )

        # Unique solved weighted: 0.9
        # Total weighted: sum(difficulty_weights) = 2.0
        # Score = 0.9 / 2.0 * 100.0 = 45.0
        assert score == pytest.approx(45.0)


class TestDominatedNoveltyKNeighbors:
    """Test k-neighbors limiting in novelty computation."""

    def test_novelty_k_neighbors_limiting(self):
        """Test that only k nearest neighbors are used."""
        solution = DNSSolution(
            model_path="test",
            fitness=0.5,
            skill_vector=[True, False, True, False, True],
        )

        # Create 5 fitter solutions with varying distances
        fitter_solutions = [
            DNSSolution(
                model_path=f"fitter_{i}",
                fitness=0.6 + i * 0.05,
                skill_vector=[True, True, True, True, True],
            )
            for i in range(5)
        ]

        k_neighbors = 2  # Only use 2 nearest

        score, fittest_model_found = compute_dominated_novelty_score(
            solution,
            fitter_solutions,
            k_neighbors,
            use_skill_ratio=False,
        )

        # All fitter solutions have same skill vector, so all distances are same
        # Distance = 2 (positions 1, 3 differ)
        # Mean of k=2 nearest = 2.0
        assert score == pytest.approx(2.0)

    def test_novelty_k_larger_than_population(self):
        """Test when k is larger than number of fitter solutions."""
        solution = DNSSolution(
            model_path="test",
            fitness=0.5,
            skill_vector=[True, False, True],
        )

        fitter_solutions = [
            DNSSolution(
                model_path="fitter",
                fitness=0.8,
                skill_vector=[True, True, True],
            )
        ]

        k_neighbors = 10  # Larger than population

        score, fittest_model_found = compute_dominated_novelty_score(
            solution, fitter_solutions, k_neighbors
        )

        # Should use all available fitter solutions (1)
        # Distance = 1
        assert score == pytest.approx(1.0)

    def test_novelty_k_zero(self):
        """Test when k=0."""
        solution = DNSSolution(
            model_path="test",
            fitness=0.5,
            skill_vector=[True, False, True],
        )

        fitter_solutions = [
            DNSSolution(
                model_path="fitter",
                fitness=0.8,
                skill_vector=[True, True, True],
            )
        ]

        k_neighbors = 0
        dominated_score = 999.0

        score, fittest_model_found = compute_dominated_novelty_score(
            solution,
            fitter_solutions,
            k_neighbors,
            dominated_score=dominated_score,
        )

        # k=0 means no neighbors, should return dominated_score
        assert score == dominated_score


class TestDominatedNoveltySubsetSkillVector:
    """Test novelty computation with skill vector subsetting."""

    def test_novelty_with_subset(self):
        """Test novelty with len_subset_skill_vector parameter."""
        solution = DNSSolution(
            model_path="test",
            fitness=0.5,
            skill_vector=[True, False, True, False, True, True],
        )

        fitter_solution = DNSSolution(
            model_path="fitter",
            fitness=0.8,
            skill_vector=[True, True, True, True, False, False],
        )

        fitter_solutions = [fitter_solution]
        k_neighbors = 1

        # Only use first 3 elements
        score, fittest_model_found = compute_dominated_novelty_score(
            solution,
            fitter_solutions,
            k_neighbors,
            use_skill_ratio=False,
            len_subset_skill_vector=3,
        )

        # Subset comparison:
        # solution:  [T, F, T]
        # fitter:    [T, T, T]
        # Distance = 1 (position 1 differs)
        assert score == pytest.approx(1.0)

    def test_novelty_subset_with_difficulty_weights(self):
        """Test subset novelty with difficulty weights."""
        solution = DNSSolution(
            model_path="test",
            fitness=0.5,
            skill_vector=[True, True, False, False, True],
        )

        fitter_solution = DNSSolution(
            model_path="fitter",
            fitness=0.8,
            skill_vector=[True, False, True, True, False],
        )

        fitter_solutions = [fitter_solution]
        k_neighbors = 1

        difficulty_weights = [0.1, 0.9, 0.3, 0.5, 0.2]

        # Use only first 3 elements
        score, fittest_model_found = compute_dominated_novelty_score(
            solution,
            fitter_solutions,
            k_neighbors,
            use_skill_ratio=True,
            use_difficulty_weights=True,
            difficulty_weights=difficulty_weights,
            len_subset_skill_vector=3,
        )

        # Subset: first 3 elements only
        # Subset difficulty weights: [0.1, 0.9, 0.3]
        # solution:  [T, T, F]
        # fitter:    [T, F, T]
        # Unique: position 1 (weight=0.9)
        # Unsolved by fitter: position 1 (weight=0.9)
        # Score = 0.9 / 0.9 * 100.0 = 100.0
        assert score == pytest.approx(100.0)
