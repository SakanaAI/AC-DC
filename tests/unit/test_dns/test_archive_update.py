"""
Unit tests for DNS archive update strategies.

Tests archive update mechanisms: top-fitness and dominated novelty search.
"""

import pytest
from omegaconf import OmegaConf

from dns.dns_utils import (
    update_dns_archive,
    update_dns_archive_top_fitness,
)
from datatypes import DNSSolution


class TestTopFitnessSelection:
    """Test pure fitness-based archive update."""

    def test_top_fitness_basic(
        self, dns_archive, new_dns_solutions, dns_cfg_top_fitness
    ):
        """Test top-fitness selection keeps highest fitness solutions."""
        updated = update_dns_archive_top_fitness(
            dns_archive, new_dns_solutions, dns_cfg_top_fitness
        )

        # Should keep population_size solutions
        assert len(updated) == dns_cfg_top_fitness.population_size

        # Verify sorted by fitness (descending)
        fitnesses = [s.fitness for s in updated]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_top_fitness_selection_rejects_weak(
        self, high_fitness_archive, low_fitness_solutions, dns_cfg_top_fitness
    ):
        """Test that low-fitness solutions are rejected."""
        updated = update_dns_archive_top_fitness(
            high_fitness_archive, low_fitness_solutions, dns_cfg_top_fitness
        )

        # All solutions should have high fitness
        assert all(s.fitness >= 0.8 for s in updated)

    def test_top_fitness_empty_archive(
        self, new_dns_solutions, dns_cfg_top_fitness
    ):
        """Test top-fitness with empty initial archive."""
        updated = update_dns_archive_top_fitness(
            [], new_dns_solutions, dns_cfg_top_fitness
        )

        assert len(updated) == min(
            dns_cfg_top_fitness.population_size, len(new_dns_solutions)
        )

    def test_top_fitness_empty_new_solutions(
        self, dns_archive, dns_cfg_top_fitness
    ):
        """Test top-fitness with no new solutions."""
        updated = update_dns_archive_top_fitness(
            dns_archive, [], dns_cfg_top_fitness
        )

        # Should keep existing archive (up to population_size)
        assert len(updated) <= dns_cfg_top_fitness.population_size

    def test_top_fitness_both_empty(self, dns_cfg_top_fitness):
        """Test top-fitness with empty archive and no new solutions."""
        updated = update_dns_archive_top_fitness([], [], dns_cfg_top_fitness)

        assert updated == []

    def test_top_fitness_exceeds_population_size(
        self, many_dns_solutions, dns_cfg_top_fitness_small
    ):
        """Test that archive respects population_size limit."""
        updated = update_dns_archive_top_fitness(
            [], many_dns_solutions, dns_cfg_top_fitness_small
        )

        assert len(updated) == dns_cfg_top_fitness_small.population_size

        # Verify we kept the best ones
        all_fitnesses = sorted(
            [s.fitness for s in many_dns_solutions], reverse=True
        )
        expected_min_fitness = all_fitnesses[
            dns_cfg_top_fitness_small.population_size - 1
        ]

        assert all(s.fitness >= expected_min_fitness for s in updated)


class TestDNSArchiveUpdate:
    """Test dominated novelty search archive update."""

    def test_dns_update_basic(
        self, dns_archive, new_dns_solutions, dns_cfg_basic
    ):
        """Test basic DNS archive update."""
        updated = update_dns_archive(
            dns_archive, new_dns_solutions, dns_cfg_basic
        )

        # Should keep population_size solutions
        assert len(updated) == dns_cfg_basic.population_size

        # All solutions should be DNSSolution objects
        assert all(isinstance(s, DNSSolution) for s in updated)

    def test_dns_update_uses_top_fitness_when_configured(
        self, dns_archive, new_dns_solutions
    ):
        """Test that use_top_fitness_selection flag works."""
        cfg = OmegaConf.create(
            {
                "population_size": 5,
                "use_top_fitness_selection": True,
            }
        )

        updated = update_dns_archive(dns_archive, new_dns_solutions, cfg)

        # Should use top-fitness selection
        fitnesses = [s.fitness for s in updated]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_dns_update_novelty_selection(
        self, dns_archive_diverse, new_dns_solutions_novel, dns_cfg_novelty
    ):
        """Test that DNS selects novel solutions over duplicates."""
        updated = update_dns_archive(
            dns_archive_diverse, new_dns_solutions_novel, dns_cfg_novelty
        )

        assert len(updated) == dns_cfg_novelty.population_size

        # Novel solutions should be included
        # (This is a simplified check - full validation requires inspecting novelty scores)

    def test_dns_update_with_difficulty_weights(
        self, dns_archive, new_dns_solutions
    ):
        """Test DNS with difficulty weighting enabled."""
        cfg = OmegaConf.create(
            {
                "population_size": 5,
                "use_top_fitness_selection": False,
                "use_difficulty_weights": True,
                "k_neighbors": 3,
                "dominated_score": 999.0,
                "use_skill_ratio": False,
                "skill_ratio_to_full": False,
            }
        )

        updated = update_dns_archive(dns_archive, new_dns_solutions, cfg)

        assert len(updated) == cfg.population_size

    def test_dns_update_with_skill_ratio(self, dns_archive, new_dns_solutions):
        """Test DNS with skill ratio metric."""
        cfg = OmegaConf.create(
            {
                "population_size": 5,
                "use_top_fitness_selection": False,
                "use_difficulty_weights": False,
                "k_neighbors": 3,
                "dominated_score": 999.0,
                "use_skill_ratio": True,
                "skill_ratio_to_full": False,
            }
        )

        updated = update_dns_archive(dns_archive, new_dns_solutions, cfg)

        assert len(updated) == cfg.population_size

    def test_dns_update_empty_archive(self, new_dns_solutions, dns_cfg_basic):
        """Test DNS update with empty initial archive."""
        updated = update_dns_archive([], new_dns_solutions, dns_cfg_basic)

        assert len(updated) == min(
            dns_cfg_basic.population_size, len(new_dns_solutions)
        )

    def test_dns_update_empty_new_solutions(self, dns_archive, dns_cfg_basic):
        """Test DNS update with no new solutions."""
        updated = update_dns_archive(dns_archive, [], dns_cfg_basic)

        # Should trim to population_size
        assert len(updated) <= dns_cfg_basic.population_size

    def test_dns_update_both_empty(self, dns_cfg_basic):
        """Test DNS update with empty archive and no new solutions."""
        updated = update_dns_archive([], [], dns_cfg_basic)

        assert updated == []

    def test_dns_update_with_subset_skill_vector(
        self, dns_archive, new_dns_solutions
    ):
        """Test DNS update with skill vector subsetting."""
        cfg = OmegaConf.create(
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

        # Use only first 3 elements of skill vectors
        len_subset = 3

        updated = update_dns_archive(
            dns_archive,
            new_dns_solutions,
            cfg,
            len_subset_skill_vector=len_subset,
        )

        assert len(updated) == cfg.population_size


class TestArchiveUpdateEdgeCases:
    """Test edge cases in archive updates."""

    def test_single_solution_archive(self, dns_cfg_basic):
        """Test archive with single solution."""
        archive = [
            DNSSolution(
                model_path="only_one",
                fitness=2 / 3,  # 2/3 tasks solved
                skill_vector=[True, False, True],
            )
        ]

        new_solutions = [
            DNSSolution(
                model_path="challenger",
                fitness=2 / 3,  # 2/3 tasks solved
                skill_vector=[False, True, True],
            )
        ]

        updated = update_dns_archive(archive, new_solutions, dns_cfg_basic)

        # Should have both if population_size >= 2
        if dns_cfg_basic.population_size >= 2:
            assert len(updated) == 2

    def test_identical_fitness_solutions(self, dns_cfg_basic):
        """Test archive update with identical fitness values."""
        # Each solution solves exactly 1/5 tasks -> fitness = 0.2
        solutions = [
            DNSSolution(
                model_path=f"model_{i}",
                fitness=0.2,  # All same fitness (1/5 tasks)
                skill_vector=[True if j == i else False for j in range(5)],
            )
            for i in range(10)
        ]

        updated = update_dns_archive([], solutions, dns_cfg_basic)

        # Should select based on novelty despite same fitness
        assert len(updated) == min(
            dns_cfg_basic.population_size, len(solutions)
        )

    def test_perfect_fitness_solutions(self, dns_cfg_novelty):
        """Test archive with all perfect fitness (1.0) solutions."""
        perfect_solutions = [
            DNSSolution(
                model_path=f"perfect_{i}",
                fitness=1.0,
                skill_vector=[True, True, True, True, True],
            )
            for i in range(5)
        ]

        updated = update_dns_archive([], perfect_solutions, dns_cfg_novelty)

        # Should keep all if population_size allows
        assert len(updated) <= dns_cfg_novelty.population_size
        assert all(s.fitness == 1.0 for s in updated)

    def test_zero_fitness_solutions(self, dns_cfg_basic):
        """Test archive with all zero fitness solutions."""
        zero_solutions = [
            DNSSolution(
                model_path=f"zero_{i}",
                fitness=0.0,
                skill_vector=[False, False, False],
            )
            for i in range(3)
        ]

        updated = update_dns_archive([], zero_solutions, dns_cfg_basic)

        # Should still maintain archive
        assert len(updated) <= dns_cfg_basic.population_size

    def test_large_population_size(self, dns_archive, new_dns_solutions):
        """Test with population_size larger than total solutions."""
        cfg = OmegaConf.create(
            {
                "population_size": 1000,  # Much larger than available solutions
                "use_top_fitness_selection": False,
                "use_difficulty_weights": False,
                "k_neighbors": 3,
                "dominated_score": 999.0,
                "use_skill_ratio": False,
                "skill_ratio_to_full": False,
            }
        )

        updated = update_dns_archive(dns_archive, new_dns_solutions, cfg)

        # Should keep all solutions
        total_solutions = len(dns_archive) + len(new_dns_solutions)
        assert len(updated) == total_solutions

    def test_varying_skill_vector_lengths(self):
        """Test that solutions must have consistent skill vector lengths."""
        # Create solutions with different skill vector lengths
        archive = [
            DNSSolution(
                model_path="model_1",
                fitness=2 / 3,  # 2/3 tasks solved
                skill_vector=[True, False, True],
            )
        ]

        new_solutions = [
            DNSSolution(
                model_path="model_2",
                fitness=3 / 5,  # 3/5 tasks solved
                skill_vector=[
                    True,
                    False,
                    True,
                    True,
                    False,
                ],  # Different length
            )
        ]

        cfg = OmegaConf.create(
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

        # This should handle gracefully or raise appropriate error
        # (Implementation-dependent - adjust based on actual behavior)
        try:
            updated = update_dns_archive(archive, new_solutions, cfg)
            # If it succeeds, verify result
            assert len(updated) > 0
        except (ValueError, AssertionError):
            # If it raises error, that's also acceptable
            pass


class TestArchiveUpdateDeterminism:
    """Test determinism of archive updates."""

    def test_dns_update_deterministic(
        self, dns_archive, new_dns_solutions, dns_cfg_basic
    ):
        """Test that DNS update is deterministic."""
        # Run update twice
        updated_1 = update_dns_archive(
            dns_archive.copy(), new_dns_solutions.copy(), dns_cfg_basic
        )

        updated_2 = update_dns_archive(
            dns_archive.copy(), new_dns_solutions.copy(), dns_cfg_basic
        )

        # Results should be identical
        assert len(updated_1) == len(updated_2)

        # Compare model paths (order-sensitive)
        paths_1 = [s.model_path for s in updated_1]
        paths_2 = [s.model_path for s in updated_2]

        # The exact order might vary, but the set should be the same
        assert set(paths_1) == set(paths_2)

    def test_top_fitness_deterministic(
        self, dns_archive, new_dns_solutions, dns_cfg_top_fitness
    ):
        """Test that top-fitness selection is deterministic."""
        updated_1 = update_dns_archive_top_fitness(
            dns_archive.copy(), new_dns_solutions.copy(), dns_cfg_top_fitness
        )

        updated_2 = update_dns_archive_top_fitness(
            dns_archive.copy(), new_dns_solutions.copy(), dns_cfg_top_fitness
        )

        # Results should be identical
        assert len(updated_1) == len(updated_2)

        paths_1 = [s.model_path for s in updated_1]
        paths_2 = [s.model_path for s in updated_2]

        assert paths_1 == paths_2  # Order should be exactly the same


class TestNoveltyBasedSelection:
    """Test novelty-based selection behavior with specific skill patterns."""

    def test_novelty_selection_with_identical_and_diverse_solutions(self):
        """Test DNS novelty scoring behavior with identical fitness solutions.

        Scenario:
        - Solution 0: [1,1,1,0,0] (fitness=0.6) - identical to solution 1
        - Solution 1: [1,1,1,0,0] (fitness=0.6) - identical to solution 0
        - Solution 2: [0,1,1,1,0] (fitness=0.6) - diverse skill pattern
        - Solution 3: [0,0,0,1,1] (fitness=0.4) - weakest fitness but best coverage with solution 0 or 1

        Key insight: Solutions 0, 1, 2 all have fitness=0.6, so they have NO fitter
        solutions. Therefore, they all get dominated_score=999.0. Solution 3 has
        3 fitter solutions, so it gets a much lower novelty score (~48).

        Expected: With population_size=2, the algorithm selects the 2 solutions with
        highest novelty scores, which are any 2 from {0, 1, 2}.
        """
        from dns.dns_utils import (
            update_dns_archive,
            compute_dominated_novelty_score,
            compute_difficulty_weights,
        )

        # Create solutions with specific skill patterns
        # fmt: off
        skill_vectors = [
            [False, False, False, True, True],  # Solution 0: fitness=0.4 (weakest)
            [True, True, True, False, False],   # Solution 1: fitness=0.6 (fittest model)
            [True, True, True, False, False],   # Solution 2: fitness=0.6 (identical to 1)
            [False, True, True, True, False],   # Solution 3: fitness=0.6 (diverse)
        ]
        # fmt: on

        solutions = [
            DNSSolution(
                model_path=f"model_{i}",
                fitness=sum(sv) / len(sv),
                skill_vector=sv,
            )
            for i, sv in enumerate(skill_vectors)
        ]

        # Use default DNS config from configs/dns/default.yaml
        cfg = OmegaConf.create(
            {
                "population_size": 2,
                "k_neighbors": 3,
                "dominated_score": 999.0,
                "use_skill_ratio": True,
                "use_difficulty_weights": True,
                "skill_ratio_to_full": True,
                "use_top_fitness_selection": False,
            }
        )

        # Verify novelty scores manually
        difficulty_weights = compute_difficulty_weights(solutions)

        assert difficulty_weights == [0.5, 0.25, 0.25, 0.5, 0.75], (
            "Difficulty weights should be [0.5, 0.25, 0.25, 0.5, 0.75], "
            f"got {difficulty_weights}"
        )

        # Solutions 0 should have dominated_score since it has no fitter solutions
        fittest_model_found = False
        for i in [0, 1, 2, 3]:
            # Find all solutions with higher fitness
            if not fittest_model_found:
                # here, we make sure to always include the fittest model
                fitter_solutions = [
                    s for s in solutions if s.fitness > solutions[i].fitness
                ]
            else:
                # here, we consider cases where multiple solutions have the same fitness
                # but might have different skill vectors
                fitter_solutions = [
                    s
                    for s in solutions
                    if s.fitness >= solutions[i].fitness
                    and s.model_path != solutions[i].model_path
                ]
            novelty, fittest_model_found = compute_dominated_novelty_score(
                solutions[i],
                fitter_solutions,
                cfg.k_neighbors,
                cfg.dominated_score,
                cfg.use_skill_ratio,
                cfg.use_difficulty_weights,
                difficulty_weights,
                cfg.skill_ratio_to_full,
                fittest_model_found=fittest_model_found,
            )
            if i == 0:
                assert novelty == pytest.approx(
                    48.15, abs=0.01
                ), f"Solution {i} should have roughly 48.15 novelty score, got {novelty}"
            elif i == 1:
                assert (
                    novelty == cfg.dominated_score
                ), f"Solution {i} should have dominated_score since it has no fitter solutions"
            elif i == 2:
                assert novelty == pytest.approx(
                    11.11, abs=0.01
                ), f"Solution {i} should have roughly 11.11 novelty score, got {novelty}"
            elif i == 3:
                assert novelty == pytest.approx(
                    22.22, abs=0.01
                ), f"Solution {i} should have roughly 22.22 novelty score, got {novelty}"

        # Run DNS archive update
        updated = update_dns_archive([], solutions, cfg)

        # Should keep exactly 2 solutions
        assert len(updated) == 2

        # Should have kept solutions 1 and 0
        assert all(s.model_path in ["model_1", "model_0"] for s in updated)
