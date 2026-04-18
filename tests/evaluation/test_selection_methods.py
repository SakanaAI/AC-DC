"""
Tests for model selection methods in evaluation/utils.py.

Tests various strategies for selecting top models from archives based on
fitness, coverage, global skill vectors, and manual generation selection.
"""

import pytest
import json
import random
from pathlib import Path

from evaluation.utils import (
    get_best_n_models_based_on_fitness,
    get_top_n_models_based_on_fitness_across_entire_archive,
    get_top_n_models_randomly,
    get_top_n_models_based_on_global_skill_vector,
    get_top_n_models_manual_gen_selection,
    get_relevant_archive_models_and_skill_vectors,
)


class TestGetBestNModelsBasedOnFitness:
    """Tests for get_best_n_models_based_on_fitness function."""

    @pytest.fixture
    def archive_with_fitness(self, tmp_path):
        """Create mock archive file with varying fitness values."""
        archive_file = tmp_path / "gen5_dns_archive.json"

        archive = []
        for i in range(5):
            solution = {
                "model_path": f"/models/gen_5_ind_{i}",
                "fitness": 0.5 + (i * 0.1),  # Fitness: 0.5, 0.6, 0.7, 0.8, 0.9
                "acdc_skill_vector": {"task_0": 0.7, "task_1": 0.6},
            }
            archive.append(solution)

        with open(archive_file, "w") as f:
            json.dump(archive, f)

        return str(archive_file)

    def test_basic_fitness_selection(self, archive_with_fitness):
        """Test basic fitness-based selection."""
        selected = get_best_n_models_based_on_fitness(
            archive_path=archive_with_fitness, n=3
        )

        assert len(selected) == 3
        # Should select models with highest fitness (ind_4, ind_3, ind_2)
        selected_names = [Path(p).name for p in selected]
        assert "gen_5_ind_4" in selected_names  # fitness=0.9
        assert "gen_5_ind_3" in selected_names  # fitness=0.8
        assert "gen_5_ind_2" in selected_names  # fitness=0.7

    def test_fitness_selection_ordering(self, archive_with_fitness):
        """Test that models are returned in fitness order."""
        selected = get_best_n_models_based_on_fitness(
            archive_path=archive_with_fitness, n=5
        )

        # Extract fitness values (they're in the archive)
        # Models should be in descending fitness order
        selected_names = [Path(p).name for p in selected]
        assert selected_names[0] == "gen_5_ind_4"  # Highest
        assert selected_names[-1] == "gen_5_ind_0"  # Lowest

    def test_fitness_selection_single_model(self, archive_with_fitness):
        """Test selecting single best model."""
        selected = get_best_n_models_based_on_fitness(
            archive_path=archive_with_fitness, n=1
        )

        assert len(selected) == 1
        assert "gen_5_ind_4" in selected[0]  # Best fitness


class TestGetTopNModelsBasedOnFitnessAcrossEntireArchive:
    """Tests for get_top_n_models_based_on_fitness_across_entire_archive."""

    @pytest.fixture
    def multi_gen_experiment(self, tmp_path):
        """Create experiment with multiple generation archives."""
        exp_dir = tmp_path / "experiment"
        archive_dir = exp_dir / "archives"
        archive_dir.mkdir(parents=True)

        # Create archives with different fitness values
        for gen in [5, 10, 15]:
            archive = []
            for ind in range(3):
                solution = {
                    "model_path": f"/models/gen_{gen}_ind_{ind}",
                    "fitness": 0.5 + (gen / 100) + (ind * 0.05),
                    "acdc_skill_vector": {"task_0": 0.7},
                }
                archive.append(solution)

            with open(archive_dir / f"gen{gen}_dns_archive.json", "w") as f:
                json.dump(archive, f)

        return str(exp_dir)

    def test_selection_across_generations(self, multi_gen_experiment):
        """Test selection across multiple generations."""
        selected = get_top_n_models_based_on_fitness_across_entire_archive(
            experiment_path=multi_gen_experiment, n=5, relevant_gens=[5, 10, 15]
        )

        assert len(selected) == 5

        # Highest fitness should be from later generations
        selected_str = " ".join(selected)
        assert "gen_15" in selected_str  # Latest gen should have highest fitness

    def test_fitness_ordering_across_archive(self, multi_gen_experiment):
        """Test that selection maintains fitness ordering."""
        selected = get_top_n_models_based_on_fitness_across_entire_archive(
            experiment_path=multi_gen_experiment, n=3, relevant_gens=[5, 10, 15]
        )

        # Should select top 3 by fitness across all generations
        assert len(selected) == 3

    def test_filter_by_relevant_gens(self, multi_gen_experiment):
        """Test filtering to only use relevant generations."""
        selected = get_top_n_models_based_on_fitness_across_entire_archive(
            experiment_path=multi_gen_experiment, n=2, relevant_gens=[5]
        )

        # Should only select from gen 5
        for model_path in selected:
            assert "gen_5" in model_path


class TestGetTopNModelsRandomly:
    """Tests for get_top_n_models_randomly function."""

    @pytest.fixture
    def random_experiment(self, tmp_path):
        """Create experiment for random selection testing."""
        exp_dir = tmp_path / "experiment"
        archive_dir = exp_dir / "archives"
        models_dir = exp_dir / "models"
        archive_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)

        for gen in [5, 10]:
            archive = []
            for ind in range(5):
                model_name = f"gen_{gen}_ind_{ind}"
                (models_dir / model_name).mkdir()

                solution = {
                    "model_path": f"/old/path/{model_name}",  # Different path
                    "fitness": 0.7,
                    "acdc_skill_vector": {"task_0": 0.7},
                }
                archive.append(solution)

            with open(archive_dir / f"gen{gen}_dns_archive.json", "w") as f:
                json.dump(archive, f)

        return exp_dir

    def test_random_selection_basic(self, random_experiment):
        """Test basic random selection."""
        selected = get_top_n_models_randomly(
            experiment_path=str(random_experiment),
            n=5,
            relevant_gens=[5, 10],
            seed=42,
        )

        assert len(selected) == 5

    def test_random_selection_reproducibility(self, random_experiment):
        """Test that same seed produces same results."""
        selected1 = get_top_n_models_randomly(
            experiment_path=str(random_experiment),
            n=3,
            relevant_gens=[5, 10],
            seed=42,
        )

        selected2 = get_top_n_models_randomly(
            experiment_path=str(random_experiment),
            n=3,
            relevant_gens=[5, 10],
            seed=42,
        )

        # Same seed should produce same selection
        assert selected1 == selected2

    def test_random_selection_different_seeds(self, random_experiment):
        """Test that different seeds produce different results."""
        selected1 = get_top_n_models_randomly(
            experiment_path=str(random_experiment),
            n=5,
            relevant_gens=[5, 10],
            seed=42,
        )

        selected2 = get_top_n_models_randomly(
            experiment_path=str(random_experiment),
            n=5,
            relevant_gens=[5, 10],
            seed=43,
        )

        # Different seeds should (likely) produce different results
        # Note: There's a small chance they could be the same, but very unlikely
        assert selected1 != selected2

    def test_random_selection_no_duplicates(self, random_experiment):
        """Test that random selection returns no duplicate models."""
        selected = get_top_n_models_randomly(
            experiment_path=str(random_experiment),
            n=5,
            relevant_gens=[5, 10],
            seed=42,
        )

        # Extract model names from paths
        model_names = [Path(p).name for p in selected]

        # Verify no duplicates
        assert len(model_names) == len(set(model_names)), (
            f"Found duplicate models in selection: {model_names}"
        )

    def test_random_selection_with_duplicate_archive_entries(self, tmp_path):
        """Test that models appearing in multiple archives are deduplicated."""
        # Create experiment where same models appear in multiple generation archives
        exp_dir = tmp_path / "experiment_with_duplicates"
        archive_dir = exp_dir / "archives"
        models_dir = exp_dir / "models"
        archive_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)

        # Create models
        for ind in range(8):
            model_name = f"gen_5_ind_{ind}"
            (models_dir / model_name).mkdir()

        # Gen 5 archive has models 0-7
        archive_gen5 = []
        for ind in range(8):
            archive_gen5.append({
                "model_path": f"/old/path/gen_5_ind_{ind}",
                "fitness": 0.7 + (ind * 0.01),
                "acdc_skill_vector": {"task_0": 0.7},
            })
        with open(archive_dir / "gen5_dns_archive.json", "w") as f:
            json.dump(archive_gen5, f)

        # Gen 10 archive has models 0-4 (duplicates!) plus some new ones
        archive_gen10 = []
        for ind in range(5):
            # These are duplicates from gen 5
            archive_gen10.append({
                "model_path": f"/old/path/gen_5_ind_{ind}",
                "fitness": 0.75 + (ind * 0.01),
                "acdc_skill_vector": {"task_0": 0.75},
            })
        with open(archive_dir / "gen10_dns_archive.json", "w") as f:
            json.dump(archive_gen10, f)

        # Now select 5 models - should get unique models even though some appear in both archives
        selected = get_top_n_models_randomly(
            experiment_path=str(exp_dir),
            n=5,
            relevant_gens=[5, 10],
            seed=42,
        )

        # Verify we got exactly 5 unique models
        assert len(selected) == 5
        model_names = [Path(p).name for p in selected]
        assert len(model_names) == len(set(model_names)), (
            f"Found duplicate models: {model_names}"
        )

        # Verify no model appears more than once
        for model_name in model_names:
            count = model_names.count(model_name)
            assert count == 1, f"Model {model_name} appears {count} times"


class TestGetTopNModelsBasedOnGlobalSkillVector:
    """Tests for get_top_n_models_based_on_global_skill_vector function."""

    @pytest.fixture
    def global_skill_experiment(self, tmp_path):
        """Create experiment with global skill vectors."""
        exp_dir = tmp_path / "experiment"
        skill_dir = exp_dir / "global_skill_vectors"
        models_dir = exp_dir / "models"
        skill_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)

        # Create skill vectors for multiple models
        for gen in range(3):
            for ind in range(3):
                model_name = f"gen_{gen}_ind_{ind}"
                (models_dir / model_name).mkdir()

                # Varying skill vectors
                skill_vector = {
                    "task_0": 0.5 + (gen * 0.1),
                    "task_1": 0.6 + (ind * 0.1),
                    "task_2": 0.7,
                }

                with open(skill_dir / f"{model_name}_skill_vector.json", "w") as f:
                    json.dump(skill_vector, f)

        return exp_dir

    def test_coverage_based_selection(self, global_skill_experiment):
        """Test coverage-based selection with global skill vectors."""
        selected = get_top_n_models_based_on_global_skill_vector(
            experiment_path=str(global_skill_experiment),
            n=3,
            selection_method="coverage",
        )

        assert len(selected) == 3

    def test_fitness_based_selection(self, global_skill_experiment):
        """Test fitness-based selection with global skill vectors."""
        selected = get_top_n_models_based_on_global_skill_vector(
            experiment_path=str(global_skill_experiment),
            n=3,
            selection_method="fitness",
        )

        assert len(selected) == 3

    def test_max_gen_filtering(self, global_skill_experiment):
        """Test max_gen parameter."""
        selected = get_top_n_models_based_on_global_skill_vector(
            experiment_path=str(global_skill_experiment),
            n=2,
            max_gen=0,
            selection_method="fitness",
        )

        # Should only select from gen 0
        for model_path in selected:
            assert "gen_0" in model_path

    def test_invalid_selection_method(self, global_skill_experiment):
        """Test error with invalid selection method."""
        with pytest.raises(ValueError, match="Invalid selection method"):
            get_top_n_models_based_on_global_skill_vector(
                experiment_path=str(global_skill_experiment),
                n=2,
                selection_method="invalid_method",
            )

    def test_max_n_parameter(self, global_skill_experiment):
        """Test max_n parameter for pre-filtering."""
        selected = get_top_n_models_based_on_global_skill_vector(
            experiment_path=str(global_skill_experiment),
            n=2,
            max_n=5,
            selection_method="coverage",
        )

        # Should return only n models, not max_n
        assert len(selected) == 2


class TestGetTopNModelsManualGenSelection:
    """Tests for get_top_n_models_manual_gen_selection function."""

    @pytest.fixture
    def manual_selection_experiment(self, tmp_path):
        """Create experiment for manual generation selection."""
        exp_dir = tmp_path / "experiment"
        archive_dir = exp_dir / "archives"
        archive_dir.mkdir(parents=True)

        # Create archives for multiple generations
        for gen in [5, 10, 15, 20]:
            archive = []
            for ind in range(5):
                solution = {
                    "model_path": f"/models/gen_{gen}_ind_{ind}",
                    "fitness": 0.6 + (ind * 0.05),
                    "acdc_skill_vector": {
                        "task_0": ind % 2,  # Binary for coverage
                        "task_1": (ind + 1) % 2,
                    },
                }
                archive.append(solution)

            with open(archive_dir / f"gen{gen}_dns_archive.json", "w") as f:
                json.dump(archive, f)

        return exp_dir

    def test_single_generation_selection(self, manual_selection_experiment):
        """Test selection from a single generation."""
        # For N=1, select from gen 5
        selected = get_top_n_models_manual_gen_selection(
            experiment_path=str(manual_selection_experiment),
            n=1,
            relevant_gens=[(5)],  # Single gen for N=1
            selection_method="fitness",
        )

        assert len(selected) == 1
        assert "gen_5" in selected[0]

    def test_multiple_generation_selection(self, manual_selection_experiment):
        """Test selection from multiple generations."""
        # For N=2, select from gens 5 and 10
        selected = get_top_n_models_manual_gen_selection(
            experiment_path=str(manual_selection_experiment),
            n=2,
            relevant_gens=[(5), (5, 10)],  # Tuple per N value
            selection_method="fitness",
        )

        assert len(selected) == 2
        # Should have one from gen 5 and one from gen 10

    def test_repeated_generation_selection(self, manual_selection_experiment):
        """Test selecting multiple models from same generation."""
        # For N=3, select 2 from gen 5 and 1 from gen 10
        selected = get_top_n_models_manual_gen_selection(
            experiment_path=str(manual_selection_experiment),
            n=3,
            relevant_gens=[(5), (5, 10), (5, 5, 10)],  # N=3: two from gen 5
            selection_method="coverage",
        )

        assert len(selected) == 3
        # Count how many from each gen
        gen_5_count = sum(1 for p in selected if "gen_5" in p)
        assert gen_5_count == 2

    def test_coverage_selection_method(self, manual_selection_experiment):
        """Test coverage-based selection method."""
        selected = get_top_n_models_manual_gen_selection(
            experiment_path=str(manual_selection_experiment),
            n=2,
            relevant_gens=[(5), (5, 10)],
            selection_method="coverage",
        )

        assert len(selected) == 2

    def test_invalid_selection_method(self, manual_selection_experiment):
        """Test error with invalid selection method."""
        with pytest.raises(ValueError, match="Invalid selection method"):
            get_top_n_models_manual_gen_selection(
                experiment_path=str(manual_selection_experiment),
                n=1,
                relevant_gens=[(5)],
                selection_method="invalid",
            )


class TestGetRelevantArchiveModelsAndSkillVectors:
    """Tests for get_relevant_archive_models_and_skill_vectors function."""

    @pytest.fixture
    def skill_vector_experiment(self, tmp_path):
        """Create experiment with skill vectors."""
        exp_dir = tmp_path / "experiment"
        skill_dir = exp_dir / "global_skill_vectors"
        models_dir = exp_dir / "models"
        skill_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)

        # Create skill vectors
        for gen in range(3):
            for ind in range(3):
                model_name = f"gen_{gen}_ind_{ind}"
                (models_dir / model_name).mkdir()

                skill_vector = {"task_0": 0.7 + (gen * 0.05), "task_1": 0.6}

                with open(skill_dir / f"{model_name}_skill_vector.json", "w") as f:
                    json.dump(skill_vector, f)

        return exp_dir

    def test_load_global_skill_vectors(self, skill_vector_experiment):
        """Test loading global skill vectors."""
        models_dir = skill_vector_experiment / "models"
        skill_dir = skill_vector_experiment / "global_skill_vectors"

        archive = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_dir), models_dir=str(models_dir)
        )

        # Should load all models
        assert len(archive) == 9  # 3 gens * 3 models

        # Check structure
        for model in archive:
            assert "model_path" in model
            assert "acdc_skill_vector" in model
            assert "fitness" in model

    def test_max_gen_filtering(self, skill_vector_experiment):
        """Test max_gen parameter."""
        models_dir = skill_vector_experiment / "models"
        skill_dir = skill_vector_experiment / "global_skill_vectors"

        archive = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_dir), models_dir=str(models_dir), max_gen=1
        )

        # Should only load from gen 0 and 1
        assert len(archive) == 6  # 2 gens * 3 models

        for model in archive:
            model_name = Path(model["model_path"]).name
            gen_num = int(model_name.split("_")[1])
            assert gen_num <= 1

    def test_include_seed_models(self, skill_vector_experiment):
        """Test include_seed_models parameter."""
        models_dir = skill_vector_experiment / "models"
        skill_dir = skill_vector_experiment / "global_skill_vectors"

        # Create a seed model
        seed_name = "gen_0_ind_Qwen2.5-7B"
        (models_dir / seed_name).mkdir()
        skill_vector = {"task_0": 0.95, "task_1": 0.9}
        with open(skill_dir / f"{seed_name}_skill_vector.json", "w") as f:
            json.dump(skill_vector, f)

        archive = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_dir),
            models_dir=str(models_dir),
            include_seed_models={seed_name},
        )

        # Check that seed model has high fitness (should be infinity)
        seed_model = next(
            m for m in archive if seed_name in m["model_path"]
        )
        assert seed_model["fitness"] == float("inf")

    def test_exclude_seed_models(self, skill_vector_experiment):
        """Test exclude_seed_models parameter."""
        models_dir = skill_vector_experiment / "models"
        skill_dir = skill_vector_experiment / "global_skill_vectors"

        # Create a seed model
        seed_name = "gen_0_ind_Qwen2.5-7B"
        (models_dir / seed_name).mkdir()
        skill_vector = {"task_0": 0.95, "task_1": 0.9}
        with open(skill_dir / f"{seed_name}_skill_vector.json", "w") as f:
            json.dump(skill_vector, f)

        archive_without_exclusion = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_dir),
            models_dir=str(models_dir),
        )

        archive_with_exclusion = get_relevant_archive_models_and_skill_vectors(
            archive_path=str(skill_dir),
            models_dir=str(models_dir),
            exclude_seed_models={seed_name},
        )

        # Should have one fewer model when excluding
        assert len(archive_with_exclusion) == len(archive_without_exclusion) - 1

        # Seed model should not be present
        for model in archive_with_exclusion:
            assert seed_name not in model["model_path"]
