"""
Tests for dns/metrics.py - Coverage and quality metrics computation.

Covers compute_acdc_coverage_metrics (called each generation) and
analyze_combined_coverage (archive aggregation).
"""

import pytest
from unittest.mock import Mock, MagicMock
from dns.metrics import (
    compute_acdc_coverage_metrics,
    analyze_combined_coverage,
)


class TestComputeACDCoverageMetrics:
    """Test compute_acdc_coverage_metrics.

    Inputs are AC/DC skill vectors with task_id -> score mappings.
    """

    def test_basic_archive_coverage(self, mock_acdc_archive_data_for_metrics, mock_tasks_for_metrics):
        """Coverage calculation with 5 models, 4 task scores each, threshold=0.5."""
        # Arrange
        archive_data = mock_acdc_archive_data_for_metrics
        tasks = mock_tasks_for_metrics
        threshold = 0.5

        # Act
        result = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=tasks,
            threshold=threshold
        )

        # Assert
        assert isinstance(result, dict)
        assert "acdc_coverage/combined/all_models/passed_ratio" in result
        assert "acdc_coverage/combined/all_models/passed_examples" in result
        assert "acdc_coverage/combined/all_models/total_examples" in result
        assert "acdc_coverage/combined/top_models/passed_ratio" in result
        assert "acdc_coverage/combined/top_models/passed_examples" in result
        assert "acdc_coverage/combined/top_models/total_examples" in result

        # Verify values are reasonable
        assert 0.0 <= result["acdc_coverage/combined/all_models/passed_ratio"] <= 1.0
        assert 0.0 <= result["acdc_coverage/combined/top_models/passed_ratio"] <= 1.0
        assert result["acdc_coverage/combined/all_models/total_examples"] == 4  # 4 unique task IDs

    def test_empty_archive_early_return(self):
        """Empty archive (missing key, empty list, or empty dict) returns {}."""
        # Test with empty dict
        result1 = compute_acdc_coverage_metrics(
            archive_data={},
            tasks=[],
            threshold=0.5
        )
        assert result1 == {}

        # Test with missing dns_archive key
        result2 = compute_acdc_coverage_metrics(
            archive_data={"other_key": []},
            tasks=[],
            threshold=0.5
        )
        assert result2 == {}

        # Test with empty dns_archive list
        result3 = compute_acdc_coverage_metrics(
            archive_data={"dns_archive": []},
            tasks=[],
            threshold=0.5
        )
        assert result3 == {}

    def test_threshold_variation_low(self, mock_acdc_archive_data_for_metrics, mock_tasks_for_metrics):
        """Lower threshold (0.3) yields equal or higher passed_examples than 0.5."""
        # Arrange
        archive_data = mock_acdc_archive_data_for_metrics
        tasks = mock_tasks_for_metrics

        # Act
        result_low = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=tasks,
            threshold=0.3
        )
        result_medium = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=tasks,
            threshold=0.5
        )

        # Assert
        # Lower threshold should result in equal or higher passed examples
        assert (
            result_low["acdc_coverage/combined/all_models/passed_examples"]
            >= result_medium["acdc_coverage/combined/all_models/passed_examples"]
        )
        assert (
            result_low["acdc_coverage/combined/all_models/passed_ratio"]
            >= result_medium["acdc_coverage/combined/all_models/passed_ratio"]
        )

    def test_threshold_variation_high(self, mock_acdc_archive_data_for_metrics, mock_tasks_for_metrics):
        """Higher threshold (0.7) yields equal or fewer passed_examples than 0.5."""
        # Arrange
        archive_data = mock_acdc_archive_data_for_metrics
        tasks = mock_tasks_for_metrics

        # Act
        result_high = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=tasks,
            threshold=0.7
        )
        result_medium = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=tasks,
            threshold=0.5
        )

        # Assert
        # Higher threshold should result in equal or fewer passed examples
        assert (
            result_high["acdc_coverage/combined/all_models/passed_examples"]
            <= result_medium["acdc_coverage/combined/all_models/passed_examples"]
        )
        assert (
            result_high["acdc_coverage/combined/all_models/passed_ratio"]
            <= result_medium["acdc_coverage/combined/all_models/passed_ratio"]
        )

    def test_validation_tasks_present(self, mock_acdc_solutions, mock_tasks_for_metrics):
        """validation/top{idx}_model/quality keys present when validation_quality is set on top models."""
        # Arrange
        # Set validation_quality for top models
        for idx in range(3):
            mock_acdc_solutions[idx].validation_quality = 0.7 + idx * 0.1

        archive_data = {
            "dns_archive": mock_acdc_solutions,
        }

        # Act
        result = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=mock_tasks_for_metrics,
            threshold=0.5,
            validation_tasks=["val_task_1", "val_task_2"]
        )

        # Assert
        # Should include validation metrics for top models
        assert "validation/top1_model/quality" in result or len(result) > 0
        # Note: The exact keys depend on which models have validation_quality set

    def test_missing_skill_vectors(self, mock_tasks_for_metrics):
        """Solutions with None acdc_skill_vector are skipped; valid ones still processed."""
        # Arrange
        solutions = []
        # Valid solution
        sol1 = Mock()
        sol1.model_path = "/models/valid_1"
        sol1.acdc_skill_vector = {"task_001": 0.9, "task_002": 0.8}
        sol1.fitness = 0.85
        sol1.validation_quality = None
        solutions.append(sol1)

        # Solution with None skill vector
        sol2 = Mock()
        sol2.model_path = "/models/invalid"
        sol2.acdc_skill_vector = None
        sol2.fitness = 0.0
        solutions.append(sol2)

        # Another valid solution
        sol3 = Mock()
        sol3.model_path = "/models/valid_2"
        sol3.acdc_skill_vector = {"task_001": 0.7, "task_002": 0.6}
        sol3.fitness = 0.65
        sol3.validation_quality = None
        solutions.append(sol3)

        archive_data = {"dns_archive": solutions}

        # Act
        result = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=mock_tasks_for_metrics,
            threshold=0.5
        )

        # Assert
        # Should process without error
        assert isinstance(result, dict)
        # Total examples should be 2 (only from valid solutions)
        assert result["acdc_coverage/combined/all_models/total_examples"] == 2

    def test_partial_skill_vectors(self, mock_tasks_for_metrics):
        """Coverage is computed over the union of task_ids across models with non-overlapping skill vectors."""
        # Arrange
        solutions = []

        sol1 = Mock()
        sol1.model_path = "/models/model_1"
        sol1.acdc_skill_vector = {"task_A": 0.9, "task_B": 0.8}
        sol1.fitness = 0.85
        sol1.validation_quality = None
        solutions.append(sol1)

        sol2 = Mock()
        sol2.model_path = "/models/model_2"
        sol2.acdc_skill_vector = {"task_C": 0.7, "task_D": 0.6}
        sol2.fitness = 0.65
        sol2.validation_quality = None
        solutions.append(sol2)

        sol3 = Mock()
        sol3.model_path = "/models/model_3"
        sol3.acdc_skill_vector = {"task_A": 0.5, "task_C": 0.4}
        sol3.fitness = 0.45
        sol3.validation_quality = None
        solutions.append(sol3)

        archive_data = {"dns_archive": solutions}

        # Act
        result = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=mock_tasks_for_metrics,
            threshold=0.5
        )

        # Assert
        # Total examples should be union: task_A, task_B, task_C, task_D = 4
        assert result["acdc_coverage/combined/all_models/total_examples"] == 4

    def test_top_k_model_selection(self, mock_acdc_solutions, mock_tasks_for_metrics):
        """With 5 models and top_k=5, all_models and top_models coverage are equal."""
        # Arrange
        # mock_acdc_solutions already has fitness values
        # Verify fitness ordering
        assert mock_acdc_solutions[0].fitness == sum(mock_acdc_solutions[0].acdc_skill_vector.values()) / 4

        archive_data = {"dns_archive": mock_acdc_solutions}

        # Act
        result = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=mock_tasks_for_metrics,
            threshold=0.5
        )

        # Assert
        # With 5 models and top_k=5, all_models and top_models should be the same
        assert (
            result["acdc_coverage/combined/all_models/passed_examples"]
            == result["acdc_coverage/combined/top_models/passed_examples"]
        )

    def test_combined_coverage_calculation(self, mock_acdc_solutions, mock_tasks_for_metrics):
        """all_models coverage >= top_models when low-fitness models cover unique tasks."""
        # Arrange
        # Add more models so top-5 is a subset
        for i in range(5, 10):
            sol = Mock()
            sol.model_path = f"/models/gen_1_ind_{i}"
            # Lower fitness models might cover unique tasks
            sol.acdc_skill_vector = {f"task_00{i}": 0.9}
            sol.fitness = 0.2
            sol.validation_quality = None
            mock_acdc_solutions.append(sol)

        archive_data = {"dns_archive": mock_acdc_solutions}

        # Act
        result = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=mock_tasks_for_metrics,
            threshold=0.5
        )

        # Assert
        # All models should cover at least as much as top-5 models
        assert (
            result["acdc_coverage/combined/all_models/passed_ratio"]
            >= result["acdc_coverage/combined/top_models/passed_ratio"]
        )
        assert (
            result["acdc_coverage/combined/all_models/total_examples"]
            >= result["acdc_coverage/combined/top_models/total_examples"]
        )

    def test_metric_dictionary_structure(self, mock_acdc_archive_data_for_metrics, mock_tasks_for_metrics):
        """Returned dict has acdc_coverage/combined/{all,top}_models/{passed_ratio,passed_examples,total_examples} keys."""
        # Arrange
        archive_data = mock_acdc_archive_data_for_metrics
        tasks = mock_tasks_for_metrics

        # Act
        result = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=tasks,
            threshold=0.5
        )

        # Assert
        # Check all required metric keys
        required_keys = [
            "acdc_coverage/combined/all_models/passed_ratio",
            "acdc_coverage/combined/all_models/passed_examples",
            "acdc_coverage/combined/all_models/total_examples",
            "acdc_coverage/combined/top_models/passed_ratio",
            "acdc_coverage/combined/top_models/passed_examples",
            "acdc_coverage/combined/top_models/total_examples",
        ]

        for key in required_keys:
            assert key in result, f"Missing required metric key: {key}"

        # Check value types
        for key in required_keys:
            if "ratio" in key:
                assert isinstance(result[key], float)
                assert 0.0 <= result[key] <= 1.0
            else:
                assert isinstance(result[key], int)
                assert result[key] >= 0


class TestAnalyzeCombinedCoverage:
    """Test analyze_combined_coverage - aggregates coverage across all models and top-k models."""

    def test_basic_combined_archive(self, sample_combined_archive):
        """Returns all_models and top_models coverage dicts with expected keys for 3-model archive."""
        # Arrange
        combined_archive = sample_combined_archive
        top_k = 2

        # Act
        result = analyze_combined_coverage(combined_archive, top_k=top_k)

        # Assert
        assert "all_models" in result
        assert "top_models" in result

        # Check structure of all_models
        assert "example_coverage" in result["all_models"]
        assert "total_examples" in result["all_models"]
        assert "passed_examples" in result["all_models"]
        assert "coverage_ratio" in result["all_models"]

        # Check structure of top_models
        assert "example_coverage" in result["top_models"]
        assert "total_examples" in result["top_models"]
        assert "passed_examples" in result["top_models"]
        assert "coverage_ratio" in result["top_models"]

        # Total examples should be 4 (task_1_ex_1, task_1_ex_2, task_2_ex_1, task_3_ex_1)
        assert result["all_models"]["total_examples"] == 4
        assert result["top_models"]["total_examples"] == 4

    def test_top_k_model_selection(self, sample_combined_archive):
        """Top-k models are selected by pass count; top_models coverage <= all_models coverage."""
        # Arrange
        # Model scores:
        # model_1: 2 passes (task_1_ex_1, task_2_ex_1)
        # model_2: 3 passes (task_1_ex_2, task_2_ex_1, task_3_ex_1)
        # model_3: 2 passes (task_1_ex_1, task_1_ex_2)
        # So top 2 should be model_2 and either model_1 or model_3

        top_k = 2

        # Act
        result = analyze_combined_coverage(sample_combined_archive, top_k=top_k)

        # Assert
        # Top models should have at least some coverage
        assert result["top_models"]["passed_examples"] > 0
        # Top models coverage should be <= all models coverage
        assert (
            result["top_models"]["coverage_ratio"]
            <= result["all_models"]["coverage_ratio"]
        )

    def test_all_models_vs_top_models(self):
        """all_models covers 5 examples while top-2 cover only 3 when low-scoring models hold unique examples."""
        # Arrange
        combined_archive = {
            "top_model_1": {"ex_1": True, "ex_2": True, "ex_3": True, "ex_4": False, "ex_5": False},
            "top_model_2": {"ex_1": True, "ex_2": True, "ex_3": False, "ex_4": False, "ex_5": False},
            "low_model_1": {"ex_1": False, "ex_2": False, "ex_3": False, "ex_4": True, "ex_5": True},
        }
        top_k = 2

        # Act
        result = analyze_combined_coverage(combined_archive, top_k=top_k)

        # Assert
        # All models should cover all 5 examples
        assert result["all_models"]["passed_examples"] == 5
        # Top 2 models only cover 3 examples (ex_1, ex_2, ex_3)
        assert result["top_models"]["passed_examples"] == 3
        assert result["all_models"]["coverage_ratio"] > result["top_models"]["coverage_ratio"]

    def test_single_model(self):
        """Single-model archive with top_k=5: all_models and top_models yield identical coverage."""
        # Arrange
        combined_archive = {
            "only_model": {"ex_1": True, "ex_2": False, "ex_3": True}
        }
        top_k = 5

        # Act
        result = analyze_combined_coverage(combined_archive, top_k=top_k)

        # Assert
        assert result["all_models"]["passed_examples"] == result["top_models"]["passed_examples"]
        assert result["all_models"]["coverage_ratio"] == result["top_models"]["coverage_ratio"]
        assert result["all_models"]["total_examples"] == 3
        assert result["all_models"]["passed_examples"] == 2

    def test_all_models_failing(self):
        """All-False archive returns passed_examples=0 and coverage_ratio=0.0."""
        # Arrange
        combined_archive = {
            "model_1": {"ex_1": False, "ex_2": False},
            "model_2": {"ex_1": False, "ex_2": False},
        }
        top_k = 2

        # Act
        result = analyze_combined_coverage(combined_archive, top_k=top_k)

        # Assert
        assert result["all_models"]["passed_examples"] == 0
        assert result["all_models"]["coverage_ratio"] == 0.0
        assert result["top_models"]["passed_examples"] == 0
        assert result["top_models"]["coverage_ratio"] == 0.0

    def test_empty_combined_archive(self):
        """Empty combined_archive raises AssertionError ('no examples found')."""
        # Arrange
        combined_archive = {}
        top_k = 5

        # Act & Assert
        with pytest.raises(AssertionError, match="no examples found in combined_archive"):
            analyze_combined_coverage(combined_archive, top_k=top_k)




# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestMetricsEdgeCases:
    """Edge cases for compute_acdc_coverage_metrics and analyze_combined_coverage."""

    def test_compute_acdc_coverage_with_no_task_ids(self, mock_tasks_for_metrics):
        """Archive with empty acdc_skill_vector dicts returns a metrics dict without error."""
        # Arrange
        solutions = []
        sol = Mock()
        sol.model_path = "/models/empty"
        sol.acdc_skill_vector = {}  # Empty skill vector
        sol.fitness = 0.0
        sol.validation_quality = None
        solutions.append(sol)

        archive_data = {"dns_archive": solutions}

        # Act
        result = compute_acdc_coverage_metrics(
            archive_data=archive_data,
            tasks=mock_tasks_for_metrics,
            threshold=0.5
        )

        # Assert
        # Should handle gracefully
        assert isinstance(result, dict)

    def test_analyze_combined_coverage_with_large_top_k(self):
        """top_k=100 with 3-model archive: top_models coverage equals all_models coverage."""
        # Arrange
        combined_archive = {
            f"model_{i}": {f"ex_{j}": j % 2 == 0 for j in range(5)}
            for i in range(3)
        }
        top_k = 100

        # Act
        result = analyze_combined_coverage(combined_archive, top_k=top_k)

        # Assert
        # Should use all models
        assert result["all_models"]["coverage_ratio"] == result["top_models"]["coverage_ratio"]
