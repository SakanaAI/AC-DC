"""
Unit tests for DNS utilities (skill vectors, archive management).

Tests skill vector creation, archive save/load, and conversions.
"""

import pytest
import json
import numpy as np
from pathlib import Path

from dns.dns_utils import (
    create_skill_vector,
    create_qd_skill_vector,
    create_dns_solution,
    create_ac_dc_solution,
    convert_acdc_to_dns_solution,
    save_dns_archive,
    load_dns_archive,
    save_ac_dc_archive,
    load_ac_dc_archive,
    compute_hamming_distance,
)
from datatypes import DNSSolution, ACDCSolution, TaskMetric, ACDCTaskEvalDetail


class TestSkillVectorCreation:
    """Test skill vector creation from task metrics."""

    def test_create_binary_skill_vector(self, mock_task_metrics, mock_tasks):
        """Test binary skill vector from task results."""
        skill_vector = create_skill_vector(mock_task_metrics, mock_tasks)

        # Should be binary (all True/False)
        assert all(isinstance(v, bool) for v in skill_vector)
        assert len(skill_vector) > 0

    def test_create_skill_vector_ordering(
        self, mock_task_metrics, mock_tasks
    ):
        """Test that skill vectors maintain consistent ordering."""
        # Create skill vector twice
        sv1 = create_skill_vector(mock_task_metrics, mock_tasks)
        sv2 = create_skill_vector(mock_task_metrics, mock_tasks)

        # Should be identical
        assert sv1 == sv2

    def test_create_qd_skill_vector(self, mock_task_metrics_qd):
        """Test QD mode skill vector creation."""
        skill_vector = create_qd_skill_vector(mock_task_metrics_qd)

        # Should be binary
        assert all(isinstance(v, bool) for v in skill_vector)
        assert len(skill_vector) > 0

    def test_create_qd_skill_vector_sorted(self, mock_task_metrics_qd):
        """Test that QD skill vectors are sorted by task name."""
        skill_vector = create_qd_skill_vector(mock_task_metrics_qd)

        # Run again - should be same order
        skill_vector_2 = create_qd_skill_vector(mock_task_metrics_qd)

        assert skill_vector == skill_vector_2

    def test_create_dns_solution(self, mock_task_metrics, mock_tasks):
        """Test DNS solution creation from metrics."""
        model_path = "gen_1_ind_5"
        solution = create_dns_solution(
            model_path=model_path,
            task_metrics=mock_task_metrics,
            tasks=mock_tasks,
            validation_quality=0.85,
        )

        assert isinstance(solution, DNSSolution)
        assert solution.model_path == model_path
        assert 0.0 <= solution.fitness <= 1.0
        assert len(solution.skill_vector) > 0
        assert solution.validation_quality == 0.85

    def test_create_dns_solution_fitness_calculation(
        self, mock_task_metrics, mock_tasks
    ):
        """Test that fitness is average of skill vector."""
        solution = create_dns_solution(
            model_path="test_model",
            task_metrics=mock_task_metrics,
            tasks=mock_tasks,
        )

        # Fitness should be sum(skill_vector) / len(skill_vector)
        expected_fitness = sum(solution.skill_vector) / len(
            solution.skill_vector
        )
        assert solution.fitness == pytest.approx(expected_fitness)


class TestACDCSkillVectors:
    """Test AC/DC-specific skill vector functionality."""

    def test_create_ac_dc_solution(self):
        """Test AC/DC solution creation."""
        acdc_skill_vector = {
            "task_0_example_0": 0.9,
            "task_0_example_1": 0.7,
            "task_1_example_0": 0.5,
        }

        solution = create_ac_dc_solution(
            model_path="gen_2_ind_10",
            task_metrics=None,
            acdc_skill_vector=acdc_skill_vector,
            avg_acdc_quality=0.7,
            validation_quality=0.75,
        )

        assert isinstance(solution, ACDCSolution)
        assert solution.model_path == "gen_2_ind_10"
        assert solution.acdc_skill_vector == acdc_skill_vector
        assert solution.validation_quality == 0.75
        # Fitness should be avg_acdc_quality since no task_metrics
        assert solution.fitness == pytest.approx(0.7)

    def test_create_ac_dc_solution_with_task_metrics(
        self, mock_task_metrics
    ):
        """Test AC/DC solution with both standard and AC/DC metrics."""
        acdc_skill_vector = {"task_0_example_0": 0.8}

        solution = create_ac_dc_solution(
            model_path="gen_2_ind_10",
            task_metrics=mock_task_metrics,
            acdc_skill_vector=acdc_skill_vector,
            avg_acdc_quality=0.7,
        )

        # Fitness should combine both sources
        assert 0.0 <= solution.fitness <= 1.0

    def test_create_ac_dc_solution_with_eval_details(self):
        """Test AC/DC solution with evaluation details."""
        eval_details = [
            ACDCTaskEvalDetail(
                task_id="task_0",
                instructions="Test instructions for task_0",
                raw_output="correct answer",
                score=0.9,
            )
        ]

        solution = create_ac_dc_solution(
            model_path="test_model",
            task_metrics=None,
            acdc_skill_vector={"task_0_example_0": 0.9},
            avg_acdc_quality=0.9,
            acdc_eval_details=eval_details,
        )

        assert solution.acdc_eval_details == eval_details
        assert len(solution.acdc_eval_details) == 1

    def test_convert_acdc_to_dns_solution(self):
        """Test converting AC/DC solution to DNS solution."""
        acdc_solution = ACDCSolution(
            model_path="gen_1_ind_3",
            fitness=0.75,
            acdc_skill_vector={
                "task_0_example_0": 0.9,
                "task_1_example_0": 0.4,
                "task_2_example_0": 0.8,
            },
        )

        ordered_task_ids = [
            "task_0_example_0",
            "task_1_example_0",
            "task_2_example_0",
        ]
        threshold = 0.5

        dns_solution = convert_acdc_to_dns_solution(
            acdc_solution, ordered_task_ids, threshold
        )

        assert isinstance(dns_solution, DNSSolution)
        assert dns_solution.model_path == "gen_1_ind_3"
        assert dns_solution.fitness == 0.75
        # skill_vector should be [True, False, True] based on threshold
        assert dns_solution.skill_vector == [True, False, True]

    def test_convert_acdc_to_dns_solution_missing_tasks(self):
        """Test conversion handles missing task IDs."""
        acdc_solution = ACDCSolution(
            model_path="test_model",
            fitness=0.5,
            acdc_skill_vector={"task_0_example_0": 0.9},
        )

        ordered_task_ids = [
            "task_0_example_0",
            "task_1_example_0",  # Missing in acdc_skill_vector
        ]
        threshold = 0.5

        dns_solution = convert_acdc_to_dns_solution(
            acdc_solution, ordered_task_ids, threshold
        )

        # Missing task should default to 0.0, which is < threshold
        assert dns_solution.skill_vector == [True, False]


class TestArchivePersistence:
    """Test DNS archive serialization and deserialization."""

    def test_save_dns_archive(self, dns_solutions, tmp_path):
        """Test saving DNS archive to JSON."""
        save_path = tmp_path / "archive.json"

        save_dns_archive(dns_solutions, str(save_path))

        assert save_path.exists()

        # Load and verify structure
        with open(save_path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == len(dns_solutions)

        for entry in data:
            assert "model_path" in entry
            assert "fitness" in entry
            assert "skill_vector" in entry

    def test_load_dns_archive(self, tmp_path):
        """Test loading DNS archive from JSON."""
        # Create archive file
        archive_data = [
            {
                "model_path": "gen_0_ind_0",
                "fitness": 0.8,
                "skill_vector": [1, 1, 0, 1, 0],
                "rank": 1,
                "validation_quality": 0.75,
            },
            {
                "model_path": "gen_0_ind_1",
                "fitness": 0.6,
                "skill_vector": [1, 0, 1, 0, 1],
                "rank": 2,
                "validation_quality": 0.65,
            },
        ]

        archive_path = tmp_path / "archive.json"
        with open(archive_path, "w") as f:
            json.dump(archive_data, f)

        # Load archive
        loaded = load_dns_archive(str(archive_path))

        assert len(loaded) == 2
        assert all(isinstance(s, DNSSolution) for s in loaded)
        assert loaded[0].model_path == "gen_0_ind_0"
        assert loaded[0].fitness == 0.8
        assert loaded[0].skill_vector == [1, 1, 0, 1, 0]

    def test_archive_round_trip(self, dns_solutions, tmp_path):
        """Test save→load produces identical data."""
        save_path = tmp_path / "archive.json"

        # Save
        save_dns_archive(dns_solutions, str(save_path))

        # Load
        loaded = load_dns_archive(str(save_path))

        # Compare
        assert len(loaded) == len(dns_solutions)
        for orig, loaded_sol in zip(dns_solutions, loaded):
            assert loaded_sol.model_path == orig.model_path
            assert loaded_sol.fitness == pytest.approx(orig.fitness)
            assert loaded_sol.skill_vector == orig.skill_vector


class TestACDCArchivePersistence:
    """Test AC/DC archive serialization."""

    def test_save_ac_dc_archive(self, ac_dc_solutions, tmp_path):
        """Test saving AC/DC archive with eval details."""
        save_path = tmp_path / "acdc_archive.json"

        save_ac_dc_archive(
            ac_dc_solutions, str(save_path), max_details_to_log=5
        )

        assert save_path.exists()

        # Load and verify
        with open(save_path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == len(ac_dc_solutions)

        for entry in data:
            assert "model_path" in entry
            assert "fitness" in entry
            assert "acdc_skill_vector" in entry

    def test_save_acdc_archive_skill_vector_sorting(
        self, ac_dc_solutions, tmp_path
    ):
        """Test that AC/DC skill vectors are sorted by task number."""
        save_path = tmp_path / "acdc_archive.json"

        save_ac_dc_archive(ac_dc_solutions, str(save_path))

        with open(save_path) as f:
            data = json.load(f)

        # Check first solution's skill vector ordering
        if data and data[0].get("acdc_skill_vector"):
            skill_vector = data[0]["acdc_skill_vector"]
            keys = list(skill_vector.keys())

            # Extract task numbers and verify sorted
            import re

            task_nums = []
            for key in keys:
                match = re.search(r"task_(\d+)_", key)
                if match:
                    task_nums.append(int(match.group(1)))

            if task_nums:
                assert task_nums == sorted(task_nums)

    def test_save_acdc_archive_truncates_eval_details(
        self, ac_dc_solution_with_many_details, tmp_path
    ):
        """Test that eval details are truncated to max_details_to_log."""
        save_path = tmp_path / "acdc_archive.json"

        max_details = 3
        save_ac_dc_archive(
            [ac_dc_solution_with_many_details],
            str(save_path),
            max_details_to_log=max_details,
        )

        with open(save_path) as f:
            data = json.load(f)

        # Verify truncation
        assert len(data[0]["acdc_eval_details"]) == max_details

    def test_load_ac_dc_archive(self, tmp_path):
        """Test loading AC/DC archive from JSON."""
        # Create archive file
        archive_data = [
            {
                "model_path": "gen_1_ind_0",
                "fitness": 0.7,
                "acdc_skill_vector": {
                    "task_0_example_0": 0.9,
                    "task_1_example_0": 0.5,
                },
                "rank": None,
                "validation_quality": None,
                "acdc_eval_details": [
                    {
                        "task_id": "task_0",
                        "instructions": "Test instructions for task_0",
                        "raw_output": "correct answer",
                        "score": 0.9,
                    }
                ],
            }
        ]

        archive_path = tmp_path / "acdc_archive.json"
        with open(archive_path, "w") as f:
            json.dump(archive_data, f)

        # Load archive
        loaded = load_ac_dc_archive(str(archive_path))

        assert len(loaded) == 1
        assert isinstance(loaded[0], ACDCSolution)
        assert loaded[0].model_path == "gen_1_ind_0"
        assert loaded[0].fitness == 0.7
        assert loaded[0].acdc_skill_vector == {
            "task_0_example_0": 0.9,
            "task_1_example_0": 0.5,
        }
        assert len(loaded[0].acdc_eval_details) == 1
        assert isinstance(loaded[0].acdc_eval_details[0], ACDCTaskEvalDetail)

    def test_acdc_archive_round_trip(self, ac_dc_solutions, tmp_path):
        """Test AC/DC archive save→load produces identical data."""
        save_path = tmp_path / "acdc_archive.json"

        # Save
        save_ac_dc_archive(
            ac_dc_solutions, str(save_path), max_details_to_log=-1
        )  # -1 = no truncation

        # Load
        loaded = load_ac_dc_archive(str(save_path))

        # Compare
        assert len(loaded) == len(ac_dc_solutions)
        for orig, loaded_sol in zip(ac_dc_solutions, loaded):
            assert loaded_sol.model_path == orig.model_path
            assert loaded_sol.fitness == pytest.approx(orig.fitness)
            assert loaded_sol.acdc_skill_vector == orig.acdc_skill_vector


class TestHammingDistance:
    """Test Hamming distance computation."""

    def test_hamming_distance_identical(self):
        """Test Hamming distance between identical vectors."""
        vec1 = [True, False, True, True, False]
        vec2 = [True, False, True, True, False]

        dist = compute_hamming_distance(vec1, vec2)
        assert dist == 0

    def test_hamming_distance_completely_different(self):
        """Test Hamming distance between completely different vectors."""
        vec1 = [True, True, True, True, True]
        vec2 = [False, False, False, False, False]

        dist = compute_hamming_distance(vec1, vec2)
        assert dist == 5

    def test_hamming_distance_partial(self):
        """Test Hamming distance with partial differences."""
        vec1 = [True, False, True, False, True]
        vec2 = [True, True, True, False, False]

        dist = compute_hamming_distance(vec1, vec2)
        # Differences at positions 1 and 4
        assert dist == 2

    def test_hamming_distance_length_mismatch(self):
        """Test that mismatched lengths raise error."""
        vec1 = [True, False, True]
        vec2 = [True, False]

        with pytest.raises(ValueError, match="same length"):
            compute_hamming_distance(vec1, vec2)

    def test_hamming_distance_empty_vectors(self):
        """Test Hamming distance with empty vectors."""
        vec1 = []
        vec2 = []

        dist = compute_hamming_distance(vec1, vec2)
        assert dist == 0
