"""
Tests for benchmark sample processing functions in evaluation/utils.py.

Tests processing of evaluation results from different benchmarks including
GSM8K, MMLU, and result aggregation functions.
"""

import pytest
import json
from pathlib import Path

from evaluation.utils import (
    process_gsm8k_samples,
    process_mmlu_samples,
    process_model_eval_results,
    process_model_metric_details,
)


class TestProcessGSM8KSamples:
    """Tests for process_gsm8k_samples function."""

    @pytest.fixture
    def gsm8k_eval_dir(self, tmp_path):
        """Create mock GSM8K evaluation directory."""
        eval_dir = tmp_path / "eval_output"
        eval_dir.mkdir()

        samples = []
        for i in range(5):
            sample = {
                "doc_id": i,
                "doc": {"question": f"Question {i}?", "answer": f"{10 * (i + 1)}"},
                "target": f"{10 * (i + 1)}",
                "resps": [[f"Let me solve this: {10 * (i + 1)}"]],
                "filtered_resps": [f"{10 * (i + 1)}" if i < 3 else f"{99}"],
                "filter": "flexible_extract",
                "exact_match": 1.0 if i < 3 else 0.0,
            }
            samples.append(sample)

        with open(eval_dir / "samples_gsm8k_llama_test.jsonl", "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        return str(eval_dir)

    def test_basic_gsm8k_processing(self, gsm8k_eval_dir):
        """Test basic GSM8K sample processing."""
        result = process_gsm8k_samples(
            eval_output_dir=gsm8k_eval_dir, filter="flexible_extract"
        )

        # Should have 5 samples
        assert len(result) == 5

        # Check structure of first sample
        assert 0 in result
        assert "correct" in result[0]
        assert "score" in result[0]
        assert "problem" in result[0]
        assert "generation" in result[0]
        assert "prediction" in result[0]
        assert "answer" in result[0]

    def test_gsm8k_correctness_parsing(self, gsm8k_eval_dir):
        """Test correctness parsing."""
        result = process_gsm8k_samples(
            eval_output_dir=gsm8k_eval_dir, filter="flexible_extract"
        )

        # First 3 should be correct
        assert result[0]["correct"] is True
        assert result[1]["correct"] is True
        assert result[2]["correct"] is True

        # Last 2 should be incorrect
        assert result[3]["correct"] is False
        assert result[4]["correct"] is False

    def test_gsm8k_score_values(self, gsm8k_eval_dir):
        """Test score values match correctness."""
        result = process_gsm8k_samples(
            eval_output_dir=gsm8k_eval_dir, filter="flexible_extract"
        )

        for doc_id, details in result.items():
            if details["correct"]:
                assert details["score"] == 1.0
            else:
                assert details["score"] == 0.0

    def test_gsm8k_filter_mismatch(self, gsm8k_eval_dir):
        """Test filter that doesn't match any samples."""
        result = process_gsm8k_samples(
            eval_output_dir=gsm8k_eval_dir, filter="nonexistent-filter"
        )

        # Should return empty dict
        assert len(result) == 0

    def test_gsm8k_version_parameter(self, tmp_path):
        """Test gsm8k_version parameter."""
        eval_dir = tmp_path / "eval_output"
        eval_dir.mkdir()

        # Create file with different version name
        samples = [
            {
                "doc_id": 0,
                "doc": {"question": "Q?", "answer": "A"},
                "target": "A",
                "resps": [["R"]],
                "filtered_resps": ["A"],
                "filter": "flexible_extract",
                "exact_match": 1.0,
            }
        ]

        with open(eval_dir / "samples_gsm8k_custom_test.jsonl", "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        result = process_gsm8k_samples(
            eval_output_dir=str(eval_dir),
            gsm8k_version="gsm8k_custom",
            filter="flexible_extract",
        )

        assert len(result) == 1


class TestProcessMMLUSamples:
    """Tests for process_mmlu_samples function."""

    @pytest.fixture
    def mmlu_eval_dir(self, tmp_path):
        """Create mock MMLU evaluation directory."""
        eval_dir = tmp_path / "eval_output"
        eval_dir.mkdir()

        subjects = ["abstract_algebra", "anatomy", "astronomy"]
        all_samples = []

        for subject_idx, subject in enumerate(subjects):
            for i in range(3):
                sample = {
                    "doc_id": i,
                    "doc": {
                        "question": f"Question {i} about {subject}",
                        "subject": subject,
                        "choices": ["A", "B", "C", "D"],
                        "answer": i % 4,
                    },
                    "target": ["A", "B", "C", "D"][i % 4],
                    "resps": [[f"Response {i}"]],
                    "filtered_resps": [["A", "B", "C", "D"][i % 4]],
                    "filter": "strict_match",
                    "acc": 1.0 if (subject_idx + i) % 2 == 0 else 0.0,
                    "exact_match": 1.0 if (subject_idx + i) % 2 == 0 else 0.0,
                }
                all_samples.append(sample)

            # Write to file
            filename = f"samples_mmlu_cot_llama_{subject}_test.jsonl"
            with open(eval_dir / filename, "w") as f:
                for sample in all_samples[-3:]:  # Last 3 samples for this subject
                    f.write(json.dumps(sample) + "\n")

        return str(eval_dir)

    def test_basic_mmlu_processing(self, mmlu_eval_dir):
        """Test basic MMLU sample processing."""
        result = process_mmlu_samples(
            eval_output_dir=mmlu_eval_dir, acc_key="exact_match"
        )

        # Should have 9 samples (3 subjects * 3 questions)
        assert len(result) == 9

        # Check structure
        first_key = next(iter(result))
        assert "correct" in result[first_key]
        assert "score" in result[first_key]
        assert "problem" in result[first_key]

    def test_mmlu_sample_id_format(self, mmlu_eval_dir):
        """Test sample ID format includes subject."""
        result = process_mmlu_samples(
            eval_output_dir=mmlu_eval_dir, acc_key="exact_match"
        )

        # Sample IDs should be in format "subject_docid"
        for sample_id in result.keys():
            assert "_" in sample_id
            subject, doc_id = sample_id.rsplit("_", 1)
            assert subject in ["abstract_algebra", "anatomy", "astronomy"]
            assert doc_id.isdigit()

    def test_mmlu_acc_key_parameter(self, mmlu_eval_dir):
        """Test different acc_key values."""
        result1 = process_mmlu_samples(
            eval_output_dir=mmlu_eval_dir, acc_key="exact_match"
        )

        result2 = process_mmlu_samples(eval_output_dir=mmlu_eval_dir, acc_key="acc")

        # Both should work (in this test they have same values)
        assert len(result1) == len(result2)


class TestProcessModelEvalResults:
    """Tests for process_model_eval_results function."""

    @pytest.fixture
    def model_results_dir(self, tmp_path):
        """Create mock model evaluation results."""
        eval_dir = tmp_path / "model_eval"
        eval_dir.mkdir()

        results = {
            "results": {
                "gsm8k_llama": {
                    "exact_match,flexible_extract": 0.75,
                    "exact_match,strict_match": 0.70,
                },
                "mmlu_cot_llama": {
                    "exact_match,strict_match": 0.80,
                    "acc,none": 0.78,
                },
            },
            "groups": {},
            "group_subtasks": {},
        }

        with open(eval_dir / "results_test.json", "w") as f:
            json.dump(results, f)

        return str(eval_dir)

    def test_basic_eval_results_processing(self, model_results_dir):
        """Test basic evaluation results processing."""
        main_metrics = {
            "gsm8k_llama": "exact_match,flexible_extract",
        }

        result = process_model_eval_results(
            eval_output_dir=model_results_dir,
            main_metrics_per_benchmark=main_metrics,
        )

        assert "gsm8k_llama" in result
        assert result["gsm8k_llama"] == 0.75

    def test_all_benchmarks_processing(self, model_results_dir):
        """Test processing all available benchmarks."""
        main_metrics = {
            "gsm8k_llama": "exact_match,flexible_extract",
            "mmlu_cot_llama": "exact_match,strict_match",
        }

        result = process_model_eval_results(
            eval_output_dir=model_results_dir,
            main_metrics_per_benchmark=main_metrics,
        )

        assert len(result) == 2
        assert result["gsm8k_llama"] == 0.75
        assert result["mmlu_cot_llama"] == 0.80

    def test_single_benchmark_processing(self, model_results_dir):
        """Test processing single benchmark."""
        main_metrics = {"gsm8k_llama": "exact_match,flexible_extract"}

        result = process_model_eval_results(
            eval_output_dir=model_results_dir,
            main_metrics_per_benchmark=main_metrics,
        )

        assert len(result) == 1
        assert "gsm8k_llama" in result


class TestProcessModelMetricDetails:
    """Tests for process_model_metric_details function."""

    @pytest.fixture
    def model_metric_details_dir(self, tmp_path):
        """Create mock detailed model metrics."""
        eval_dir = tmp_path / "model_eval"
        eval_dir.mkdir()

        results = {
            "results": {
                "mmlu_cot_llama": {
                    "exact_match,strict_match": 0.80,
                    "acc,none": 0.78,
                },
                "mmlu_cot_llama_abstract_algebra": {
                    "exact_match,strict_match": 0.75,
                },
                "mmlu_cot_llama_anatomy": {
                    "exact_match,strict_match": 0.85,
                },
            },
            "groups": {
                "mmlu_cot_llama": {
                    "exact_match,strict_match": 0.80,
                }
            },
            "group_subtasks": {
                "mmlu_cot_llama": ["abstract_algebra", "anatomy"]
            },
        }

        with open(eval_dir / "results_test.json", "w") as f:
            json.dump(results, f)

        return str(eval_dir)

    def test_basic_metric_details_processing(self, model_metric_details_dir):
        """Test basic metric details processing."""
        result = process_model_metric_details(
            eval_output_dir=model_metric_details_dir,
            benchmark_name="mmlu_cot_llama",
        )

        assert "results" in result
        assert "groups" in result
        assert "group_subtasks" in result

    def test_benchmark_specific_results(self, model_metric_details_dir):
        """Test extraction of benchmark-specific results."""
        result = process_model_metric_details(
            eval_output_dir=model_metric_details_dir,
            benchmark_name="mmlu_cot_llama",
        )

        # Results should contain the benchmark metrics
        assert "exact_match,strict_match" in result["results"]
        assert result["results"]["exact_match,strict_match"] == 0.80

    def test_group_subtasks_extraction(self, model_metric_details_dir):
        """Test extraction of group subtasks."""
        result = process_model_metric_details(
            eval_output_dir=model_metric_details_dir,
            benchmark_name="mmlu_cot_llama",
        )

        # Group subtasks should be present
        assert len(result["group_subtasks"]) == 2
        assert "abstract_algebra" in result["group_subtasks"]
        assert "anatomy" in result["group_subtasks"]

    def test_no_benchmark_name(self, model_metric_details_dir):
        """Test processing without specifying benchmark name."""
        result = process_model_metric_details(
            eval_output_dir=model_metric_details_dir,
            benchmark_name=None,
        )

        # Should return all results
        assert "results" in result
        assert "mmlu_cot_llama" in result["results"]
