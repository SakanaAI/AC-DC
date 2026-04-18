"""
Tests for core coverage metrics computation functions.

Tests the get_coverage_metrics and related functions that compute
individual accuracies, coverage accuracy, majority vote, and contributions.
"""

import pytest
import json

from evaluation.coverage import (
    get_coverage_metrics,
    get_coverage_metrics_per_benchmark,
    missing_model_exists,
    get_missing_models,
    get_model_eval_path,
)


class TestMissingModelHelpers:
    """Tests for helper functions that check for missing models."""

    def test_missing_model_exists_with_none(self):
        """Test detection of None in model paths."""
        model_paths = ["/path/model1", None, "/path/model3"]
        assert missing_model_exists(model_paths) is True

    def test_missing_model_exists_all_present(self):
        """Test when all models are present."""
        model_paths = ["/path/model1", "/path/model2", "/path/model3"]
        assert missing_model_exists(model_paths) is False

    def test_missing_model_exists_empty_list(self):
        """Test with empty list."""
        assert missing_model_exists([]) is False

    def test_get_missing_models_basic(self):
        """Test identification of missing models."""
        model_paths = [
            "/path/models/gen_0_ind_0",
            "/path/models/gen_0_ind_1",
            "/path/models/gen_0_ind_2",
        ]
        model_eval_paths = [
            "/path/eval/gen_0_ind_0",
            "/path/eval/gen_0_ind_2",
        ]

        missing = get_missing_models(model_paths, model_eval_paths)
        assert missing == {"gen_0_ind_1"}

    def test_get_missing_models_none_missing(self):
        """Test when no models are missing."""
        model_paths = ["/path/models/gen_0_ind_0", "/path/models/gen_0_ind_1"]
        model_eval_paths = ["/path/eval/gen_0_ind_0", "/path/eval/gen_0_ind_1"]

        missing = get_missing_models(model_paths, model_eval_paths)
        assert missing == set()


class TestGetModelEvalPath:
    """Tests for get_model_eval_path function."""

    def test_get_model_eval_path_exists(self, tmp_path):
        """Test getting eval path when it exists."""
        # Create directory structure
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        model_dir = eval_dir / "gen_0_ind_0"
        model_dir.mkdir()

        saved_model_path = "/path/models/gen_0_ind_0"
        base_eval_path = str(eval_dir)

        result = get_model_eval_path(saved_model_path, base_eval_path)
        assert result == str(model_dir)

    def test_get_model_eval_path_not_exists(self, tmp_path):
        """Test error when eval path doesn't exist."""
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        saved_model_path = "/path/models/gen_0_ind_999"
        base_eval_path = str(eval_dir)

        with pytest.raises(ValueError, match="does not exist"):
            get_model_eval_path(saved_model_path, base_eval_path)


class TestGetCoverageMetrics:
    """Tests for get_coverage_metrics function."""

    @pytest.fixture
    def mock_eval_dirs(self, tmp_path):
        """Create mock evaluation directories with sample files."""
        eval_base = tmp_path / "eval"
        eval_base.mkdir()

        model_dirs = []
        for i in range(3):
            model_dir = eval_base / f"model_{i}"
            model_dir.mkdir()

            # Create sample files for each model
            samples = []
            for doc_id in range(5):
                # Models have different correctness patterns
                correct = (doc_id + i) % 3 != 0
                sample = {
                    "doc_id": doc_id,
                    "doc": {
                        "question": f"Question {doc_id}?",
                        "answer": f"Answer {doc_id}",
                    },
                    "target": f"Answer {doc_id}",
                    "resps": [[f"Response {doc_id}"]],
                    "filtered_resps": [
                        f"Answer {doc_id}" if correct else f"Wrong {doc_id}"
                    ],
                    "filter": "flexible_extract",
                    "exact_match": 1.0 if correct else 0.0,
                }
                samples.append(sample)

            with open(model_dir / "samples_gsm8k_llama_test.jsonl", "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            # Create results JSON
            results = {
                "results": {
                    "gsm8k_llama": {"exact_match,flexible_extract": 0.6 + (i * 0.1)}
                },
                "groups": {},
                "group_subtasks": {"gsm8k_llama": {}},
            }
            with open(model_dir / "results_gsm8k.json", "w") as f:
                json.dump(results, f)

            model_dirs.append(str(model_dir))

        return model_dirs

    def test_basic_coverage_computation(self, mock_eval_dirs, tmp_path):
        """Test basic coverage metrics computation."""
        result = get_coverage_metrics(
            model_eval_paths=mock_eval_dirs,
            benchmark_name="gsm8k_llama",
            main_metric="exact_match,flexible_extract",
            experiment_path=str(tmp_path),
        )

        # Check structure
        assert "num_models_analyzed" in result
        assert "num_unique_samples" in result
        assert "individual_accuracies" in result
        assert "coverage_accuracy" in result
        assert "majority_vote_accuracy" in result
        assert "coverage_contributions" in result
        assert "unique_contributions" in result

        # Check values
        assert result["num_models_analyzed"] == 3
        assert result["num_unique_samples"] == 5
        assert 0 <= result["coverage_accuracy"] <= 1
        assert 0 <= result["majority_vote_accuracy"] <= 1

    def test_coverage_all_correct(self, tmp_path):
        """Test when all models get everything correct."""
        eval_base = tmp_path / "eval"
        eval_base.mkdir()

        model_dirs = []
        for i in range(2):
            model_dir = eval_base / f"model_{i}"
            model_dir.mkdir()

            # All samples correct
            samples = []
            for doc_id in range(3):
                sample = {
                    "doc_id": doc_id,
                    "doc": {"question": f"Q{doc_id}", "answer": f"A{doc_id}"},
                    "target": f"A{doc_id}",
                    "resps": [[f"A{doc_id}"]],
                    "filtered_resps": [f"A{doc_id}"],
                    "filter": "flexible_extract",
                    "exact_match": 1.0,
                }
                samples.append(sample)

            with open(model_dir / "samples_gsm8k_llama_test.jsonl", "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            results = {
                "results": {"gsm8k_llama": {"exact_match,flexible_extract": 1.0}},
                "groups": {},
                "group_subtasks": {"gsm8k_llama": {}},
            }
            with open(model_dir / "results_gsm8k.json", "w") as f:
                json.dump(results, f)

            model_dirs.append(str(model_dir))

        result = get_coverage_metrics(
            model_eval_paths=model_dirs,
            benchmark_name="gsm8k_llama",
            main_metric="exact_match,flexible_extract",
            experiment_path=str(tmp_path),
        )

        assert result["coverage_accuracy"] == 1.0
        assert result["majority_vote_accuracy"] == 1.0
        assert result["coverage_correct_count"] == 3

    def test_coverage_none_correct(self, tmp_path):
        """Test when no models get anything correct."""
        eval_base = tmp_path / "eval"
        eval_base.mkdir()

        model_dirs = []
        for i in range(2):
            model_dir = eval_base / f"model_{i}"
            model_dir.mkdir()

            # All samples incorrect
            samples = []
            for doc_id in range(3):
                sample = {
                    "doc_id": doc_id,
                    "doc": {"question": f"Q{doc_id}", "answer": f"A{doc_id}"},
                    "target": f"A{doc_id}",
                    "resps": [[f"Wrong{doc_id}"]],
                    "filtered_resps": [f"Wrong{doc_id}"],
                    "filter": "flexible_extract",
                    "exact_match": 0.0,
                }
                samples.append(sample)

            with open(model_dir / "samples_gsm8k_llama_test.jsonl", "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            results = {
                "results": {"gsm8k_llama": {"exact_match,flexible_extract": 0.0}},
                "groups": {},
                "group_subtasks": {"gsm8k_llama": {}},
            }
            with open(model_dir / "results_gsm8k.json", "w") as f:
                json.dump(results, f)

            model_dirs.append(str(model_dir))

        result = get_coverage_metrics(
            model_eval_paths=model_dirs,
            benchmark_name="gsm8k_llama",
            main_metric="exact_match,flexible_extract",
            experiment_path=str(tmp_path),
        )

        assert result["coverage_accuracy"] == 0.0
        assert result["majority_vote_accuracy"] == 0.0
        assert result["coverage_correct_count"] == 0

    def test_unique_contributions(self, tmp_path):
        """Test unique contribution calculation."""
        eval_base = tmp_path / "eval"
        eval_base.mkdir()

        # Model 0: correct on samples 0, 1
        # Model 1: correct on samples 1, 2
        # Sample 0: only model 0 (unique)
        # Sample 1: both models (not unique)
        # Sample 2: only model 1 (unique)

        model_correctness = [
            [1.0, 1.0, 0.0],  # model_0
            [0.0, 1.0, 1.0],  # model_1
        ]

        model_dirs = []
        for i, correctness in enumerate(model_correctness):
            model_dir = eval_base / f"model_{i}"
            model_dir.mkdir()

            samples = []
            for doc_id, correct_val in enumerate(correctness):
                sample = {
                    "doc_id": doc_id,
                    "doc": {"question": f"Q{doc_id}", "answer": f"A{doc_id}"},
                    "target": f"A{doc_id}",
                    "resps": [[f"R{doc_id}"]],
                    "filtered_resps": [f"A{doc_id}" if correct_val else f"Wrong"],
                    "filter": "flexible_extract",
                    "exact_match": correct_val,
                }
                samples.append(sample)

            with open(model_dir / "samples_gsm8k_llama_test.jsonl", "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            results = {
                "results": {
                    "gsm8k_llama": {
                        "exact_match,flexible_extract": sum(correctness) / len(correctness)
                    }
                },
                "groups": {},
                "group_subtasks": {"gsm8k_llama": {}},
            }
            with open(model_dir / "results_gsm8k.json", "w") as f:
                json.dump(results, f)

            model_dirs.append(str(model_dir))

        result = get_coverage_metrics(
            model_eval_paths=model_dirs,
            benchmark_name="gsm8k_llama",
            main_metric="exact_match,flexible_extract",
            experiment_path=str(tmp_path),
        )

        # Each model should have 1 unique contribution
        unique_contribs = result["unique_contributions"]
        assert len(unique_contribs) == 2
        # The actual model names will be based on the directory structure

    def test_majority_vote_accuracy(self, tmp_path):
        """Test majority vote accuracy calculation."""
        eval_base = tmp_path / "eval"
        eval_base.mkdir()

        # 3 models, majority needs 2+ correct
        # Sample 0: 3 correct -> majority
        # Sample 1: 2 correct -> majority
        # Sample 2: 1 correct -> not majority
        model_correctness = [
            [1.0, 1.0, 0.0],  # model_0
            [1.0, 1.0, 0.0],  # model_1
            [1.0, 0.0, 1.0],  # model_2
        ]

        model_dirs = []
        for i, correctness in enumerate(model_correctness):
            model_dir = eval_base / f"model_{i}"
            model_dir.mkdir()

            samples = []
            for doc_id, correct_val in enumerate(correctness):
                sample = {
                    "doc_id": doc_id,
                    "doc": {"question": f"Q{doc_id}", "answer": f"A{doc_id}"},
                    "target": f"A{doc_id}",
                    "resps": [[f"R{doc_id}"]],
                    "filtered_resps": [f"A{doc_id}" if correct_val else f"Wrong"],
                    "filter": "flexible_extract",
                    "exact_match": correct_val,
                }
                samples.append(sample)

            with open(model_dir / "samples_gsm8k_llama_test.jsonl", "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            results = {
                "results": {
                    "gsm8k_llama": {"exact_match,flexible_extract": sum(correctness) / 3}
                },
                "groups": {},
                "group_subtasks": {"gsm8k_llama": {}},
            }
            with open(model_dir / "results_gsm8k.json", "w") as f:
                json.dump(results, f)

            model_dirs.append(str(model_dir))

        result = get_coverage_metrics(
            model_eval_paths=model_dirs,
            benchmark_name="gsm8k_llama",
            main_metric="exact_match,flexible_extract",
            experiment_path=str(tmp_path),
        )

        # Sample 0: 3/3 correct -> majority
        # Sample 1: 2/3 correct -> majority
        # Sample 2: 2/3 correct -> majority (models 0 and 2 got it, not model 1)
        # Wait, let me recalculate:
        # Sample 0: models 0,1,2 -> 3 correct -> majority ✓
        # Sample 1: models 0,1 -> 2 correct -> majority ✓
        # Sample 2: model 2 -> 1 correct -> not majority ✗
        assert result["majority_vote_accuracy"] == 2 / 3
        assert result["majority_vote_correct_count"] == 2

    def test_duplicate_models_raises_error(self, tmp_path):
        """Test that duplicate models in input raises a ValueError."""
        eval_base = tmp_path / "eval"
        eval_base.mkdir()

        # Create one model directory
        model_dir = eval_base / "model_0"
        model_dir.mkdir()

        samples = []
        for doc_id in range(3):
            sample = {
                "doc_id": doc_id,
                "doc": {"question": f"Q{doc_id}", "answer": f"A{doc_id}"},
                "target": f"A{doc_id}",
                "resps": [[f"R{doc_id}"]],
                "filtered_resps": [f"A{doc_id}"],
                "filter": "flexible_extract",
                "exact_match": 1.0,
            }
            samples.append(sample)

        with open(model_dir / "samples_gsm8k_llama_test.jsonl", "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        results = {
            "results": {"gsm8k_llama": {"exact_match,flexible_extract": 1.0}},
            "groups": {},
            "group_subtasks": {"gsm8k_llama": {}},
        }
        with open(model_dir / "results_gsm8k.json", "w") as f:
            json.dump(results, f)

        # Pass the same model path TWICE - this should raise an error
        model_dirs_with_duplicate = [str(model_dir), str(model_dir)]

        with pytest.raises(ValueError, match="Duplicate models detected"):
            get_coverage_metrics(
                model_eval_paths=model_dirs_with_duplicate,
                benchmark_name="gsm8k_llama",
                main_metric="exact_match,flexible_extract",
                experiment_path=str(tmp_path),
            )


class TestGetCoverageMetricsPerBenchmark:
    """Tests for get_coverage_metrics_per_benchmark function."""

    @pytest.fixture
    def multi_benchmark_eval_dirs(self, tmp_path):
        """Create evaluation directories with multiple benchmark samples."""
        eval_base = tmp_path / "eval"
        eval_base.mkdir()

        model_dirs = []
        for i in range(2):
            model_dir = eval_base / f"model_{i}"
            model_dir.mkdir()

            # GSM8K samples
            gsm8k_samples = []
            for doc_id in range(3):
                correct = (doc_id + i) % 2 == 0
                sample = {
                    "doc_id": doc_id,
                    "doc": {"question": f"Q{doc_id}", "answer": f"A{doc_id}"},
                    "target": f"A{doc_id}",
                    "resps": [[f"R{doc_id}"]],
                    "filtered_resps": [f"A{doc_id}" if correct else f"Wrong"],
                    "filter": "flexible_extract",
                    "exact_match": 1.0 if correct else 0.0,
                }
                gsm8k_samples.append(sample)

            with open(model_dir / "samples_gsm8k_llama_test.jsonl", "w") as f:
                for sample in gsm8k_samples:
                    f.write(json.dumps(sample) + "\n")

            # IFEVAL samples
            ifeval_samples = []
            for doc_id in range(3):
                correct = doc_id < 2
                sample = {
                    "doc_id": doc_id,
                    "doc": {
                        "prompt": f"Instruction {doc_id}",
                        "instruction_id_list": [f"inst_{doc_id}"],
                        "kwargs": [],
                    },
                    "target": "target",
                    "resps": [[f"R{doc_id}"]],
                    "filtered_resps": [f"filtered_{doc_id}"],
                    "filter": "none",
                    "prompt_level_strict_acc": correct,
                    "inst_level_strict_acc": [correct],
                    "prompt_level_loose_acc": correct,
                    "inst_level_loose_acc": [correct],
                }
                ifeval_samples.append(sample)

            with open(model_dir / "samples_ifeval_test.jsonl", "w") as f:
                for sample in ifeval_samples:
                    f.write(json.dumps(sample) + "\n")

            # Results
            results = {
                "results": {
                    "gsm8k_llama": {"exact_match,flexible_extract": 0.5 + (i * 0.1)},
                    "ifeval": {"prompt_level_loose_acc,none": 0.6 + (i * 0.05)},
                },
                "groups": {},
                "group_subtasks": {"gsm8k_llama": {}, "ifeval": {}},
            }
            with open(model_dir / "results_test.json", "w") as f:
                json.dump(results, f)

            model_dirs.append(str(model_dir))

        return model_dirs

    def test_multiple_benchmarks(self, multi_benchmark_eval_dirs, tmp_path):
        """Test coverage computation across multiple benchmarks."""
        main_metrics = {
            "gsm8k_llama": "exact_match,flexible_extract",
            "ifeval": "prompt_level_loose_acc,none",
        }

        result = get_coverage_metrics_per_benchmark(
            model_eval_paths=multi_benchmark_eval_dirs,
            main_metrics_per_benchmark=main_metrics,
            experiment_path=str(tmp_path),
        )

        # Check structure
        assert "gsm8k_llama" in result
        assert "ifeval" in result

        # Each benchmark should have complete metrics
        for benchmark in ["gsm8k_llama", "ifeval"]:
            assert "coverage_accuracy" in result[benchmark]
            assert "majority_vote_accuracy" in result[benchmark]
            assert "individual_accuracies" in result[benchmark]

    def test_single_benchmark(self, multi_benchmark_eval_dirs, tmp_path):
        """Test with single benchmark."""
        main_metrics = {"gsm8k_llama": "exact_match,flexible_extract"}

        result = get_coverage_metrics_per_benchmark(
            model_eval_paths=multi_benchmark_eval_dirs,
            main_metrics_per_benchmark=main_metrics,
            experiment_path=str(tmp_path),
        )

        assert len(result) == 1
        assert "gsm8k_llama" in result

    def test_duplicate_models_raises_error_per_benchmark(
        self, multi_benchmark_eval_dirs, tmp_path
    ):
        """Test that duplicate models raise error in per-benchmark function."""
        main_metrics = {"gsm8k_llama": "exact_match,flexible_extract"}

        # Use the first model twice
        model_dirs_with_duplicate = [
            multi_benchmark_eval_dirs[0],
            multi_benchmark_eval_dirs[0],
        ]

        with pytest.raises(ValueError, match="Duplicate models"):
            get_coverage_metrics_per_benchmark(
                model_eval_paths=model_dirs_with_duplicate,
                main_metrics_per_benchmark=main_metrics,
                experiment_path=str(tmp_path),
            )
