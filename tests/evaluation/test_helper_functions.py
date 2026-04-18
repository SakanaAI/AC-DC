"""
Tests for helper utility functions in evaluation/utils.py.

Tests path utilities, model identification, text processing, and archive helpers.
"""

import pytest
import json
from pathlib import Path

from evaluation.utils import (
    is_seed_model,
    get_model_name_from_lm_harness_path,
    get_model_details_from_entire_relevant_archive,
    remove_chat_template_from_question,
    get_question_and_model_answer_from_sample_details,
    get_active_task_names_up_to_gen,
)


class TestIsSeedModel:
    """Tests for is_seed_model function."""

    def test_seed_model_with_name(self):
        """Test identification of seed model with model name."""
        assert is_seed_model("gen_0_ind_Qwen2.5-7B") is True
        assert is_seed_model("gen_0_ind_Qwen2.5-7B-Instruct") is True
        assert is_seed_model("gen_0_ind_Meta-Llama-3-8B-Instruct") is True

    def test_not_seed_model_with_number(self):
        """Test that numeric indices are not seed models."""
        assert is_seed_model("gen_0_ind_0") is False
        assert is_seed_model("gen_0_ind_3") is False
        assert is_seed_model("gen_0_ind_15") is False
        assert is_seed_model("gen_1_ind_5") is False

    def test_seed_model_with_path(self):
        """Test with full paths."""
        assert is_seed_model("/path/to/models/gen_0_ind_Qwen2.5-7B") is True
        assert is_seed_model("/path/to/models/gen_0_ind_5") is False

    def test_seed_model_edge_cases(self):
        """Test edge cases in model naming."""
        # Model names with numbers in them (but not purely numeric after ind_)
        assert is_seed_model("gen_0_ind_Qwen2.5-7B") is True
        assert is_seed_model("gen_0_ind_Llama-3-8B") is True

        # Purely numeric
        assert is_seed_model("gen_0_ind_123") is False


class TestGetModelNameFromLmHarnessPath:
    """Tests for get_model_name_from_lm_harness_path function."""

    def test_with_models_separator(self):
        """Test extraction with __models__ separator."""
        path = "outputs/2025-05-27/08-31-08/eval/lm_harness/outputs__2025-05-27__08-31-08__models__gen_0_ind_10"
        result = get_model_name_from_lm_harness_path(path)
        assert result == "gen_0_ind_10"

    def test_without_models_separator(self):
        """Test extraction without __models__ separator."""
        path = "outputs/2025-05-27/08-31-08/eval/lm_harness/gen_0_ind_10"
        result = get_model_name_from_lm_harness_path(path)
        assert result == "gen_0_ind_10"

    def test_with_none_path(self):
        """Test with None path."""
        result = get_model_name_from_lm_harness_path(None)
        assert result is None

    def test_with_empty_path(self):
        """Test with empty string."""
        result = get_model_name_from_lm_harness_path("")
        assert result is None

    def test_with_complex_model_name(self):
        """Test with complex model names."""
        path = "eval/lm_harness/outputs__2025__models__gen_0_ind_Qwen2.5-7B-Instruct"
        result = get_model_name_from_lm_harness_path(path)
        assert result == "gen_0_ind_Qwen2.5-7B-Instruct"


class TestRemoveChatTemplateFromQuestion:
    """Tests for remove_chat_template_from_question function."""

    def test_basic_chat_template_removal(self):
        """Test removal of basic chat template."""
        question = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "What is 2+2?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
        )

        result = remove_chat_template_from_question(question)
        assert "What is 2+2?" in result
        assert "<|start_header_id|>" not in result
        assert "<|eot_id|>" not in result
        assert "<|begin_of_text|>" not in result

    def test_user_only_template(self):
        """Test with only user message."""
        question = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Solve this problem.<|eot_id|>"
        )

        result = remove_chat_template_from_question(question)
        assert "Solve this problem" in result
        assert "<|start_header_id|>" not in result

    def test_no_template(self):
        """Test with plain text (no template)."""
        question = "What is the capital of France?"
        result = remove_chat_template_from_question(question)
        assert result == question

    def test_multiple_turns(self):
        """Test with multiple conversation turns."""
        question = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>First question<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>First answer<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>Second question<|eot_id|>"
        )

        result = remove_chat_template_from_question(question)
        # Should extract the last user message
        assert "Second question" in result
        assert "<|start_header_id|>" not in result


class TestGetQuestionAndModelAnswer:
    """Tests for get_question_and_model_answer_from_sample_details function."""

    def test_basic_extraction(self):
        """Test basic question and answer extraction."""
        sample_details = {
            "arguments": {
                "gen_args_0": {
                    "arg_0": "<|start_header_id|>user<|end_header_id|>What is 2+2?<|eot_id|>"
                }
            },
            "resps": [["The answer is 4"]],
            "filtered_resps": ["4"],
        }

        question, answer = get_question_and_model_answer_from_sample_details(
            sample_details
        )

        assert "What is 2+2?" in question
        assert answer == "4"

    def test_full_answer_extraction(self):
        """Test extraction of full answer."""
        sample_details = {
            "arguments": {"gen_args_0": {"arg_0": "Question text"}},
            "resps": [["Long detailed answer with multiple sentences"]],
            "filtered_resps": ["Short answer"],
        }

        question, answer = get_question_and_model_answer_from_sample_details(
            sample_details, get_full_answer=True
        )

        assert answer == "Long detailed answer with multiple sentences"

    def test_nested_filtered_resps(self):
        """Test with nested list in filtered_resps (e.g., humaneval)."""
        sample_details = {
            "arguments": {"gen_args_0": {"arg_0": "Question"}},
            "resps": [["Full response"]],
            "filtered_resps": [["Nested answer"]],  # Double nested
        }

        question, answer = get_question_and_model_answer_from_sample_details(
            sample_details
        )

        assert answer == "Nested answer"


class TestGetModelDetailsFromEntireRelevantArchive:
    """Tests for get_model_details_from_entire_relevant_archive function."""

    @pytest.fixture
    def multi_gen_archives(self, tmp_path):
        """Create multi-generation archive files."""
        archive_dir = tmp_path / "archives"
        archive_dir.mkdir()

        for gen in [5, 10, 15, 20]:
            archive = []
            for ind in range(3):
                solution = {
                    "model_path": f"/models/gen_{gen}_ind_{ind}",
                    "fitness": 0.7 + (ind * 0.05),
                    "acdc_skill_vector": {
                        "task_0": 0.8,
                        "task_1": 0.6 + (ind * 0.1),
                    },
                }
                archive.append(solution)

            with open(archive_dir / f"gen{gen}_dns_archive.json", "w") as f:
                json.dump(archive, f)

        return str(archive_dir)

    def test_basic_retrieval(self, multi_gen_archives):
        """Test basic model details retrieval."""
        relevant_gens = [5, 10]
        result = get_model_details_from_entire_relevant_archive(
            archive_path=multi_gen_archives, relevant_gens=relevant_gens
        )

        # Should have models from gen 5 and 10 only
        assert len(result) == 6  # 3 models per gen * 2 gens

        # Check structure
        for key, details in result.items():
            assert "model_path" in details
            assert "fitness" in details
            assert "acdc_skill_vector" in details
            assert "gen_num" in details

    def test_filter_by_relevant_gens(self, multi_gen_archives):
        """Test filtering by relevant generations."""
        relevant_gens = [5]
        result = get_model_details_from_entire_relevant_archive(
            archive_path=multi_gen_archives, relevant_gens=relevant_gens
        )

        # Should only have models from gen 5
        assert len(result) == 3

        # All should be from gen 5
        for details in result.values():
            assert details["gen_num"] == 5

    def test_all_gens(self, multi_gen_archives):
        """Test retrieving all generations."""
        relevant_gens = [5, 10, 15, 20]
        result = get_model_details_from_entire_relevant_archive(
            archive_path=multi_gen_archives, relevant_gens=relevant_gens
        )

        assert len(result) == 12  # 3 models per gen * 4 gens

    def test_empty_relevant_gens(self, multi_gen_archives):
        """Test with empty relevant_gens list."""
        result = get_model_details_from_entire_relevant_archive(
            archive_path=multi_gen_archives, relevant_gens=[]
        )

        assert len(result) == 0


class TestGetActiveTaskNamesUpToGen:
    """Tests for get_active_task_names_up_to_gen function."""

    @pytest.fixture
    def task_pool_dir(self, tmp_path):
        """Create mock task pool directory with active_pool files."""
        exp_dir = tmp_path / "experiment"
        pool_dir = exp_dir / "generated_tasks" / "pool"
        pool_dir.mkdir(parents=True)

        # Create active pool files for different generations
        for gen in [0, 5, 10]:
            tasks = [f"/path/to/task_{i}_gen{gen}" for i in range(3)]
            with open(pool_dir / f"active_pool_gen_{gen}.json", "w") as f:
                json.dump(tasks, f)

        return str(exp_dir)

    def test_basic_retrieval(self, task_pool_dir):
        """Test basic active task names retrieval."""
        result = get_active_task_names_up_to_gen(
            experiment_dir=task_pool_dir, max_gen=5
        )

        # Should include tasks from gen 0 and 5
        assert len(result) == 6  # 3 tasks per gen * 2 gens

        # Check that task names are extracted correctly
        for task_name in result:
            assert "task_" in task_name
            assert "gen" in task_name

    def test_filter_by_max_gen(self, task_pool_dir):
        """Test filtering by max_gen."""
        result = get_active_task_names_up_to_gen(
            experiment_dir=task_pool_dir, max_gen=0
        )

        # Should only include tasks from gen 0
        assert len(result) == 3

        for task_name in result:
            assert "gen0" in task_name

    def test_all_gens(self, task_pool_dir):
        """Test including all generations."""
        result = get_active_task_names_up_to_gen(
            experiment_dir=task_pool_dir, max_gen=10
        )

        # Should include all tasks
        assert len(result) == 9  # 3 tasks per gen * 3 gens

    def test_max_gen_beyond_available(self, task_pool_dir):
        """Test max_gen higher than available generations."""
        result = get_active_task_names_up_to_gen(
            experiment_dir=task_pool_dir, max_gen=100
        )

        # Should include all available tasks
        assert len(result) == 9

    def test_empty_pool_dir(self, tmp_path):
        """Test with empty task pool directory."""
        exp_dir = tmp_path / "experiment"
        pool_dir = exp_dir / "generated_tasks" / "pool"
        pool_dir.mkdir(parents=True)

        result = get_active_task_names_up_to_gen(
            experiment_dir=str(exp_dir), max_gen=10
        )

        assert len(result) == 0
