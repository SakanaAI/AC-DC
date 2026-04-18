"""
Tests for Best-of-N answer selection methods.

This module tests LLM-based and reward model-based selection methods for
choosing the best answer from a population of model responses.

Covered modules:
- evaluation/single_answer_from_pop_analysis.py (LLM-based selection)
- evaluation/single_answer_from_pop_rm_based.py (RM-based selection)
"""

import pytest
import json
import os
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

from evaluation.single_answer_from_pop_analysis import (
    get_correct_model_and_sample_details_w_monarchical_llm,
    get_correct_model_and_sample_details_devide_and_conquer,
    get_correct_model_and_sample_details_answer_scoring,
    compute_accuracy_of_single_ans_from_pop,
    load_data_efficiently,
    save_data_to_file,
    get_task_to_model_results_files,
    get_single_answer_from_pop_results,
)

from evaluation.single_answer_from_pop_rm_based import (
    evaluate_response_with_reward_model,
    already_evaluated,
    get_relevant_model_details,
    get_single_ans_from_pop_results as get_single_ans_from_pop_results_rm,
    assign_scores_to_responses_in_eval_details,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_model_answers():
    """Sample model answers for testing selection methods."""
    return {
        "model_a": {
            "arguments": {
                "gen_args_0": {
                    "arg_0": "What is 2+2?",
                }
            },
            "resps": [["The answer is 4."]],
            "filtered_resps": ["4"],
            "doc": {"question": "What is 2+2?"},
        },
        "model_b": {
            "arguments": {
                "gen_args_0": {
                    "arg_0": "What is 2+2?",
                }
            },
            "resps": [["The answer is 5."]],
            "filtered_resps": ["5"],
            "doc": {"question": "What is 2+2?"},
        },
        "model_c": {
            "arguments": {
                "gen_args_0": {
                    "arg_0": "What is 2+2?",
                }
            },
            "resps": [["The answer is 4."]],
            "filtered_resps": ["4"],
            "doc": {"question": "What is 2+2?"},
        },
    }


@pytest.fixture
def sample_accuracy_results():
    """Sample results for accuracy computation."""
    return [
        {"exact_match": True, "filter": "strict-match", "model_name": "model_a"},
        {"exact_match": True, "filter": "strict-match", "model_name": "model_b"},
        {"exact_match": False, "filter": "strict-match", "model_name": "model_a"},
        {"exact_match": True, "filter": "strict-match", "model_name": "model_c"},
        {"exact_match": False, "filter": "other-match", "model_name": "model_a"},
    ]


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create a temporary directory for file I/O tests."""
    return tmp_path


@pytest.fixture
def sample_eval_details():
    """Sample eval_details for RM-based testing."""
    return {
        "sample_0": {
            "problem": "What is 2+2?",
            "generation": "The answer is 4.",
            "correct": True,
        },
        "sample_1": {
            "problem": "What is 3+3?",
            "generation": "The answer is 6.",
            "correct": True,
        },
        "sample_2": {
            "problem": "What is 5+5?",
            "generation": "The answer is 10.",
            "correct": False,
        },
    }


@pytest.fixture
def mock_reward_model():
    """Mock reward model and tokenizer."""
    mock_model = Mock()
    mock_tokenizer = Mock()

    # Mock tokenizer behavior
    mock_tokenizer.bos_token = "<s>"
    mock_tokenizer.apply_chat_template = Mock(
        side_effect=lambda conv, tokenize: f"<s>User: {conv[0]['content']}\nAssistant: {conv[1]['content']}"
    )

    # Create a dict-like mock object that has a .to() method
    class MockTokenized(dict):
        def to(self, device):
            return self

    # Mock tokenizer to return appropriate batch sizes
    def mock_tokenize(prompts_or_formatted, **kwargs):
        if isinstance(prompts_or_formatted, list):
            batch_size = len(prompts_or_formatted)
        else:
            batch_size = 1
        return MockTokenized({
            "input_ids": torch.ones(batch_size, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        })

    mock_tokenizer.side_effect = mock_tokenize

    # Mock model behavior - return different batch sizes based on input
    def mock_forward(**kwargs):
        mock_output = Mock()
        # Determine batch size from input_ids
        batch_size = kwargs.get("input_ids", torch.tensor([[1]])).shape[0]
        # Return appropriate number of scores
        if batch_size == 1:
            mock_output.logits = torch.tensor([[0.8]])
        elif batch_size == 3:
            mock_output.logits = torch.tensor([[0.8], [0.6], [0.9]])
        else:
            # Default: return batch_size scores
            mock_output.logits = torch.ones(batch_size, 1) * 0.5
        return mock_output

    mock_model.side_effect = mock_forward
    mock_model.name_or_path = "test/reward-model"

    return mock_model, mock_tokenizer


# ============================================================================
# Tests for LLM-based Selection Methods (single_answer_from_pop_analysis.py)
# ============================================================================


class TestJudgeExtractionLogic:
    """Tests for judge response extraction logic."""

    @patch("evaluation.monarchical_judge.do_request")
    def test_monarchical_extract_valid_response(
        self, mock_do_request, sample_model_answers
    ):
        """Test extraction of valid monarchical judge response."""
        # Mock the API to return a realistic response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
THOUGHT:
Looking at the answers, model 2 provides the correct calculation.

DECISION:
2
"""
        mock_do_request.return_value = mock_response

        selected_model, _ = get_correct_model_and_sample_details_w_monarchical_llm(
            sample_model_answers, use_full_answers=False
        )

        # Should extract "2" and convert to 0-indexed = 1 (model_b)
        assert selected_model == "model_b"

    @patch("evaluation.monarchical_judge.do_request")
    def test_monarchical_extract_first_model(
        self, mock_do_request, sample_model_answers
    ):
        """Test extraction when first model (index 1) is selected."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
THOUGHT:
Model 1 is correct.

DECISION:
1
"""
        mock_do_request.return_value = mock_response

        selected_model, _ = get_correct_model_and_sample_details_w_monarchical_llm(
            sample_model_answers, use_full_answers=False
        )

        # Should extract "1" and convert to 0-indexed = 0 (model_a)
        assert selected_model == "model_a"

    @patch("evaluation.monarchical_judge.do_request")
    def test_monarchical_extract_missing_decision(
        self, mock_do_request, sample_model_answers
    ):
        """Test extraction with missing DECISION tag."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
THOUGHT:
This is just a thought with no decision.
"""
        mock_do_request.return_value = mock_response

        selected_model, _ = get_correct_model_and_sample_details_w_monarchical_llm(
            sample_model_answers, use_full_answers=False
        )

        # Should fall back to random selection
        assert selected_model in sample_model_answers.keys()

    @patch("evaluation.monarchical_judge.do_request")
    def test_monarchical_extract_invalid_number(
        self, mock_do_request, sample_model_answers
    ):
        """Test extraction with invalid number format."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
THOUGHT:
Invalid format

DECISION:
not_a_number
"""
        mock_do_request.return_value = mock_response

        selected_model, _ = get_correct_model_and_sample_details_w_monarchical_llm(
            sample_model_answers, use_full_answers=False
        )

        # Should fall back to random selection
        assert selected_model in sample_model_answers.keys()

    @patch("evaluation.monarchical_judge.do_request")
    def test_monarchical_extract_out_of_bounds(
        self, mock_do_request, sample_model_answers
    ):
        """Test extraction with out-of-bounds index."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
THOUGHT:
Selecting non-existent model

DECISION:
99
"""
        mock_do_request.return_value = mock_response

        selected_model, _ = get_correct_model_and_sample_details_w_monarchical_llm(
            sample_model_answers, use_full_answers=False
        )

        # Should fall back to random selection
        assert selected_model in sample_model_answers.keys()

    @patch("evaluation.monarchical_judge.do_request")
    def test_divide_conquer_extract_answer_a(self, mock_do_request):
        """Test extraction of [[A]] from divide & conquer response."""
        two_models = {
            "model_a": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_a"]],
                "filtered_resps": ["answer_a"],
                "doc": {"question": "Test?"},
            },
            "model_b": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_b"]],
                "filtered_resps": ["answer_b"],
                "doc": {"question": "Test?"},
            },
        }

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
THOUGHT:
Assistant A provides a better answer.

DECISION:
[[A]]
"""
        mock_do_request.return_value = mock_response

        selected_model, _ = get_correct_model_and_sample_details_devide_and_conquer(
            two_models, use_full_answers=False
        )

        # Should extract [[A]] and return 0, which corresponds to whichever model was in position A
        # Note: Due to random shuffling, we can only assert it's one of the two
        assert selected_model in ["model_a", "model_b"]

    @patch("evaluation.monarchical_judge.do_request")
    def test_divide_conquer_extract_answer_b(self, mock_do_request):
        """Test extraction of [[B]] from divide & conquer response."""
        two_models = {
            "model_a": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_a"]],
                "filtered_resps": ["answer_a"],
                "doc": {"question": "Test?"},
            },
            "model_b": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_b"]],
                "filtered_resps": ["answer_b"],
                "doc": {"question": "Test?"},
            },
        }

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
THOUGHT:
Assistant B is more accurate.

DECISION:
[[B]]
"""
        mock_do_request.return_value = mock_response

        selected_model, _ = get_correct_model_and_sample_details_devide_and_conquer(
            two_models, use_full_answers=False
        )

        # Should extract [[B]] and return 1
        assert selected_model in ["model_a", "model_b"]

    @patch("evaluation.monarchical_judge.do_request")
    def test_divide_conquer_extract_missing_brackets(self, mock_do_request):
        """Test extraction with missing bracket format."""
        two_models = {
            "model_a": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_a"]],
                "filtered_resps": ["answer_a"],
                "doc": {"question": "Test?"},
            },
            "model_b": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_b"]],
                "filtered_resps": ["answer_b"],
                "doc": {"question": "Test?"},
            },
        }

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
THOUGHT:
Just saying A without brackets

DECISION:
A
"""
        mock_do_request.return_value = mock_response

        selected_model, _ = get_correct_model_and_sample_details_devide_and_conquer(
            two_models, use_full_answers=False
        )

        # Should fall back to random selection
        assert selected_model in ["model_a", "model_b"]

    @patch("evaluation.monarchical_judge.do_request")
    def test_divide_conquer_extract_invalid_format(self, mock_do_request):
        """Test extraction with completely invalid format."""
        two_models = {
            "model_a": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_a"]],
                "filtered_resps": ["answer_a"],
                "doc": {"question": "Test?"},
            },
            "model_b": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_b"]],
                "filtered_resps": ["answer_b"],
                "doc": {"question": "Test?"},
            },
        }

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
THOUGHT:
Invalid response

DECISION:
[[C]]
"""
        mock_do_request.return_value = mock_response

        selected_model, _ = get_correct_model_and_sample_details_devide_and_conquer(
            two_models, use_full_answers=False
        )

        # Should fall back to random selection
        assert selected_model in ["model_a", "model_b"]


class TestMonarchicalLLMSelection:
    """Tests for monarchical LLM-based answer selection."""

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_monarchical_llm_basic_selection(
        self, mock_ask_judge, sample_model_answers
    ):
        """Test basic monarchical LLM selection."""
        # Mock judge to select model_a (index 0)
        mock_ask_judge.return_value = (0, "Selected model A because...")

        selected_model, selected_details = (
            get_correct_model_and_sample_details_w_monarchical_llm(
                sample_model_answers, use_full_answers=False
            )
        )

        assert selected_model == "model_a"
        assert selected_details == sample_model_answers["model_a"]
        assert mock_ask_judge.called

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_monarchical_llm_all_same_answers(self, mock_ask_judge):
        """Test monarchical LLM when all models give identical answers."""
        identical_answers = {
            "model_a": {
                "arguments": {"gen_args_0": {"arg_0": "Test question"}},
                "resps": [["same"]],
                "filtered_resps": ["same"],
                "doc": {"question": "Test question"},
            },
            "model_b": {
                "arguments": {"gen_args_0": {"arg_0": "Test question"}},
                "resps": [["same"]],
                "filtered_resps": ["same"],
                "doc": {"question": "Test question"},
            },
        }

        selected_model, selected_details = (
            get_correct_model_and_sample_details_w_monarchical_llm(
                identical_answers, use_full_answers=False
            )
        )

        # Should select randomly without calling judge
        assert selected_model in ["model_a", "model_b"]
        assert not mock_ask_judge.called

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_monarchical_llm_invalid_judge_response(
        self, mock_ask_judge, sample_model_answers
    ):
        """Test monarchical LLM with invalid judge response (fallback to random)."""
        # Mock judge to return invalid index
        mock_ask_judge.return_value = (-1, "Invalid response")

        selected_model, selected_details = (
            get_correct_model_and_sample_details_w_monarchical_llm(
                sample_model_answers, use_full_answers=False
            )
        )

        # Should still select a valid model (random fallback)
        assert selected_model in sample_model_answers.keys()
        assert selected_details in sample_model_answers.values()

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_monarchical_llm_with_full_answers(
        self, mock_ask_judge, sample_model_answers
    ):
        """Test monarchical LLM selection with full answers."""
        mock_ask_judge.return_value = (1, "Selected model B")

        selected_model, selected_details = (
            get_correct_model_and_sample_details_w_monarchical_llm(
                sample_model_answers, use_full_answers=True
            )
        )

        assert selected_model == "model_b"
        assert mock_ask_judge.called

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_monarchical_llm_three_models_single_call(
        self, mock_ask_judge, sample_model_answers
    ):
        """Test monarchical LLM with 3 models requires only 1 judge call."""
        # Monarchical approach asks judge once to select from all N models
        mock_ask_judge.return_value = (2, "DECISION: 3")  # Select model_c (index 2)

        selected_model, selected_details = (
            get_correct_model_and_sample_details_w_monarchical_llm(
                sample_model_answers, use_full_answers=False
            )
        )

        # Should select model_c
        assert selected_model == "model_c"
        # Monarchical approach: only 1 judge call needed (judge selects from all N)
        assert mock_ask_judge.call_count == 1


class TestDivideAndConquerSelection:
    """Tests for divide-and-conquer answer selection."""

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_divide_and_conquer_basic(self, mock_ask_judge, sample_model_answers):
        """Test basic divide-and-conquer selection."""
        # Mock judge to prefer first model in each comparison
        mock_ask_judge.return_value = (0, "A is better")

        selected_model, selected_details = (
            get_correct_model_and_sample_details_devide_and_conquer(
                sample_model_answers, use_full_answers=False
            )
        )

        # Should select one of the models
        assert selected_model in sample_model_answers.keys()
        assert selected_details in sample_model_answers.values()
        assert mock_ask_judge.called

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_divide_and_conquer_two_models(self, mock_ask_judge):
        """Test divide-and-conquer with exactly 2 models."""
        two_models = {
            "model_a": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_a"]],
                "filtered_resps": ["answer_a"],
                "doc": {"question": "Test?"},
            },
            "model_b": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_b"]],
                "filtered_resps": ["answer_b"],
                "doc": {"question": "Test?"},
            },
        }

        # Mock judge to select first option (index 0)
        mock_ask_judge.return_value = (0, "First wins")

        selected_model, selected_details = (
            get_correct_model_and_sample_details_devide_and_conquer(
                two_models, use_full_answers=False
            )
        )

        # Should select one of the models (order depends on random shuffle)
        assert selected_model in ["model_a", "model_b"]
        assert mock_ask_judge.call_count == 1  # Only one comparison needed

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_divide_and_conquer_identical_answers(self, mock_ask_judge):
        """Test divide-and-conquer with identical answers (no judge needed)."""
        identical = {
            "model_a": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["same"]],
                "filtered_resps": ["same"],
                "doc": {"question": "Test?"},
            },
            "model_b": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["same"]],
                "filtered_resps": ["same"],
                "doc": {"question": "Test?"},
            },
        }

        selected_model, selected_details = (
            get_correct_model_and_sample_details_devide_and_conquer(
                identical, use_full_answers=False
            )
        )

        # Should select randomly without calling judge (identical answers)
        assert selected_model in ["model_a", "model_b"]

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_divide_and_conquer_invalid_judge_response(
        self, mock_ask_judge, sample_model_answers
    ):
        """Test divide-and-conquer with invalid judge response."""
        # Mock judge to return invalid index (triggers random selection)
        mock_ask_judge.return_value = (-1, "Invalid")

        selected_model, selected_details = (
            get_correct_model_and_sample_details_devide_and_conquer(
                sample_model_answers, use_full_answers=False
            )
        )

        # Should still select a valid model
        assert selected_model in sample_model_answers.keys()

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_divide_and_conquer_three_models_two_calls(
        self, mock_ask_judge, sample_model_answers
    ):
        """Test divide-and-conquer with 3 models requires 2 judge calls."""
        # Divide-and-conquer does tournament-style: 2 models battle, winner faces 3rd
        # Round 1: 2 models compete (1 call)
        # Round 2: Winner vs remaining model (1 call)
        mock_ask_judge.side_effect = [
            (0, "DECISION: [[A]]"),  # First battle winner
            (1, "DECISION: [[B]]"),  # Final battle winner
        ]

        selected_model, selected_details = (
            get_correct_model_and_sample_details_devide_and_conquer(
                sample_model_answers, use_full_answers=False
            )
        )

        # Should select one of the models
        assert selected_model in sample_model_answers.keys()
        # Divide-and-conquer with 3 models: 2 judge calls needed (tournament style)
        assert mock_ask_judge.call_count == 2


class TestAnswerScoringSelection:
    """Tests for answer scoring-based selection."""

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_answer_scoring_basic(self, mock_ask_judge, sample_model_answers):
        """Test basic answer scoring selection."""
        # Mock judge to return scores: model_a=8, model_b=5, model_c=7
        mock_ask_judge.side_effect = [
            (8, "Score: 8"),
            (5, "Score: 5"),
            (7, "Score: 7"),
        ]

        selected_model, selected_details = (
            get_correct_model_and_sample_details_answer_scoring(
                sample_model_answers, use_full_answers=False
            )
        )

        # Should select model_a (highest score)
        assert selected_model == "model_a"
        assert mock_ask_judge.call_count == 3

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_answer_scoring_tied_scores(self, mock_ask_judge):
        """Test answer scoring with tied scores."""
        models = {
            "model_a": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_a"]],
                "filtered_resps": ["answer_a"],
                "doc": {"question": "Test?"},
            },
            "model_b": {
                "arguments": {"gen_args_0": {"arg_0": "Test?"}},
                "resps": [["answer_b"]],
                "filtered_resps": ["answer_b"],
                "doc": {"question": "Test?"},
            },
        }

        # Both get same score
        mock_ask_judge.side_effect = [(5, "Score: 5"), (5, "Score: 5")]

        selected_model, selected_details = (
            get_correct_model_and_sample_details_answer_scoring(
                models, use_full_answers=False
            )
        )

        # Should select one of them (deterministic based on dict ordering)
        assert selected_model in ["model_a", "model_b"]

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_answer_scoring_invalid_scores(self, mock_ask_judge, sample_model_answers):
        """Test answer scoring with some invalid scores."""
        # Mock judge to return -1 for invalid, valid scores for others
        mock_ask_judge.side_effect = [
            (-1, "Invalid"),  # model_a gets default score 1
            (8, "Score: 8"),  # model_b gets 8
            (6, "Score: 6"),  # model_c gets 6
        ]

        selected_model, selected_details = (
            get_correct_model_and_sample_details_answer_scoring(
                sample_model_answers, use_full_answers=False
            )
        )

        # Should select model_b (highest valid score)
        assert selected_model == "model_b"


# ============================================================================
# Tests for Utility Functions
# ============================================================================


class TestComputeAccuracy:
    """Tests for compute_accuracy_of_single_ans_from_pop."""

    def test_accuracy_all_correct(self):
        """Test accuracy computation with all correct answers."""
        results = [
            {"exact_match": True, "filter": "strict-match", "model_name": "model_a"},
            {"exact_match": True, "filter": "strict-match", "model_name": "model_b"},
        ]

        accuracy, model_dist = compute_accuracy_of_single_ans_from_pop(
            results, acc_key="exact_match", filter_value="strict-match"
        )

        assert accuracy == 1.0
        assert model_dist["model_a"] == 50.0
        assert model_dist["model_b"] == 50.0

    def test_accuracy_mixed(self, sample_accuracy_results):
        """Test accuracy with mixed correct/incorrect."""
        accuracy, model_dist = compute_accuracy_of_single_ans_from_pop(
            sample_accuracy_results, acc_key="exact_match", filter_value="strict-match"
        )

        # 3 out of 4 strict-match samples are correct
        assert accuracy == 0.75
        assert "model_a" in model_dist
        assert "model_b" in model_dist
        assert "model_c" in model_dist

    def test_accuracy_all_incorrect(self):
        """Test accuracy with all incorrect answers."""
        results = [
            {"exact_match": False, "filter": "strict-match", "model_name": "model_a"},
            {"exact_match": False, "filter": "strict-match", "model_name": "model_b"},
        ]

        accuracy, model_dist = compute_accuracy_of_single_ans_from_pop(
            results, acc_key="exact_match", filter_value="strict-match"
        )

        assert accuracy == 0.0

    def test_accuracy_empty_results(self):
        """Test accuracy with empty results."""
        accuracy, model_dist = compute_accuracy_of_single_ans_from_pop(
            [], acc_key="exact_match", filter_value="strict-match"
        )

        assert accuracy == 0

    def test_accuracy_filter_value(self, sample_accuracy_results):
        """Test accuracy filtering by filter_value."""
        # Only 1 sample with "other-match" and it's incorrect
        accuracy, model_dist = compute_accuracy_of_single_ans_from_pop(
            sample_accuracy_results, acc_key="exact_match", filter_value="other-match"
        )

        assert accuracy == 0.0

    def test_accuracy_no_acc_key(self):
        """Test accuracy using 'correct' field instead of acc_key."""
        results = [
            {"correct": True, "model_name": "model_a"},
            {"correct": False, "model_name": "model_b"},
            {"correct": True, "model_name": "model_a"},
        ]

        accuracy, model_dist = compute_accuracy_of_single_ans_from_pop(
            results, acc_key=None, filter_value=None
        )

        assert accuracy == 2 / 3


class TestLoadDataEfficiently:
    """Tests for load_data_efficiently."""

    def test_load_single_file(self, temp_results_dir):
        """Test loading a single JSONL file."""
        # Create test file
        test_file = temp_results_dir / "model_a.jsonl"
        samples = [
            {"filter": "strict-match", "data": "sample1"},
            {"filter": "strict-match", "data": "sample2"},
            {"filter": "other-match", "data": "sample3"},
        ]

        with open(test_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        paths = {"model_a": str(test_file)}
        data = load_data_efficiently(paths, filter_value="strict-match")

        assert "model_a" in data
        assert len(data["model_a"]) == 2  # Only strict-match samples

    def test_load_multiple_files(self, temp_results_dir):
        """Test loading multiple JSONL files."""
        # Create multiple test files
        for model_name in ["model_a", "model_b"]:
            test_file = temp_results_dir / f"{model_name}.jsonl"
            with open(test_file, "w") as f:
                f.write(json.dumps({"filter": "strict-match", "data": model_name}) + "\n")

        paths = {
            "model_a": str(temp_results_dir / "model_a.jsonl"),
            "model_b": str(temp_results_dir / "model_b.jsonl"),
        }

        data = load_data_efficiently(paths, filter_value="strict-match")

        assert len(data) == 2
        assert "model_a" in data
        assert "model_b" in data

    def test_load_empty_file(self, temp_results_dir):
        """Test loading an empty file."""
        test_file = temp_results_dir / "empty.jsonl"
        test_file.touch()

        paths = {"empty": str(test_file)}
        data = load_data_efficiently(paths, filter_value="strict-match")

        assert data["empty"] == []

    def test_load_no_matching_filter(self, temp_results_dir):
        """Test loading when no samples match filter."""
        test_file = temp_results_dir / "model_a.jsonl"
        with open(test_file, "w") as f:
            f.write(json.dumps({"filter": "other-match", "data": "sample"}) + "\n")

        paths = {"model_a": str(test_file)}
        data = load_data_efficiently(paths, filter_value="strict-match")

        assert data["model_a"] == []


class TestSaveDataToFile:
    """Tests for save_data_to_file."""

    def test_save_new_file(self, temp_results_dir):
        """Test saving data to a new file."""
        data = {
            "benchmark1": {
                "monarchical_llm": 0.85,
                "divide_and_conquer": 0.87,
            }
        }

        save_data_to_file(
            data=data,
            path_to_save_dir=str(temp_results_dir),
            model_group_name="N8",
            overwrite_json_files=False,
            task_force_selection_method="global_skill_vector_coverage",
        )

        result_file = (
            temp_results_dir
            / "benchmark1"
            / "global_skill_vector_coverage"
            / "results_N8.json"
        )
        assert result_file.exists()

        with open(result_file, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data["monarchical_llm"] == 0.85
        assert loaded_data["divide_and_conquer"] == 0.87

    def test_save_update_existing_file(self, temp_results_dir):
        """Test updating an existing file (merge mode)."""
        # Create initial file
        result_dir = (
            temp_results_dir / "benchmark1" / "global_skill_vector_coverage"
        )
        result_dir.mkdir(parents=True)
        result_file = result_dir / "results_N8.json"

        initial_data = {"monarchical_llm": 0.80}
        with open(result_file, "w") as f:
            json.dump(initial_data, f)

        # Update with new data
        new_data = {
            "benchmark1": {
                "divide_and_conquer": 0.85,
            }
        }

        save_data_to_file(
            data=new_data,
            path_to_save_dir=str(temp_results_dir),
            model_group_name="N8",
            overwrite_json_files=False,
            task_force_selection_method="global_skill_vector_coverage",
        )

        # Check merged data
        with open(result_file, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data["monarchical_llm"] == 0.80  # Preserved
        assert loaded_data["divide_and_conquer"] == 0.85  # Added

    def test_save_overwrite_file(self, temp_results_dir):
        """Test overwriting an existing file."""
        # Create initial file
        result_dir = (
            temp_results_dir / "benchmark1" / "global_skill_vector_coverage"
        )
        result_dir.mkdir(parents=True)
        result_file = result_dir / "results_N8.json"

        initial_data = {"old_key": "old_value"}
        with open(result_file, "w") as f:
            json.dump(initial_data, f)

        # Overwrite
        new_data = {"benchmark1": {"new_key": "new_value"}}

        save_data_to_file(
            data=new_data,
            path_to_save_dir=str(temp_results_dir),
            model_group_name="N8",
            overwrite_json_files=True,
            task_force_selection_method="global_skill_vector_coverage",
        )

        # Check overwritten data
        with open(result_file, "r") as f:
            loaded_data = json.load(f)

        assert "old_key" not in loaded_data
        assert loaded_data["new_key"] == "new_value"

    def test_save_multiple_benchmarks(self, temp_results_dir):
        """Test saving data for multiple benchmarks."""
        data = {
            "benchmark1": {"metric": 0.85},
            "benchmark2": {"metric": 0.90},
        }

        save_data_to_file(
            data=data,
            path_to_save_dir=str(temp_results_dir),
            model_group_name="N8",
            overwrite_json_files=False,
            task_force_selection_method="global_skill_vector_coverage",
        )

        # Both benchmark files should exist
        for bench in ["benchmark1", "benchmark2"]:
            result_file = (
                temp_results_dir
                / bench
                / "global_skill_vector_coverage"
                / "results_N8.json"
            )
            assert result_file.exists()


class TestGetTaskToModelResultsFiles:
    """Tests for get_task_to_model_results_files."""

    def test_basic_file_discovery(self, temp_results_dir):
        """Test basic discovery of result files."""
        # Create model result directories and files
        for model_name in ["model_a", "model_b"]:
            model_dir = temp_results_dir / model_name
            model_dir.mkdir()

            # Create sample files
            (model_dir / "samples_mmlu_2025-01-01.jsonl").touch()
            (model_dir / "samples_gsm8k_2025-01-01.jsonl").touch()

        paths_to_models = [
            str(temp_results_dir / "model_a"),
            str(temp_results_dir / "model_b"),
        ]

        task_to_models = get_task_to_model_results_files(
            paths_to_models, benchmark="mmlu"
        )

        assert "samples_mmlu" in task_to_models
        assert "model_a" in task_to_models["samples_mmlu"]
        assert "model_b" in task_to_models["samples_mmlu"]

    def test_filter_llm_as_judge_files(self, temp_results_dir):
        """Test filtering out llm_as_a_judge files."""
        model_dir = temp_results_dir / "model_a"
        model_dir.mkdir()

        # Create regular and judge files
        (model_dir / "samples_mmlu_2025-01-01.jsonl").touch()
        (model_dir / "samples_mmlu_llm_as_a_judge_2025-01-01.jsonl").touch()

        paths_to_models = [str(model_dir)]

        task_to_models = get_task_to_model_results_files(
            paths_to_models, benchmark="mmlu"
        )

        # Should only have the regular file, not the judge file
        assert len(task_to_models) == 1
        assert "samples_mmlu" in task_to_models

    def test_no_matching_files(self, temp_results_dir):
        """Test when no matching files are found."""
        model_dir = temp_results_dir / "model_a"
        model_dir.mkdir()

        paths_to_models = [str(model_dir)]

        with pytest.raises(AssertionError, match="No results files found"):
            get_task_to_model_results_files(paths_to_models, benchmark="nonexistent")

    def test_multiple_tasks(self, temp_results_dir):
        """Test discovery of multiple task files."""
        model_dir = temp_results_dir / "model_a"
        model_dir.mkdir()

        # Create files for different subtasks
        (model_dir / "samples_mmlu_abstract_algebra_2025-01-01.jsonl").touch()
        (model_dir / "samples_mmlu_anatomy_2025-01-01.jsonl").touch()

        paths_to_models = [str(model_dir)]

        task_to_models = get_task_to_model_results_files(
            paths_to_models, benchmark="mmlu"
        )

        # Should have separate entries for each subtask
        assert len(task_to_models) >= 2


class TestGetSingleAnswerFromPopResults:
    """Tests for get_single_answer_from_pop_results."""

    @patch("evaluation.single_answer_from_pop_analysis.ask_judge")
    def test_parallel_processing(self, mock_ask_judge, temp_results_dir):
        """Test parallel processing of multiple samples."""
        # Create sample data
        model_to_sample_details = {
            "model_a": [
                {
                    "arguments": {"gen_args_0": {"arg_0": "Q1"}},
                    "resps": [["ans1"]],
                    "filtered_resps": ["ans1"],
                    "doc": {"question": "Q1"},
                },
                {
                    "arguments": {"gen_args_0": {"arg_0": "Q2"}},
                    "resps": [["ans2"]],
                    "filtered_resps": ["ans2"],
                    "doc": {"question": "Q2"},
                },
            ],
            "model_b": [
                {
                    "arguments": {"gen_args_0": {"arg_0": "Q1"}},
                    "resps": [["ans3"]],
                    "filtered_resps": ["ans3"],
                    "doc": {"question": "Q1"},
                },
                {
                    "arguments": {"gen_args_0": {"arg_0": "Q2"}},
                    "resps": [["ans4"]],
                    "filtered_resps": ["ans4"],
                    "doc": {"question": "Q2"},
                },
            ],
        }

        # Mock judge to always select first model
        mock_ask_judge.return_value = (0, "Selected first")

        results = get_single_answer_from_pop_results(
            model_to_sample_details=model_to_sample_details,
            selection_method="monarchical_llm",
            num_workers=2,
        )

        assert len(results) == 2  # Two samples processed
        assert all("model_name" in r for r in results)


# ============================================================================
# Tests for RM-based Methods (single_answer_from_pop_rm_based.py)
# ============================================================================


class TestEvaluateResponseWithRewardModel:
    """Tests for evaluate_response_with_reward_model."""

    def test_single_response_evaluation(self, mock_reward_model):
        """Test evaluating a single response."""
        model, tokenizer = mock_reward_model

        prompts = ["What is 2+2?"]
        responses = ["The answer is 4."]

        scores = evaluate_response_with_reward_model(
            prompts=prompts,
            responses=responses,
            tokenizer=tokenizer,
            reward_model=model,
            device="cpu",
        )

        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_batch_response_evaluation(self, mock_reward_model):
        """Test evaluating multiple responses in batch."""
        model, tokenizer = mock_reward_model

        prompts = ["What is 2+2?", "What is 3+3?", "What is 5+5?"]
        responses = ["4", "6", "10"]

        scores = evaluate_response_with_reward_model(
            prompts=prompts,
            responses=responses,
            tokenizer=tokenizer,
            reward_model=model,
            device="cpu",
        )

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)


class TestAlreadyEvaluated:
    """Tests for already_evaluated function."""

    def test_not_evaluated_missing_rm_score(self, sample_eval_details):
        """Test detection when rm_score key is missing."""
        result = already_evaluated(sample_eval_details, rm_name="test-rm")
        assert result is False

    def test_not_evaluated_missing_rm_name(self, sample_eval_details):
        """Test detection when specific RM name is missing."""
        # Add rm_score but for different model
        sample_eval_details["sample_0"]["rm_score"] = {"other-rm": 0.8}

        result = already_evaluated(sample_eval_details, rm_name="test-rm")
        assert result is False

    def test_already_evaluated(self, sample_eval_details):
        """Test detection when all samples are evaluated."""
        # Add rm_score for all samples
        for sample_details in sample_eval_details.values():
            sample_details["rm_score"] = {"test-rm": 0.8}

        result = already_evaluated(sample_eval_details, rm_name="test-rm")
        assert result is True

    def test_partially_evaluated(self, sample_eval_details):
        """Test detection when only some samples are evaluated."""
        # Only evaluate first sample
        sample_eval_details["sample_0"]["rm_score"] = {"test-rm": 0.8}

        result = already_evaluated(sample_eval_details, rm_name="test-rm")
        assert result is False


class TestGetRelevantModelDetails:
    """Tests for get_relevant_model_details."""

    def test_filter_to_subset(self, temp_results_dir):
        """Test filtering to a subset of models."""
        # Create dummy eval_details files
        for model_name in ["model_a", "model_b", "model_c"]:
            (temp_results_dir / f"{model_name}_eval_details.json").touch()

        all_paths = [
            str(temp_results_dir / f"{name}_eval_details.json")
            for name in ["model_a", "model_b", "model_c"]
        ]

        relevant = get_relevant_model_details(
            all_paths, model_names=["model_a", "model_c"]
        )

        assert len(relevant) == 2
        assert any("model_a" in p for p in relevant)
        assert any("model_c" in p for p in relevant)
        assert not any("model_b" in p for p in relevant)

    def test_all_models_match(self, temp_results_dir):
        """Test when all models match."""
        for model_name in ["model_a", "model_b"]:
            (temp_results_dir / f"{model_name}_eval_details.json").touch()

        all_paths = [
            str(temp_results_dir / f"{name}_eval_details.json")
            for name in ["model_a", "model_b"]
        ]

        relevant = get_relevant_model_details(
            all_paths, model_names=["model_a", "model_b"]
        )

        assert len(relevant) == 2

    def test_no_models_match(self, temp_results_dir):
        """Test when no models match."""
        (temp_results_dir / "model_a_eval_details.json").touch()

        all_paths = [str(temp_results_dir / "model_a_eval_details.json")]

        relevant = get_relevant_model_details(
            all_paths, model_names=["model_x", "model_y"]
        )

        assert len(relevant) == 0


class TestGetSingleAnsFromPopResultsRM:
    """Tests for RM-based get_single_ans_from_pop_results."""

    def test_select_highest_score(self):
        """Test selection of model with highest RM score."""
        model_eval_details = {
            "model_a": {
                "sample_0": {
                    "problem": "Q1",
                    "generation": "A1",
                    "rm_score": {"test-rm": 0.7},
                },
                "sample_1": {
                    "problem": "Q2",
                    "generation": "A2",
                    "rm_score": {"test-rm": 0.6},
                },
            },
            "model_b": {
                "sample_0": {
                    "problem": "Q1",
                    "generation": "B1",
                    "rm_score": {"test-rm": 0.9},
                },
                "sample_1": {
                    "problem": "Q2",
                    "generation": "B2",
                    "rm_score": {"test-rm": 0.8},
                },
            },
        }

        results = get_single_ans_from_pop_results_rm(
            model_eval_details, rm_name="test-rm"
        )

        assert len(results) == 2
        # Sample 0 should select model_b (0.9 > 0.7)
        assert results[0]["model_name"] == "model_b"
        assert results[0]["generation"] == "B1"
        # Sample 1 should select model_b (0.8 > 0.6)
        assert results[1]["model_name"] == "model_b"
        assert results[1]["generation"] == "B2"

    def test_tied_scores(self):
        """Test selection when scores are tied."""
        model_eval_details = {
            "model_a": {
                "sample_0": {
                    "problem": "Q1",
                    "generation": "A1",
                    "rm_score": {"test-rm": 0.8},
                }
            },
            "model_b": {
                "sample_0": {
                    "problem": "Q1",
                    "generation": "B1",
                    "rm_score": {"test-rm": 0.8},
                }
            },
        }

        results = get_single_ans_from_pop_results_rm(
            model_eval_details, rm_name="test-rm"
        )

        # Should deterministically select one
        assert len(results) == 1
        assert results[0]["model_name"] in ["model_a", "model_b"]


class TestAssignScoresToResponses:
    """Tests for assign_scores_to_responses_in_eval_details."""

    @patch("evaluation.single_answer_from_pop_rm_based.evaluate_response_with_reward_model")
    def test_assign_scores_new_file(
        self, mock_evaluate, temp_results_dir, sample_eval_details, mock_reward_model
    ):
        """Test assigning scores to a new eval_details file."""
        model, tokenizer = mock_reward_model

        # Create eval_details file
        eval_file = temp_results_dir / "model_a_eval_details.json"
        with open(eval_file, "w") as f:
            json.dump(sample_eval_details, f)

        # Mock scores
        mock_evaluate.return_value = [0.8, 0.7, 0.9]

        result = assign_scores_to_responses_in_eval_details(
            path_to_model_eval_details=str(eval_file),
            rm=model,
            tokenizer=tokenizer,
            device="cpu",
            batch_size=32,
        )

        # Check that scores were assigned
        for sample_details in result.values():
            assert "rm_score" in sample_details
            assert "reward-model" in sample_details["rm_score"]

    @patch("evaluation.single_answer_from_pop_rm_based.evaluate_response_with_reward_model")
    def test_skip_already_evaluated(
        self, mock_evaluate, temp_results_dir, sample_eval_details, mock_reward_model
    ):
        """Test skipping already evaluated files."""
        model, tokenizer = mock_reward_model

        # Add scores to eval_details
        for sample_details in sample_eval_details.values():
            sample_details["rm_score"] = {"reward-model": 0.8}

        eval_file = temp_results_dir / "model_a_eval_details.json"
        with open(eval_file, "w") as f:
            json.dump(sample_eval_details, f)

        result = assign_scores_to_responses_in_eval_details(
            path_to_model_eval_details=str(eval_file),
            rm=model,
            tokenizer=tokenizer,
            device="cpu",
            batch_size=32,
        )

        # Should not call evaluate since already evaluated
        assert not mock_evaluate.called


# ============================================================================
# Fixtures for parametrized tests
# ============================================================================


@pytest.fixture
def sample_skill_vectors():
    """Sample skill vectors for testing (reused from test_taskforce_selection.py)."""
    return [
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
    ]
