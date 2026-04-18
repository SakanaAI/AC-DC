"""
To run this analysis, please make sure to first run the following script:

```bash
bash evaluation/evaluate_w_logprobs_for_fixed_models_lm_harness.sh
```

When using that script, make sure to configure the desired tasks/benchmarks and the models you want to evaluate.

After running the script, you can run this script to analyze the results.

"""

import os
import sys
import json
import numpy as np
from typing import Any, Optional
from glob import glob
from tqdm import tqdm
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from itertools import combinations
import random
import re
import argparse
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.utils import get_question_and_model_answer_from_sample_details
from evaluation.monarchical_judge import ask_judge, setup_judge_logging
from evaluation.selection_llm_prompts import (
    SELECT_FROM_N_SYSTEM_PROMPT,
    SELECT_FROM_N_USER_PROMPT,
    SELECT_FROM_2_SYSTEM_PROMPT,
    SELECT_FROM_2_USER_PROMPT,
    ANSWER_SCORING_SYSTEM_PROMPT,
    ANSWER_SCORING_USER_PROMPT,
)

# Set up basic logging to output to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to terminal
    ],
)

logger = logging.getLogger(__name__)


def compute_sequence_logprob(
    sequence_logprobs: list[dict[str, list[float, str]]],
) -> float:
    """
    Compute the log probability of a sequence of tokens.

    Args:
        sequence_logprobs: A list of dictionaries, where each dictionary maps a token to a list of log probabilities and the token itself.
            Example:
            [
                {
                    "token_id": [logprob, token_decoded],
                    "token_id": [logprob, token_decoded],
                },
                {
                    "token_id": [logprob, token_decoded],
                    "token_id": [logprob, token_decoded],
                },
                ...
            ]

    Returns:
        The log probability of the sequence of tokens.
    """
    total_logprob = 0.0

    for token_dict in sequence_logprobs:
        # Each dictionary can have multiple key-value pairs.
        # We here only care about the first key-value pair
        # which is the token that was actually used in the response.
        for token_key, logprob_data in token_dict.items():
            # logprob_data is [logprob, token], we want the logprob (first element)
            logprob = logprob_data[0]
            total_logprob += logprob
            break  # Only process the first item in the dict

    return total_logprob


def compute_sequence_self_certainty(
    sequence_logprobs: list[dict[str, list[float, str]]],
) -> float:
    """
    Compute the self-certainty of a sequence of tokens. [https://arxiv.org/abs/2502.18581]

    `-1/(nV) * sum_i sum_j log(V * p(j|x, y_<i))`, \ 
    where `n` is the number of tokens in the sequence, `V` is the vocabulary size,
    and `p(j|x, y_<i)` is the probability of the `j`-th token given the sequence `x` and the previous tokens `y_<i`.

    Note: This implementation allows for approximating the self-certainty of a sequence of tokens\
    by considering only the top `approx_V` tokens in the vocabulary.
        
    Args:
        sequence_logprobs: A list of dictionaries, where each dictionary maps a token to a list of log probabilities and the token itself.
            Example:
            [
                {
                    "token_id": [logprob, token_decoded],
                    "token_id": [logprob, token_decoded],
                },
                ...
            ]

    Returns:
        The self-certainty of the sequence of tokens.
    """

    n = len(sequence_logprobs)
    approx_V = len(sequence_logprobs[0])
    # logger.info(f"approx_V: {approx_V}, n: {n}")

    if n == 0 or approx_V == 0:
        return 0.0

    sum_results = 0.0

    # Sum over all `n` tokens in the sequence
    for token_dict in sequence_logprobs:
        # Sum over all `V` tokens in the vocabulary
        for token_id, logprob_data in token_dict.items():
            # We need log(V * p(j|x, y_<i)) = log(V) + log(p(j|x, y_<i))
            logprob_v_times_prob = np.log(approx_V) + logprob_data[0]
            sum_results += logprob_v_times_prob

    self_certainty = -sum_results / (n * approx_V)

    assert (
        self_certainty > 0
    ), f"Self-certainty is {self_certainty}, but needs to be positive"
    # assert self_certainty <= 0, f"Self-certainty is {self_certainty}, but needs to be negative"
    return self_certainty


def get_correct_model_and_sample_details_w_monarchical_llm(
    sample_details_per_model: dict[str, dict[str, Any]],
    use_full_answers: bool = False,
) -> tuple[str, dict]:
    """
    Get the correct model and sample details based on a monarchical-llm.
    """

    def construct_messages(question: str, answers: list[str]) -> list[dict]:
        """
        Construct the messages for the monarchical-llm.
        """

        assistant_submission_template = """[[START OF ASSISTANT SUBMISSION {index}]]
{answer}
[[END OF ASSISTANT SUBMISSION {index}]]
"""

        submissions = "\n".join(
            [
                assistant_submission_template.format(index=i + 1, answer=answer)
                for i, answer in enumerate(answers)
            ]
        )

        indices = [i + 1 for i in range(len(answers))]

        messages = [
            {
                "role": "system",
                "content": SELECT_FROM_N_SYSTEM_PROMPT.format(indices=indices),
            },
            {
                "role": "user",
                "content": SELECT_FROM_N_USER_PROMPT.format(
                    question=question,
                    submissions=submissions,
                ),
            },
        ]

        return messages

    def extract_judge_decision(response: str) -> int:
        """
        Extract the judge's decision from the response and return it as an integer.
        """
        try:
            selected_model_idx = response.split("DECISION:")[-1].strip()
            return (
                int(selected_model_idx) - 1
            )  # -1 because the index is 1-indexed
        except Exception as e:
            logger.warning(
                f"Error extracting judge decision: {e}\nResponse: {response}"
            )
            return -1

    # Get the question and all answers
    question = None
    model_answers = []
    all_model_names = []
    for model_name, sample_details in sample_details_per_model.items():
        question, model_answer = (
            get_question_and_model_answer_from_sample_details(
                sample_details,
                get_full_answer=use_full_answers,
            )
        )

        if question is None:
            raise ValueError(f"Question is None for {sample_details}")

        model_answers.append(model_answer)
        all_model_names.append(model_name)

    # If all answers are the exact same, then we just return a random answer
    if len(set(model_answers)) == 1:
        selected_model_idx = np.random.randint(0, len(all_model_names))
        selected_model_name = all_model_names[selected_model_idx]
        selected_sample_details = sample_details_per_model[selected_model_name]
        return selected_model_name, selected_sample_details

    # Ask the judge
    messages = construct_messages(question, model_answers)
    context_info = {
        "selection_method": "monarchical_llm",
        "use_full_answers": use_full_answers,
        "num_models": len(all_model_names),
        "question_preview": (
            question[:100] + "..." if len(question) > 100 else question
        ),
    }

    selected_model_idx, judge_full_response = ask_judge(
        messages=messages,
        extract_judge_decision=extract_judge_decision,
        context_info=context_info,
    )

    if selected_model_idx == -1 or selected_model_idx >= len(all_model_names):
        logger.warning(
            f"No correct answer found for question: {question}\n"
            "Or the selected model index is out of bounds.\n"
            "Selecting a random answer from the population.\n"
            f"Judge full response:\n{judge_full_response}"
        )
        selected_model_idx = np.random.randint(0, len(all_model_names))

    selected_model_name = all_model_names[selected_model_idx]

    # Get the correct sample details
    selected_sample_details = sample_details_per_model[selected_model_name]

    return selected_model_name, selected_sample_details

# TODO: Fix typo in function name
def get_correct_model_and_sample_details_devide_and_conquer(
    sample_details_per_model: dict[str, dict[str, Any]],
    use_full_answers: bool = False,
) -> tuple[str, dict]:
    """
    Get the correct model and sample details based on a divide and conquer approach.
    We let the judge decide the "more correct" model in a 1 vs. 1 comparison.
    The winners of each comparison are then compared again, until we have a single winner.
    """

    # Helper functions #################################################

    def construct_messages(question: str, answers: list[str]) -> list[dict]:
        """
        Construct the messages for the monarchical-llm.

        Args:
            question: The question to which we get the proposed answers.
            answers: The 2 answers to compare.

        Returns:
            messages: The messages to send to the judge.
        """

        assert len(answers) == 2, "We only support 2 answers for now"

        messages = [
            {
                "role": "system",
                "content": SELECT_FROM_2_SYSTEM_PROMPT.format(
                    question=question,
                ),
            },
            {
                "role": "user",
                "content": SELECT_FROM_2_USER_PROMPT.format(
                    question=question, answer_a=answers[0], answer_b=answers[1]
                ),
            },
        ]

        return messages

    def extract_judge_decision(response: str) -> int:
        """
        Extract the judge's decision from the response and return it as an integer.
        """
        try:
            selected_model_letter = response.split("DECISION:")[-1].strip()
            if "[[A]]" in selected_model_letter:
                return 0
            elif "[[B]]" in selected_model_letter:
                return 1
            else:
                raise ValueError(
                    f"Invalid judge decision: {selected_model_letter}"
                )
        except Exception as e:
            logger.warning(
                f"Error extracting judge decision: {e}\nResponse: {response}"
            )
            return -1

    def get_1_vs_1_battle_results(
        question: str,
        model_a_name: str,
        model_b_name: str,
        model_a_answer: str,
        model_b_answer: str,
    ) -> str:
        """
        Get the winner of a 1 vs. 1 battle between 2 models.

        Args:
            question: The question to which we get the proposed answers.
            model_a_name: The name of the first model.
            model_b_name: The name of the second model.
            model_a_answer: The answer of the first model.
            model_b_answer: The answer of the second model.

        Returns:
            The name of the winner model.
        """

        # If the answers are the same, then we just return a random winner
        if model_a_answer == model_b_answer:
            winner_name = np.random.choice([model_a_name, model_b_name])
            looser_name = (
                model_a_name if winner_name == model_b_name else model_b_name
            )
            return winner_name, looser_name

        messages = construct_messages(
            question, [model_a_answer, model_b_answer]
        )
        context_info = {
            "selection_method": "divide_and_conquer",
            "model_a": model_a_name,
            "model_b": model_b_name,
            "model_a_answer": model_a_answer,
            "model_b_answer": model_b_answer,
            "question_preview": (
                question[:100] + "..." if len(question) > 100 else question
            ),
        }

        selected_model_idx, judge_full_response = ask_judge(
            messages=messages,
            extract_judge_decision=extract_judge_decision,
            context_info=context_info,
        )

        if selected_model_idx == -1:
            # print(
            #     f"No winner found for question: {question}\nSelecting random winner."
            # )
            logger.warning(
                "No winner found in 1 vs. 1 battle. Selecting random winner."
            )
            winner_name = np.random.choice([model_a_name, model_b_name])
            looser_name = (
                model_a_name if winner_name == model_b_name else model_b_name
            )
        else:
            winner_name = (
                model_a_name if selected_model_idx == 0 else model_b_name
            )
            looser_name = (
                model_b_name if winner_name == model_a_name else model_a_name
            )
        return winner_name, looser_name

    ####################################################################

    # Get the question and all answers
    question = None
    model_names_to_answers = {}
    for model_name, sample_details in sample_details_per_model.items():
        question, model_answer = (
            get_question_and_model_answer_from_sample_details(
                sample_details,
                get_full_answer=use_full_answers,
            )
        )
        if question is None:
            raise ValueError(f"Question is None for {sample_details}")
        model_names_to_answers[model_name] = model_answer

    # Loop until we have a single winner
    while len(model_names_to_answers) > 1:
        # Select all 1 vs. 1 combinations of models w/o repeating the same model
        shuffled_model_names = np.random.permutation(
            list(model_names_to_answers.keys())
        )
        battles = []
        for i in range(0, len(shuffled_model_names) - 1, 2):
            battles.append(
                (
                    shuffled_model_names[i],
                    shuffled_model_names[i + 1],
                    model_names_to_answers[shuffled_model_names[i]],
                    model_names_to_answers[shuffled_model_names[i + 1]],
                )
            )

        # Get the results of all battles in parallel threads
        with ThreadPoolExecutor(max_workers=len(battles)) as executor:
            # with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_battle = {
                executor.submit(
                    get_1_vs_1_battle_results, question, *battle
                ): battle
                for battle in battles
            }

            # Collect results as they complete
            for future in as_completed(future_to_battle):
                winner_name, loser_name = future.result()

                # Remove the looser from the model_names_to_answers dictionary
                del model_names_to_answers[loser_name]

    # Get the winner of the final battle
    winner_name = list(model_names_to_answers.keys())[0]
    selected_model_name = winner_name
    selected_sample_details = sample_details_per_model[selected_model_name]

    return selected_model_name, selected_sample_details


def get_correct_model_and_sample_details_answer_scoring(
    sample_details_per_model: dict[str, dict[str, Any]],
    use_full_answers: bool = False,
    use_reward_model: bool = False,
) -> tuple[str, dict]:
    """
    Get the correct model and sample details based on answer scoring.
    First, we score the answers using a LLM or a reward model.
    Then, we select the answer with the highest score.
    """

    # Helper functions #################################################

    def construct_messages(question: str, answer: str) -> list[dict]:
        """
        Construct the messages for the monarchical-llm.
        """
        messages = [
            {"role": "system", "content": ANSWER_SCORING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ANSWER_SCORING_USER_PROMPT.format(
                    question=question, answer=answer
                ),
            },
        ]

        return messages

    def extract_judge_decision(response: str) -> int:
        """
        Extract the judge's decision from the response and return it as an integer.
        """
        try:
            answer_score_in_double_brackets = response.split("DECISION:")[
                -1
            ].strip()
            regex_pattern = r"\[\[([0-9]+)\]\]"
            match = re.search(regex_pattern, answer_score_in_double_brackets)
            if match:
                answer_score = match.group(1)
            else:
                raise ValueError(
                    f"No score found in {answer_score_in_double_brackets}"
                )
            return int(answer_score)
        except Exception as e:
            logger.warning(
                f"Error extracting judge decision: {e}\nResponse: {response}"
            )
            return -1

    def get_answer_score(question: str, answer: str, model_name: str) -> int:
        messages = construct_messages(question, answer)
        context_info = {
            "selection_method": "answer_scoring",
            "use_full_answers": use_full_answers,
            "use_reward_model": use_reward_model,
            "model_name": model_name,
            "question_preview": (
                question[:100] + "..." if len(question) > 100 else question
            ),
        }

        answer_score, judge_full_response = ask_judge(
            messages=messages,
            extract_judge_decision=extract_judge_decision,
            context_info=context_info,
        )
        return answer_score

    def get_answer_score_with_reward_model(
        question: str, answer: str, model_name: str
    ) -> int:
        """
        Get the score of an answer using a reward model.
        """
        raise NotImplementedError(
            "This selection method is not implemented yet"
        )

    ####################################################################

    # Get the question and all answers
    question = None
    model_names_to_answers = {}
    for model_name, sample_details in sample_details_per_model.items():
        question, model_answer = (
            get_question_and_model_answer_from_sample_details(
                sample_details,
                get_full_answer=use_full_answers,
            )
        )
        if question is None:
            raise ValueError(f"Question is None for {sample_details}")
        model_names_to_answers[model_name] = model_answer

    if use_reward_model:
        answer_extraction_fn = get_answer_score_with_reward_model
    else:
        answer_extraction_fn = get_answer_score

    # Get the scores of all answers in parallel threads
    model_names_to_scores = {}
    with ThreadPoolExecutor(
        max_workers=len(model_names_to_answers)
    ) as executor:
        future_to_model_name = {
            executor.submit(
                answer_extraction_fn, question, answer, model_name
            ): model_name
            for model_name, answer in model_names_to_answers.items()
        }
        for future in as_completed(future_to_model_name):
            model_name = future_to_model_name[future]
            score = future.result()
            if score == -1:
                logger.warning(
                    f"No score found for {model_name}. Using score 1."
                )
                # score = np.random.randint(1, 11)
                score = 1
            model_names_to_scores[model_name] = score

    # Select the answer with the highest score
    selected_model_name = max(
        model_names_to_scores, key=model_names_to_scores.get
    )
    selected_sample_details = sample_details_per_model[selected_model_name]

    return selected_model_name, selected_sample_details


def get_single_answer_from_pop(
    sample_details_per_model: dict[str, dict[str, Any]],
    selection_method: str = "logprob",
) -> tuple[str, dict]:
    """
    Get the single answer from the population of answers.
    Picks the answer based on the highest log probability or self-certainty or based on a monarchical-llm.

    Args:
        sample_details_per_model: A dictionary, where each key is a model name\
            and each value is a dictionary containing the sample details for that model for one sample.
        selection_method: The method to use for selecting the single answer from the population.
            Can be "logprob" or "self_certainty" or "monarchical_llm" or "monarchical_llm_with_full_answers".

    Returns:
        tuple[str, dict]: A tuple containing the model name and the sample details for the selected single answer.
    """

    selected_sample_details = None
    selected_model_name = None
    highest_certainty = -float("inf")

    if selection_method == "logprob" or selection_method == "self_certainty":
        for model_name, sample in sample_details_per_model.items():
            logprobs = sample.get("logprobs")[0]

            if selection_method == "logprob":
                sequence_certainty = compute_sequence_logprob(logprobs)
            elif selection_method == "self_certainty":
                sequence_certainty = compute_sequence_self_certainty(logprobs)

            if selected_sample_details is None:
                selected_model_name = model_name
                selected_sample_details = sample
                highest_certainty = sequence_certainty
            else:
                if sequence_certainty > highest_certainty:
                    selected_model_name = model_name
                    selected_sample_details = sample
                    highest_certainty = sequence_certainty
    elif selection_method == "monarchical_llm":
        selected_model_name, selected_sample_details = (
            get_correct_model_and_sample_details_w_monarchical_llm(
                sample_details_per_model=sample_details_per_model,
            )
        )
    elif selection_method == "monarchical_llm_with_full_answers":
        selected_model_name, selected_sample_details = (
            get_correct_model_and_sample_details_w_monarchical_llm(
                sample_details_per_model=sample_details_per_model,
                use_full_answers=True,
            )
        )
    elif selection_method == "divide_and_conquer":
        selected_model_name, selected_sample_details = (
            get_correct_model_and_sample_details_devide_and_conquer(
                sample_details_per_model=sample_details_per_model,
            )
        )
    elif selection_method == "divide_and_conquer_with_full_answers":
        selected_model_name, selected_sample_details = (
            get_correct_model_and_sample_details_devide_and_conquer(
                sample_details_per_model=sample_details_per_model,
                use_full_answers=True,
            )
        )
    elif selection_method == "answer_scoring_llm_based":
        selected_model_name, selected_sample_details = (
            get_correct_model_and_sample_details_answer_scoring(
                sample_details_per_model=sample_details_per_model,
                use_full_answers=False,
                use_reward_model=False,
            )
        )
    elif selection_method == "answer_scoring_llm_based_with_full_answers":
        selected_model_name, selected_sample_details = (
            get_correct_model_and_sample_details_answer_scoring(
                sample_details_per_model=sample_details_per_model,
                use_full_answers=True,
                use_reward_model=False,
            )
        )
    elif selection_method == "abs_answer_scoring_argmax_reward_model_based":
        selected_model_name, selected_sample_details = (
            get_correct_model_and_sample_details_answer_scoring(
                sample_details_per_model=sample_details_per_model,
                use_full_answers=False,
                use_reward_model=True,
            )
        )
    elif (
        selection_method
        == "abs_answer_scoring_argmax_reward_model_based_with_full_answers"
    ):
        selected_model_name, selected_sample_details = (
            get_correct_model_and_sample_details_answer_scoring(
                sample_details_per_model=sample_details_per_model,
                use_full_answers=True,
                use_reward_model=True,
            )
        )
    else:
        raise ValueError(f"Invalid selection method: {selection_method}")

    return selected_model_name, selected_sample_details


def _process_single_sample_worker(args):
    """
    Worker function to process a single sample in parallel.

    Args:
        args: Tuple containing (i, model_to_sample_details, selection_method)

    Returns:
        Tuple of (i, model_name, sample_details) where i is the original index
    """
    i, model_sample_details, selection_method = args

    model_name, sample_details = get_single_answer_from_pop(
        sample_details_per_model=model_sample_details,
        selection_method=selection_method,
    )
    sample_details["model_name"] = model_name

    return i, model_name, sample_details


def get_single_answer_from_pop_results(
    model_to_sample_details: dict[str, list[dict[str, Any]]],
    selection_method: str = "logprob",
    num_workers: int = 128,
) -> list[dict[str, Any]]:
    """
    Get the single answer from the population of answers.
    Picks the answer with the highest log probability.

    Args:
        model_to_sample_details: A dictionary, where each key is a model name\
            and each value is a list of sample details for that model.

    Returns:
        list[dict[str, Any]]: A list of sample details where each sample is the selected single answer.
    """

    num_samples = len(
        model_to_sample_details[list(model_to_sample_details.keys())[0]]
    )

    # Initialize results list with None placeholders
    single_ans_from_pop_results = [None] * num_samples

    # Prepare arguments for parallel processing
    args_list = []
    for i in range(num_samples):
        relevant_model_to_sample_details = {
            model_name: model_to_sample_details[model_name][i]
            for model_name in model_to_sample_details.keys()
        }
        args_list.append(
            (i, relevant_model_to_sample_details, selection_method)
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_process_single_sample_worker, args): args[0]
            for args in args_list
        }

        # Collect results as they complete
        # processed_samples = 0
        for future in tqdm(
            as_completed(future_to_index),
            total=num_samples,
            desc="Processing samples in parallel",
        ):
            # processed_samples += 1
            # if processed_samples == num_samples - 1:
            #     print("breakpoint")

            i, model_name, sample_details = future.result()
            single_ans_from_pop_results[i] = sample_details

    return single_ans_from_pop_results


def compute_accuracy_of_single_ans_from_pop(
    single_ans_from_pop_results: list[dict[str, Any]],
    acc_key: str = "exact_match",
    filter_value: str = "strict-match",
) -> float:
    """
    Compute the accuracy of the single answer from the population.
    """
    num_correct = 0
    num_samples = 0
    model_to_count = {}
    model_names = set()
    for result in single_ans_from_pop_results:
        model_name = result["model_name"]
        model_names.add(model_name)

        if filter_value and result.get("filter") != filter_value:
            continue

        ### Check if the answer is correct
        if acc_key:
            # lm-harness files based on acc_key
            correct = result.get(acc_key)
        else:
            # eval_details files based on correct
            correct = result.get("correct")
        if correct:
            num_correct += 1

        # Track which model contributed to the accuracy
        if model_name not in model_to_count:
            model_to_count[model_name] = 0
        model_to_count[model_name] += 1

        # Count the number of samples
        num_samples += 1

    # Make counts into percentages
    model_to_count = {
        model_name: round((count / num_samples) * 100, 2)
        for model_name, count in model_to_count.items()
    }

    # Fill up missing models with 0
    for model_name in model_names:
        if model_name not in model_to_count:
            model_to_count[model_name] = 0

    # Sort by model name
    model_to_count = dict(sorted(model_to_count.items()))

    # Compute accuracy
    if num_samples > 0:
        accuracy = num_correct / num_samples
    else:
        logger.warning(
            f"No successful samples found for {acc_key} with filter {filter_value}"
        )
        accuracy = 0

    return accuracy, model_to_count


def get_task_to_model_results_files(
    paths_to_model_results: list[str], benchmark: str
) -> dict[str, dict[str, str]]:
    """
    Get the task to model results files.

    Returns:

        Format:
        {
            "(sub)task_name": {
                "model_name": "path_to_benchmark_results_file",
                "model_name": "path_to_benchmark_results_file",
                ...
            }
        }
    """

    model_to_benchmark_results_files = {}
    model_names = set()
    for path_to_model_results in paths_to_model_results:
        model_name = os.path.basename(path_to_model_results)
        for path_to_benchmark_results in glob(
            os.path.join(path_to_model_results, f"samples_{benchmark}*.jsonl")
        ):
            results_file_name = os.path.basename(path_to_benchmark_results)
            if (
                "llm_as_a_judge" in results_file_name
                and "llm_as_a_judge" not in benchmark
            ):
                continue
            task_name = results_file_name.split("_2025")[0]
            if task_name not in model_to_benchmark_results_files:
                model_to_benchmark_results_files[task_name] = {}
            model_to_benchmark_results_files[task_name][
                model_name
            ] = path_to_benchmark_results
            model_names.add(model_name)

    assert (
        len(model_to_benchmark_results_files) > 0
    ), f"No results files found for {benchmark}"

    logger.info(f"🤖 Considering the following models: {model_names}")

    return model_to_benchmark_results_files


def load_data_efficiently(paths_to_results_files, filter_value):
    """Load all data with simple progress tracking"""
    all_data = {}
    total_files = len(paths_to_results_files)

    with tqdm(total=total_files, desc="Loading files") as pbar:
        for model_name, file_path in paths_to_results_files.items():
            all_data[model_name] = []
            with open(file_path, "r") as f:
                for line in f:
                    sample = json.loads(line)
                    if sample.get("filter") == filter_value:
                        all_data[model_name].append(sample)
            pbar.update(1)

    return all_data


def full_eval_single_ans_from_pop(
    paths_to_model_results: list[str],
    benchmark_to_acckey_and_filter_value: dict[str, tuple[str, str]],
    selection_methods: list[str],
    num_workers: int = 128,
    path_to_save_dir: str = None,
    model_group_name: str = None,
    overwrite_json_files: bool = False,
    task_force_selection_method: str = "global_skill_vector_coverage",
) -> dict[str, dict[str, float]]:
    """
    Evaluate the single answer selection from population for a given benchmark.
    """
    results = {}
    for benchmark, (
        acc_key,
        filter_value,
    ) in benchmark_to_acckey_and_filter_value.items():
        logger.info(f"🎁 Processing benchmark: {benchmark}")
        benchmark_data = {}
        single_ans_from_pop_results_per_selection_method = {
            selection_method: [] for selection_method in selection_methods
        }

        task_to_model_results_files = get_task_to_model_results_files(
            paths_to_model_results, benchmark
        )

        n_tasks_processed = 0
        n_tasks = len(task_to_model_results_files)
        for task_name in tqdm(
            task_to_model_results_files,
            desc=f"Aggregating single answers",
            leave=True,
            total=n_tasks,
        ):
            ### Get the model results files for the current task for each model
            model_to_results_file_dict = task_to_model_results_files[task_name]

            ### Load the data from the results files
            model_to_sample_details = load_data_efficiently(
                model_to_results_file_dict, filter_value
            )

            ### Get the single answer from the population for all selection methods
            n_tasks_processed += 1

            # Parallelize selection methods processing
            def process_selection_method(selection_method):
                logger.info(
                    f"🤖 Getting single answer from population for {selection_method} ({n_tasks_processed}/{n_tasks})"
                )
                return selection_method, get_single_answer_from_pop_results(
                    model_to_sample_details=model_to_sample_details,
                    selection_method=selection_method,
                    num_workers=num_workers,
                )

            # Process all selection methods in parallel
            with ThreadPoolExecutor(
                max_workers=len(selection_methods)
            ) as executor:
                future_to_selection_method = {
                    executor.submit(
                        process_selection_method, selection_method
                    ): selection_method
                    for selection_method in selection_methods
                }

                for future in as_completed(future_to_selection_method):
                    selection_method, future_results = future.result()
                    single_ans_from_pop_results_per_selection_method[
                        selection_method
                    ].extend(future_results)

        ### Compute accuracy for all selection methods
        for selection_method in selection_methods:
            logger.info(f"🔍 Computing accuracy for {selection_method}")
            acc_selection_method, model_distribution_selection_method = (
                compute_accuracy_of_single_ans_from_pop(
                    single_ans_from_pop_results_per_selection_method[
                        selection_method
                    ],
                    acc_key,
                    filter_value,
                )
            )
            benchmark_data.update(
                {
                    selection_method: acc_selection_method,
                    f"model_distribution_{selection_method}": model_distribution_selection_method,
                }
            )
            logger.info(
                f"Accuracy {selection_method}: {acc_selection_method*100:.2f}%"
            )
            logger.info("Model distribution:")
            for (
                model_name,
                count,
            ) in model_distribution_selection_method.items():
                logger.info(f"{model_name}: {count}")

        # Save the results for the current benchmark
        save_data_to_file(
            data={benchmark: benchmark_data},
            path_to_save_dir=path_to_save_dir,
            model_group_name=model_group_name,
            overwrite_json_files=overwrite_json_files,
            task_force_selection_method=task_force_selection_method,
        )

        results[benchmark] = benchmark_data

    return results


def save_data_to_file(
    data: dict,
    path_to_save_dir: str,
    model_group_name: str,
    overwrite_json_files: bool = False,
    task_force_selection_method: str = "global_skill_vector_coverage",
):
    """
    Save the data to a file.

    Args:
        data: The data to save, where the keys are the benchmark names\
            and the values are dictionaries with the results as values.
        path_to_save_dir: The path to the directory to save the data to.
        model_group_name: The name of the model group to save the data for.
        overwrite_json_files: Whether to overwrite the json files with the new results.
    """
    # Save data in seperate dirs for benchmarks
    for benchmark, results in data.items():
        logger.info(f"💾 Saving results for {benchmark} to {path_to_save_dir}")
        path_to_save_dir_benchmark = os.path.join(path_to_save_dir, benchmark)
        os.makedirs(
            os.path.join(
                path_to_save_dir_benchmark, task_force_selection_method
            ),
            exist_ok=True,
        )

        file_path = os.path.join(
            path_to_save_dir_benchmark,
            task_force_selection_method,
            f"results_{model_group_name}.json",
        )

        # if the file already exists, load it and update it otherwise, completely overwrite it
        if os.path.exists(file_path) and not overwrite_json_files:
            with open(file_path, "r") as f:
                existing_data = json.load(f)
            existing_data.update(results)
            results = existing_data

        # Save the results
        with open(file_path, "w") as f:
            json.dump(results, f)


def get_relevant_model_names(
    experiment_dir: str, task_force_selection_method: str, n_models: int
) -> list[str]:
    """
    Get the relevant model names from the experiment directory.
    """
    # Get the dir for the first benchmark to all selection methods
    path_to_selection_methods = glob(
        os.path.join(experiment_dir, "eval", "coverage", "*")
    )[0]

    # Get the top N models for the specified selection method
    path_to_results_json = os.path.join(
        path_to_selection_methods,
        task_force_selection_method,
        f"results_N{n_models}.json",
    )

    # Check if path_to_results_json exists
    if not os.path.exists(path_to_results_json):
        raise FileNotFoundError(
            f"Results json file {path_to_results_json} not found."
        )

    with open(path_to_results_json, "r") as f:
        selection_method_results = json.load(f)
    model_names = list(
        selection_method_results["coverage_contributions"].keys()
    )

    return model_names


def filter_for_already_evaluated_benchmarks(
    benchmark_to_acckey_and_filter_value: dict[str, tuple[str, str]],
    path_to_save_dir: str,
    model_group_name: str,
    task_force_selection_method: str,
    is_baseline_eval: bool = False,
) -> dict[str, tuple[str, str]]:
    """
    Filter for already evaluated benchmarks.
    Check if the SAS files for the given selection method and model_group_name already exist
    and contain the required keys (monarchical_llm, divide_and_conquer).
    "results_{model_group_name}.json"
    """
    return_dict = {}
    for benchmark, (
        acc_key,
        filter_value,
    ) in benchmark_to_acckey_and_filter_value.items():
        path_to_sas_file = os.path.join(
            path_to_save_dir,
            benchmark,
            task_force_selection_method,
            f"results_{model_group_name}.json",
        )
        if not os.path.exists(path_to_sas_file):
            # logger.info(f"SAS file for {benchmark} does not exist")
            return_dict[benchmark] = [acc_key, filter_value]
        else:
            # Check if the file contains the required keys
            try:
                with open(path_to_sas_file, "r") as f:
                    results_data = json.load(f)

                # Check if required keys exist in the JSON data
                if (
                    "monarchical_llm" not in results_data
                    or "divide_and_conquer" not in results_data
                ):
                    # logger.info(f"SAS file for {benchmark} missing required keys - adding to evaluation list")
                    return_dict[benchmark] = [acc_key, filter_value]
                # else:
                #     logger.info(
                #         f"SAS file for {benchmark} already exists with required keys - skipping..."
                #     )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Error reading SAS file {path_to_sas_file}: {e} - adding to evaluation list"
                )
                return_dict[benchmark] = [acc_key, filter_value]
    return return_dict


def filter_for_exsiting_benchmarks(
    benchmark_to_acckey_and_filter_value_full: dict[str, tuple[str, str]],
    experiment_path: str,
    is_baseline_eval: bool = False,
) -> dict[str, tuple[str, str]]:
    """
    Filter for existing benchmarks.
    """
    if not is_baseline_eval:
        all_benchmark_dirs = glob(
            os.path.join(experiment_path, "eval", "coverage", "*")
        )
    else:
        all_benchmark_dirs = glob(
            os.path.join(experiment_path, "coverage", "*")
        )

    all_benchmark_names = set(
        [
            os.path.basename(benchmark_dir)
            for benchmark_dir in all_benchmark_dirs
        ]
    )
    benchmark_to_acckey_and_filter_value = {}
    for benchmark in benchmark_to_acckey_and_filter_value_full.keys():
        if benchmark not in all_benchmark_names:
            logger.warning(
                f"Benchmark {benchmark} not found in {all_benchmark_names}"
            )
            continue
        benchmark_to_acckey_and_filter_value[benchmark] = (
            benchmark_to_acckey_and_filter_value_full[benchmark]
        )

    return benchmark_to_acckey_and_filter_value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", "-e", type=str, required=True)
    parser.add_argument(
        "--benchmarks_file", "-b", type=str, default="benchmarks_main.yaml"
    )
    # task_force_selection_method
    parser.add_argument(
        "--task_force_selection_method",
        "-t",
        type=str,
        default="global_skill_vector_coverage",
    )
    parser.add_argument(
        "--selection_methods_file",
        "-s",
        type=str,
        # default="selection_methods_main_llm_as_a_judge.yaml",
        default="selection_methods_reduced.yaml",
    )
    parser.add_argument(
        "--lm_harness_name", "-l", type=str, default="lm_harness"
    )
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--overwrite_json_files", action="store_true")
    parser.add_argument(
        "--baseline_model_names_config",
        type=str,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up judge logging at the start
    log_file = setup_judge_logging(
        "evaluation/.logs/answer_selection_judge_interactions"
    )
    logger.info(f"Judge interactions will be logged to: {log_file}")

    ### Load the benchmarks and selection methods ######################
    base_path_to_config = "evaluation/single_answer_selection_configs"
    with open(f"{base_path_to_config}/{args.benchmarks_file}", "r") as f:
        benchmark_to_acckey_and_filter_value_full = yaml.safe_load(f)

    # Check for which benchmarks actually exist
    benchmark_to_acckey_and_filter_value = filter_for_exsiting_benchmarks(
        benchmark_to_acckey_and_filter_value_full,
        args.experiment_path,
        is_baseline_eval=args.baseline_model_names_config is not None,
    )
    logger.info(
        f"Remaining benchmarks: {benchmark_to_acckey_and_filter_value.keys()}"
    )

    # Load the single answer selection methods
    with open(f"{base_path_to_config}/{args.selection_methods_file}", "r") as f:
        selection_methods = yaml.safe_load(f)
    if args.baseline_model_names_config is None:
        ####################################################################
        ### Ours ###########################################################
        ####################################################################
        path_to_save_dir = os.path.join(
            args.experiment_path, "eval/single_answer_from_pop"
        )
        os.makedirs(path_to_save_dir, exist_ok=True)

        results_dir_path = os.path.join(
            args.experiment_path, "eval", args.lm_harness_name
        )

        ## N = 8 ###########################################################
        logger.info("-" * 100)
        logger.info(
            f"Evaluating our top 8 models for {args.task_force_selection_method}"
        )
        start_time = time.time()

        model_names = get_relevant_model_names(
            experiment_dir=args.experiment_path,
            task_force_selection_method=args.task_force_selection_method,
            n_models=8,
        )
        logger.info(f"Model names: {model_names}")

        our_N8_results_dir_paths = [
            os.path.join(results_dir_path, model_name)
            for model_name in model_names
        ]

        # Filter out benchmarks that already have the SAS files
        filtered_benchmark_to_acckey_and_filter_value = (
            filter_for_already_evaluated_benchmarks(
                benchmark_to_acckey_and_filter_value,
                path_to_save_dir,
                model_group_name="N8",
                task_force_selection_method=args.task_force_selection_method,
                is_baseline_eval=False,
            )
        )

        if len(filtered_benchmark_to_acckey_and_filter_value) > 0:
            logger.info(
                f"🔍 Remaining unevaluated benchmarks: {filtered_benchmark_to_acckey_and_filter_value.keys()}"
            )
            results_our_N8 = full_eval_single_ans_from_pop(
                paths_to_model_results=our_N8_results_dir_paths,
                benchmark_to_acckey_and_filter_value=filtered_benchmark_to_acckey_and_filter_value,
                selection_methods=selection_methods,
                num_workers=args.num_workers,
                path_to_save_dir=path_to_save_dir,
                model_group_name="N8",
                overwrite_json_files=args.overwrite_json_files,
                task_force_selection_method=args.task_force_selection_method,
            )
        else:
            logger.info("🎉 SAS for N8 for all benchmarks already exists.")
        # save_data_to_file(
        #     data=results_our_N8,
        #     path_to_save_dir=path_to_save_dir,
        #     model_group_name="N8",
        #     overwrite_json_files=args.overwrite_json_files,
        # )
        end_time = time.time()
        time_taken_minutes = (end_time - start_time) / 60
        logger.info(f"🕒 Time taken: {time_taken_minutes:.2f} minutes.\n")

        ## N=3 #############################################################
        logger.info("-" * 100)
        logger.info(
            f"Evaluating our top 3 models for {args.task_force_selection_method}"
        )
        start_time = time.time()

        model_names = get_relevant_model_names(
            experiment_dir=args.experiment_path,
            task_force_selection_method=args.task_force_selection_method,
            n_models=3,
        )
        logger.info(f"Model names: {model_names}")

        our_N3_results_dir_paths = [
            os.path.join(results_dir_path, model_name)
            for model_name in model_names
        ]

        filtered_benchmark_to_acckey_and_filter_value = (
            filter_for_already_evaluated_benchmarks(
                benchmark_to_acckey_and_filter_value,
                path_to_save_dir,
                model_group_name="N3",
                task_force_selection_method=args.task_force_selection_method,
                is_baseline_eval=False,
            )
        )

        if len(filtered_benchmark_to_acckey_and_filter_value) > 0:
            logger.info(
                f"🔍 Remaining unevaluated benchmarks: {list(filtered_benchmark_to_acckey_and_filter_value.keys())}"
            )
            results_our_N3 = full_eval_single_ans_from_pop(
                paths_to_model_results=our_N3_results_dir_paths,
                benchmark_to_acckey_and_filter_value=filtered_benchmark_to_acckey_and_filter_value,
                selection_methods=selection_methods,
                num_workers=args.num_workers,
                path_to_save_dir=path_to_save_dir,
                model_group_name="N3",
                overwrite_json_files=args.overwrite_json_files,
                task_force_selection_method=args.task_force_selection_method,
            )
        else:
            logger.info("🎉 SAS for N3 for all benchmarks already exists.")
        # save_data_to_file(
        #     data=results_our_N3,
        #     path_to_save_dir=path_to_save_dir,
        #     model_group_name="N3",
        #     overwrite_json_files=args.overwrite_json_files,
        # )
        end_time = time.time()
        time_taken_minutes = (end_time - start_time) / 60
        logger.info(f"🕒 Time taken: {time_taken_minutes:.2f} minutes.\n")

    else:
        ################################################################
        ### Baselines ##################################################
        ################################################################

        results_dir_path = os.path.join(
            args.experiment_path, args.lm_harness_name
        )
        assert os.path.exists(
            results_dir_path
        ), f"Results dir {results_dir_path} not found"
        path_to_save_dir = os.path.join(
            args.experiment_path, "eval", "single_answer_from_pop"
        )

        # Baseline model names config
        base_path_to_baseline_model_names_config = (
            "evaluation/pass@kModels_configs/baseline_model_names"
        )
        assert os.path.exists(
            f"{base_path_to_baseline_model_names_config}/{args.baseline_model_names_config}"
        ), f"Baseline model names config {args.baseline_model_names_config} not found in {base_path_to_baseline_model_names_config}"
        with open(
            f"{base_path_to_baseline_model_names_config}/{args.baseline_model_names_config}",
            "r",
        ) as f:
            baseline_model_names_config = yaml.safe_load(f)

        os.makedirs(path_to_save_dir, exist_ok=True)

        # control ######################################################
        logger.info("-" * 100)
        logger.info(f"Evaluating control N=8 models")
        start_time = time.time()

        # Get paths to the control models
        path_to_first_benchmark_dir = glob(
            os.path.join(args.experiment_path, "coverage", "*")
        )[0]

        path_to_results_json = os.path.join(
            path_to_first_benchmark_dir, "control_results_N8.json"
        )

        with open(path_to_results_json, "r") as f:
            selection_method_results = json.load(f)
        model_names = list(
            selection_method_results["coverage_contributions"].keys()
        )

        control_results_dir_paths = []
        control_model_name = baseline_model_names_config["control"]

        if control_model_name is not None:
            for model_name in model_names:
                assert os.path.exists(
                    results_dir_path + f"/{control_model_name}/{model_name}"
                ), f"Results dir {results_dir_path + f'/{control_model_name}/{model_name}'} not found"
                control_results_dir_paths.append(
                    results_dir_path + f"/{control_model_name}/{model_name}"
                )

            results_control = full_eval_single_ans_from_pop(
                paths_to_model_results=control_results_dir_paths,
                benchmark_to_acckey_and_filter_value=benchmark_to_acckey_and_filter_value,
                selection_methods=selection_methods,
                num_workers=args.num_workers,
                path_to_save_dir=path_to_save_dir,
                model_group_name="control_N8",
                overwrite_json_files=args.overwrite_json_files,
            )
            # save_data_to_file(
            #     data=results_control,
            #     path_to_save_dir=path_to_save_dir,
            #     model_group_name="control_N8",
            #     overwrite_json_files=args.overwrite_json_files,
            # )
            end_time = time.time()
            time_taken_minutes = (end_time - start_time) / 60
            logger.info(f"🕒 Time taken: {time_taken_minutes:.2f} minutes.\n")

            logger.info("-" * 100)
            logger.info(f"Evaluating control N=3 models")
            start_time = time.time()
            results_control = full_eval_single_ans_from_pop(
                paths_to_model_results=control_results_dir_paths[:3],
                benchmark_to_acckey_and_filter_value=benchmark_to_acckey_and_filter_value,
                selection_methods=selection_methods,
                num_workers=args.num_workers,
                path_to_save_dir=path_to_save_dir,
                model_group_name="control_N3",
                overwrite_json_files=args.overwrite_json_files,
            )
            # save_data_to_file(
            #     data=results_control,
            #     path_to_save_dir=path_to_save_dir,
            #     model_group_name="control_N3",
            #     overwrite_json_files=args.overwrite_json_files,
            # )
            end_time = time.time()
            time_taken_minutes = (end_time - start_time) / 60
            logger.info(f"🕒 Time taken: {time_taken_minutes:.2f} minutes.\n")
        else:
            logger.warning(
                "🚨 No control models found in the baseline model names config."
            )

        # experts ######################################################
        print("-" * 100)
        logger.info(f"Evaluating expert models")
        expert_model_name_1 = baseline_model_names_config["expert_1"]
        expert_model_name_2 = baseline_model_names_config["expert_2"]
        expert_model_name_3 = baseline_model_names_config["expert_3"]
        if (
            expert_model_name_1 is not None
            and expert_model_name_2 is not None
            and expert_model_name_3 is not None
        ):
            start_time = time.time()
            experts_results_dir_paths = [
                results_dir_path
                + f"/{baseline_model_names_config['expert_1']}",
                results_dir_path
                + f"/{baseline_model_names_config['expert_2']}",
                results_dir_path
                + f"/{baseline_model_names_config['expert_3']}",
            ]
            results_experts = full_eval_single_ans_from_pop(
                paths_to_model_results=experts_results_dir_paths,
                benchmark_to_acckey_and_filter_value=benchmark_to_acckey_and_filter_value,
                selection_methods=selection_methods,
                num_workers=args.num_workers,
                path_to_save_dir=path_to_save_dir,
                model_group_name="experts_N3",
                overwrite_json_files=args.overwrite_json_files,
            )
            # save_data_to_file(
            #     data=results_experts,
            #     path_to_save_dir=path_to_save_dir,
            #     model_group_name="experts_N3",
            #     overwrite_json_files=args.overwrite_json_files,
            # )
            end_time = time.time()
            time_taken_minutes = (end_time - start_time) / 60
            logger.info(f"🕒 Time taken: {time_taken_minutes:.2f} minutes.\n")
        else:
            logger.warning(
                "🚨 No expert models found in the baseline model names config."
            )


if __name__ == "__main__":
    main()
