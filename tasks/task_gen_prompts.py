base_system_msg = """You are an expert in designing task families to assess the capabilities of a particular new large language model (LLM).
Your goal is to create novel, diverse task families that can reveal the breadth and depth of the model's skills.
"""

task_creation_system_msg = (
    base_system_msg
    + """We especially want you to uncover tasks that could reveal surprising capabilities or failures.
Successfully written task families will be added to a task archive.
In each generation, previously generated task families will be provided as context so that you can generate different tasks.
Be creative, explore different domains, and design tasks that can distinguish between models with varying levels of expertise.
Make sure each task is coherent and does not try to combine too many concepts into one.

Each task family should be designed according to the METR Standard, which requires the following Python format:

```python
class TaskFamily:
    @staticmethod
    def get_tasks() -> dict[str, dict]:
        # Returns a mapping from task names to arbitrary data that is specific to each task.
        # Each task family should contain 2 tasks in the dictionary, labelled "1" and "2".
        # ... implementation ...

    @staticmethod
    def get_instructions(t: dict) -> str:
        # This method should return a string containing initial task instructions for the agent.
        # ... implementation ...

    @staticmethod
    def score(t: dict, submission: str) -> float | None:
        # `score` is called after the agent submits a solution. Return 0 on failure and 1 on success. Do not give partial scores.
        # Throwing an exception will result in a score of 0.
        # For math or code related tasks, implements the step-by-step code to compute the correct answer given the task instructions.
        # ... implementation ...
```

When applicable, for example, for any math or code related tasks, the `score` function MUST implement the code to get the correct answer given the task instructions.
You can use the standard python operations, but also libraries like `numpy` or `sympy` to do more complex computations.
Don't forget to import them.
Also, don't forget that `sympy` returns fractions by default, so you need to convert the results it returns to floats.
Make sure to add comments to your code to explain what you are doing.
Make sure to handle different answer types, e.g. latex style answers such as \\sqrt{{a}} or \\frac{{a}}{{b}}, or text based answers, such as `p - q`, or `p + q`.

The `score` function may optionally call a helper function that calls a GPT-4 based LLM judge.
```python
# Automated LLM judge helper function
def eval_with_llm_judge(
        instructions: str,  # The instructions for the task
        submission: str,  # The submission to evaluate
        criteria: Optional[List[str]] = None,  # Optional additional criteria to evaluate the submission against
) -> bool:
    # Returns a boolean indicating whether the agent is deemed to have succeeded at the task, and meets any additional criteria.
```

You should use this function unless the task is better evaluated through code.
DO NOT RE-IMPLEMENT THIS FUNCTION!
There is no need to repeat any criteria that are already in the instructions.
One possible use of the criteria field is if you already have an answer in mind for the task.
An example usage in `score` with an optional criteria is as follows:
```python
@staticmethod
def score(t: dict, submission: str) -> float | None:
    from sandbox_eval_helper import eval_with_llm_judge
    instructions = TaskFamily.get_instructions(t)
    criteria = ["The response should include the name Alice."]
    return 1.0 if eval_with_llm_judge(instructions, submission, criteria) else 0.0
```

Additionaly, the `score` function may call the `get_function_name_to_callable` function to get a mapping from function names to their callables.
```python
@staticmethod
def get_function_name_to_callable(
    func_string: str, # A string containing one or multiple python function definitions.
) -> dict[str, Callable]:
    # Returns a dictionary mapping function names to their callables.
```

You should use this function if the task instruction requests the agent to write code.
You may use this example function as a reference for how to implement the `score` function.
Reimplement it for simple tasks, or extend it for more complex tasks.
An example usage in `score` is as follows:
```python
@staticmethod
def score(t: dict, submission: str) -> float | None:
    from sandbox_eval_helper import get_function_name_to_callable
    
    # Test cases from get_tasks
    test_input, expected_output = t["test_cases"][0]

    # Get function name to callable mapping
    function_name_to_callable = get_function_name_to_callable(
        submission
    )

    # Run test case
    try:
        # Execute the function
        submission_result = function_name_to_callable[t["expected_func_name"]](
            test_input
        )

        # Compare with expected output
        if submission_result == expected_output:
            return 1.0

    except Exception:
        # If function fails on a test case, return 0.0
        return 0.0
```

Respond precisely in the following format including the JSON start and end markers:

THOUGHT:
<THOUGHT>

RESPONSE JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly think and reason about what kind of task family you want to propose. Thoughts may also include (but are not limited to): your motivation for investigating the capability, whether you think the model will succeed or fail at this, its novelty relative to what you have already generated, how to ensure the tasks are valid, whether it is suitable to invoke an LLM judge for scoring.

In <JSON>, provide a JSON response with the following fields:
- "name_of_task": A concise, descriptive label (lowercase, no spaces, e.g., "name_capital_city").
- "description_of_task": A clear explanation of what the task entails. (e.g., "Return the capital city of a country").
- "capability_being_measured": The specific LLM capability being evaluated (e.g., knowledge, reasoning, creativity, ...).
- "estimated_human_difficulty": An estimate of the difficulty of the task on a 1-5 scale. 1 = very easy (simple factual recall), 2 = easy (basic understanding, some inference), 3 = moderate (application of knowledge, multiple steps), 4 = difficult (analysis, synthesis, creative problem-solving), 5 = very difficult (highly specialized knowledge, complex reasoning).
- "done": By default, this is set to "False". You will have {num_rounds} rounds to refine the task family but do not need to use them all. Tasks will only be saved if they are flagged "done" by the end. Do not return "True" until you are satisfied with and have received feedback on the task family.
- "task_family": The fully implemented Python code for the TaskFamily class. Write good human-readable code.
- "example_instruction": An example instruction for the task that we would expect from the output of `get_instructions`. This should be a string.

All values in the JSON should be strings.
You may only use standard Python packages and libraries to implement the tasks.
Required library imports should be included either at the top of the file or in the class method where they are used. An import at the start of the class has no effect.
DO NOT download additional data from the internet, or access the file system.
Your response will be automatically parsed and used for evaluation, so ensure all components MUST be fully implemented and adhere to the METR standard."""
)

# Version of task_creation_system_msg without suggesting the LLM judge helper
task_creation_system_msg_no_judge = (
    base_system_msg
    + """We especially want you to uncover tasks that could reveal surprising capabilities or failures.
Successfully written task families will be added to a task archive.
In each generation, previously generated task families will be provided as context so that you can generate different tasks.
Be creative, explore different domains, and design tasks that can distinguish between models with varying levels of expertise.
Make sure each task is coherent and does not try to combine too many concepts into one.

Each task family should be designed according to the METR Standard, which requires the following Python format:

```python
class TaskFamily:
    @staticmethod
    def get_tasks() -> dict[str, dict]:
        # Returns a mapping from task names to arbitrary data that is specific to each task.
        # Each task family should contain 2 tasks in the dictionary, labelled "1" and "2".
        # ... implementation ...

    @staticmethod
    def get_instructions(t: dict) -> str:
        # This method should return a string containing initial task instructions for the agent.
        # ... implementation ...

    @staticmethod
    def score(t: dict, submission: str) -> float | None:
        # `score` is called after the agent submits a solution. Return 0 on failure and 1 on success. Do not give partial scores.
        # Throwing an exception will result in a score of 0.
        # ... implementation ...
```

Respond precisely in the following format including the JSON start and end markers:

THOUGHT:
<THOUGHT>

RESPONSE JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly think and reason about what kind of task family you want to propose. Thoughts may also include (but are not limited to): your motivation for investigating the capability, whether you think the model will succeed or fail at this, its novelty relative to what you have already generated, how to ensure the tasks are valid.

In <JSON>, provide a JSON response with the following fields:
- "name_of_task": A concise, descriptive label (lowercase, no spaces, e.g., "name_capital_city").
- "description_of_task": A clear explanation of what the task entails. (e.g., "Return the capital city of a country").
- "capability_being_measured": The specific LLM capability being evaluated (e.g., knowledge, reasoning, creativity, ...).
- "estimated_human_difficulty": An estimate of the difficulty of the task on a 1-5 scale. 1 = very easy (simple factual recall), 2 = easy (basic understanding, some inference), 3 = moderate (application of knowledge, multiple steps), 4 = difficult (analysis, synthesis, creative problem-solving), 5 = very difficult (highly specialized knowledge, complex reasoning).
- "done": By default, this is set to "False". You will have {num_rounds} rounds to refine the task family but do not need to use them all. Tasks will only be saved if they are flagged "done" by the end. Do not return "True" until you are satisfied with and have received feedback on the task family.
- "task_family": The fully implemented Python code for the TaskFamily class. Write good human-readable code.
- "example_instruction": An example instruction for the task that we would expect from the output of `get_instructions`. This should be a string.

All values in the JSON should be strings.
You may only use standard Python packages and libraries to implement the tasks.
Required library imports should be included either at the top of the file or in the class method where they are used. An import at the start of the class has no effect.
DO NOT download additional data from the internet, or access the file system.
Your response will be automatically parsed and used for evaluation, so ensure all components MUST be fully implemented and adhere to the METR standard."""
)


initial_task_prompt_original = '''A previous generated task family that the agent succeeded at is provided below (with code):
"""
{prev_json}
"""

Summaries of other previously generated tasks for context are:
"""
{other_task_jsons}
"""

Remember if a previous task failed, either the agent couldn't solve the task or the task was incorrectly implemented.
Generate the next interestingly new task family.'''

# For initialization, there is no other similar task jsons used, so it doesn't make sense to include
initial_task_prompt = '''A previous generated task family is provided below (with code):
"""
{prev_json}
"""

Remember if a previous task failed, either the agent couldn't solve the task or the task was incorrectly implemented.
Generate the next interestingly new task family.'''

old_initial_task_gen_prompt_completely_novel = '''A previous generated task family is provided below (with code):
"""
{prev_json}
"""

Remember if a previous task failed, either the agent couldn't solve the task or the task was incorrectly implemented.
Generate the next interestingly new task family.'''

initial_task_gen_prompt_completely_novel = '''A previous generated task family is provided below (with code):
"""
{prev_json}
"""

Generate the next interestingly new task family.'''

old_initial_task_gen_prompt_adapt_similar = '''A previous generated task family is provided below (with code):
"""
{prev_json}
"""

Remember if a previous task failed, either the agent couldn't solve the task or the task was incorrectly implemented.
Inspired by the previous task family, generate the next new task family that provides a more interesting challenge that is more complex but still related to the previous task family.'''

initial_task_gen_prompt_adapt_similar = '''A previous generated task family is provided below (with code):
"""
{prev_json}
"""

Generate a new task family that is inspired by the previous task family, so that it provides a more interesting challenge that is more complex or explores beyond what the current task family is evaluating in terms of model capabilities.'''

# original prompt
task_creation_reflexion_prompt = '''Current round = {current_round}/{num_rounds}.
In your thoughts, first carefully consider the correctness of the task family you just created. Factors may include but are not limited to:
- Whether sufficient synthetic data has been given in the instructions to challenge the models.
- Incomplete class definitions, syntax errors, misplaced import statements.
- Feasibility of the evaluation criteria.
- Any ambiguity or issues in the clarity of the tasks.
- The consistency and accuracy of all the fields in the JSON.
- Whether precisely two entries in `get_tasks` have been defined.
- Whether agents can get a successful score in an unintended manner.
- Whether the desired response format for the agent has been given in the instructions.
- Under no circumstances should the instructions contain the answer or any direct hints leading to the answer. Ensure that the task remains challenging without giving away the solution.
- If the agent fails at the task, it is OK as long as the task is valid and the agent's failure is due to the difficulty of the task. You are also trying to find surprising failures!

Here is the result of attempting to evaluate the tasks on the given model:
"""
{eval_response}
"""

In the next attempt, try and improve on the implementation.
Respond in the same JSON format as before.
If there is nothing to improve, simply repeat the previous response EXACTLY after the thought and set the "done" flag to "True".
Do not set the "done" flag to "True" if you are still improving the task on the current turn.
Do not set the "done" flag to "True" on the final step if there are major issues with the task family.'''
# Our own prompt with automatic answer extraction.
# task_creation_reflexion_prompt = '''Current round = {current_round}/{num_rounds}.
# In your thoughts, first carefully consider the correctness of the task family you just created. Factors may include but are not limited to:
# - Whether sufficient synthetic data has been given in the instructions to challenge the models.
# - Incomplete class definitions, syntax errors, misplaced import statements.
# - Feasibility of the evaluation criteria.
# - Any ambiguity or issues in the clarity of the tasks.
# - The consistency and accuracy of all the fields in the JSON.
# - Whether precisely two entries in `get_tasks` have been defined.
# - Whether agents can get a successful score in an unintended manner.
# - Whether the desired response format for the agent has been given in the instructions.
# - Under no circumstances should the instructions contain the answer or any direct hints leading to the answer. Ensure that the task remains challenging without giving away the solution.
# - If the agent fails at the task, it is OK as long as the task is valid and the agent's failure is due to the difficulty of the task. You are also trying to find surprising failures!

# Here is the result of attempting to evaluate the tasks on the given model:
# """
# {eval_response}
# """

# Did the agent pass the task, i.e. achieve a good score?
# If not, have a very close look at the agent's answer.
# If the angents text answer should is correct and should have gotten a good score, then you need to improve the parsing and answer extraction of the simple text submission.
# It might also be usefull, to request the agent to provide the final answer in a specific format, e.g. "Answer: <answer>", or, for math tasks "\\boxed{{answer}}".
# Make VERY CLEAR that the agent HAS TO FOLLOW THE FORMAT EXACTLY!
# This is VERY IMPORTANT!

# In the next attempt, try and improve on the implementation.
# Respond in the same JSON format as before.
# If there is nothing to improve, simply repeat the previous response EXACTLY after the thought and set the "done" flag to "True".
# Do not set the "done" flag to "True" if you are still improving the task on the current turn.
# Do not set the "done" flag to "True" on the final step if there are major issues with the task family.'''

task_creation_reflexion_multi_response_prompt = '''Current round = {current_round}/{num_rounds}.
In your thoughts, first carefully consider the correctness of the task family you just created. Factors may include but are not limited to:
- Whether sufficient synthetic data has been given in the instructions to challenge the models.
- Incomplete class definitions, syntax errors, misplaced import statements.
- Feasibility of the evaluation criteria.
- Any ambiguity or issues in the clarity of the tasks.
- The consistency and accuracy of all the fields in the JSON.
- Whether precisely two entries in `get_tasks` have been defined.
- Whether agents can get a successful score in an unintended manner.
- Whether the desired response format for the agent has been given in the instructions.
- Under no circumstances should the instructions contain the answer or any direct hints leading to the answer. Ensure that the task remains challenging without giving away the solution.
- If the agent fails at the task, it is OK as long as the task is valid and the agent's failure is due to the difficulty of the task. You are also trying to find surprising failures!

Here are some responses from a sample of models on the primary task instance:
"""
{population_responses_details}
"""

In the next attempt, try and improve on the implementation based on the evaluation summary and the diversity/quality of population responses.
Respond in the same JSON format as before.
If there is nothing to improve, simply repeat the previous response EXACTLY after the thought and set the "done" flag to "True".
Do not set the "done" flag to "True" if you are still improving the task on the current turn.
Do not set the "done" flag to "True" on the final step if there are major issues with the task family.'''

eval_zs_system_msg = """You are a helpful assistant.
You are given a task to complete.

Please provide your final answer directly starting with 'Answer: '.
Only the text after 'Answer: ' will be evaluated. This will be automatically parsed so ensure you follow the format exactly.
Respect any other output constraints given in the task prompt after 'Answer: '.
"""

eval_cot_system_msg_old = """You are a helpful assistant.
You are given a task to complete.

First, reason about the problem and your plan for solving the task, and then provide your final answer starting with 'Answer: '.
Only the text after 'Answer: ' will be evaluated. This will be automatically parsed so ensure you follow the format exactly.
Respect any other output constraints given in the task prompt after 'Answer: '.
"""

eval_cot_system_msg = """You are a helpful assistant.
You are given a task to complete.

First, reason about the problem and your plan for solving the task, and then provide your final answer starting with 'Answer: '.
If you are asked to write code, write the code after the 'Answer: ' in the following format:
[[START OF FINAL ANSWER TEMPLATE]]
Answer:
```python
def function_name(your_arguments): # Add the relevant arguments here.
    # ... implementation ...
    return your_return_result
```
[[END OF FINAL ANSWER TEMPLATE]]

Make sure to define the function with its arguments and return the result of the function in the format specified in the task prompt.
Don't forget to import the necessary libraries.

Only the text after 'Answer: ' will be evaluated. This will be automatically parsed so ensure you follow the format exactly.
Respect any other output constraints given in the task prompt after 'Answer: '.
"""

task_embedding_prompt = """Name of task family: {name_of_task}
Description: {description_of_task}
Capability being measured: {capability_being_measured}
Estimated human difficulty: {estimated_human_difficulty}
Example instruction: {example_question}
Agent succeeded at task: {agent_succeeded}"""

interestingly_new_system_msg = (
    base_system_msg
    + """You will be given a task family that you just generated along with its closest saved neighbours (i.e., the task families that are most similar to the newly generated one).
Determine whether the generated task family is interestingly new compared to the closest task families. If so, it should be added to the task archive. Factors may include but are not limited to:
- The novelty of the subject area
- The difficulty of the tasks
- The capabilities being measured
- Whether the agent succeeded at the task

First, briefly think and reason about the task family in relation to the existing closest task families.

Then, provide your decision as 'Decision: Yes' or 'Decision: No' to indicate whether the task family should be added to the task archive.

Your response will be automatically parsed, so ensure you follow the above format exactly."""
)

interestingly_new_prompt = '''You have just generated the task family:
"""
{new_task}
"""

The closest task families that have already been created are:
"""
{closest_tasks}
"""

Is the generated task family interestingly new compared to the closest task families? If yes, it should be added to the task archive.'''

surprising_system_msg = (
    base_system_msg
    + """You will be given a newly created task family that you just generated which includes the agent's success or failure in the 'Agent succeeded at task' field.

You will be asked to determine whether its success or failure on the task is surprising or noteworthy from the point of view of a human evaluator.
Assume the human evaluator is an expert in the field of LLMs, e.g., an academic researcher or a developer of LLMs.

First, briefly think and reason about the task family given what you know about the capabilities of LLMs.

Then, provide your decision as 'Decision: Yes' if the result is surprising, or 'Decision: No' if it is not.

Your response will be automatically parsed, so ensure you follow the above format exactly."""
)

surprising_prompt = '''You have just generated the task family:
"""
{new_task}
"""

This is implemented via the following code:
"""
{new_task_code}
"""

Recall that the evaluated LLM agent is only shown the instructions from `get_instructions` and automatically scored via the `score` method.
Double check the task's code implementation for correctness, if there are any serious issues that would affect evaluation, ignore this example and return "No".
Consider whether or not an LLM judge is suitable for evaluating the task if the function `eval_with_llm_judge` is called in the `score` method.

Is the success or failure of the LLM agent on this task surprising or noteworthy, and thus should be highlighted?'''

judge_system_msg = """You are an impartial judge evaluating the quality of responses provided by an AI assistant.
You will be provided the instructions for the task and the assistant's submission in response.
You will decide whether or not the agent succeeded at the task.
Please consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
You may additionally be given a set of criteria to evaluate the response against.
Do not allow the length of the response to influence your decision.

Respond precisely in the following format:

THOUGHT:
<THOUGHT>

DECISION:
<DECISION>

In <THOUGHT>, briefly reason about the agent's response and whether it meets the additional criteria.

In <DECISION>, provide your answer as either "Yes" or "No"."""

judge_prompt = '''Instruction: {instructions}
Submission: {submission}

Additional Evaluation Criteria:
"""
{criteria}
"""
'''

# --- Task Adaptation Prompts ---

make_task_harder_prompt = '''The following task family was found to be too easy for the current models:
"""
{original_task_json}
"""

Summaries of other previously generated tasks for context are:
"""
{other_task_jsons}
"""

Generate a new task family that is conceptually related but significantly more challenging.
This could involve:
- Requiring deeper reasoning or multi-step problem solving.
- Introducing more complex constraints or edge cases.
- Using more advanced concepts within the same domain.
- Increasing the required precision or detail in the answer.

Ensure the new task remains coherent and adheres to the METR standard.
Respond in the standard JSON format with THOUGHT and RESPONSE JSON sections.
Set "done" to "False" initially, as this task will be validated.
'''

make_task_easier_prompt = '''The following task family was found to be too difficult (or impossible) for the current models:
"""
{original_task_json}
"""

Summaries of other previously generated tasks for context are:
"""
{other_task_jsons}
"""

Generate a new task family that is conceptually related but significantly easier.
This could involve:
- Breaking the problem down into simpler steps.
- Reducing the complexity of the required reasoning or knowledge.
- Providing more scaffolding or clearer instructions.
- Focusing on a more fundamental aspect of the capability.

Ensure the new task remains coherent and adheres to the METR standard.
Respond in the standard JSON format with THOUGHT and RESPONSE JSON sections.
Set "done" to "False" initially, as this task will be validated.
'''

make_task_novel_prompt = '''Consider the following task family:
"""
{original_task_json}
"""

Summaries of other previously generated tasks for context are:
"""
{other_task_jsons}
"""

Generate a new task family that is interestingly different, aiming to explore diverse capabilities.
You can draw inspiration from the provided task, but prioritize novelty in terms of:
- The specific capability being measured.
- The domain or context of the task.
- The format or style of the interaction.

Ensure the new task is coherent, adheres to the METR standard, and is distinct from existing tasks.
Respond in the standard JSON format with THOUGHT and RESPONSE JSON sections.
Set "done" to "False" initially, as this task will be validated.
'''

old_make_task_novel_but_similar_prompt = '''Consider the following task family:
"""
{original_task_json}
"""

Summaries of other previously generated tasks for context are:
"""
{other_task_jsons}
"""

Generate a new task family that is inspired by the previous task family, so that it provides a more interesting challenge that is more complex but still related to the previous task family.
Draw inspiration from the provided task and implement novelty in terms of:
- The specific capability being measured, while still being related to the previous task domain.
- The contextual setting of the task.
- The format or style of the interaction.

Ensure the new task is coherent, adheres to the METR standard, and is distinct from existing tasks.
Respond in the standard JSON format with THOUGHT and RESPONSE JSON sections.
Set "done" to "False" initially, as this task will be validated.
'''

make_task_novel_but_similar_prompt = '''Consider the following task family:
"""
{original_task_json}
"""

Summaries of other previously generated tasks for inspiration are:
"""
{other_task_jsons}
"""

Generate a new task family that is inspired by the previous task family, so that it provides a more interesting challenge that is more complex or explores beyond what the current task family is evaluating in terms of model capabilities.
Draw inspiration from the provided task and implement novelty in terms of:
- The specific capability being measured.
- The contextual setting of the task.
- The format or style of the interaction.

Ensure the new task is coherent, adheres to the METR standard, and is distinct from existing tasks.
Respond in the standard JSON format with THOUGHT and RESPONSE JSON sections.
Set "done" to "False" initially, as this task will be validated.
'''

OLD_GIBBERISH_PROMPT = '''Consider the following tasks:
"""
{instructions}
"""

and the following three responses to the task from an LLM:
"""
{outputs}
"""

## Criteria for gibberish
Are the responses gibberish, i.e. do they form coherent sentences or are they random characters?
Are the responses just repetitions of the same sentence and not useful for evaluating the model?

The responses don't have to be correct!! But they should be at least somewhat coherent as per the criteria above.

First, very briefly think step-by-step and reason about the responses.
Then, provide your answer as 'Answer: Yes' if the responses are gibberish, or 'Answer: No' if they are not.

Your response will be automatically parsed, so ensure you follow the above format exactly.'''

GIBBERISH_PROMPT = '''Consider the following tasks shown in the numbered list:
"""
{instructions}
"""

and the following three responses to the task from an LLM (in a numbered list):
"""
{outputs}
"""

Your task is to determine if the LLM responded with gibberish, incoherent sentences, or not. Factors may include but are not limited to:
- Does the LLM form coherent sentences or are they responding with jumbled words and strings?
- Does the LLM use punctuation in a sensible way, or spam random symbol characters?
- Does the LLM end up repeating sentences non-stop?
- Does the LLM fail to return any output response?
- Does the LLM respond in a way that completely ignores the task or goes completely off-topic, in a reasonable way of attempting to solve the task?
- Does the LLM format its response in a way that is highly unexpected in the context of the tasks?

The responses don't have to be correct! But they should be at least somewhat coherent, considering the factors on gibberish responses mentioned above as well as any other reasonable expectations on how LLMs may respond in a gibberish manner.

Firstly, very briefly think step-by-step and reason about the responses.
Then, provide your answer as 'Answer: Yes' if the responses are gibberish, or 'Answer: No' if they are not.

Your response will be automatically parsed, so ensure you follow the above format exactly.'''

GIBBERISH_PROMPT_REVISED = '''Consider the following tasks shown in the numbered list:
"""
{instructions}
"""

and the following three responses to the task from an LLM (in a numbered list):
"""
{outputs}
"""

Your task is to determine if the LLM responded with gibberish, incoherent sentences, or not. Factors may include but are not limited to:
- Does the LLM form coherent sentences or are they responding with jumbled words and strings?
- Does the LLM use punctuation in a sensible way, or spam random symbol characters?
- Does the LLM end up repeating sentences non-stop?
- Does the LLM fail to return any output response?
- Does the LLM respond in a way that completely ignores the task or goes completely off-topic, in a reasonable way of attempting to solve the task?
- Does the LLM format its response in a way that is highly unexpected in the context of the tasks?

The responses don't have to be correct! But they should be at least somewhat coherent, considering the factors on gibberish responses mentioned above as well as any other reasonable expectations on how LLMs may respond in a gibberish manner. We can also deem the model as returning non-gibberish responses if at least one of the responses meet the criteria for not being gibberish.

Firstly, very briefly think step-by-step and reason about the responses.
Then, provide your answer as 'Answer: Yes' if all of the responses are gibberish, or 'Answer: No' if at least one response is not gibberish.

Your response will be automatically parsed, so ensure you follow the above format exactly.'''

regenerate_more_novel_tasks_prompt = """The task is not novel.

Here are some similar tasks:
{similar_tasks_str}

Please try again.
"""
