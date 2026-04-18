class TaskFamily:
    @staticmethod
    def get_tasks() -> dict[str, dict]:
        return {
            "1": {
                "question": "Write a python function called `first_repeated_character` to find the first repeated character in a given string.",
                "test_cases": [("abcabc", "a"), ("abc", None), ("123123", "1")],
                "expected_func_name": "first_repeated_character",
            }
        }

    @staticmethod
    def get_instructions(t: dict) -> str:
        return f"""{t["question"]}"""

    @staticmethod
    def score(t: dict, submission: str) -> float | None:
        from sandbox_eval_helper import get_function_name_to_callable

        # Test cases from get_tasks
        test_cases = t["test_cases"]

        correct_count = 0
        num_test_cases = len(test_cases)

        # Get function name to callable mapping
        function_name_to_callable = get_function_name_to_callable(submission)

        if len(function_name_to_callable) == 0:
            return 0.0

        # Get function names
        func_names = set(function_name_to_callable.keys())

        if t["expected_func_name"] not in func_names:
            return 0.0

        # Run test cases
        for test_input, expected_output in test_cases:
            try:
                # Execute the function
                submission_result = function_name_to_callable[
                    t["expected_func_name"]
                ](test_input)

                # Compare with expected output
                if submission_result == expected_output:
                    correct_count += 1

            except Exception:
                # If function fails on a test case, skip it
                continue

        pass_rate = correct_count / num_test_cases
        min_pass_rate = 1.0

        return 1.0 if pass_rate >= min_pass_rate else 0.0
