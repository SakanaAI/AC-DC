class TaskFamily:
    @staticmethod
    def get_tasks() -> dict[str, dict]:
        return {
            "1": {
                "question": "Write a python function to find the first repeated character in a given string.",
                "test_cases": [("abcabc", "a"), ("abc", None), ("123123", "1")],
            }
        }

    @staticmethod
    def get_instructions(t: dict) -> str:
        return f"""{t["question"]}"""

    @staticmethod
    def score(t: dict, submission: str) -> float | None:
        from sandbox_eval_helper import extract_and_execute_function_string

        # Test cases from get_tasks
        test_cases = t["test_cases"]

        correct_count = 0

        for test_input, expected_output in test_cases:
            try:
                # Execute the submitted function
                submission_result = extract_and_execute_function_string(
                    submission, test_input
                )

                # Compare with expected output
                if submission_result == expected_output:
                    correct_count += 1

            except Exception:
                # If function fails on any test case, skip it
                continue

        return 1.0 if correct_count > 0 else 0.0
