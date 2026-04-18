class TaskFamily:
    @staticmethod
    def get_tasks() -> dict[str, dict]:
        return {
            "1": {
                "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            }
        }

    @staticmethod
    def get_instructions(t: dict) -> str:
        return f"""{t["question"]}"""

    @staticmethod
    def score(t: dict, submission: str) -> float | None:
        try:
            submission_val = int(submission)
            # Compute the correct answer based on the given reasoning
            april_sales = 48

            # Calculate May sales: half of April's sales
            may_sales = april_sales / 2

            # Calculate total sales for both months
            total_sales = april_sales + may_sales

            return 1.0 if submission_val == total_sales else 0.0
        except ValueError:
            return 0.0
