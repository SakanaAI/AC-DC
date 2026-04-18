class TaskFamily:
    @staticmethod
    def get_tasks() -> dict[str, dict]:
        return {
            "1": {
                "question": "Given that the storage space for a circular queue is the array A[21], with front pointing to the position before the head element and rear pointing to the tail element, assuming the current values of front and rear are 8 and 3, respectively, the length of the queue is ().",
                "options": {"A": "5", "B": "6", "C": "16", "D": "17"},
                "answer": "C",
            }
        }

    @staticmethod
    def get_instructions(t: dict) -> str:
        return f"""{t["question"]}\nA: {t["options"]["A"]}\nB: {t["options"]["B"]}\nC: {t["options"]["C"]}\nD: {t["options"]["D"]}\n\nReturn the letter of the correct option."""

    @staticmethod
    def score(t: dict, submission: str) -> float | None:
        return 1.0 if t["answer"].lower() == submission.lower().strip() else 0.0
