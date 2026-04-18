class TaskFamily:
    @staticmethod
    def get_tasks() -> dict[str, dict]:
        return {
            "1": {
                "question": "House owns his home in City. On the lawn in front of his home and within five feet of the public sidewalk there was a large tree. The roots of the tree caused the sidewalk to buckle severely and become dangerous. An ordinance of City requires adjacent landowners to keep sidewalks in safe condition. House engaged Contractor to repair the sidewalk, leaving it to Contractor to decide how the repair should be made. Contractor dug up the sidewalk, cut back the roots of the tree, and laid a new sidewalk. Two days after House had paid Contractor the agreed price of the repair, the tree fell over onto the street and damaged a parked car belonging to Driver. Driver has asserted claims against House and Contractor, and both defendants admit that cutting the roots caused the tree to fall.If Driver recovers a judgment against House, does House have any recourse against Contractor?",
                "options": {
                    "A": "No, if payment by House was an acceptance of the work.",
                    "B": "No, because House selected Contractor to do the work.",
                    "C": "Yes, if the judgment against House was based on vicarious liability.",
                    "D": "Yes, because House's conduct was not a factual cause of the harm.",
                },
                "answer": "C",
            }
        }

    @staticmethod
    def get_instructions(t: dict) -> str:
        return f"""{t["question"]}\nA: {t["options"]["A"]}\nB: {t["options"]["B"]}\nC: {t["options"]["C"]}\nD: {t["options"]["D"]}\n\nReturn the letter of the correct option."""

    @staticmethod
    def score(t: dict, submission: str) -> float | None:
        return 1.0 if t["answer"].lower() == submission.lower().strip() else 0.0
