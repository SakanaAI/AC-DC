import numpy as np
from typing import Dict, List

from fishspawn import run_merge

from .base import BaseModelMerger


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ModelwiseSlerpMerge(BaseModelMerger):

    def __init__(self):
        self.num_merge_params = 1

    def _generate_merge_params(self) -> np.ndarray:
        return np.array(np.random.uniform(0, 1))

    def _merge(self,
               task_vectors: List[Dict],
               merge_params: np.ndarray) -> Dict:
        base_value = float(sigmoid(merge_params))
        weight_dict = {f"MODEL{i}": x for i, x in enumerate(task_vectors)}
        merge_config = {
            "slices": [
                {
                    "sources": [
                        {"model": f"MODEL{i}"} for i in range(len(task_vectors))
                    ],
                }
            ],
            "merge_method": "slerp",
            "parameters": {
                "t_values": {},
                "t_value": base_value,
            },
            "dtype": "bfloat16",
        }
        return run_merge(
            config_dict=merge_config,
            weight_dict=weight_dict,
            seed=42,
            use_cuda=False,
        )