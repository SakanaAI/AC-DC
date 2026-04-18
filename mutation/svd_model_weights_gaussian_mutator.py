import torch
import numpy as np
import logging
from typing import Dict, Optional
import time

from .base import BaseMutator


class SVDModelWeightsGaussianMutator(BaseMutator):

    def __init__(
        self,
        mutation_rate: float,
        keep_rank: int = 512,
        include_bias_mutation: bool = False,
    ):
        self.num_mutation_params = 1
        self.mutation_rate = mutation_rate
        self.keep_rank = keep_rank
        self.include_bias_mutation = include_bias_mutation
        self.logger = logging.getLogger(__name__)
        self.warning_printed = False

        self.logger.info(
            f"SVDModelWeightsGaussianMutator initialized with "
            f"mutation_rate={mutation_rate}, keep_rank={keep_rank}, "
            f"include_bias_mutation={include_bias_mutation}"
        )

    def _generate_mutation_params(self) -> np.ndarray:
        return np.array(self.mutation_rate)

    def _mutate(
        self,
        weight_dict: Dict,
        q_name: str,
        mutation_params: np.ndarray,
    ) -> Dict:
        cpu = torch.device("cpu")
        for key, value in weight_dict.items():
            if (
                "norm" in key
                or ("_proj.bias" in key and "qwen" in q_name)
                # skip 1-D vectors (e.g. bias)
                or value.ndim == 1
            ):
                continue

            # Store original dtype
            original_dtype = value.dtype

            # Convert to float32 for SVD and move to GPU
            value_f32 = value.to(torch.float32).cuda()

            # Get the SVD
            u, s, vt = torch.linalg.svd(value_f32, full_matrices=False)

            ### Scale the singular values using a Gaussian distribution
            # if keep_rank is None, use the current numerical rank
            rank = min(value_f32.shape)
            if self.keep_rank is not None and self.keep_rank <= rank:
                rank = self.keep_rank
            elif self.keep_rank is not None and not self.warning_printed:
                self.warning_printed = True
                self.logger.warning(
                    f"`keep_rank={self.keep_rank}` is greater than the current "
                    f"numerical rank ({rank}), "
                    f"for <{key}>. Using the current numerical rank."
                )

            # Only perturb the top-k singular values
            s_perturbed = s.clone()
            s_perturbed[:rank] += torch.randn(rank, device=s.device) * float(
                mutation_params
            )
            s_perturbed = torch.clamp(s_perturbed, min=0)

            # Reconstruct the perturbed matrix and move back to CPU
            value_f32 = (u @ torch.diag_embed(s_perturbed) @ vt).to(cpu)

            # Convert back to original dtype
            value = value_f32.to(original_dtype)

            # Update model weights
            weight_dict[key] = value

        return weight_dict
