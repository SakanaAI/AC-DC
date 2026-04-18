"""
Unit tests for SVDModelWeightsGaussianMutator.

Tests the SVD-based mutation that computes SVD on-the-fly from model weights.
"""

import pytest
import torch
import numpy as np
import copy

from mutation.svd_model_weights_gaussian_mutator import (
    SVDModelWeightsGaussianMutator,
)


@pytest.mark.requires_gpu
class TestSVDModelWeightsGaussianMutator:
    """Test suite for SVDModelWeightsGaussianMutator class."""

    def test_initialization(self):
        """Test that SVDModelWeightsGaussianMutator initializes correctly."""
        mutation_rate = 0.01
        keep_rank = 256
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=mutation_rate,
            keep_rank=keep_rank,
            include_bias_mutation=False,
        )

        assert mutator.mutation_rate == mutation_rate
        assert mutator.keep_rank == keep_rank
        assert mutator.include_bias_mutation == False
        assert mutator.num_mutation_params == 1

    def test_initialization_with_bias(self):
        """Test initialization with bias mutation enabled."""
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=0.01, keep_rank=512, include_bias_mutation=True
        )

        assert mutator.include_bias_mutation == True

    def test_generate_mutation_params(self):
        """Test mutation parameter generation."""
        mutation_rate = 0.05
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=mutation_rate, keep_rank=256
        )

        params = mutator._generate_mutation_params()

        assert isinstance(params, np.ndarray)
        assert params == mutation_rate

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_mutate_preserves_shapes(self, tiny_weight_dict):
        """Test that mutation preserves tensor shapes."""
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=0.01, keep_rank=32
        )
        mutator.update_seed(42)

        original_shapes = {k: v.shape for k, v in tiny_weight_dict.items()}
        mutated = mutator.mutate(copy.deepcopy(tiny_weight_dict), q_name="test")

        assert set(mutated.keys()) == set(tiny_weight_dict.keys())

        for key in mutated:
            assert (
                mutated[key].shape == original_shapes[key]
            ), f"Shape mismatch for {key}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_mutate_preserves_dtype(self, tiny_weight_dict):
        """Test that mutation preserves tensor dtypes."""
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=0.01, keep_rank=32
        )
        mutator.update_seed(42)

        original_dtypes = {k: v.dtype for k, v in tiny_weight_dict.items()}
        mutated = mutator.mutate(copy.deepcopy(tiny_weight_dict), q_name="test")

        for key in mutated:
            assert (
                mutated[key].dtype == original_dtypes[key]
            ), f"Dtype mismatch for {key}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_mutate_changes_weights(self, tiny_weight_dict):
        """Test that mutation actually modifies the weights."""
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=0.1, keep_rank=32
        )
        mutator.update_seed(42)

        original = copy.deepcopy(tiny_weight_dict)
        mutated = mutator.mutate(copy.deepcopy(tiny_weight_dict), q_name="test")

        # Check that weights have changed (excluding norm and 1D tensors)
        changed_count = 0
        for key in mutated:
            if "norm" not in key and tiny_weight_dict[key].ndim >= 2:
                if not torch.allclose(
                    mutated[key].float(), original[key].float(), rtol=1e-3
                ):
                    changed_count += 1

        assert changed_count > 0, "No weights were mutated"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_mutate_skips_norm_layers(self, tiny_weight_dict_with_norms):
        """Test that mutation skips normalization layers."""
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=0.1, keep_rank=32
        )
        mutator.update_seed(42)

        original = copy.deepcopy(tiny_weight_dict_with_norms)
        mutated = mutator.mutate(
            copy.deepcopy(tiny_weight_dict_with_norms), q_name="test"
        )

        # Check that norm layers are unchanged
        for key in mutated:
            if "norm" in key:
                assert torch.allclose(
                    mutated[key], original[key]
                ), f"Norm layer {key} was modified"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_mutate_skips_1d_tensors(self, tiny_weight_dict):
        """Test that mutation skips 1D tensors (like biases)."""
        # Add some 1D tensors to weight dict
        weights_with_bias = copy.deepcopy(tiny_weight_dict)
        weights_with_bias["model.layers.0.mlp.bias"] = torch.randn(
            128, dtype=torch.bfloat16
        )

        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=0.1, keep_rank=32, include_bias_mutation=False
        )
        mutator.update_seed(42)

        original = copy.deepcopy(weights_with_bias)
        mutated = mutator.mutate(
            copy.deepcopy(weights_with_bias), q_name="test"
        )

        # 1D tensors should be unchanged
        assert torch.allclose(
            mutated["model.layers.0.mlp.bias"],
            original["model.layers.0.mlp.bias"],
        ), "1D tensor was modified"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_singular_values_clamped_positive(self, tiny_weight_dict):
        """Test that singular values are clamped to be non-negative."""
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=10.0,  # Large rate to potentially create negative values
            keep_rank=32,
        )
        mutator.update_seed(42)

        # This shouldn't raise an error due to clamping
        mutated = mutator.mutate(copy.deepcopy(tiny_weight_dict), q_name="test")

        # Check that all mutated matrices are valid
        for key, value in mutated.items():
            if "norm" not in key and value.ndim >= 2:
                # Should not contain NaNs or Infs
                assert not torch.isnan(value).any(), f"NaN found in {key}"
                assert not torch.isinf(value).any(), f"Inf found in {key}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_keep_rank_parameter(self, tiny_weight_dict):
        """Test that keep_rank parameter limits the rank of mutations."""
        low_rank = 16
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=0.05, keep_rank=low_rank
        )
        mutator.update_seed(42)

        mutated = mutator.mutate(copy.deepcopy(tiny_weight_dict), q_name="test")

        # The mutations should be applied in low-rank space
        # (difficult to test directly, but we can check it doesn't error)
        assert mutated is not None
        assert len(mutated) == len(tiny_weight_dict)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_mutate_seed_reproducibility(self, tiny_weight_dict):
        """Test that same seed produces same mutation."""
        mutator1 = SVDModelWeightsGaussianMutator(
            mutation_rate=0.05, keep_rank=32
        )
        mutator1.update_seed(123)
        weights1 = copy.deepcopy(tiny_weight_dict)
        mutated1 = mutator1.mutate(weights1, q_name="test")

        mutator2 = SVDModelWeightsGaussianMutator(
            mutation_rate=0.05, keep_rank=32
        )
        mutator2.update_seed(123)
        weights2 = copy.deepcopy(tiny_weight_dict)
        mutated2 = mutator2.mutate(weights2, q_name="test")

        # Same seed should produce identical results
        for key in mutated1:
            if "norm" not in key and tiny_weight_dict[key].ndim >= 2:
                assert torch.allclose(
                    mutated1[key].float(), mutated2[key].float(), rtol=1e-4
                ), f"Different results for {key} with same seed"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    @pytest.mark.parametrize("keep_rank", [16, 32, 64, 128])
    def test_various_keep_ranks(self, tiny_weight_dict, keep_rank):
        """Test mutation with various keep_rank values."""
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=0.01, keep_rank=keep_rank
        )
        mutator.update_seed(42)

        mutated = mutator.mutate(copy.deepcopy(tiny_weight_dict), q_name="test")

        assert mutated is not None
        assert len(mutated) == len(tiny_weight_dict)

        for key in mutated:
            assert mutated[key].shape == tiny_weight_dict[key].shape


# CPU-only tests (don't require GPU)
class TestSVDModelWeightsGaussianMutatorCPU:
    """CPU-only tests for SVDModelWeightsGaussianMutator."""

    def test_initialization_no_gpu(self):
        """Test that initialization works without GPU."""
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=0.01, keep_rank=256
        )

        assert mutator.mutation_rate == 0.01
        assert mutator.keep_rank == 256

    def test_invalid_mutation_params_size(self, tiny_weight_dict):
        """Test that providing wrong number of mutation params raises error."""
        mutator = SVDModelWeightsGaussianMutator(
            mutation_rate=0.01, keep_rank=256
        )
        mutator.update_seed(42)

        with pytest.raises(AssertionError):
            mutator.mutate(
                copy.deepcopy(tiny_weight_dict),
                q_name="test",
                mutation_params=np.array([0.01, 0.02]),
            )
