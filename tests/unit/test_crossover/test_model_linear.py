"""
Unit tests for ModelwiseLinearMerge (Chat Vector Merge).

Tests the linear combination merge operator for model weights.
"""

import pytest
import torch
import numpy as np
import copy
from unittest.mock import MagicMock, patch

from crossover.model_linear import ModelwiseLinearMerge


class TestModelwiseLinearMerge:
    """ModelwiseLinearMerge: linear combination of task vectors using (mean, std) merge params."""

    def test_initialization(self):
        """Test that ModelwiseLinearMerge initializes correctly."""
        std = 0.01
        merger = ModelwiseLinearMerge(std=std)

        assert merger.std == std
        assert merger.num_merge_params == 2  # Mean and std

    def test_initialization_default_std(self):
        """Test initialization with default std value."""
        merger = ModelwiseLinearMerge()

        assert merger.std == 0.01  # Default value
        assert merger.num_merge_params == 2

    def test_generate_merge_params(self):
        """Test merge parameter generation."""
        std = 0.05
        merger = ModelwiseLinearMerge(std=std)

        params = merger._generate_merge_params()

        assert isinstance(params, np.ndarray)
        assert len(params) == 2
        assert params[0] == 0.0  # Mean
        assert params[1] == std  # Std

    def test_get_task_vector(
        self, base_model_weights, finetuned_model_weights_1
    ):
        """Test task vector computation."""
        merger = ModelwiseLinearMerge()

        task_vector = merger._get_task_vector(
            base_model_weights, finetuned_model_weights_1
        )

        # Task vector should be the difference
        for key in base_model_weights:
            expected = finetuned_model_weights_1[key] - base_model_weights[key]
            assert torch.allclose(
                task_vector[key], expected
            ), f"Task vector mismatch for {key}"

    def test_merge_task_vectors(self, task_vector_1, task_vector_2):
        """Test merging of task vectors."""
        merger = ModelwiseLinearMerge(std=0.0)
        merger.update_seed(42)

        task_vectors = [task_vector_1, task_vector_2]
        merge_params = np.array([0.0, 0.0])  # No randomness

        merged = merger._merge(task_vectors, merge_params)

        # With zero std and zero mean, should be average of task vectors
        for key in task_vector_1:
            expected = (task_vector_1[key] + task_vector_2[key]) / 2.0
            assert torch.allclose(
                merged[key], expected, rtol=1e-4
            ), f"Merged vector mismatch for {key}"

    def test_merge_preserves_shapes(self, task_vector_1, task_vector_2):
        """Test that merging preserves tensor shapes."""
        merger = ModelwiseLinearMerge(std=0.01)
        merger.update_seed(42)

        task_vectors = [task_vector_1, task_vector_2]
        merge_params = np.array([0.0, 0.01])

        merged = merger._merge(task_vectors, merge_params)

        # Check that shapes are preserved
        for key in task_vector_1:
            assert (
                merged[key].shape == task_vector_1[key].shape
            ), f"Shape mismatch for {key}"

    def test_merge_with_custom_params(self, task_vector_1, task_vector_2):
        """Test merging with custom parameters."""
        merger = ModelwiseLinearMerge(std=0.01)
        merger.update_seed(42)

        task_vectors = [task_vector_1, task_vector_2]

        # Custom params: mean=0.5, std=0.1
        custom_params = np.array([0.5, 0.1])

        merged = merger._merge(task_vectors, custom_params)

        assert merged is not None
        assert len(merged) == len(task_vector_1)

    def test_merge_seed_reproducibility(self, task_vector_1, task_vector_2):
        """Test that same seed produces same merge."""
        merger1 = ModelwiseLinearMerge(std=0.05)
        merger1.update_seed(123)
        task_vectors_1 = [
            copy.deepcopy(task_vector_1),
            copy.deepcopy(task_vector_2),
        ]
        merged1 = merger1._merge(
            task_vectors_1, merger1._generate_merge_params()
        )

        merger2 = ModelwiseLinearMerge(std=0.05)
        merger2.update_seed(123)
        task_vectors_2 = [
            copy.deepcopy(task_vector_1),
            copy.deepcopy(task_vector_2),
        ]
        merged2 = merger2._merge(
            task_vectors_2, merger2._generate_merge_params()
        )

        # Same seed should produce identical results
        for key in merged1:
            assert torch.allclose(
                merged1[key], merged2[key], rtol=1e-5
            ), f"Different results for {key} with same seed"

    def test_merge_seed_difference(self, task_vector_1, task_vector_2):
        """Test that different seeds produce different merges."""
        merger1 = ModelwiseLinearMerge(std=0.05)
        merger1.update_seed(123)

        merger2 = ModelwiseLinearMerge(std=0.05)
        merger2.update_seed(456)

        task_vectors_1 = [
            copy.deepcopy(task_vector_1),
            copy.deepcopy(task_vector_2),
        ]
        task_vectors_2 = [
            copy.deepcopy(task_vector_1),
            copy.deepcopy(task_vector_2),
        ]

        merged1 = merger1._merge(
            task_vectors_1, merger1._generate_merge_params()
        )
        merged2 = merger2._merge(
            task_vectors_2, merger2._generate_merge_params()
        )

        # Different seeds should produce different results
        any_different = False
        for key in merged1:
            if not torch.allclose(merged1[key], merged2[key], rtol=1e-5):
                any_different = True
                break

        assert any_different, "Different seeds produced identical merges"

    def test_merge_multiple_task_vectors(
        self, task_vector_1, task_vector_2, base_model_weights
    ):
        """Test merging more than 2 task vectors."""
        merger = ModelwiseLinearMerge(std=0.0)
        merger.update_seed(42)

        # Create a third task vector
        task_vector_3 = {
            k: torch.randn_like(v) * 0.01 for k, v in base_model_weights.items()
        }

        task_vectors = [task_vector_1, task_vector_2, task_vector_3]
        merge_params = np.array([0.0, 0.0])

        merged = merger._merge(task_vectors, merge_params)

        # With zero variance, should be weighted average
        assert merged is not None
        assert len(merged) == len(task_vector_1)

    def test_merge_single_task_vector(self, task_vector_1):
        """Test merging a single task vector."""
        merger = ModelwiseLinearMerge(std=0.0)
        merger.update_seed(42)

        task_vectors = [task_vector_1]
        merge_params = np.array([0.0, 0.0])

        merged = merger._merge(task_vectors, merge_params)

        # Single task vector should return itself (with unit weight)
        for key in task_vector_1:
            assert torch.allclose(
                merged[key], task_vector_1[key], rtol=1e-5
            ), f"Single task vector merge failed for {key}"

    def test_weights_normalized(self, task_vector_1, task_vector_2):
        """Test that merge weights are normalized (sum preserving)."""
        merger = ModelwiseLinearMerge(std=0.0)
        merger.update_seed(42)

        task_vectors = [task_vector_1, task_vector_2]
        merge_params = np.array([0.0, 0.0])

        # With zero std, weights should all be 1.0
        # Result should be average of task vectors
        merged = merger._merge(task_vectors, merge_params)

        for key in task_vector_1:
            expected = (task_vector_1[key] + task_vector_2[key]) / 2.0
            assert torch.allclose(
                merged[key], expected, rtol=1e-4
            ), f"Weights not properly normalized for {key}"

    def test_invalid_merge_params_size(self, task_vector_1, task_vector_2):
        """Test that providing wrong number of merge params raises error."""
        merger = ModelwiseLinearMerge(std=0.01)
        merger.update_seed(42)

        task_vectors = [task_vector_1, task_vector_2]

        # ModelwiseLinearMerge expects 2 parameters, provide 1
        with pytest.raises(AssertionError):
            merger._merge(task_vectors, np.array([0.01]))

        # Provide 3 parameters
        with pytest.raises(AssertionError):
            merger._merge(task_vectors, np.array([0.0, 0.01, 0.02]))

    @pytest.mark.parametrize("std", [0.001, 0.01, 0.05, 0.1])
    def test_various_std_values(self, task_vector_1, task_vector_2, std):
        """Test merging with various std values."""
        merger = ModelwiseLinearMerge(std=std)
        merger.update_seed(42)

        task_vectors = [task_vector_1, task_vector_2]
        merge_params = merger._generate_merge_params()

        merged = merger._merge(task_vectors, merge_params)

        assert merged is not None
        assert len(merged) == len(task_vector_1)

        for key in merged:
            assert merged[key].shape == task_vector_1[key].shape

    @pytest.mark.parametrize("num_vectors", [1, 2, 3, 5])
    def test_various_num_task_vectors(self, base_model_weights, num_vectors):
        """Test merging different numbers of task vectors."""
        merger = ModelwiseLinearMerge(std=0.01)
        merger.update_seed(42)

        # Create multiple task vectors
        task_vectors = [
            {
                k: torch.randn_like(v) * 0.01
                for k, v in base_model_weights.items()
            }
            for _ in range(num_vectors)
        ]

        merge_params = merger._generate_merge_params()
        merged = merger._merge(task_vectors, merge_params)

        assert merged is not None
        assert len(merged) == len(base_model_weights)

    def test_merge_preserves_dtype(self, task_vector_1, task_vector_2):
        """Test that merging preserves tensor dtypes."""
        merger = ModelwiseLinearMerge(std=0.01)
        merger.update_seed(42)

        task_vectors = [task_vector_1, task_vector_2]
        original_dtypes = {k: v.dtype for k, v in task_vector_1.items()}

        merged = merger._merge(task_vectors, merger._generate_merge_params())

        for key in merged:
            # Dtype might change to float32 during computation, but should be consistent
            assert merged[key].dtype in [
                torch.float32,
                torch.bfloat16,
                original_dtypes[key],
            ]


class TestModelwiseLinearMergeIntegration:
    """Integration tests for the full merge pipeline."""

    def test_full_merge_pipeline(
        self,
        base_model_weights,
        finetuned_model_weights_1,
        finetuned_model_weights_2,
    ):
        """
        Test full merge pipeline: compute task vectors and merge them.

        This simulates the complete workflow:
        1. Start with base model
        2. Compute task vectors from fine-tuned models
        3. Merge task vectors
        4. Add merged vector back to base
        """
        merger = ModelwiseLinearMerge(std=0.01)
        merger.update_seed(42)

        # Step 1: Compute task vectors
        task_vector_1 = merger._get_task_vector(
            base_model_weights, finetuned_model_weights_1
        )
        task_vector_2 = merger._get_task_vector(
            base_model_weights, finetuned_model_weights_2
        )

        # task vectors should be different
        any_different = False
        for key in task_vector_1:
            if not torch.allclose(task_vector_1[key], task_vector_2[key]):
                any_different = True
                break
        assert any_different, "Task vectors are identical"

        # Step 2: Merge task vectors
        task_vectors = [task_vector_1, task_vector_2]
        merge_params = np.array([0.0, 0.01])
        merged_task_vector = merger._merge(task_vectors, merge_params)

        # Step 3: Add merged vector back to base
        final_weights = {
            k: base_model_weights[k] + merged_task_vector[k]
            for k in base_model_weights
        }

        # Verify the result
        assert final_weights is not None
        assert len(final_weights) == len(base_model_weights)

        # Check that final weights are different from base
        any_different = False
        for key in final_weights:
            if not torch.allclose(final_weights[key], base_model_weights[key]):
                any_different = True
                break

        assert any_different, "Merged weights are identical to base weights"

        # Check that shapes are preserved
        for key in final_weights:
            assert final_weights[key].shape == base_model_weights[key].shape

    def test_merge_with_zero_std(
        self,
        base_model_weights,
        finetuned_model_weights_1,
        finetuned_model_weights_2,
    ):
        """
        Test merge with zero std produces deterministic average.

        With std=0, all weights should be 1.0, so the result should be
        the simple average of the two task vectors.
        """
        merger = ModelwiseLinearMerge(std=0.0)
        merger.update_seed(42)

        # Compute task vectors
        task_vector_1 = merger._get_task_vector(
            base_model_weights, finetuned_model_weights_1
        )
        task_vector_2 = merger._get_task_vector(
            base_model_weights, finetuned_model_weights_2
        )

        # Merge with zero variance
        merged = merger._merge(
            [task_vector_1, task_vector_2], np.array([0.0, 0.0])
        )

        # Result should be exact average
        for key in merged:
            expected_avg = (task_vector_1[key] + task_vector_2[key]) / 2.0
            assert torch.allclose(merged[key], expected_avg, rtol=1e-5)

    def test_merge_three_models(
        self,
        base_model_weights,
        finetuned_model_weights_1,
        finetuned_model_weights_2,
    ):
        """Test merging three models together."""
        merger = ModelwiseLinearMerge(std=0.01)
        merger.update_seed(42)

        # Create a third fine-tuned model
        finetuned_model_weights_3 = {
            k: v + torch.randn_like(v) * 0.01
            for k, v in base_model_weights.items()
        }

        # Compute task vectors
        task_vectors = [
            merger._get_task_vector(base_model_weights, ft_weights)
            for ft_weights in [
                finetuned_model_weights_1,
                finetuned_model_weights_2,
                finetuned_model_weights_3,
            ]
        ]

        # Merge
        merged = merger._merge(task_vectors, np.array([0.0, 0.01]))

        # Verify result
        assert merged is not None
        assert len(merged) == len(base_model_weights)

        # Add to base to create final model
        final_weights = {
            k: base_model_weights[k] + merged[k] for k in base_model_weights
        }

        # Check shapes preserved
        for key in final_weights:
            assert final_weights[key].shape == base_model_weights[key].shape

    def test_task_vector_properties(
        self, base_model_weights, finetuned_model_weights_1
    ):
        """Test that task vectors have expected properties."""
        merger = ModelwiseLinearMerge()

        # Compute task vector
        task_vector = merger._get_task_vector(
            base_model_weights, finetuned_model_weights_1
        )

        # Task vector should have same keys as base
        assert set(task_vector.keys()) == set(base_model_weights.keys())

        # Task vector + base should equal fine-tuned
        reconstructed = {
            k: base_model_weights[k] + task_vector[k]
            for k in base_model_weights
        }

        for key in reconstructed:
            assert torch.allclose(
                reconstructed[key], finetuned_model_weights_1[key], rtol=1e-5
            ), f"Task vector property failed for {key}"

    def test_reproducible_merge_pipeline(
        self,
        base_model_weights,
        finetuned_model_weights_1,
        finetuned_model_weights_2,
    ):
        """Test that the full pipeline is reproducible with same seed."""
        # First run
        merger1 = ModelwiseLinearMerge(std=0.05)
        merger1.update_seed(999)

        task_vector_1a = merger1._get_task_vector(
            base_model_weights, finetuned_model_weights_1
        )
        task_vector_2a = merger1._get_task_vector(
            base_model_weights, finetuned_model_weights_2
        )
        merged1 = merger1._merge(
            [task_vector_1a, task_vector_2a], merger1._generate_merge_params()
        )

        # Second run with same seed
        merger2 = ModelwiseLinearMerge(std=0.05)
        merger2.update_seed(999)

        task_vector_1b = merger2._get_task_vector(
            base_model_weights, finetuned_model_weights_1
        )
        task_vector_2b = merger2._get_task_vector(
            base_model_weights, finetuned_model_weights_2
        )
        merged2 = merger2._merge(
            [task_vector_1b, task_vector_2b], merger2._generate_merge_params()
        )

        # Results should be identical
        for key in merged1:
            assert torch.allclose(merged1[key], merged2[key], rtol=1e-6)

    @patch("crossover.base.AutoModelForCausalLM")
    def test_full_merge_function_with_model_loading(
        self,
        mock_model_class,
        base_model_weights,
        finetuned_model_weights_1,
        finetuned_model_weights_2,
    ):
        """
        Test the full merge() function from BaseModelMerger.

        This tests the complete pipeline including:
        1. Model loading (mocked)
        2. Task vector computation
        3. Merging
        4. Adding back to base
        """
        # Mock the model loading to return our test weights
        def create_mock_model(state_dict):
            mock_model = MagicMock()
            mock_model.state_dict.return_value = state_dict
            return mock_model

        # Setup mock to return different models for different paths
        model_weights_map = {
            "/path/to/model1": finetuned_model_weights_1,
            "/path/to/model2": finetuned_model_weights_2,
        }

        def mock_from_pretrained(model_path, **kwargs):
            return create_mock_model(model_weights_map[model_path])

        mock_model_class.from_pretrained.side_effect = mock_from_pretrained

        # Now test the full merge() function
        merger = ModelwiseLinearMerge(std=0.01)
        merger.update_seed(42)

        model_paths = ["/path/to/model1", "/path/to/model2"]
        merge_params = np.array([0.0, 0.01])

        # Call the full merge function (from BaseModelMerger)
        final_weights = merger.merge(
            base_param=base_model_weights,
            model_paths=model_paths,
            merge_params=merge_params,
        )

        # Verify the results
        assert final_weights is not None
        assert len(final_weights) == len(base_model_weights)

        # Verify model loading was called
        assert mock_model_class.from_pretrained.call_count == 2
        mock_model_class.from_pretrained.assert_any_call(
            "/path/to/model1", torch_dtype=torch.bfloat16
        )
        mock_model_class.from_pretrained.assert_any_call(
            "/path/to/model2", torch_dtype=torch.bfloat16
        )

        # Verify final weights are different from base
        any_different = False
        for key in final_weights:
            if not torch.allclose(final_weights[key], base_model_weights[key]):
                any_different = True
                break

        assert any_different, "Final weights are identical to base weights"

        # Verify shapes are preserved
        for key in final_weights:
            assert final_weights[key].shape == base_model_weights[key].shape

        # Verify the result is mathematically correct
        # Manually compute what the result should be
        task_vector_1 = {
            k: finetuned_model_weights_1[k] - base_model_weights[k]
            for k in base_model_weights
        }
        task_vector_2 = {
            k: finetuned_model_weights_2[k] - base_model_weights[k]
            for k in base_model_weights
        }

        # Merge with same seed and params
        merger_verify = ModelwiseLinearMerge(std=0.01)
        merger_verify.update_seed(42)
        merged_task_vector = merger_verify._merge(
            [task_vector_1, task_vector_2], merge_params
        )

        expected_final = {
            k: base_model_weights[k] + merged_task_vector[k]
            for k in base_model_weights
        }

        # Final weights should match expected
        for key in final_weights:
            assert torch.allclose(
                final_weights[key], expected_final[key], rtol=1e-5
            ), f"Final weights don't match expected for {key}"

    @patch("crossover.base.AutoModelForCausalLM")
    def test_full_merge_with_default_params(
        self,
        mock_model_class,
        base_model_weights,
        finetuned_model_weights_1,
        finetuned_model_weights_2,
    ):
        """
        Test full merge() with default parameters (merge_params=None).

        When merge_params is None, it should use _generate_merge_params().
        """
        # Setup mocks
        def create_mock_model(state_dict):
            mock_model = MagicMock()
            mock_model.state_dict.return_value = state_dict
            return mock_model

        model_weights_map = {
            "/path/to/model1": finetuned_model_weights_1,
            "/path/to/model2": finetuned_model_weights_2,
        }

        def mock_from_pretrained(model_path, **kwargs):
            return create_mock_model(model_weights_map[model_path])

        mock_model_class.from_pretrained.side_effect = mock_from_pretrained

        # Test with merge_params=None (should use defaults)
        merger = ModelwiseLinearMerge(std=0.02)
        merger.update_seed(123)

        model_paths = ["/path/to/model1", "/path/to/model2"]

        # Call merge without specifying merge_params
        final_weights = merger.merge(
            base_param=base_model_weights,
            model_paths=model_paths,
            # merge_params=None  # Not provided, should use defaults
        )

        # Verify it worked
        assert final_weights is not None
        assert len(final_weights) == len(base_model_weights)

        # Verify shapes preserved
        for key in final_weights:
            assert final_weights[key].shape == base_model_weights[key].shape

        # Verify models were loaded
        assert mock_model_class.from_pretrained.call_count == 2
