"""
Tests for utils/helpers.py

Covers state dict conversions (HF to vLLM format), weight loading,
archive management (save/load), model cleanup, and generation tracking.
"""

import pytest
import torch
import json
import os
import re
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import dataclass
from pathlib import Path

# Import functions to test
from utils.helpers import (
    state_dict_hf_to_vllm_qwen,
    state_dict_hf_to_vllm_llama,
    load_hf_params_to_vllm,
    load_qwen_hf_params_to_vllm,
    update_vllm_weights_general,
    save_archive_map,
    load_archive_map,
    delete_models_not_in_archive,
    delete_outdated_models,
    cleanup_old_models,
    get_latest_generation,
    get_largest_gen_file,
)


# ============================================================================
# State Dict Conversions
# ============================================================================


class TestStateDictConversions:
    """Tests for HF to vLLM state dict conversion functions."""

    def test_state_dict_hf_to_vllm_qwen_basic_conversion(
        self, mock_hf_state_dict_qwen_simple
    ):
        """Test basic Qwen HF to vLLM conversion.

        Setup:
            - HF state dict with q, k, v projections
            - HF state dict with gate, up projections

        Expected:
            - qkv_proj created by concatenating q, k, v
            - gate_up_proj created by concatenating gate, up
            - Other weights copied as-is
        """
        result = state_dict_hf_to_vllm_qwen(mock_hf_state_dict_qwen_simple)

        # Verify qkv projection was created
        assert "model.layers.0.self_attn.qkv_proj.weight" in result
        assert result["model.layers.0.self_attn.qkv_proj.weight"].shape == (
            512 * 3,
            512,
        )

        # Verify gate_up projection was created
        assert "model.layers.0.mlp.gate_up_proj.weight" in result
        assert result["model.layers.0.mlp.gate_up_proj.weight"].shape == (2048 * 2, 512)

        # Verify o_proj and down_proj copied correctly
        assert "model.layers.0.self_attn.o_proj.weight" in result
        assert "model.layers.0.mlp.down_proj.weight" in result

    def test_state_dict_hf_to_vllm_qwen_with_biases(
        self, mock_hf_state_dict_qwen_with_bias
    ):
        """Test Qwen conversion with bias terms.

        Setup:
            - HF state dict with q, k, v biases

        Expected:
            - qkv_proj.bias created by concatenating q, k, v biases
        """
        result = state_dict_hf_to_vllm_qwen(mock_hf_state_dict_qwen_with_bias)

        # Verify qkv bias was created
        assert "model.layers.0.self_attn.qkv_proj.bias" in result
        assert result["model.layers.0.self_attn.qkv_proj.bias"].shape == (512 * 3,)

        # Verify weight still created correctly
        assert "model.layers.0.self_attn.qkv_proj.weight" in result

    def test_state_dict_hf_to_vllm_qwen_lm_head_handling(
        self, mock_hf_state_dict_qwen_no_lm_head
    ):
        """Test Qwen conversion when lm_head is missing (word embedding tying).

        Setup:
            - HF state dict without lm_head.weight
            - Has model.embed_tokens.weight

        Expected:
            - lm_head.weight copied from model.embed_tokens.weight
            - Warning logged about word embedding tying
        """
        with patch("utils.helpers.logger") as mock_logger:
            result = state_dict_hf_to_vllm_qwen(mock_hf_state_dict_qwen_no_lm_head)

            # Verify lm_head was added
            assert "lm_head.weight" in result
            assert torch.equal(
                result["lm_head.weight"], result["model.embed_tokens.weight"]
            )

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "lm_head.weight not found" in str(mock_logger.warning.call_args)

    def test_state_dict_hf_to_vllm_llama_basic_conversion(
        self, mock_hf_state_dict_llama
    ):
        """Test basic Llama HF to vLLM conversion.

        Setup:
            - Llama HF state dict (no biases)

        Expected:
            - qkv_proj created without biases
            - gate_up_proj created
            - All weights properly concatenated
        """
        result = state_dict_hf_to_vllm_llama(mock_hf_state_dict_llama)

        # Verify qkv projection (no bias for Llama)
        assert "model.layers.0.self_attn.qkv_proj.weight" in result
        assert "model.layers.0.self_attn.qkv_proj.bias" not in result

        # Verify gate_up projection
        assert "model.layers.0.mlp.gate_up_proj.weight" in result

        # Verify embed_tokens copied
        assert "model.embed_tokens.weight" in result

    def test_state_dict_hf_to_vllm_llama_qkv_concatenation(
        self, mock_hf_state_dict_llama
    ):
        """Test that Llama q, k, v projections are correctly concatenated.

        Setup:
            - Known q, k, v weight values

        Expected:
            - qkv_proj is torch.cat([q, k, v], dim=0)
        """
        result = state_dict_hf_to_vllm_llama(mock_hf_state_dict_llama)

        # Get original tensors
        q = mock_hf_state_dict_llama["model.layers.0.self_attn.q_proj.weight"]
        k = mock_hf_state_dict_llama["model.layers.0.self_attn.k_proj.weight"]
        v = mock_hf_state_dict_llama["model.layers.0.self_attn.v_proj.weight"]

        # Verify concatenation
        expected_qkv = torch.cat([q, k, v], dim=0)
        assert torch.equal(
            result["model.layers.0.self_attn.qkv_proj.weight"], expected_qkv
        )

    def test_state_dict_hf_to_vllm_llama_gate_up_concatenation(
        self, mock_hf_state_dict_llama
    ):
        """Test that Llama gate and up projections are correctly concatenated.

        Setup:
            - Known gate, up weight values

        Expected:
            - gate_up_proj is torch.cat([gate, up], dim=0)
        """
        result = state_dict_hf_to_vllm_llama(mock_hf_state_dict_llama)

        # Get original tensors
        gate = mock_hf_state_dict_llama["model.layers.0.mlp.gate_proj.weight"]
        up = mock_hf_state_dict_llama["model.layers.0.mlp.up_proj.weight"]

        # Verify concatenation
        expected_gate_up = torch.cat([gate, up], dim=0)
        assert torch.equal(
            result["model.layers.0.mlp.gate_up_proj.weight"], expected_gate_up
        )

    def test_state_dict_conversion_preserves_other_keys(
        self, mock_hf_state_dict_qwen_simple
    ):
        """Test that non-attention, non-MLP keys are preserved.

        Setup:
            - State dict with layernorm, embed_tokens, etc.

        Expected:
            - All non-converted keys copied as-is
        """
        result = state_dict_hf_to_vllm_qwen(mock_hf_state_dict_qwen_simple)

        # Verify layernorms preserved
        assert "model.layers.0.input_layernorm.weight" in result
        assert "model.layers.0.post_attention_layernorm.weight" in result

        # Verify embeddings preserved
        assert "model.embed_tokens.weight" in result
        assert "model.norm.weight" in result

    def test_state_dict_conversion_with_extra_keys(self, mock_hf_state_dict_llama):
        """Test conversion with extra keys that should be copied as-is.

        Setup:
            - State dict with some custom keys

        Expected:
            - Extra keys copied without modification
        """
        # Add a custom key
        mock_hf_state_dict_llama["custom.layer.weight"] = torch.randn(10, 10)

        result = state_dict_hf_to_vllm_llama(mock_hf_state_dict_llama)

        # Verify custom key was preserved
        assert "custom.layer.weight" in result


# ============================================================================
# Weight Loading
# ============================================================================


class TestWeightLoading:
    """Tests for weight loading functions that transfer HF params to vLLM."""

    def test_load_hf_params_to_vllm_basic_conversion(
        self, mock_vllm_for_weight_loading, mock_hf_params_full
    ):
        """Test load_hf_params_to_vllm transfers parameters correctly.

        Setup:
            - Mock vLLM model with 2 layers
            - Complete HF parameter dict

        Expected:
            - All parameters accessed via get_parameter
            - Parameters copied with correct dtype and device
        """
        mock_llm, mock_model = mock_vllm_for_weight_loading

        load_hf_params_to_vllm(mock_hf_params_full, mock_llm)

        # Verify embeddings were accessed
        calls = [str(c) for c in mock_model.get_parameter.call_args_list]
        assert any("embed_tokens" in c for c in calls)
        assert any("lm_head" in c for c in calls)

        # Verify at least one layer was processed
        assert any("layers.0" in c for c in calls) or any("layers.1" in c for c in calls)

    def test_load_hf_params_to_vllm_dtype_conversion(
        self, mock_vllm_for_weight_loading, mock_hf_params_full
    ):
        """Test that parameters are converted to model dtype.

        Setup:
            - vLLM model with mocked parameters
            - HF params in float32

        Expected:
            - copy_ is called on model parameters (dtype conversion happens via .to() calls)
        """
        mock_llm, mock_model = mock_vllm_for_weight_loading

        # Don't override get_parameter - use the fixture's side_effect which creates
        # parameters with mocked copy_ methods
        load_hf_params_to_vllm(mock_hf_params_full, mock_llm)

        # Verify that get_parameter was called (parameters were accessed)
        assert mock_model.get_parameter.called

        # Verify copy_ was called on at least one of the created parameters
        # The fixture's get_param function mocks copy_ for each parameter
        copy_called = False
        for call_args in mock_model.get_parameter.call_args_list:
            param_name = call_args[0][0]  # Get the parameter name from the call
            # Get the parameter that was created (it's cached in the fixture)
            param = mock_model.get_parameter(param_name)
            if hasattr(param, 'copy_') and hasattr(param.copy_, 'called') and param.copy_.called:
                copy_called = True
                break

        assert copy_called, "copy_ should have been called on at least one parameter"

    def test_load_hf_params_to_vllm_layer_iteration(
        self, mock_vllm_for_weight_loading, mock_hf_params_full
    ):
        """Test that all layers are processed in the loop.

        Setup:
            - Model with 2 layers

        Expected:
            - Parameters for layers 0 and 1 accessed
        """
        mock_llm, mock_model = mock_vllm_for_weight_loading
        mock_model.config.num_hidden_layers = 2

        load_hf_params_to_vllm(mock_hf_params_full, mock_llm)

        calls = [str(c) for c in mock_model.get_parameter.call_args_list]

        # Verify both layers were processed
        assert any("layers.0" in c for c in calls)
        assert any("layers.1" in c for c in calls)

    def test_load_qwen_hf_params_to_vllm_with_biases(
        self, mock_vllm_for_weight_loading, mock_qwen_params_with_bias
    ):
        """Test load_qwen_hf_params_to_vllm handles biases correctly.

        Setup:
            - Qwen model with qkv biases

        Expected:
            - Both qkv_proj.weight and qkv_proj.bias accessed
        """
        mock_llm, mock_model = mock_vllm_for_weight_loading

        load_qwen_hf_params_to_vllm(mock_qwen_params_with_bias, mock_llm)

        calls = [str(c) for c in mock_model.get_parameter.call_args_list]

        # Verify bias parameters were accessed
        assert any("qkv_proj.bias" in c for c in calls)
        assert any("qkv_proj.weight" in c for c in calls)

    def test_update_vllm_weights_general(
        self, mock_vllm_for_weight_loading, mock_hf_params_full
    ):
        """Test update_vllm_weights_general with custom conversion function.

        Setup:
            - Custom conversion function
            - Mock vLLM model

        Expected:
            - Conversion function called
            - apply_model called with update function
        """
        mock_llm, mock_model = mock_vllm_for_weight_loading

        # Mock conversion function
        def mock_conversion(params):
            return {"converted": torch.randn(10, 10)}

        update_vllm_weights_general(mock_hf_params_full, mock_llm, mock_conversion)

        # Verify apply_model was called
        assert mock_llm.apply_model.call_count == 1

    def test_update_vllm_weights_error_handling(
        self, mock_vllm_for_weight_loading, mock_hf_params_full
    ):
        """Test that weight update errors are properly raised.

        Setup:
            - Mock that raises exception during apply_model

        Expected:
            - Exception re-raised with descriptive message
        """
        mock_llm, mock_model = mock_vllm_for_weight_loading

        # Make apply_model raise an exception
        mock_llm.apply_model.side_effect = RuntimeError("Shape mismatch")

        def mock_conversion(params):
            return params

        with pytest.raises(Exception, match="Failed to update VLLM weights"):
            update_vllm_weights_general(mock_hf_params_full, mock_llm, mock_conversion)


# ============================================================================
# Archive Management
# ============================================================================


class TestArchiveManagement:
    """Tests for archive save/load operations."""

    def test_save_archive_map_serialization(self, tmp_path, mock_archive_data):
        """Test save_archive_map creates valid JSON.

        Setup:
            - Archive data with nested dicts
            - Tuple keys that need conversion to strings

        Expected:
            - JSON file created
            - Tuple keys converted to comma-separated strings
        """
        save_path = tmp_path / "archive_map.json"

        save_archive_map(mock_archive_data, str(save_path))

        # Verify file exists
        assert save_path.exists()

        # Verify JSON structure
        with open(save_path, "r") as f:
            data = json.load(f)

        assert "task_0" in data
        assert "0,1" in data["task_0"]  # Tuple (0, 1) converted to string

    def test_load_archive_map_deserialization(self, tmp_path, mock_archive_data):
        """Test load_archive_map reconstructs data correctly.

        Setup:
            - Saved archive map JSON file

        Expected:
            - Data loaded with tuple keys
            - Dataclass instances reconstructed
        """
        save_path = tmp_path / "archive_map.json"
        save_archive_map(mock_archive_data, str(save_path))

        # Define the dataclass for loading
        @dataclass
        class MockArchive:
            model_path: str
            fitness: float

        loaded = load_archive_map(str(save_path), MockArchive)

        # Verify structure
        assert "task_0" in loaded
        assert (0, 1) in loaded["task_0"]
        assert isinstance(loaded["task_0"][(0, 1)], MockArchive)
        assert loaded["task_0"][(0, 1)].model_path == "/models/gen_1_ind_0"

    def test_save_load_archive_map_round_trip(self, tmp_path, mock_archive_data):
        """Test save → load preserves all data.

        Setup:
            - Complex archive data

        Expected:
            - All keys and values preserved
            - Nested structure intact
        """
        save_path = tmp_path / "archive_map.json"

        @dataclass
        class MockArchive:
            model_path: str
            fitness: float

        # Save
        save_archive_map(mock_archive_data, str(save_path))

        # Load
        loaded = load_archive_map(str(save_path), MockArchive)

        # Verify all data preserved
        assert len(loaded) == len(mock_archive_data)
        for task in mock_archive_data:
            assert task in loaded
            assert len(loaded[task]) == len(mock_archive_data[task])

    def test_archive_map_with_complex_nested_structures(self, tmp_path):
        """Test save/load with multiple tasks and BC dimensions.

        Setup:
            - Archive with 3 tasks, various BC tuples

        Expected:
            - All tuples correctly serialized/deserialized
        """
        @dataclass
        class TestData:
            value: int

        complex_data = {
            "task_0": {(0, 0): TestData(1), (1, 1): TestData(2)},
            "task_1": {(5, 10): TestData(3)},
            "task_2": {(0, 0, 0): TestData(4)},  # 3D BC
        }

        save_path = tmp_path / "complex_archive.json"
        save_archive_map(complex_data, str(save_path))
        loaded = load_archive_map(str(save_path), TestData)

        assert loaded["task_0"][(0, 0)].value == 1
        assert loaded["task_1"][(5, 10)].value == 3
        assert loaded["task_2"][(0, 0, 0)].value == 4


# ============================================================================
# Model Cleanup
# ============================================================================


class TestModelCleanup:
    """Tests for model cleanup and deletion operations."""

    def test_delete_models_not_in_archive_basic_cleanup(self, tmp_path):
        """Test delete_models_not_in_archive removes old models.

        Setup:
            - 4 model directories (gen 1-4)
            - Archive keeps gen 3 and 4
            - Threshold = 3

        Expected:
            - gen_1 and gen_2 deleted
            - gen_3 and gen_4 preserved
        """
        # Create model directories
        for i in range(1, 5):
            (tmp_path / f"gen_{i}_ind_0").mkdir()

        keep_paths = [str(tmp_path / "gen_3_ind_0"), str(tmp_path / "gen_4_ind_0")]

        deleted = delete_models_not_in_archive(
            model_dir=str(tmp_path), keep_model_paths=keep_paths, threshold=3
        )

        assert len(deleted) == 2
        assert not (tmp_path / "gen_1_ind_0").exists()
        assert not (tmp_path / "gen_2_ind_0").exists()
        assert (tmp_path / "gen_3_ind_0").exists()
        assert (tmp_path / "gen_4_ind_0").exists()

    def test_delete_models_with_skip_interval_preservation(self, tmp_path):
        """Test skip_interval preserves every Nth generation.

        Setup:
            - Models gen 1-6
            - skip_interval = 2 (keep gen 2, 4, 6)
            - Threshold = 5

        Expected:
            - gen_1, gen_3 deleted
            - gen_2, gen_4 preserved (multiples of 2)
            - gen_5, gen_6 preserved (above threshold or in keep list)
        """
        for i in range(1, 7):
            (tmp_path / f"gen_{i}_ind_0").mkdir()

        keep_paths = [str(tmp_path / f"gen_{i}_ind_0") for i in [5, 6]]

        deleted = delete_models_not_in_archive(
            model_dir=str(tmp_path),
            keep_model_paths=keep_paths,
            threshold=5,
            skip_interval=2,
        )

        # gen_1, gen_3 should be deleted (odd numbers below threshold)
        assert not (tmp_path / "gen_1_ind_0").exists()
        assert not (tmp_path / "gen_3_ind_0").exists()

        # gen_2, gen_4 preserved (skip_interval)
        assert (tmp_path / "gen_2_ind_0").exists()
        assert (tmp_path / "gen_4_ind_0").exists()

        # gen_5, gen_6 preserved (keep_paths)
        assert (tmp_path / "gen_5_ind_0").exists()
        assert (tmp_path / "gen_6_ind_0").exists()

    def test_delete_models_preserves_parent_mapping_dir(self, tmp_path):
        """Test that parent_models_mapping directory is never deleted.

        Setup:
            - parent_models_mapping directory
            - Old model directories

        Expected:
            - parent_models_mapping preserved
        """
        (tmp_path / "parent_models_mapping").mkdir()
        (tmp_path / "gen_1_ind_0").mkdir()

        deleted = delete_models_not_in_archive(
            model_dir=str(tmp_path), keep_model_paths=[], threshold=5
        )

        assert (tmp_path / "parent_models_mapping").exists()
        assert not (tmp_path / "gen_1_ind_0").exists()

    def test_delete_models_preserves_archive_models(self, tmp_path):
        """Test that models in archive are never deleted.

        Setup:
            - gen_1, gen_2 in archive
            - gen_3 not in archive

        Expected:
            - gen_1, gen_2 preserved
            - gen_3 deleted
        """
        for i in range(1, 4):
            (tmp_path / f"gen_{i}_ind_0").mkdir()

        keep_paths = [str(tmp_path / f"gen_{i}_ind_0") for i in [1, 2]]

        deleted = delete_models_not_in_archive(
            model_dir=str(tmp_path), keep_model_paths=keep_paths, threshold=5
        )

        assert (tmp_path / "gen_1_ind_0").exists()
        assert (tmp_path / "gen_2_ind_0").exists()
        assert not (tmp_path / "gen_3_ind_0").exists()

    def test_delete_outdated_models_from_qd_archive(self, tmp_path):
        """Test delete_outdated_models for QD archive format.

        Setup:
            - QD archive map (dict of dicts with tuple keys)
            - Model directories

        Expected:
            - Models not in archive deleted
        """
        # Create mock QD archive map
        @dataclass
        class QDData:
            model_path: str

        data_map = {
            "task_0": {
                (0, 1): QDData(model_path=str(tmp_path / "gen_3_ind_0")),
                (1, 2): QDData(model_path=str(tmp_path / "gen_3_ind_1")),
            }
        }

        # Create model directories
        for i in range(1, 4):
            (tmp_path / f"gen_{i}_ind_0").mkdir()
        (tmp_path / "gen_3_ind_1").mkdir()

        deleted = delete_outdated_models(
            data_map=data_map, model_dir=str(tmp_path), threshold=3
        )

        # gen_1, gen_2 should be deleted
        assert not (tmp_path / "gen_1_ind_0").exists()
        assert not (tmp_path / "gen_2_ind_0").exists()

        # gen_3 models preserved (in archive)
        assert (tmp_path / "gen_3_ind_0").exists()
        assert (tmp_path / "gen_3_ind_1").exists()

    def test_cleanup_old_models_with_different_modes(self, tmp_path):
        """Test cleanup_old_models delegates to correct function based on mode.

        Setup:
            - Different optimization modes (BDMA, DNS, QD)

        Expected:
            - Correct cleanup function called for each mode
        """
        # This test mocks the cleanup functions to verify correct delegation
        with patch("utils.helpers.delete_models_not_in_archive") as mock_dns, patch(
            "utils.helpers.delete_outdated_models"
        ) as mock_qd:
            # Test DNS mode
            @dataclass
            class DNSSolution:
                model_path: str

            cfg_dns = Mock()
            cfg_dns.run_dns = True
            cfg_dns.run_bdma = False

            archive_data_dns = {
                "dirs": {"model_dir": str(tmp_path)},
                "dns_archive": [DNSSolution(model_path="/models/gen_1_ind_0")],
            }

            cleanup_old_models(cfg_dns, 5, archive_data_dns)
            assert mock_dns.called

            # Test QD mode (run_dns=False, run_bdma=False)
            mock_dns.reset_mock()
            mock_qd.reset_mock()

            cfg_qd = Mock()
            cfg_qd.run_dns = False
            cfg_qd.run_bdma = False

            archive_data_qd = {
                "dirs": {"model_dir": str(tmp_path)},
                "archive_map": {},
            }

            cleanup_old_models(cfg_qd, 5, archive_data_qd)
            assert mock_qd.called


# ============================================================================
# Generation Tracking
# ============================================================================


class TestGenerationTracking:
    """Tests for generation number parsing and tracking."""

    def test_get_latest_generation_basic(self, tmp_path):
        """Test get_latest_generation finds highest generation.

        Setup:
            - Model directories: gen_1, gen_3, gen_2

        Expected:
            - Returns 3
        """
        (tmp_path / "gen_1_ind_0").mkdir()
        (tmp_path / "gen_3_ind_2").mkdir()
        (tmp_path / "gen_2_ind_1").mkdir()
        (tmp_path / "other_file.txt").touch()  # Should be ignored

        latest = get_latest_generation(str(tmp_path))

        assert latest == 3

    def test_get_latest_generation_with_no_models(self, tmp_path):
        """Test get_latest_generation raises error when no models found.

        Setup:
            - Empty directory

        Expected:
            - ValueError raised
        """
        with pytest.raises(ValueError, match="No model files found"):
            get_latest_generation(str(tmp_path))

    def test_get_largest_gen_file(self, tmp_path):
        """Test get_largest_gen_file finds largest archive generation.

        Setup:
            - Archive files: gen1_archive_map.json, gen5_archive_map.json

        Expected:
            - Returns ('gen5_archive_map.json', 5)
        """
        (tmp_path / "gen1_archive_map.json").touch()
        (tmp_path / "gen5_archive_map.json").touch()
        (tmp_path / "gen3_archive_map.json").touch()

        filename, gen_num = get_largest_gen_file(str(tmp_path))

        assert gen_num == 5
        assert filename == "gen5_archive_map.json"

    def test_get_latest_generation_ignores_non_matching_files(self, tmp_path):
        """Test that non-matching files are ignored.

        Setup:
            - gen_5_ind_0 (valid)
            - other_dir (no gen pattern)
            - gen_archive.json (different pattern)

        Expected:
            - Returns 5 (only valid gen_ directory counted)
        """
        (tmp_path / "gen_5_ind_0").mkdir()
        (tmp_path / "other_dir").mkdir()
        (tmp_path / "gen_archive.json").touch()

        latest = get_latest_generation(str(tmp_path))

        assert latest == 5
