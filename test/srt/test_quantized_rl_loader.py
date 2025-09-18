#!/usr/bin/env python3
"""
Comprehensive tests for QuantizedRLModelLoader integration with SGLang.

Tests the complete workflow from configuration to weight updates,
including automatic model.load_weights override and FlashRL state management.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn as nn

# Use existing SGLang test utilities
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,  # Standardized test model
)
from sglang.test.test_utils import (
    CustomTestCase,  # Provides retry logic and better error handling
)
from sglang.test.test_utils import auto_config_device  # Standardized device detection
from sglang.test.test_utils import try_cached_model  # Support for cached models


class TestQuantizedRLLoaderIntegration(CustomTestCase):
    """Test QuantizedRLModelLoader integration with SGLang configuration system."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Use standardized model with caching support
        cls.model_path = try_cached_model(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        # Setup SGLang path
        cls._setup_sglang_path()

    @classmethod
    def _setup_sglang_path(cls):
        """Setup SGLang path for testing."""
        current_dir = Path(__file__).parent
        sglang_path = (current_dir / "../../../python").resolve()
        if str(sglang_path) not in sys.path:
            sys.path.insert(0, str(sglang_path))

    def setUp(self):
        """Set up each test."""
        # Import SGLang components
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader import get_model, get_model_loader
        from sglang.srt.model_loader.loader import QuantizedRLModelLoader

        self.LoadConfig = LoadConfig
        self.LoadFormat = LoadFormat
        self.ModelConfig = ModelConfig
        self.DeviceConfig = DeviceConfig
        self.get_model = get_model
        self.get_model_loader = get_model_loader
        self.QuantizedRLModelLoader = QuantizedRLModelLoader

        # Use standardized device detection
        device_str = auto_config_device()
        self.device = torch.device(device_str)

    def test_complete_quantized_rl_workflow(self):
        """Test the complete QuantizedRL workflow: loader selection, model loading, and weight updates."""
        # 1. Verify LoadFormat.FLASHRL routes to QuantizedRLModelLoader
        load_config = self.LoadConfig(load_format=self.LoadFormat.FLASHRL)
        loader = self.get_model_loader(load_config)
        self.assertIsInstance(loader, self.QuantizedRLModelLoader)

        # 2. Load model using QuantizedRLModelLoader
        device_config = self.DeviceConfig(device=self.device.type)

        # Patch _initialize_model to return our test model
        test_model = self._create_test_model()
        import sglang.srt.model_loader.loader as loader_module

        original_initialize_model = loader_module._initialize_model
        loader_module._initialize_model = lambda model_config, load_config: test_model

        try:
            model = loader.load_model(
                model_config=self._create_mock_model_config(),
                device_config=device_config,
            )

            # Verify FlashRL state was initialized
            self._assert_flashrl_state_initialized(model)
        finally:
            loader_module._initialize_model = original_initialize_model

        # 3. Test weight updates with overridden load_weights method
        initial_state = self._capture_model_state(model)

        # First update
        update1_weights = self._create_updated_weights_simulation(model, update_id=1)
        model.load_weights(update1_weights)
        self._assert_flashrl_state_preserved(model)
        self._assert_weights_changed(model, initial_state, "first update")

        # Second update
        update1_state = self._capture_model_state(model)
        update2_weights = self._create_updated_weights_simulation(model, update_id=2)
        model.load_weights(update2_weights)
        self._assert_flashrl_state_preserved(model)
        self._assert_weights_changed(model, update1_state, "second update")

    def test_quantized_rl_state_reset_and_recovery(self):
        """Test FlashRL state reset and recovery workflow."""
        load_config = self.LoadConfig(load_format=self.LoadFormat.FLASHRL)
        loader = self.get_model_loader(load_config)
        device_config = self.DeviceConfig(device=self.device.type)

        # Patch _initialize_model to return our test model
        test_model = self._create_test_model()
        import sglang.srt.model_loader.loader as loader_module

        original_initialize_model = loader_module._initialize_model
        loader_module._initialize_model = lambda model_config, load_config: test_model

        try:
            model = loader.load_model(
                model_config=self._create_mock_model_config(),
                device_config=device_config,
            )
            self._assert_flashrl_state_initialized(model)
        finally:
            loader_module._initialize_model = original_initialize_model

        # Reset FlashRL state
        reset_result = loader.quantized_rl_reset_state(model)
        self.assertTrue(reset_result, "Reset should return True for initialized model")

        # Load weights again - should re-initialize FlashRL state
        new_weights = self._create_updated_weights_simulation(model, update_id=999)
        model.load_weights(new_weights)
        self._assert_flashrl_state_initialized(model)

        # Verify subsequent updates still work after reset
        post_reset_state = self._capture_model_state(model)
        final_weights = self._create_updated_weights_simulation(model, update_id=1000)
        model.load_weights(final_weights)
        self._assert_flashrl_state_preserved(model)
        self._assert_weights_changed(model, post_reset_state, "post-reset update")

    # Helper methods to reduce code duplication

    def _create_mock_model_config(self):
        """Create a mock model config for testing."""

        # This would normally be a real ModelConfig, but for testing we'll create a minimal mock
        class MockModelConfig:
            def __init__(self, model_path):
                self.model_path = model_path
                self.revision = None
                self.dtype = torch.bfloat16

        return MockModelConfig(self.model_path)

    def _create_test_model(self) -> nn.Module:
        """Create a standardized test model similar to real transformer models."""
        model = TestTransformerModel(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=2048,
            dtype=torch.bfloat16,
            device=self.device,
        )
        return model

    def _create_weights_from_disk_simulation(
        self, model: nn.Module
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Simulate loading weights from disk."""
        weights = []
        for name, param in model.named_parameters():
            # Simulate weights loaded from checkpoint file
            disk_weight = torch.randn_like(param) * 0.02  # Small random weights
            weights.append((name, disk_weight))

        return iter(weights)

    def _create_updated_weights_simulation(
        self, model: nn.Module, update_id: int = 1
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Simulate updated weights from RL training."""
        weights = []
        for name, param in model.named_parameters():
            # Create significantly different weights to ensure they're detectable
            updated_weight = torch.randn_like(param) * 0.1 + update_id * 0.1
            weights.append((name, updated_weight))

        return iter(weights)

    def _capture_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Capture current model parameter values for comparison."""
        return {name: param.data.clone() for name, param in model.named_parameters()}

    # Assertion helpers for better test readability

    def _assert_no_flashrl_state(self, model: nn.Module) -> None:
        """Assert that model has no FlashRL state."""
        self.assertFalse(hasattr(model, "original_weights_rebuild_keys"))

    def _assert_flashrl_state_initialized(self, model: nn.Module) -> None:
        """Assert that FlashRL state was properly initialized."""
        self.assertTrue(hasattr(model, "original_weights_rebuild_keys"))
        self.assertGreater(len(model.original_weights_rebuild_keys), 0)
        self.assertTrue(hasattr(model, "recorded_loader"))

    def _assert_flashrl_state_preserved(self, model: nn.Module) -> None:
        """Assert that FlashRL state is preserved during updates."""
        self.assertTrue(hasattr(model, "original_weights_rebuild_keys"))
        self.assertTrue(hasattr(model, "recorded_loader"))

    def _assert_weights_changed(
        self, model: nn.Module, initial_values: Dict[str, torch.Tensor], context: str
    ) -> None:
        """Assert that model weights have changed from initial values."""
        weights_changed = False
        changed_params = []
        unchanged_params = []

        for name, param in model.named_parameters():
            if not torch.equal(param.data, initial_values[name]):
                weights_changed = True
                changed_params.append(name)
            else:
                unchanged_params.append(name)

        if not weights_changed:
            print(f"DEBUG: No weights changed in {context}")
            print(f"DEBUG: Total parameters: {len(list(model.named_parameters()))}")
            print(
                f"DEBUG: Unchanged parameters: {unchanged_params[:5]}..."
            )  # Show first 5

        self.assertTrue(
            weights_changed,
            f"Weights should have changed in {context}. Changed: {len(changed_params)}, Unchanged: {len(unchanged_params)}",
        )


class TestTransformerModel(nn.Module):
    """Standardized test transformer model for quantized RL loader testing."""

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        super().__init__()

        # Simulate transformer layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, intermediate_size, dtype=dtype),
                nn.Linear(intermediate_size, hidden_size, dtype=dtype),
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, dtype=dtype)
        self.lm_head = nn.Linear(hidden_size, vocab_size, dtype=dtype)

        # Add quantization methods to simulate FP8 quantization
        for layer in self.layers:
            layer.quant_method = MockQuantMethod()
        self.lm_head.quant_method = MockQuantMethod()

        if device is not None:
            self.to(device)

    def load_weights(self, weights: Iterator[Tuple[str, torch.Tensor]]) -> set:
        """Load weights into the model."""
        loaded_params = set()
        state_dict = dict(self.named_parameters())

        for name, tensor in weights:
            if name in state_dict:
                param = state_dict[name]
                if tensor.shape == param.shape:
                    param.data.copy_(tensor.to(param.device, param.dtype))
                    loaded_params.add(name)
                else:
                    print(
                        f"Warning: Shape mismatch for {name}: {tensor.shape} vs {param.shape}"
                    )

        return loaded_params


class MockQuantMethod:
    """Mock quantization method for testing."""

    def __init__(self):
        self.process_count = 0

    def process_weights_after_loading(self, module: nn.Module) -> None:
        """Simulate FP8 quantization process."""
        self.process_count += 1

        # Add FP8 metadata to simulate quantization
        for name, param in module.named_parameters():
            setattr(param, "fp8_quantized", True)
            setattr(param, "quantization_step", self.process_count)


def main():
    """Run focused QuantizedRLModelLoader integration tests."""
    print("Testing QuantizedRLModelLoader Integration")
    print("=" * 45)
    print("Focused tests covering:")
    print("• Complete workflow: config → loader → model loading → weight updates")
    print("• FlashRL state management and reset functionality")
    print("• Automatic model.load_weights override in real workflows")
    print("=" * 45)

    # Run tests
    import unittest

    unittest.main(verbosity=2, exit=False)


if __name__ == "__main__":
    main()
