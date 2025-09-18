#!/usr/bin/env python3
"""
Real test for QuantizedRLModelLoader.quantized_rl_load_weights function.

This test:
1. Tests quantized_rl_load_weights function directly
2. Verifies the complete workflow: load from disk -> weight updates -> FlashRL state management
3. Uses existing test utilities to reduce code duplication
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from typing import Iterator, Tuple, Dict, Any

# Use existing SGLang test utilities
from sglang.test.test_utils import (
    CustomTestCase,  # Provides retry logic and better error handling
    auto_config_device,  # Standardized device detection
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,  # Standardized test model
    try_cached_model,  # Support for cached models
)


class TestQuantizedRLLoaderFunction(CustomTestCase):
    """Test the actual QuantizedRLModelLoader.quantized_rl_load_weights function."""
    
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
        from sglang.srt.model_loader.loader import QuantizedRLModelLoader
        
        self.loader_class = QuantizedRLModelLoader
        
        # Use standardized device detection
        device_str = auto_config_device()
        self.device = torch.device(device_str)
    
    def test_quantized_rl_load_weights_first_time(self):
        """Test first time loading with quantized_rl_load_weights."""
        # Create a test model
        model = self._create_test_model()
        
        # Verify initial state (no FlashRL state)
        self._assert_no_flashrl_state(model)
        
        # Create weights to load (simulating loading from disk)
        weights = self._create_weights_from_disk_simulation(model)
        
        # Test the actual quantized_rl_load_weights function
        result = self.loader_class.quantized_rl_load_weights(
            model, weights, model.load_weights
        )
        
        # Verify FlashRL state was initialized
        self._assert_flashrl_state_initialized(model)
    
    def test_quantized_rl_load_weights_subsequent_updates(self):
        """Test subsequent weight updates using quantized_rl_load_weights."""
        # Create and initialize model
        model = self._create_test_model()
        
        # First load to initialize FlashRL state
        self._initialize_model_with_flashrl(model)
        
        # Record initial parameter values for comparison
        initial_param_values = self._capture_model_state(model)
        
        # Simulate multiple weight updates (like in RL training)
        num_updates = 3
        for i in range(num_updates):
            # Create updated weights (simulating new weights from RL training)
            updated_weights = self._create_updated_weights_simulation(model, update_id=i+1)
            
            # Call quantized_rl_load_weights for subsequent update
            result = self.loader_class.quantized_rl_load_weights(
                model, updated_weights, model.load_weights
            )
            
            # Verify FlashRL state is preserved and weights changed
            self._assert_flashrl_state_preserved(model)
            self._assert_weights_changed(model, initial_param_values, f"update {i+1}")
    
    def test_quantized_rl_reset_and_reload(self):
        """Test FlashRL state reset and reload (forced re-quantization)."""
        # Create and initialize model
        model = self._create_test_model()
        
        # Initial load
        self._initialize_model_with_flashrl(model)
        self._assert_flashrl_state_initialized(model)
        
        # Reset FlashRL state
        reset_result = self.loader_class.quantized_rl_reset_state(model)
        self.assertTrue(reset_result, "Reset should return True for initialized model")
        
        # Load weights again (should trigger re-initialization)
        new_weights = self._create_updated_weights_simulation(model, update_id=999)
        result = self.loader_class.quantized_rl_load_weights(
            model, new_weights, model.load_weights
        )
        
        # Verify FlashRL state was re-established
        self._assert_flashrl_state_initialized(model)
    
    # Helper methods to reduce code duplication
    
    def _create_test_model(self) -> nn.Module:
        """Create a standardized test model similar to real transformer models."""
        model = TestTransformerModel(
            vocab_size=1000,
            hidden_size=512,
            intermediate_size=2048,
            dtype=torch.bfloat16,
            device=self.device
        )
        return model
    
    def _initialize_model_with_flashrl(self, model: nn.Module) -> None:
        """Initialize model with FlashRL state using first load."""
        initial_weights = self._create_weights_from_disk_simulation(model)
        self.loader_class.quantized_rl_load_weights(
            model, initial_weights, model.load_weights
        )
    
    def _create_weights_from_disk_simulation(self, model: nn.Module) -> Iterator[Tuple[str, torch.Tensor]]:
        """Simulate loading weights from disk."""
        weights = []
        for name, param in model.named_parameters():
            # Simulate weights loaded from checkpoint file
            disk_weight = torch.randn_like(param) * 0.02  # Small random weights
            weights.append((name, disk_weight))
        
        return iter(weights)
    
    def _create_updated_weights_simulation(self, model: nn.Module, update_id: int = 1) -> Iterator[Tuple[str, torch.Tensor]]:
        """Simulate updated weights from RL training."""
        weights = []
        for name, param in model.named_parameters():
            # Simulate updated weights with slight modifications
            updated_weight = param.data.clone() + torch.randn_like(param) * 0.001 * update_id
            weights.append((name, updated_weight))
        
        return iter(weights)
    
    def _capture_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Capture current model parameter values for comparison."""
        return {name: param.data.clone() for name, param in model.named_parameters()}
    
    # Assertion helpers for better test readability
    
    def _assert_no_flashrl_state(self, model: nn.Module) -> None:
        """Assert that model has no FlashRL state."""
        self.assertFalse(hasattr(model, 'original_weights_rebuild_keys'))
    
    def _assert_flashrl_state_initialized(self, model: nn.Module) -> None:
        """Assert that FlashRL state was properly initialized."""
        self.assertTrue(hasattr(model, 'original_weights_rebuild_keys'))
        self.assertGreater(len(model.original_weights_rebuild_keys), 0)
        self.assertTrue(hasattr(model, 'recorded_loader'))
    
    def _assert_flashrl_state_preserved(self, model: nn.Module) -> None:
        """Assert that FlashRL state is preserved during updates."""
        self.assertTrue(hasattr(model, 'original_weights_rebuild_keys'))
        self.assertTrue(hasattr(model, 'recorded_loader'))
    
    def _assert_weights_changed(self, model: nn.Module, initial_values: Dict[str, torch.Tensor], context: str) -> None:
        """Assert that model weights have changed from initial values."""
        weights_changed = False
        for name, param in model.named_parameters():
            if not torch.equal(param.data, initial_values[name]):
                weights_changed = True
                break
        
        self.assertTrue(weights_changed, f"Weights should have changed in {context}")


class TestTransformerModel(nn.Module):
    """Standardized test transformer model for quantized RL loader testing."""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 512, 
                 intermediate_size: int = 2048, dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = None):
        super().__init__()
        
        # Simulate transformer layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=dtype)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, dtype=dtype),
            nn.Linear(intermediate_size, hidden_size, dtype=dtype)
        ])
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
                    print(f"Warning: Shape mismatch for {name}: {tensor.shape} vs {param.shape}")
        
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
            setattr(param, 'fp8_quantized', True)
            setattr(param, 'quantization_step', self.process_count)


def main():
    """Run the QuantizedRLModelLoader function tests."""
    print("Testing QuantizedRLModelLoader.quantized_rl_load_weights Function")
    print("=" * 70)
    print("This test verifies:")
    print("1. First-time loading with FlashRL state initialization")
    print("2. Subsequent weight updates using FlashRL rebinding")
    print("3. State reset and reload functionality")
    print("=" * 70)
    
    # Run tests
    import unittest
    unittest.main(verbosity=2, exit=False)


if __name__ == "__main__":
    main()
