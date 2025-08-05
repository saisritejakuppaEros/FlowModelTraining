"""
Test script for Flow Model Forward Pass
Tests forward pass with mock data and model integration
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from typing import Dict, Any

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent))

from base_model import BaseFlowModel
from flow_model import FlowModel
from controlnet_flow_model import ControlNetFlowModel


def create_mock_batch(batch_size=2, img_size=64):
    """Create mock batch data for testing"""
    # Create mock images (normalized to [-1, 1])
    images = torch.randn(batch_size, 3, img_size, img_size) * 2 - 1
    
    # Create mock captions
    captions = [f"test caption {i}" for i in range(batch_size)]
    
    return images, captions


def create_mock_controlnet_batch(batch_size=2, img_size=64):
    """Create mock ControlNet batch data for testing"""
    # Create mock images
    images = torch.randn(batch_size, 3, img_size, img_size) * 2 - 1
    
    # Create mock control signals (Canny edges)
    control_signals = torch.randn(batch_size, 3, img_size, img_size) * 2 - 1
    
    # Create mock captions
    captions = [f"test caption {i}" for i in range(batch_size)]
    
    return images, control_signals, captions


def test_base_model_forward():
    """Test base model forward pass with simple model"""
    print("Testing BaseFlowModel forward pass...")
    
    try:
        # Create a simple model that inherits from BaseFlowModel
        class SimpleTestModel(BaseFlowModel):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 20)
                self.layer2 = nn.Linear(20, 1)
            
            def forward(self, x):
                return self.layer2(self.layer1(x))
            
            def training_step(self, batch, batch_idx):
                x, y = batch
                output = self.forward(x)
                loss = nn.MSELoss()(output, y)
                self.log('train_loss', loss)
                return loss
            
            def validation_step(self, batch, batch_idx):
                x, y = batch
                output = self.forward(x)
                loss = nn.MSELoss()(output, y)
                self.log('val_loss', loss)
                return loss
        
        model = SimpleTestModel()
        
        # Create mock data
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        batch = (x, y)
        
        # Test forward pass
        output = model.forward(x)
        assert output.shape == (4, 1), f"Expected shape (4, 1), got {output.shape}"
        print("✅ BaseFlowModel forward pass successful")
        
        # Test training step
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor), "Training step should return tensor"
        print("✅ BaseFlowModel training step successful")
        
        return True
        
    except Exception as e:
        print(f"❌ BaseFlowModel forward pass test failed: {e}")
        return False


def test_flow_model_forward_without_setup():
    """Test flow model forward pass without actual model loading"""
    print("\nTesting FlowModel forward pass (without setup)...")
    
    try:
        # Create flow model
        model = FlowModel(
            model_name="flux-dev",
            learning_rate=1e-5
        )
        
        # Mock the setup to avoid loading actual models
        model.is_setup = True
        
        # Create mock batch
        images, captions = create_mock_batch()
        batch = (images, captions)
        
        # Test that forward pass raises appropriate error when models aren't loaded
        try:
            model.forward(batch)
            print("❌ Forward pass should fail when models aren't loaded")
            return False
        except (AttributeError, TypeError):
            print("✅ Forward pass correctly fails when models aren't loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ FlowModel forward pass test failed: {e}")
        return False


def test_controlnet_model_forward_without_setup():
    """Test ControlNet model forward pass without actual model loading"""
    print("\nTesting ControlNetFlowModel forward pass (without setup)...")
    
    try:
        # Create ControlNet model
        model = ControlNetFlowModel(
            model_name="flux-dev",
            learning_rate=2e-5
        )
        
        # Mock the setup to avoid loading actual models
        model.is_setup = True
        
        # Create mock batch
        images, control_signals, captions = create_mock_controlnet_batch()
        batch = (images, control_signals, captions)
        
        # Test that forward pass raises appropriate error when models aren't loaded
        try:
            model.forward(batch)
            print("❌ Forward pass should fail when models aren't loaded")
            return False
        except (AttributeError, TypeError):
            print("✅ Forward pass correctly fails when models aren't loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ ControlNetFlowModel forward pass test failed: {e}")
        return False


def test_loss_computation_with_mock_data():
    """Test loss computation with mock data"""
    print("\nTesting loss computation with mock data...")
    
    try:
        # Create model
        model = FlowModel(
            model_name="flux-dev",
            snr_gamma=5.0
        )
        
        # Create mock data
        batch_size = 4
        channels = 3
        height = 64
        width = 64
        
        predicted_noise = torch.randn(batch_size, channels, height, width)
        target_noise = torch.randn(batch_size, channels, height, width)
        timesteps = torch.rand(batch_size)
        
        # Ensure timesteps are valid (not zero)
        timesteps = torch.clamp(timesteps, min=0.01, max=0.99)
        
        # Compute loss
        loss = model.compute_loss(predicted_noise, target_noise, timesteps)
        
        # Check loss properties
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"
        
        print(f"✅ Loss computation successful: {loss.item():.4f}")
        
        # Test with different SNR gamma values
        for gamma in [1.0, 2.0, 5.0, 10.0]:
            model.snr_gamma = gamma
            loss = model.compute_loss(predicted_noise, target_noise, timesteps)
            assert not torch.isnan(loss), f"Loss should not be NaN with gamma={gamma}"
            print(f"✅ Loss computation with gamma={gamma}: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss computation test failed: {e}")
        return False


def test_model_parameter_groups():
    """Test parameter group creation"""
    print("\nTesting parameter group creation...")
    
    try:
        # Create a simple model for testing
        class TestModel(BaseFlowModel):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 20)
                self.layer2 = nn.Linear(20, 1)
                # Freeze one layer
                self.layer1.requires_grad_(False)
            
            def forward(self, x):
                return self.layer2(self.layer1(x))
        
        model = TestModel()
        
        # Get parameter groups
        param_groups = model.get_param_groups()
        
        # Check parameter groups
        assert len(param_groups) == 1, "Should have one parameter group"
        assert 'params' in param_groups[0], "Parameter group should have 'params' key"
        assert 'lr' in param_groups[0], "Parameter group should have 'lr' key"
        
        # Count parameters in group
        params_in_group = param_groups[0]['params']
        expected_params = sum(p.numel() for p in model.layer2.parameters())
        actual_params = sum(p.numel() for p in params_in_group)
        
        assert actual_params == expected_params, f"Expected {expected_params} parameters, got {actual_params}"
        print(f"✅ Parameter groups created correctly: {actual_params} trainable parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter group test failed: {e}")
        return False


def test_optimizer_configuration():
    """Test optimizer configuration with different models"""
    print("\nTesting optimizer configuration...")
    
    try:
        # Test BaseFlowModel optimizer
        base_model = BaseFlowModel(learning_rate=1e-4)
        base_optimizer = base_model.configure_optimizers()
        
        assert 'optimizer' in base_optimizer, "Should have optimizer"
        assert 'lr_scheduler' in base_optimizer, "Should have lr_scheduler"
        assert isinstance(base_optimizer['optimizer'], torch.optim.AdamW), "Should be AdamW"
        print("✅ BaseFlowModel optimizer configuration successful")
        
        # Test FlowModel optimizer (without setup)
        flow_model = FlowModel(model_name="flux-dev", learning_rate=1e-5)
        flow_model.is_setup = True  # Mock setup
        flow_optimizer = flow_model.configure_optimizers()
        
        assert 'optimizer' in flow_optimizer, "Should have optimizer"
        assert 'lr_scheduler' in flow_optimizer, "Should have lr_scheduler"
        print("✅ FlowModel optimizer configuration successful")
        
        # Test ControlNet optimizer (without setup)
        controlnet_model = ControlNetFlowModel(model_name="flux-dev", learning_rate=2e-5)
        controlnet_model.is_setup = True  # Mock setup
        controlnet_optimizer = controlnet_model.configure_optimizers()
        
        assert 'optimizer' in controlnet_optimizer, "Should have optimizer"
        assert 'lr_scheduler' in controlnet_optimizer, "Should have lr_scheduler"
        print("✅ ControlNetFlowModel optimizer configuration successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimizer configuration test failed: {e}")
        return False


def test_model_device_handling():
    """Test model device handling"""
    print("\nTesting model device handling...")
    
    try:
        # Create model
        model = BaseFlowModel()
        
        # Test device property
        device = model.device
        assert device is not None, "Device should not be None"
        print(f"✅ Model device: {device}")
        
        # Test moving model to device (if CUDA available)
        if torch.cuda.is_available():
            model = model.cuda()
            assert model.device.type == 'cuda', "Model should be on CUDA"
            print("✅ Model moved to CUDA successfully")
        else:
            print("✅ CUDA not available, skipping CUDA test")
        
        return True
        
    except Exception as e:
        print(f"❌ Device handling test failed: {e}")
        return False


def test_model_hyperparameter_saving():
    """Test that hyperparameters are properly saved"""
    print("\nTesting hyperparameter saving...")
    
    try:
        # Create model with specific hyperparameters
        model = FlowModel(
            model_name="flux-dev",
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=1000,
            snr_gamma=5.0,
            guidance_scale=7.5,
            sample_every_n_steps=500
        )
        
        # Check that hyperparameters are saved
        assert model.learning_rate == 1e-5, "Learning rate not saved"
        assert model.weight_decay == 0.01, "Weight decay not saved"
        assert model.warmup_steps == 100, "Warmup steps not saved"
        assert model.max_steps == 1000, "Max steps not saved"
        assert model.snr_gamma == 5.0, "SNR gamma not saved"
        assert model.guidance_scale == 7.5, "Guidance scale not saved"
        assert model.sample_every_n_steps == 500, "Sample every n steps not saved"
        
        print("✅ All hyperparameters saved correctly")
        
        # Check that hyperparameters are in hparams
        hparams = model.hparams
        assert hparams.learning_rate == 1e-5, "Learning rate not in hparams"
        assert hparams.weight_decay == 0.01, "Weight decay not in hparams"
        
        print("✅ Hyperparameters in hparams correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Hyperparameter saving test failed: {e}")
        return False


def main():
    """Run all forward pass tests"""
    print("Running Flow Model Forward Pass tests...\n")
    
    tests = [
        test_base_model_forward,
        test_flow_model_forward_without_setup,
        test_controlnet_model_forward_without_setup,
        test_loss_computation_with_mock_data,
        test_model_parameter_groups,
        test_optimizer_configuration,
        test_model_device_handling,
        test_model_hyperparameter_saving
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
        print()
    
    print(f"Forward Pass Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All forward pass tests passed!")
        return True
    else:
        print("❌ Some forward pass tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 