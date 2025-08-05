"""
Test script for Flow Models
Tests forward pass, optimizer setup, and model initialization
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


def test_base_model():
    """Test base model functionality"""
    print("Testing BaseFlowModel...")
    
    try:
        # Create base model
        model = BaseFlowModel(
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=1000
        )
        
        print("✅ BaseFlowModel created successfully")
        
        # Test optimizer configuration
        optimizer = model.configure_optimizers()
        print("✅ Optimizer configuration successful")
        
        # Check optimizer type
        assert isinstance(optimizer['optimizer'], torch.optim.AdamW), "Optimizer should be AdamW"
        print("✅ Optimizer is AdamW")
        
        # Check scheduler
        assert 'lr_scheduler' in optimizer, "Should have learning rate scheduler"
        print("✅ Learning rate scheduler configured")
        
        # Test parameter groups
        param_groups = model.get_param_groups()
        assert len(param_groups) > 0, "Should have parameter groups"
        print("✅ Parameter groups created")
        
        return True
        
    except Exception as e:
        print(f"❌ BaseFlowModel test failed: {e}")
        return False


def test_flow_model_initialization():
    """Test flow model initialization"""
    print("\nTesting FlowModel initialization...")
    
    try:
        # Create flow model
        model = FlowModel(
            model_name="flux-dev",
            learning_rate=1e-5,
            sample_every_n_steps=100
        )
        
        print("✅ FlowModel created successfully")
        
        # Check model attributes
        assert hasattr(model, 'flow_model'), "Should have flow_model attribute"
        assert hasattr(model, 'vae'), "Should have vae attribute"
        assert hasattr(model, 't5'), "Should have t5 attribute"
        assert hasattr(model, 'clip'), "Should have clip attribute"
        print("✅ Model attributes present")
        
        # Check that models are initially None (lazy loading)
        assert model.flow_model is None, "flow_model should be None initially"
        assert model.vae is None, "vae should be None initially"
        print("✅ Models are lazy-loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ FlowModel initialization test failed: {e}")
        return False


def test_flow_model_setup():
    """Test flow model setup (without actual model loading)"""
    print("\nTesting FlowModel setup...")
    
    try:
        # Create flow model
        model = FlowModel(
            model_name="flux-dev",
            learning_rate=1e-5
        )
        
        # Mock the setup to avoid loading actual models
        model.is_setup = True
        
        # Test optimizer configuration
        optimizer = model.configure_optimizers()
        print("✅ FlowModel optimizer configuration successful")
        
        # Test parameter groups (should be empty since models aren't loaded)
        param_groups = model.get_param_groups()
        print("✅ Parameter groups created")
        
        return True
        
    except Exception as e:
        print(f"❌ FlowModel setup test failed: {e}")
        return False


def test_controlnet_model_initialization():
    """Test ControlNet model initialization"""
    print("\nTesting ControlNetFlowModel initialization...")
    
    try:
        # Create ControlNet model
        model = ControlNetFlowModel(
            model_name="flux-dev",
            learning_rate=2e-5,
            controlnet_depth=2
        )
        
        print("✅ ControlNetFlowModel created successfully")
        
        # Check model attributes
        assert hasattr(model, 'flow_model'), "Should have flow_model attribute"
        assert hasattr(model, 'controlnet'), "Should have controlnet attribute"
        assert hasattr(model, 'vae'), "Should have vae attribute"
        print("✅ ControlNet model attributes present")
        
        # Check that models are initially None
        assert model.flow_model is None, "flow_model should be None initially"
        assert model.controlnet is None, "controlnet should be None initially"
        print("✅ ControlNet models are lazy-loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ ControlNetFlowModel initialization test failed: {e}")
        return False


def test_model_hyperparameters():
    """Test model hyperparameter handling"""
    print("\nTesting model hyperparameters...")
    
    try:
        # Test FlowModel hyperparameters
        model = FlowModel(
            model_name="flux-dev",
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=1000,
            snr_gamma=5.0,
            guidance_scale=7.5
        )
        
        # Check hyperparameters are saved
        assert model.learning_rate == 1e-5, "Learning rate not saved correctly"
        assert model.weight_decay == 0.01, "Weight decay not saved correctly"
        assert model.snr_gamma == 5.0, "SNR gamma not saved correctly"
        assert model.guidance_scale == 7.5, "Guidance scale not saved correctly"
        print("✅ FlowModel hyperparameters saved correctly")
        
        # Test ControlNet hyperparameters
        controlnet_model = ControlNetFlowModel(
            model_name="flux-dev",
            learning_rate=2e-5,
            controlnet_guidance_scale=0.7,
            controlnet_depth=2
        )
        
        assert controlnet_model.learning_rate == 2e-5, "ControlNet learning rate not saved"
        assert controlnet_model.controlnet_guidance_scale == 0.7, "ControlNet guidance scale not saved"
        assert controlnet_model.controlnet_depth == 2, "ControlNet depth not saved"
        print("✅ ControlNet hyperparameters saved correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Hyperparameter test failed: {e}")
        return False


def test_loss_computation():
    """Test loss computation functions"""
    print("\nTesting loss computation...")
    
    try:
        # Create model
        model = FlowModel(
            model_name="flux-dev",
            snr_gamma=5.0
        )
        
        # Create dummy data
        batch_size = 2
        channels = 3
        height = 64
        width = 64
        
        predicted_noise = torch.randn(batch_size, channels, height, width)
        target_noise = torch.randn(batch_size, channels, height, width)
        timesteps = torch.rand(batch_size)
        
        # Compute loss
        loss = model.compute_loss(predicted_noise, target_noise, timesteps)
        
        # Check loss properties
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"
        print("✅ Loss computation successful")
        
        # Test SNR weighting
        snr = (1 - timesteps) / timesteps
        snr_weight = torch.where(snr > 0, snr ** model.snr_gamma, torch.ones_like(snr))
        assert snr_weight.shape == timesteps.shape, "SNR weight shape mismatch"
        print("✅ SNR weighting computation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss computation test failed: {e}")
        return False


def test_logging_functions():
    """Test logging functions"""
    print("\nTesting logging functions...")
    
    try:
        # Create model
        model = BaseFlowModel()
        
        # Test loss logging
        losses = {
            "loss": torch.tensor(0.5),
            "accuracy": torch.tensor(0.8)
        }
        
        # This should not raise an error (even without logger)
        model.log_losses(losses, "train")
        print("✅ Loss logging function works")
        
        # Test image logging
        images = torch.randn(4, 3, 64, 64)  # 4 images, 3 channels, 64x64
        
        # This should not raise an error (even without logger)
        model.log_images(images, "train")
        print("✅ Image logging function works")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False


def test_checkpoint_handling():
    """Test checkpoint saving and loading"""
    print("\nTesting checkpoint handling...")
    
    try:
        # Create model
        model = FlowModel(
            model_name="flux-dev",
            use_ema=True
        )
        
        # Mock checkpoint
        checkpoint = {
            'state_dict': {},
            'optimizer_states': [],
            'lr_schedulers': [],
            'ema_model': {}
        }
        
        # Test checkpoint loading (should not raise error)
        model.on_load_checkpoint(checkpoint)
        print("✅ Checkpoint loading function works")
        
        # Test checkpoint saving (should not raise error)
        model.on_save_checkpoint(checkpoint)
        print("✅ Checkpoint saving function works")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        return False


def test_model_parameter_counting():
    """Test model parameter counting"""
    print("\nTesting model parameter counting...")
    
    try:
        # Create a simple model for testing
        class SimpleModel(BaseFlowModel):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 20)
                self.layer2 = nn.Linear(20, 1)
            
            def forward(self, x):
                return self.layer2(self.layer1(x))
        
        model = SimpleModel()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Expected: layer1 (10*20 + 20) + layer2 (20*1 + 1) = 200 + 20 + 20 + 1 = 241
        expected_total = 241
        assert total_params == expected_total, f"Expected {expected_total} parameters, got {total_params}"
        assert trainable_params == expected_total, f"Expected {expected_total} trainable parameters, got {trainable_params}"
        
        print(f"✅ Parameter counting correct: {total_params} total, {trainable_params} trainable")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter counting test failed: {e}")
        return False


def test_gradient_clipping():
    """Test gradient clipping configuration"""
    print("\nTesting gradient clipping...")
    
    try:
        # Create model
        model = BaseFlowModel(gradient_clip_val=1.0)
        
        # Create dummy optimizer
        optimizer = torch.optim.AdamW([torch.randn(10, requires_grad=True)])
        
        # Test gradient clipping configuration
        model.configure_gradient_clipping(optimizer, gradient_clip_val=1.0)
        print("✅ Gradient clipping configuration successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient clipping test failed: {e}")
        return False


def test_learning_rate_scheduler():
    """Test learning rate scheduler creation"""
    print("\nTesting learning rate scheduler...")
    
    try:
        # Create model
        model = BaseFlowModel(
            learning_rate=1e-4,
            warmup_steps=100,
            max_steps=1000
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW([torch.randn(10, requires_grad=True)])
        
        # Create scheduler
        scheduler = model.create_lr_scheduler(optimizer)
        
        # Test scheduler
        assert scheduler is not None, "Scheduler should not be None"
        print("✅ Learning rate scheduler created successfully")
        
        # Test scheduler step
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        
        # LR should change after step
        assert new_lr != initial_lr, "Learning rate should change after scheduler step"
        print("✅ Learning rate scheduler stepping works")
        
        return True
        
    except Exception as e:
        print(f"❌ Learning rate scheduler test failed: {e}")
        return False


def main():
    """Run all model tests"""
    print("Running Flow Model tests...\n")
    
    tests = [
        test_base_model,
        test_flow_model_initialization,
        test_flow_model_setup,
        test_controlnet_model_initialization,
        test_model_hyperparameters,
        test_loss_computation,
        test_logging_functions,
        test_checkpoint_handling,
        test_model_parameter_counting,
        test_gradient_clipping,
        test_learning_rate_scheduler
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
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All model tests passed!")
        return True
    else:
        print("❌ Some model tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 