"""
Test script for configuration loading
Tests if config.yaml and model_config.yaml load properly and contain expected fields
"""

import sys
import os
from pathlib import Path

# Add the configs directory to the path
sys.path.append(str(Path(__file__).parent))

from config_loader import ConfigLoader, validate_and_load_config


def test_config_loading():
    """Test basic configuration loading"""
    print("Testing configuration loading...")
    
    loader = ConfigLoader()
    
    # Test loading base config
    try:
        config = loader.load_config("config")
        print("✅ Base config loaded successfully")
        
        # Check required fields
        required_fields = [
            "model_name", "train_batch_size", "max_train_steps", 
            "learning_rate", "lr_scheduler", "data_config"
        ]
        
        for field in required_fields:
            if hasattr(config, field):
                print(f"✅ Found required field: {field}")
            else:
                print(f"❌ Missing required field: {field}")
                return False
                
    except Exception as e:
        print(f"❌ Failed to load base config: {e}")
        return False
    
    # Test loading model config
    try:
        model_config = loader.load_model_config()
        print("✅ Model config loaded successfully")
        
        # Check required fields
        required_model_fields = [
            "flux_params", "model_variants", "flow_params", "loss_params"
        ]
        
        for field in required_model_fields:
            if hasattr(model_config, field):
                print(f"✅ Found required model field: {field}")
            else:
                print(f"❌ Missing required model field: {field}")
                return False
                
    except Exception as e:
        print(f"❌ Failed to load model config: {e}")
        return False
    
    return True


def test_config_validation():
    """Test configuration validation"""
    print("\nTesting configuration validation...")
    
    loader = ConfigLoader()
    
    try:
        config = loader.load_config("config")
        loader.validate_config(config)
        print("✅ Configuration validation passed")
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def test_model_params():
    """Test model parameters loading"""
    print("\nTesting model parameters loading...")
    
    loader = ConfigLoader()
    
    try:
        # Test flux-dev parameters
        params = loader.get_model_params("flux-dev")
        print(f"✅ Flux-dev parameters loaded: hidden_size={params.hidden_size}")
        
        # Test flux-schnell parameters
        params = loader.get_model_params("flux-schnell")
        print(f"✅ Flux-schnell parameters loaded: hidden_size={params.hidden_size}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load model parameters: {e}")
        return False


def test_lora_config():
    """Test LoRA configuration loading"""
    print("\nTesting LoRA configuration loading...")
    
    loader = ConfigLoader()
    
    try:
        config = loader.load_config("lora_config")
        print("✅ LoRA config loaded successfully")
        
        # Check LoRA specific fields
        if hasattr(config, "lora_config"):
            print("✅ Found LoRA configuration section")
        else:
            print("❌ Missing LoRA configuration section")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Failed to load LoRA config: {e}")
        return False


def test_controlnet_config():
    """Test ControlNet configuration loading"""
    print("\nTesting ControlNet configuration loading...")
    
    loader = ConfigLoader()
    
    try:
        config = loader.load_config("controlnet_config")
        print("✅ ControlNet config loaded successfully")
        
        # Check ControlNet specific fields
        if hasattr(config, "controlnet_config"):
            print("✅ Found ControlNet configuration section")
        else:
            print("❌ Missing ControlNet configuration section")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Failed to load ControlNet config: {e}")
        return False


def test_config_merging():
    """Test configuration merging"""
    print("\nTesting configuration merging...")
    
    loader = ConfigLoader()
    
    try:
        # Create a simple override config
        override_config = {
            "model_name": "flux-schnell",
            "learning_rate": 2e-5
        }
        
        # Load base config
        base_config = loader.load_config("config")
        
        # Merge configurations
        merged = loader.merge_configs("config", "config")  # Merge with itself for testing
        
        print("✅ Configuration merging works")
        return True
    except Exception as e:
        print(f"❌ Configuration merging failed: {e}")
        return False


def main():
    """Run all configuration tests"""
    print("Running configuration loading tests...\n")
    
    tests = [
        test_config_loading,
        test_config_validation,
        test_model_params,
        test_lora_config,
        test_controlnet_config,
        test_config_merging
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All configuration tests passed!")
        return True
    else:
        print("❌ Some configuration tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 