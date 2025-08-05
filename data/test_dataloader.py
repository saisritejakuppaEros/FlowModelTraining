"""
Test script for dataloader functionality
Tests dataset loading, batching, and PyTorch Lightning integration
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json

# Add the data directory to the path
sys.path.append(str(Path(__file__).parent))

from dataset import FlowImageDataset, ControlNetDataset, create_dataloader
from transforms import get_training_transforms, get_validation_transforms
from datamodule import FlowDataModule, ControlNetDataModule


def create_test_data(data_dir: str = "test_data", num_samples: int = 10):
    """Create test data for validation"""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    print(f"Creating test data in {data_path}")
    
    for i in range(num_samples):
        # Create a simple test image
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save image
        img_path = data_path / f"test_image_{i:03d}.jpg"
        img.save(img_path)
        
        # Create caption file
        caption_data = {
            "caption": f"This is test image number {i} with a beautiful landscape"
        }
        caption_path = data_path / f"test_image_{i:03d}.json"
        with open(caption_path, 'w') as f:
            json.dump(caption_data, f)
    
    print(f"Created {num_samples} test samples")


def test_basic_dataset():
    """Test basic dataset functionality"""
    print("Testing basic dataset functionality...")
    
    # Create test data
    test_data_dir = "test_data"
    create_test_data(test_data_dir, num_samples=5)
    
    try:
        # Test FlowImageDataset
        dataset = FlowImageDataset(
            img_dir=test_data_dir,
            img_size=512,
            caption_type='json',
            random_ratio=False,
            center_crop=False
        )
        
        print(f"✅ Dataset created successfully with {len(dataset)} samples")
        
        # Test getting an item
        image, caption = dataset[0]
        print(f"✅ Sample loaded: image shape {image.shape}, caption: '{caption[:50]}...'")
        
        # Test image normalization
        assert image.min() >= -1.0 and image.max() <= 1.0, "Image not normalized to [-1, 1]"
        print("✅ Image normalization working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


def test_dataloader():
    """Test DataLoader functionality"""
    print("\nTesting DataLoader functionality...")
    
    try:
        # Create dataset
        dataset = FlowImageDataset(
            img_dir="test_data",
            img_size=512,
            caption_type='json'
        )
        
        # Create DataLoader
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            shuffle=True
        )
        
        print(f"✅ DataLoader created successfully")
        
        # Test batch loading
        for batch_idx, (images, captions) in enumerate(dataloader):
            print(f"✅ Batch {batch_idx}: images shape {images.shape}, {len(captions)} captions")
            
            # Check batch dimensions
            assert images.shape[0] == 2, f"Expected batch size 2, got {images.shape[0]}"
            assert len(captions) == 2, f"Expected 2 captions, got {len(captions)}"
            
            if batch_idx >= 1:  # Just test first few batches
                break
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False


def test_transforms():
    """Test transform functionality"""
    print("\nTesting transforms functionality...")
    
    try:
        # Create test image tensor
        test_image = torch.randn(3, 512, 512)
        
        # Test training transforms
        train_transforms = get_training_transforms(
            img_size=512,
            random_flip=True,
            random_rotation=False,
            color_jitter=False,
            normalize=True
        )
        
        transformed_image = train_transforms(test_image)
        print(f"✅ Training transforms applied: shape {transformed_image.shape}")
        
        # Test validation transforms
        val_transforms = get_validation_transforms(
            img_size=512,
            normalize=True
        )
        
        transformed_image = val_transforms(test_image)
        print(f"✅ Validation transforms applied: shape {transformed_image.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transforms test failed: {e}")
        return False


def test_controlnet_dataset():
    """Test ControlNet dataset functionality"""
    print("\nTesting ControlNet dataset functionality...")
    
    try:
        # Create ControlNet dataset
        dataset = ControlNetDataset(
            img_dir="test_data",
            img_size=512,
            caption_type='json',
            control_type='canny'
        )
        
        print(f"✅ ControlNet dataset created successfully with {len(dataset)} samples")
        
        # Test getting an item
        image, control_signal, caption = dataset[0]
        print(f"✅ ControlNet sample loaded: image shape {image.shape}, control shape {control_signal.shape}")
        
        # Test control signal generation
        assert control_signal.shape == image.shape, "Control signal should have same shape as image"
        print("✅ Control signal generation working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ ControlNet dataset test failed: {e}")
        return False


def test_pytorch_lightning_datamodule():
    """Test PyTorch Lightning DataModule"""
    print("\nTesting PyTorch Lightning DataModule...")
    
    try:
        # Create DataModule
        datamodule = FlowDataModule(
            data_dir="test_data",
            img_size=512,
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            train_split=0.8,
            caption_type='json'
        )
        
        print("✅ DataModule created successfully")
        
        # Setup for training
        datamodule.setup(stage='fit')
        
        # Test training dataloader
        train_loader = datamodule.train_dataloader()
        print(f"✅ Training dataloader created with {len(train_loader)} batches")
        
        # Test validation dataloader
        val_loader = datamodule.val_dataloader()
        print(f"✅ Validation dataloader created with {len(val_loader)} batches")
        
        # Test batch loading
        for batch_idx, (images, captions) in enumerate(train_loader):
            print(f"✅ Training batch {batch_idx}: images shape {images.shape}")
            if batch_idx >= 1:
                break
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch Lightning DataModule test failed: {e}")
        return False


def test_controlnet_datamodule():
    """Test ControlNet PyTorch Lightning DataModule"""
    print("\nTesting ControlNet PyTorch Lightning DataModule...")
    
    try:
        # Create ControlNet DataModule
        datamodule = ControlNetDataModule(
            data_dir="test_data",
            img_size=512,
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            train_split=0.8,
            caption_type='json',
            control_type='canny'
        )
        
        print("✅ ControlNet DataModule created successfully")
        
        # Setup for training
        datamodule.setup(stage='fit')
        
        # Test training dataloader
        train_loader = datamodule.train_dataloader()
        print(f"✅ ControlNet training dataloader created with {len(train_loader)} batches")
        
        # Test batch loading
        for batch_idx, (images, control_signals, captions) in enumerate(train_loader):
            print(f"✅ ControlNet training batch {batch_idx}: images shape {images.shape}, control shape {control_signals.shape}")
            if batch_idx >= 1:
                break
        
        return True
        
    except Exception as e:
        print(f"❌ ControlNet PyTorch Lightning DataModule test failed: {e}")
        return False


def cleanup_test_data():
    """Clean up test data"""
    import shutil
    test_data_dir = Path("test_data")
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
        print("🧹 Cleaned up test data")


def main():
    """Run all dataloader tests"""
    print("Running dataloader tests...\n")
    
    tests = [
        test_basic_dataset,
        test_dataloader,
        test_transforms,
        test_controlnet_dataset,
        test_pytorch_lightning_datamodule,
        test_controlnet_datamodule
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
    
    # Cleanup
    cleanup_test_data()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All dataloader tests passed!")
        return True
    else:
        print("❌ Some dataloader tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 