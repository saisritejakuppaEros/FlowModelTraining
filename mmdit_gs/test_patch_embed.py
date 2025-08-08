#!/usr/bin/env python3
"""
Test script for PatchEmbed module with 64x64x4 latent space input.
This script tests various configurations of the PatchEmbed module to ensure
it works correctly with different patch sizes and embedding dimensions.
"""

import torch
import torch.nn as nn
from modules import PatchEmbed
import numpy as np

def test_patch_embed_basic():
    """Test basic functionality with 64x64x4 input"""
    print("=" * 60)
    print("Testing PatchEmbed with 64x64x4 latent space input")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    img_size = 64
    in_channels = 4  # Latent space channels
    embed_dim = 768
    
    # Create test input: (batch_size, channels, height, width)
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"Input shape: {x.shape}")
    
    # Test different patch sizes
    patch_sizes = [2, 4, 8, 16]
    
    for patch_size in patch_sizes:
        print(f"\n--- Testing with patch_size={patch_size} ---")
        
        # Initialize PatchEmbed
        patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            flatten=True
        )
        
        # Forward pass
        output = patch_embed(x)
        
        # Expected output shape
        expected_patches = (img_size // patch_size) ** 2
        expected_shape = (batch_size, expected_patches, embed_dim)
        
        print(f"Expected output shape: {expected_shape}")
        print(f"Actual output shape: {output.shape}")
        print(f"Number of patches: {patch_embed.num_patches}")
        print(f"Grid size: {patch_embed.grid_size}")
        
        # Verify output shape
        assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
        print("✓ Shape test passed!")
        
        # Test that output is not all zeros
        assert not torch.allclose(output, torch.zeros_like(output)), "Output is all zeros!"
        print("✓ Non-zero output test passed!")
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        print("✓ Gradient flow test passed!")

def test_patch_embed_no_flatten():
    """Test PatchEmbed without flattening"""
    print("\n" + "=" * 60)
    print("Testing PatchEmbed without flattening")
    print("=" * 60)
    
    batch_size = 1
    img_size = 64
    in_channels = 4
    embed_dim = 768
    patch_size = 4
    
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"Input shape: {x.shape}")
    
    patch_embed = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_channels,
        embed_dim=embed_dim,
        flatten=False
    )
    
    output = patch_embed(x)
    expected_shape = (batch_size, embed_dim, img_size // patch_size, img_size // patch_size)
    
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {output.shape}")
    print(f"Grid size: {patch_embed.grid_size}")
    
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print("✓ No flatten test passed!")

def test_patch_embed_different_embed_dims():
    """Test with different embedding dimensions"""
    print("\n" + "=" * 60)
    print("Testing PatchEmbed with different embedding dimensions")
    print("=" * 60)
    
    batch_size = 1
    img_size = 64
    in_channels = 4
    patch_size = 2
    
    embed_dims = [256, 512, 768, 1024]
    
    for embed_dim in embed_dims:
        print(f"\n--- Testing with embed_dim={embed_dim} ---")
        
        x = torch.randn(batch_size, in_channels, img_size, img_size)
        
        patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            flatten=True
        )
        
        output = patch_embed(x)
        expected_patches = (img_size // patch_size) ** 2
        expected_shape = (batch_size, expected_patches, embed_dim)
        
        print(f"Expected output shape: {expected_shape}")
        print(f"Actual output shape: {output.shape}")
        
        assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
        print("✓ Embedding dimension test passed!")

def test_patch_embed_no_img_size():
    """Test PatchEmbed without specifying img_size"""
    print("\n" + "=" * 60)
    print("Testing PatchEmbed without img_size specification")
    print("=" * 60)
    
    batch_size = 1
    img_size = 64
    in_channels = 4
    embed_dim = 768
    patch_size = 4
    
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"Input shape: {x.shape}")
    
    patch_embed = PatchEmbed(
        img_size=None,  # Don't specify img_size
        patch_size=patch_size,
        in_chans=in_channels,
        embed_dim=embed_dim,
        flatten=True
    )
    
    output = patch_embed(x)
    expected_patches = (img_size // patch_size) ** 2
    expected_shape = (batch_size, expected_patches, embed_dim)
    
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {output.shape}")
    print(f"num_patches: {patch_embed.num_patches}")  # Should be None
    
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    assert patch_embed.num_patches is None, "num_patches should be None when img_size is None"
    print("✓ No img_size test passed!")

def test_patch_embed_device_dtype():
    """Test PatchEmbed with different devices and dtypes"""
    print("\n" + "=" * 60)
    print("Testing PatchEmbed with different devices and dtypes")
    print("=" * 60)
    
    batch_size = 1
    img_size = 64
    in_channels = 4
    embed_dim = 768
    patch_size = 4
    
    # Test different dtypes
    dtypes = [torch.float32, torch.float16]
    
    for dtype in dtypes:
        print(f"\n--- Testing with dtype={dtype} ---")
        
        x = torch.randn(batch_size, in_channels, img_size, img_size, dtype=dtype)
        
        patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            dtype=dtype
        )
        
        output = patch_embed(x)
        
        print(f"Input dtype: {x.dtype}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output shape: {output.shape}")
        
        assert output.dtype == dtype, f"Output dtype mismatch! Expected {dtype}, got {output.dtype}"
        print("✓ Dtype test passed!")

def test_patch_embed_edge_cases():
    """Test edge cases and error conditions"""
    print("\n" + "=" * 60)
    print("Testing PatchEmbed edge cases")
    print("=" * 60)
    
    # Test with patch_size larger than image size
    print("\n--- Testing patch_size > image_size ---")
    try:
        patch_embed = PatchEmbed(
            img_size=64,
            patch_size=128,  # Larger than image size
            in_chans=4,
            embed_dim=768
        )
        x = torch.randn(1, 4, 64, 64)
        output = patch_embed(x)
        print("✓ Large patch_size test passed!")
    except Exception as e:
        print(f"Expected error with large patch_size: {e}")
    
    # Test with odd image size
    print("\n--- Testing odd image size ---")
    try:
        patch_embed = PatchEmbed(
            img_size=63,  # Odd size
            patch_size=2,
            in_chans=4,
            embed_dim=768
        )
        x = torch.randn(1, 4, 63, 63)
        output = patch_embed(x)
        print(f"Output shape with odd image size: {output.shape}")
        print("✓ Odd image size test passed!")
    except Exception as e:
        print(f"Error with odd image size: {e}")

def main():
    """Run all tests"""
    print("Starting PatchEmbed tests...")
    
    try:
        test_patch_embed_basic()
        test_patch_embed_no_flatten()
        test_patch_embed_different_embed_dims()
        test_patch_embed_no_img_size()
        test_patch_embed_device_dtype()
        test_patch_embed_edge_cases()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 