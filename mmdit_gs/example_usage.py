#!/usr/bin/env python3
"""
Simple example demonstrating PatchEmbed usage with 64x64x4 latent space input.
This script shows the basic usage pattern for embedding image patches.
"""

import torch
from modules import PatchEmbed

def main():
    print("PatchEmbed Example with 64x64x4 Latent Space Input")
    print("=" * 50)
    
    # Configuration for your latent space
    batch_size = 1
    img_size = 64          # Your image size
    in_channels = 4        # Your latent space channels
    patch_size = 2         # Patch size (you can adjust this)
    embed_dim = 768        # Embedding dimension
    
    # Create a sample latent representation
    # Shape: (batch_size, channels, height, width)
    latent = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"Input latent shape: {latent.shape}")
    
    # Initialize PatchEmbed
    patch_embed = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_channels,
        embed_dim=embed_dim,
        flatten=True
    )
    
    print(f"Patch size: {patch_embed.patch_size}")
    print(f"Grid size: {patch_embed.grid_size}")
    print(f"Number of patches: {patch_embed.num_patches}")
    
    # Forward pass
    embedded_patches = patch_embed(latent)
    print(f"Output shape: {embedded_patches.shape}")
    
    # Calculate expected shape
    num_patches = (img_size // patch_size) ** 2
    expected_shape = (batch_size, num_patches, embed_dim)
    print(f"Expected shape: {expected_shape}")
    
    # Verify the output
    assert embedded_patches.shape == expected_shape, "Shape mismatch!"
    print("✓ Shape verification passed!")
    
    # Show some statistics
    print(f"\nOutput statistics:")
    print(f"  Mean: {embedded_patches.mean().item():.4f}")
    print(f"  Std:  {embedded_patches.std().item():.4f}")
    print(f"  Min:  {embedded_patches.min().item():.4f}")
    print(f"  Max:  {embedded_patches.max().item():.4f}")
    
    print("\n✓ PatchEmbed example completed successfully!")

if __name__ == "__main__":
    main() 