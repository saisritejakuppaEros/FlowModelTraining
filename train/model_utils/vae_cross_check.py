import matplotlib.pyplot as plt

from train.data_cache import load_ae
import os

def test_vae_encoding_decoding(image_path: str, save_dir: str = "/tmp", device: str = "cuda"):
    """
    Test function to load an image, encode it with VAE, decode it back, and save the result.
    
    Args:
        image_path: Path to the input image
        save_dir: Directory to save the test results
        device: Device to run the test on
    """
    import torch
    import numpy as np
    from PIL import Image
    import os
    from einops import rearrange
    
    print(f"🔍 Testing VAE encoding/decoding with image: {image_path}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Load the autoencoder
        print("Loading autoencoder...")
        autoencoder = load_ae("flux-schnell", device=device)
        autoencoder.eval()
        print(f"Autoencoder loaded successfully on {device}")
        
        # Load and preprocess the image
        print("Loading and preprocessing image...")
        image = Image.open(image_path).convert('RGB')
        
        # Resize image to 512x512
        print("Resizing image to 512x512...")
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        image_array = np.array(image)
        
        # Convert to tensor and normalize to [-1, 1]
        image_tensor = torch.from_numpy(image_array).float() / 255.0
        image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        image_tensor = image_tensor.to(device)
        
        print(f"Original image shape: {image_tensor.shape}")
        
        # Encode the image to latents
        print("Encoding image to latents...")
        with torch.no_grad():
            latents = autoencoder.encode(image_tensor)
        
        print(f"Encoded latents shape: {latents.shape}")
        
        # Save the encoded latents
        latents_save_path = os.path.join(save_dir, "encoded_latents.npy")
        np.save(latents_save_path, latents.cpu().numpy())
        print(f"Encoded latents saved to: {latents_save_path}")
        
        # Decode the latents back to image
        print("Decoding latents back to image...")
        with torch.no_grad():
            decoded_image = autoencoder.decode(latents)
        
        print(f"Decoded image shape: {decoded_image.shape}")
        
        # Convert back to numpy and normalize to [0, 1]
        decoded_image_np = decoded_image.cpu().float().numpy()
        decoded_image_np = (decoded_image_np + 1) / 2  # Convert from [-1, 1] to [0, 1]
        decoded_image_np = np.clip(decoded_image_np, 0, 1)
        
        # Convert from CHW to HWC
        decoded_image_np = decoded_image_np[0].transpose(1, 2, 0)
        
        # Save the original and decoded images
        original_save_path = os.path.join(save_dir, "original_image.png")
        decoded_save_path = os.path.join(save_dir, "decoded_image.png")
        
        # Convert original image back to [0, 1] for saving
        original_image_np = (image_tensor.cpu().numpy()[0].transpose(1, 2, 0) + 1) / 2
        original_image_np = np.clip(original_image_np, 0, 1)
        
        # Save images
        plt.imsave(original_save_path, original_image_np)
        plt.imsave(decoded_save_path, decoded_image_np)
        
        print(f"Original image saved to: {original_save_path}")
        print(f"Decoded image saved to: {decoded_save_path}")
        
        # Calculate and print some statistics
        mse = np.mean((original_image_np - decoded_image_np) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        print(f"Reconstruction MSE: {mse:.6f}")
        print(f"Reconstruction PSNR: {psnr:.2f} dB")
        
        print("✅ VAE encoding/decoding test completed successfully!")
        
        return {
            'original_image': original_image_np,
            'decoded_image': decoded_image_np,
            'latents': latents.cpu().numpy(),
            'mse': mse,
            'psnr': psnr
        }
        
    except Exception as e:
        print(f"❌ Error during VAE test: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test the latent scaling issue with the specified image
    test_image_path = "/data0/teja_works/diffusion_training/nvidia_tools_training/mosicml_code/FlowModelTraining/sa_2848197.jpg"
    test_save_dir = "./test_latent_scaling_results"
    
    print("🚀 Starting latent scaling test...")
    result = test_vae_encoding_decoding(
        image_path=test_image_path,
        save_dir=test_save_dir,
        device="cuda"
    )
    
    if result is not None:
        print("🎉 Test completed successfully!")
    else:
        print("💥 Test failed!")
