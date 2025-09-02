import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import os
import json
from tqdm.auto import tqdm
from dit import load_flow_model2
import argparse
import sys
sys.path.append('/data0/teja_codes/ImmersoAiResearch/flow_matching_v2/FlowModelTraining/dataset_preparation')
from data_cache import load_ae
from PIL import Image
import matplotlib.animation as animation
import math


from torch import Tensor
import math
from typing import Callable





def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

from tqdm import tqdm

def denoise(
    model,  # Flux model
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1.0,
    neg_ip_scale: Tensor | float = 1.0
):
    i = 0
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps)-1):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale, 
        )
        if i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec, 
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
        # img = img + (t_prev - t_curr) * pred
        img = img - (t_prev - t_curr) * pred

        i += 1
    return img


class FlowMatchingInference:
    """
    Comprehensive inference pipeline for flow matching models.
    Supports loading trained checkpoints and generating samples with configurable steps.
    """
    
    def __init__(self, checkpoint_dir="model_checkpoints", device="cuda"):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.model = None
        self.data = None
        self.autoencoder = None
        self.original_h = None
        self.original_w = None
        
    def find_latest_checkpoint(self):
        """Find the latest checkpoint based on epoch number"""
        if not os.path.exists(self.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")
        
        # Look for model checkpoint files
        model_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
        
        if not model_files:
            raise FileNotFoundError("No model checkpoint files found")
        
        # Extract epoch numbers and find the latest
        epochs = []
        for f in model_files:
            try:
                epoch_num = int(f.split('_')[2].split('.')[0])
                epochs.append((epoch_num, f))
            except:
                continue
        
        if not epochs:
            raise FileNotFoundError("No valid epoch checkpoint files found")
        
        latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
        return latest_epoch, os.path.join(self.checkpoint_dir, latest_file)
    
    def load_model(self, checkpoint_path=None):
        """Load the flow model with trained weights"""
        print("Loading flow model...")
        
        # Load base model
        self.model = load_flow_model2("flux-schnell")
        self.model.eval()
        
        # Find latest checkpoint if not specified
        if checkpoint_path is None:
            latest_epoch, checkpoint_path = self.find_latest_checkpoint()
            print(f"Loading latest checkpoint from epoch {latest_epoch}: {checkpoint_path}")
        
        # Load trained weights
        print(f"Loading weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        # Move to device and convert to bfloat16 for consistency
        self.model = self.model.to(self.device, dtype=torch.bfloat16)
        print(f"Model loaded successfully on {self.device} with dtype {next(self.model.parameters()).dtype}")
        
        return self.model
    
    def load_autoencoder(self):
        """Load the autoencoder for decoding latents to images"""
        print("Loading autoencoder...")
        self.autoencoder = load_ae("flux-schnell", device=self.device)
        self.autoencoder.eval()
        print(f"Autoencoder loaded successfully on {self.device}")
        return self.autoencoder
    
    def load_data(self, data_path="/data0/teja_codes/ImmersoAiResearch/flow_matching_v2/FlowModelTraining/dataset_preparation/data_cache/data.pt"):
        """Load the same data used during training"""
        print("Loading data...")
        try:
            self.data = torch.load(data_path, map_location='cpu')
            print("Data loaded successfully!")
            
            # Print data shapes for verification
            print(f"Data shapes:")
            print(f"  img_latent: {self.data['img_latent'].shape}")
            print(f"  text_embed: {self.data['text_embed'].shape}")
            print(f"  clip_embed: {self.data['clip_embed'].shape}")
            print(f"  img_ids: {self.data['inp']['img_ids'].shape}")
            print(f"  txt_ids: {self.data['inp']['txt_ids'].shape}")
            print(f"  vec: {self.data['inp']['vec'].shape}")
            
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def prepare_inference_inputs(self, batch_idx=0):
        """Prepare inputs for inference from the loaded data"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Extract data (same preprocessing as training)
        x_1 = self.data["img_latent"]
        
        # Store original dimensions for later use in decode_latents
        _, _, orig_h, orig_w = x_1.shape
        self.original_h = orig_h // 2  # After rearranging with ph=2
        self.original_w = orig_w // 2  # After rearranging with pw=2
        
        x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        text_embed = self.data["text_embed"]
        clip_embed = self.data["clip_embed"]
        inp = self.data["inp"]
        
        # Get specific batch or use first sample
        if batch_idx >= x_1.shape[0]:
            batch_idx = 0
            print(f"Warning: batch_idx {batch_idx} out of range, using batch 0")
        
        # Prepare inputs for inference (ensure consistent dtype with model)
        model_dtype = torch.bfloat16  # Match the model dtype
        inputs = {
            'target_img': x_1[batch_idx:batch_idx+1].to(self.device, dtype=model_dtype),  # Ground truth for comparison
            'text_embed': text_embed[batch_idx:batch_idx+1].to(self.device, dtype=model_dtype),
            'clip_embed': clip_embed[batch_idx:batch_idx+1].to(self.device, dtype=model_dtype),
            'img_ids': inp['img_ids'][batch_idx:batch_idx+1].to(self.device),  # Keep as int/long
            'txt_ids': inp['txt_ids'][batch_idx:batch_idx+1].to(self.device),  # Keep as int/long
            'vec': inp['vec'][batch_idx:batch_idx+1].to(self.device, dtype=model_dtype)
        }
        
        print(f"Prepared inputs for batch {batch_idx}")
        print(f"  target_img shape: {inputs['target_img'].shape}")
        print(f"  text_embed shape: {inputs['text_embed'].shape}")
        print(f"  vec shape: {inputs['vec'].shape}")
        
        return inputs
    
    @torch.no_grad()
    def flow_sampling(self, inputs, num_steps=1000, guidance_scale=4.0):
        """
        Flow sampling using the same tensor format as training.
        target_img is already in rearranged format: [batch, seq_len, features]
        """
        device = inputs['target_img'].device
        dtype = inputs['target_img'].dtype  # This should now be bfloat16
        seed = 42

        # Create noise with the same shape as the rearranged target_img
        # target_img shape: [batch, seq_len, features] where seq_len = h*w and features = c*ph*pw
        noise = torch.randn(inputs['target_img'].shape, 
                               device=device, 
                               dtype=dtype, 
                               generator=torch.Generator(device=device).manual_seed(seed))

        # Calculate image sequence length for timestep scheduling
        # seq_len represents the number of patches (h*w)
        image_seq_len = inputs['target_img'].shape[1]  # This is h*w after rearranging
        
        timesteps = get_schedule(
            num_steps,
            image_seq_len,
            shift=True,
        )

        
        # Prepare negative prompts for CFG (using empty text as default)
        # Create empty text embeddings for negative prompts
        neg_txt = torch.zeros_like(inputs['text_embed'])  # Will inherit bfloat16 dtype
        neg_txt_ids = inputs['txt_ids']  # Same IDs structure
        neg_vec = torch.zeros_like(inputs['vec'])  # Empty conditioning vector (bfloat16)
        
        # Use denoise function for the sampling process
        img_latent = denoise(
            model=self.model,
            img=noise,
            img_ids=inputs['img_ids'],
            txt=inputs['text_embed'],
            txt_ids=inputs['txt_ids'],
            vec=inputs['vec'],
            neg_txt=neg_txt,
            neg_txt_ids=neg_txt_ids,
            neg_vec=neg_vec,
            timesteps=timesteps,
            guidance=guidance_scale,
            true_gs=1.0,  # CFG strength
            timestep_to_start_cfg=0,  # Start CFG from beginning
            image_proj=None,  # No IP-adapter
            neg_image_proj=None,
            ip_scale=1.0,
            neg_ip_scale=1.0
        )
        
        print(f"Generated latent shape: {img_latent.shape}")
        return img_latent
    
    def decode_latents(self, latents):
        """Decode latents to images using the autoencoder"""
        if self.autoencoder is None:
            raise ValueError("Autoencoder not loaded. Call load_autoencoder() first.")
        
        if self.original_h is None or self.original_w is None:
            raise ValueError("Original dimensions not set. Call prepare_inference_inputs() first.")
        
        # Rearrange latents back to image format using stored dimensions
        latents = rearrange(latents, "b (h w) (c ph pw) -> b c (h ph) (w pw)", 
                          ph=2, pw=2, h=self.original_h, w=self.original_w)
        
        # Convert latents to float32 to match autoencoder dtype
        latents = latents.float()
        
        # Decode to image
        with torch.no_grad():
            images = self.autoencoder.decode(latents)
        
        # Convert to numpy and normalize to [0, 1]
        images = images.cpu().float().numpy()
        images = (images + 1) / 2  # Assuming images are in [-1, 1] range
        images = np.clip(images, 0, 1)
        
        return images
    
    def generate_sample(self, batch_idx=0, num_steps=500, guidance_scale=4.0, save_path=None):
        """Complete generation pipeline: load model, sample, and decode"""
        print(f"Starting generation with {num_steps} steps, guidance scale {guidance_scale}")
        
        # Prepare inputs
        inputs = self.prepare_inference_inputs(batch_idx)
        
        # Generate latents
        generated_latents = self.flow_sampling(inputs, num_steps, guidance_scale)
        
        # Decode to images
        generated_images = self.decode_latents(generated_latents)
        
        # Also decode target for comparison
        target_images = self.decode_latents(inputs['target_img'])
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Generated image
        axes[0].imshow(generated_images[0].transpose(1, 2, 0))
        axes[0].set_title(f"Generated (steps={num_steps}, guidance={guidance_scale})")
        axes[0].axis('off')
        
        # Target image
        axes[1].imshow(target_images[0].transpose(1, 2, 0))
        axes[1].set_title("Target")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")
        
        plt.show()
        
        return generated_images, target_images


def main():
    """Example usage of the inference pipeline"""
    parser = argparse.ArgumentParser(description='Flow Matching Inference')
    parser.add_argument('--checkpoint_dir', default='model_checkpoints', help='Directory containing model checkpoints')
    parser.add_argument('--batch_idx', type=int, default=0, help='Batch index to use for generation')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('--guidance_scale', type=float, default=4.0, help='Guidance scale for generation')
    parser.add_argument('--device', default='cuda', help='Device to use for inference')
    parser.add_argument('--save_path', default='test_generation_comparison.png', help='Path to save generated image')
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    inference = FlowMatchingInference(checkpoint_dir=args.checkpoint_dir, device=args.device)
    
    try:
        # Load components
        print("Loading model...")
        inference.load_model()
        
        print("Loading autoencoder...")
        inference.load_autoencoder()
        
        print("Loading data...")
        inference.load_data()
        
        # Generate sample
        print("Generating sample...")
        generated_images, target_images = inference.generate_sample(
            batch_idx=args.batch_idx,
            num_steps=50,
            guidance_scale=args.guidance_scale,
            save_path=args.save_path
        )
        
        print("Generation completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()
