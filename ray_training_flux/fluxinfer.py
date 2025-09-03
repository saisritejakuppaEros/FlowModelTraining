import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import os
import json
import pandas as pd
from tqdm.auto import tqdm
import argparse
import sys
from typing import Callable, List, Tuple, Optional
import math
from torch import Tensor
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft

# Add paths for your modules
sys.path.append('/data0/teja_codes/ImmersoAiResearch/flow_matching_v2/FlowModelTraining')
from model_utils.dit import load_flow_model2

sys.path.append('/data0/teja_codes/ImmersoAiResearch/flow_matching_v2/FlowModelTraining/working_flux/dataset_preparation')
from data_cache import load_ae
from autoencoder import AutoEncoder

# Model configurations
from dataclasses import dataclass

def sample_timesteps(bs, device, use_flux_schedule=True):
    """Sample timesteps matching inference schedule"""
    if use_flux_schedule:
        # Sample uniformly then apply time shift (matches FLUX)
        t_uniform = torch.rand((bs,), device=device)
        # Apply same time shift as inference
        mu = get_lin_function(y1=0.5, y2=1.15)(256)  # Use your seq_len
        t = time_shift(mu, 1.0, t_uniform)
    else:
        # Simple uniform sampling
        t = torch.rand((bs,), device=device)
    return t

@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: List[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float

@dataclass
class ModelSpec:
    ae_params: AutoEncoderParams
    repo_id: str
    repo_id_ae: str
    repo_ae: str

configs = {
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-schnell",
        repo_ae="ae.safetensors",
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    )
}

class HFEmbedder(torch.nn.Module):
    """Text encoder using HuggingFace models"""
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        
        if self.is_clip:
            self.tokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer = T5TokenizerFast.from_pretrained(version, max_length=max_length)
            self.hf_module = T5EncoderModel.from_pretrained(version, **hf_kwargs)
        
        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: List[str]) -> torch.Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]

def load_flux_ae(name: str = "flux-schnell", device: str = "cuda", hf_download: bool = True):
    """Load FLUX AutoEncoder"""
    ckpt_path = os.getenv("AE")
    if (ckpt_path is None and configs[name].repo_id_ae is not None 
        and configs[name].repo_ae is not None and hf_download):
        ckpt_path = hf_hub_download(configs[name].repo_id_ae, configs[name].repo_ae)
    
    print("Loading FLUX AutoEncoder...")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)
    
    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        if missing or unexpected:
            print(f"AE load - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    return ae

def time_shift(mu: float, sigma: float, t: Tensor):
    """Time shift function for timestep scheduling"""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    """Get linear function for timestep scheduling"""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule(num_steps: int, image_seq_len: int, base_shift: float = 0.5, 
                max_shift: float = 1.15, shift: bool = True) -> List[float]:
    """Get timestep schedule for sampling"""
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()

def simple_schedule(num_steps: int) -> List[float]:
    """Simple uniform schedule matching training"""
    return torch.linspace(1, 0, num_steps + 1).tolist()

def denoise(model, img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor, 
           vec: Tensor, timesteps: List[float], guidance: float = 4.0, 
           use_cfg: bool = False):
    """Simplified denoising function"""
    model.eval()
    
    with torch.no_grad():
        for i, (t_curr, t_prev) in enumerate(tqdm(zip(timesteps[:-1], timesteps[1:]), 
                                                 total=len(timesteps)-1, desc="Denoising")):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
            
            # Forward pass
            pred = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )
            
            # Update image using flow matching
            img = img + (t_prev - t_curr) * pred
    
    return img

class FluxInference:
    """Fixed FLUX inference pipeline"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.autoencoder = None
        self.clip_encoder = None
        self.t5_encoder = None
        
        # Model configuration
        self.feature_dim = 64  # Must match training
        self.resolution = 512
        
    def load_model(self):
        """Load the flow model with trained weights"""
        print("Loading flow model...")
        
        # Load base model
        self.model = load_flow_model2("flux-schnell")
        self.model.eval()
        
        # Load trained weights
        print(f"Loading weights from: {self.checkpoint_path}")
        
        # Find model file in checkpoint directory
        model_files = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.pt')]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.checkpoint_path}")
        
        # Try to find the main model file
        model_file = None
        for f in model_files:
            if 'model' in f.lower() and not f.startswith('optimizer'):
                model_file = f
                break
        
        if model_file is None:
            model_file = model_files[0]
        
        model_path = os.path.join(self.checkpoint_path, model_file)
        print(f"Loading model from: {model_path}")
        
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Handle module prefix
        if 'module.' in list(state_dict.keys())[0]:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device, dtype=torch.bfloat16)
        print(f"Model loaded successfully on {self.device}")
        
    def load_text_encoders(self):
        """Load CLIP and T5 text encoders"""
        print("Loading text encoders...")
        
        self.clip_encoder = HFEmbedder(
            "openai/clip-vit-large-patch14", 
            max_length=77,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        self.t5_encoder = HFEmbedder(
            "xlabs-ai/xflux_text_encoders", 
            max_length=512,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        print("Text encoders loaded successfully")
        
    def load_autoencoder(self):
        """Load the autoencoder for decoding latents to images"""
        print("Loading autoencoder...")
        try:
            self.autoencoder = load_flux_ae("flux-schnell", device=self.device)
            self.autoencoder = self.autoencoder.to(self.device, dtype=torch.bfloat16).eval()
            print("Autoencoder loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load autoencoder: {e}")
            self.autoencoder = None
    
    def encode_text(self, caption: str) -> dict:
        """Encode text caption into embeddings"""
        if self.clip_encoder is None or self.t5_encoder is None:
            raise ValueError("Text encoders not loaded. Call load_text_encoders() first.")
        
        with torch.no_grad():
            # Encode text
            clip_embeds = self.clip_encoder([caption])  # (1, 768)
            t5_embeds = self.t5_encoder([caption])      # (1, 512, 4096)
            
            return {
                'clip_embeds': clip_embeds,
                't5_embeds': t5_embeds
            }
    
    def create_spatial_ids(self, seq_len: int, batch_size: int = 1) -> dict:
        """Create spatial position IDs for FLUX model"""
        h = w = int(np.sqrt(seq_len))  # e.g., 16x16 for seq_len=256
        
        # Create image position IDs
        img_ids = torch.zeros(h, w, 3)
        img_ids[..., 1] = torch.arange(h)[:, None]
        img_ids[..., 2] = torch.arange(w)[None, :]
        img_ids = img_ids.reshape(1, -1, 3).repeat(batch_size, 1, 1).to(self.device)
        
        # Create text position IDs  
        txt_ids = torch.zeros(batch_size, 512, 3).to(self.device)
        
        return {
            'img_ids': img_ids,
            'txt_ids': txt_ids
        }
    
    def prepare_inputs(self, caption: str, seq_len: int = 256) -> dict:
        """Prepare all inputs for the model from a text caption"""
        
        # Encode text
        text_data = self.encode_text(caption)
        
        # Create spatial IDs
        spatial_data = self.create_spatial_ids(seq_len, batch_size=1)
        
        # Create initial noise
        noise = torch.randn(1, seq_len, self.feature_dim, 
                          device=self.device, dtype=torch.bfloat16)
        
        return {
            'img': noise,
            'img_ids': spatial_data['img_ids'],
            'txt': text_data['t5_embeds'],
            'txt_ids': spatial_data['txt_ids'],
            'vec': text_data['clip_embeds'],
        }
    
    def generate_latents(self, caption: str, num_steps: int = 50, 
                        guidance_scale: float = 4.0, use_simple_schedule: bool = True) -> Tensor:
        """Generate latents from text caption"""
        print(f"Generating from caption: '{caption}'")
        
        # Prepare inputs
        inputs = self.prepare_inputs(caption, seq_len=256)
        


        use_simple_schedule = True

        # Get timestep schedule
        if use_simple_schedule:
            timesteps = simple_schedule(num_steps)
            print("Using simple uniform schedule (matches training)")
        else:
            timesteps = get_schedule(num_steps, inputs['img'].shape[1], shift=True)
            print("Using FLUX time-shifted schedule")
        
        # Denoise
        print(f"Running {num_steps} denoising steps with guidance={guidance_scale}")
        generated_latents = denoise(
            model=self.model,
            img=inputs['img'],
            img_ids=inputs['img_ids'],
            txt=inputs['txt'],
            txt_ids=inputs['txt_ids'],
            vec=inputs['vec'],
            timesteps=timesteps,
            guidance=guidance_scale
        )
        
        print(f"Generated latent shape: {generated_latents.shape}")
        print(f"Generated latent stats: mean={generated_latents.mean():.4f}, std={generated_latents.std():.4f}")
        
        return generated_latents
    
    def decode_latents(self, latents: Tensor) -> np.ndarray:
        """Decode latents to images using autoencoder"""
        if self.autoencoder is None:
            raise ValueError("Autoencoder not loaded. Call load_autoencoder() first.")
        
        print("Decoding latents to image...")
        
        # Get original dimensions
        seq_len = latents.shape[1]
        h = w = int(np.sqrt(seq_len))
        
        # Rearrange from FLUX format back to image format
        latents = rearrange(latents, "b (h w) (c ph pw) -> b c (h ph) (w pw)", 
                          ph=2, pw=2, h=h, w=w)
        
        # Reverse FLUX scaling applied during training
        ae_params = configs["flux-schnell"].ae_params
        latents = latents / ae_params.scale_factor + ae_params.shift_factor
        
        # Decode with autoencoder
        with torch.no_grad():
            latents = latents.detach().to(dtype=torch.bfloat16)  # Keep bfloat16 for autoencoder
            images = self.autoencoder.decode(latents)
        
        # Convert to numpy and normalize to [0, 1]
        images = images.cpu().float().numpy()
        images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
        images = np.clip(images, 0, 1)
        
        print(f"Decoded image shape: {images.shape}")
        return images
    
    def generate_image(self, caption: str, num_steps: int = 50, 
                      guidance_scale: float = 4.0, save_path: Optional[str] = None) -> Tuple[Tensor, Optional[np.ndarray]]:
        """Complete generation pipeline: text -> latents -> image"""
        
        # Generate latents
        latents = self.generate_latents(caption, num_steps, guidance_scale)
        
        # Try to decode to image
        images = None
        if self.autoencoder is not None:
            try:
                images = self.decode_latents(latents)
                
                # Save image if path provided
                if save_path and images is not None:
                    img_array = (images[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                    Image.fromarray(img_array).save(save_path)
                    print(f"Image saved to: {save_path}")
                    
            except Exception as e:
                print(f"Warning: Could not decode latents: {e}")
                images = None
        
        return latents, images
    



def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='FLUX Flow Matching Inference')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint directory')
    parser.add_argument('--caption', type=str, help='Single caption to generate from')
    parser.add_argument('--captions_file', type=str, help='CSV file with captions')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples from CSV')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--guidance', type=float, default=4.0, help='Guidance scale')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output_dir', default='./generated_images', help='Output directory')
    parser.add_argument('--test_forward', action='store_true', help='Test model forward pass only')
    parser.add_argument('--simple_schedule', action='store_true', help='Use simple uniform schedule')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = FluxInference(args.checkpoint, args.device)
    
    try:
        # Load components
        print("Loading model...")
        inference.load_model()
        
        print("Loading text encoders...")
        inference.load_text_encoders()
        
        print("Loading autoencoder...")
        inference.load_autoencoder()
        
        # Test forward pass if requested
        if args.test_forward:
            test_caption = args.caption or "A beautiful sunset over mountains"
            success = inference.test_model_forward(test_caption)
            if not success:
                print("Model forward pass failed. Check model loading and input shapes.")
                return
        
        # Generate from single caption
        if args.caption:
            print(f"\nGenerating from caption: '{args.caption}'")
            save_path = os.path.join(args.output_dir, "single_generation.png")
            os.makedirs(args.output_dir, exist_ok=True)
            
            latents, images = inference.generate_image(
                args.caption, 
                args.num_steps, 
                args.guidance,
                save_path
            )
            
            if images is not None:
                print(f"Image generated and saved to: {save_path}")
            else:
                print("Only latents generated (no autoencoder)")
        
        # Generate from CSV file
        elif args.captions_file:
            print(f"Loading captions from: {args.captions_file}")
            df = pd.read_csv(args.captions_file)
            
            # Sample captions
            if len(df) > args.num_samples:
                sample_df = df.sample(n=args.num_samples, random_state=42)
            else:
                sample_df = df
            
            captions = sample_df['caption'].tolist()
            
            # Generate batch
            results = inference.generate_batch(
                captions, 
                args.num_steps, 
                args.guidance, 
                args.output_dir
            )
            
        else:
            # Default test generation
            test_captions = [
                "A beautiful sunset over mountains",
                "A cat sitting on a windowsill",
                "Abstract geometric patterns in blue and gold"
            ]
            
            results = inference.generate_batch(
                test_captions, 
                args.num_steps, 
                args.guidance, 
                args.output_dir
            )
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()