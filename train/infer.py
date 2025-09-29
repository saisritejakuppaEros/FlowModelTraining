#!/usr/bin/env python3
"""
FLUX Inference Script

Generates images from text prompts using a trained FLUX model.
Loads checkpoints in MosaicML Composer format and performs text-to-image generation.

Usage:
    python infer.py --prompt "A beautiful landscape with mountains" --output output.png
    python infer.py --prompt "A cat sitting on a chair" --steps 20 --guidance 3.5
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange, repeat

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MosaicML Composer imports for checkpoint loading
from composer import Trainer
from composer.utils.checkpoint import load_checkpoint

# Local imports
from fluxtrain import FluxComposerModel, load_config
from inference_utils import (
    denoise, get_schedule
)
from data_cache import load_ae

# FLUX text encoder imports
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel


class FluxInferenceEngine:
    """
    FLUX Inference Engine for text-to-image generation.
    Handles text encoding, model inference, and image decoding.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "config.yaml",
        device: str = "cuda",
        dtype: str = "bfloat16"
    ):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        
        print(f"🚀 Initializing FLUX Inference Engine")
        print(f"📍 Device: {self.device}")
        print(f"🔢 Dtype: {self.dtype}")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load trained model from checkpoint
        self.model = self._load_model_from_checkpoint(checkpoint_path)
        
        # Load text encoders
        self._load_text_encoders()
        
        # Load autoencoder for VAE decoding
        self._load_autoencoder()
        
        print("✅ FLUX Inference Engine initialized successfully!")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"📝 Loaded configuration from: {config_path}")
        return config
    
    def _load_model_from_checkpoint(self, checkpoint_path: str) -> FluxComposerModel:
        """Load trained FLUX model from MosaicML Composer checkpoint"""
        print(f"📦 Loading model from checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Initialize the model architecture
        model = FluxComposerModel(model_name=self.config['model']['name'])
        
        # Load the checkpoint using Composer's load_checkpoint utility
        print("🔄 Loading checkpoint state...")
        try:
            # Try Composer's load_checkpoint function first
            state_dict = load_checkpoint(checkpoint_path)
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            elif 'state' in state_dict:
                state_dict = state_dict['state']['model']
        except Exception as e:
            print(f"⚠️  Composer load failed, trying direct torch.load: {e}")
            # Fallback to direct torch.load with weights_only=False for compatibility
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract model state dict from Composer checkpoint format
            if 'state' in checkpoint:
                # Composer checkpoint format
                state_dict = checkpoint['state']['model']
            elif 'model' in checkpoint:
                # Direct model state dict
                state_dict = checkpoint['model']
            else:
                # Assume the entire checkpoint is the state dict
                state_dict = checkpoint
        
        # Load state dict into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys in checkpoint: {missing_keys[:5]}..." if len(missing_keys) > 5 else missing_keys)
        if unexpected_keys:
            print(f"⚠️  Unexpected keys in checkpoint: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else unexpected_keys)
        
        # Move to device and set dtype
        model = model.to(self.device, dtype=self.dtype)
        model.eval()
        
        print("✅ Model loaded successfully!")
        return model
    
    def _load_text_encoders(self):
        """Load CLIP and T5 text encoders"""
        print("🔤 Loading text encoders...")
        
        # Text encoder configs from training config
        clip_model = self.config.get('text_encoders', {}).get('clip_model', "openai/clip-vit-large-patch14")
        t5_model = self.config.get('text_encoders', {}).get('t5_model', "xlabs-ai/xflux_text_encoders")
        max_length_clip = self.config.get('text_encoders', {}).get('max_length_clip', 77)
        max_length_t5 = self.config.get('text_encoders', {}).get('max_length_t5', 512)
        
        # Load CLIP encoder
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.clip_encoder = CLIPTextModel.from_pretrained(clip_model, torch_dtype=self.dtype)
        self.clip_encoder = self.clip_encoder.to(self.device).eval()
        
        # Load T5 encoder
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_model, torch_dtype=self.dtype)
        self.t5_encoder = self.t5_encoder.to(self.device).eval()
        
        self.max_length_clip = max_length_clip
        self.max_length_t5 = max_length_t5
        
        print("✅ Text encoders loaded successfully!")
    
    def _load_autoencoder(self):
        """Load FLUX autoencoder for VAE decoding"""
        print("🖼️  Loading autoencoder...")
        try:
            self.autoencoder = load_ae("flux-schnell", device=self.device)
            self.autoencoder.eval()
            print("✅ Autoencoder loaded successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Could not load autoencoder: {e}")
            print("Will save raw latents without VAE decoding")
            self.autoencoder = None
    
    def encode_text(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Encode text prompt using CLIP and T5 encoders"""
        with torch.no_grad():
            # CLIP encoding
            clip_inputs = self.clip_tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length_clip,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            
            clip_outputs = self.clip_encoder(
                input_ids=clip_inputs["input_ids"].to(self.device),
                attention_mask=clip_inputs["attention_mask"].to(self.device),
            )
            clip_embeds = clip_outputs.pooler_output  # [1, 768]
            
            # T5 encoding
            t5_inputs = self.t5_tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length_t5,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            
            t5_outputs = self.t5_encoder(
                input_ids=t5_inputs["input_ids"].to(self.device),
                attention_mask=t5_inputs["attention_mask"].to(self.device),
            )
            t5_embeds = t5_outputs.last_hidden_state  # [1, seq_len, 4096]
        
        return {
            "clip_embeds": clip_embeds.to(self.dtype),
            "t5_embeds": t5_embeds.to(self.dtype)
        }
    
    def prepare_flux_inputs(
        self,
        clip_embeds: torch.Tensor,
        t5_embeds: torch.Tensor,
        height: int = 256,
        width: int = 256
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs in FLUX format for generation"""
        
        # Calculate latent dimensions (FLUX uses 8x downsampling)
        latent_h = height // 8
        latent_w = width // 8
        
        # Create image position IDs for patchified latents (ph=pw=2)
        patch_h, patch_w = latent_h // 2, latent_w // 2
        img_ids = torch.zeros(patch_h, patch_w, 3, device=self.device, dtype=self.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(patch_h, device=self.device)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(patch_w, device=self.device)[None, :]
        img_ids = img_ids.view(1, patch_h * patch_w, 3)  # [1, seq_len, 3]
        
        # Create text position IDs
        txt_ids = torch.zeros(1, t5_embeds.shape[1], 3, device=self.device, dtype=self.dtype)
        
        # Create initial noise in FLUX format
        # Latent shape: [1, (h//2)*(w//2), 16*2*2] for patchified format
        img_seq_len = patch_h * patch_w
        img_features = 16 * 4  # c * ph * pw where ph=pw=2
        noise = torch.randn(1, img_seq_len, img_features, device=self.device, dtype=self.dtype)
        
        return {
            "img": noise,
            "img_ids": img_ids,
            "txt": t5_embeds,
            "txt_ids": txt_ids,
            "vec": clip_embeds,
        }
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 256,
        width: int = 256,
        num_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate image from text prompt"""
        
        print(f"🎨 Generating image...")
        print(f"📝 Prompt: {prompt}")
        print(f"📏 Size: {width}x{height}")
        print(f"🔢 Steps: {num_steps}")
        print(f"🎯 Guidance: {guidance_scale}")
        
        if seed is not None:
            torch.manual_seed(seed)
            print(f"🌱 Seed: {seed}")
        
        # Encode text prompts
        print("🔤 Encoding text...")
        text_embeds = self.encode_text(prompt)
        
        # Prepare negative embeddings (empty prompt)
        if negative_prompt:
            neg_text_embeds = self.encode_text(negative_prompt)
        else:
            neg_text_embeds = self.encode_text("")
        
        # Prepare FLUX inputs
        flux_inputs = self.prepare_flux_inputs(
            text_embeds["clip_embeds"],
            text_embeds["t5_embeds"],
            height=height,
            width=width
        )
        
        # Generate timestep schedule
        image_seq_len = flux_inputs["img"].shape[1]
        timesteps = get_schedule(
            num_steps=num_steps,
            image_seq_len=image_seq_len,
            shift=True
        )
        
        print(f"🕐 Generated {len(timesteps)} timesteps")
        
        # Run denoising process
        print("🔄 Running denoising process...")
        with torch.no_grad():
            generated_latent = denoise(
                model=self.model.flux_model,
                img=flux_inputs["img"],
                img_ids=flux_inputs["img_ids"],
                txt=flux_inputs["txt"],
                txt_ids=flux_inputs["txt_ids"],
                vec=flux_inputs["vec"],
                neg_txt=neg_text_embeds["t5_embeds"],
                neg_txt_ids=flux_inputs["txt_ids"],
                neg_vec=neg_text_embeds["clip_embeds"],
                timesteps=timesteps,
                guidance=guidance_scale,
                true_gs=1.0,
                timestep_to_start_cfg=0,
                image_proj=None,
                neg_image_proj=None,
                ip_scale=1.0,
                neg_ip_scale=1.0
            )
        
        print("✅ Denoising completed!")
        
        # Decode latents to image if autoencoder is available
        image = None
        if self.autoencoder is not None:
            print("🖼️  Decoding latents to image...")
            try:
                image = self._decode_latent_to_image(generated_latent, height, width)
                print("✅ Image decoded successfully!")
            except Exception as e:
                print(f"⚠️  Warning: Could not decode latents to image: {e}")
                image = None
        else:
            print("⚠️  No autoencoder available - only returning raw latents")
        
        return {
            "latent": generated_latent.cpu().float().numpy(),  # Convert BFloat16 to Float32 for numpy
            "image": image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": seed
        }
    
    def _decode_latent_to_image(self, latents: torch.Tensor, height: int, width: int) -> np.ndarray:
        """Decode FLUX latents to image using autoencoder"""
        # Calculate latent dimensions
        latent_h, latent_w = height // 8, width // 8
        
        print(f"🔄 Rearranging latents from FLUX format...")
        print(f"📊 Input latents shape: {latents.shape}")
        print(f"📏 Target dimensions: h={latent_h//2}, w={latent_w//2} (patchified)")
        
        # Rearrange from FLUX format back to image format
        latents_reshaped = rearrange(
            latents, 
            "b (h w) (c ph pw) -> b c (h ph) (w pw)", 
            ph=2, pw=2, h=latent_h//2, w=latent_w//2
        )
        
        print(f"📊 Reshaped latents shape: {latents_reshaped.shape}")
        
        # Convert to float32 for autoencoder
        latents_reshaped = latents_reshaped.float()
        
        # Decode to image
        with torch.no_grad():
            images = self.autoencoder.decode(latents_reshaped)
        
        # Convert to numpy and normalize
        images = images.cpu().float().numpy()
        images = np.clip(images, 0, 1)
        
        # Convert from CHW to HWC
        image = images[0].transpose(1, 2, 0)
        
        return image
    
    def save_result(self, result: Dict[str, Any], output_path: str):
        """Save generation result to file"""
        if result["image"] is not None:
            print(f"💾 Saving image to: {output_path}")
            plt.imsave(output_path, result["image"])
            
            # Save metadata
            metadata_path = output_path.replace('.png', '_metadata.txt')
            with open(metadata_path, 'w') as f:
                f.write(f"Prompt: {result['prompt']}\n")
                f.write(f"Negative Prompt: {result['negative_prompt']}\n")
                f.write(f"Size: {result['width']}x{result['height']}\n")
                f.write(f"Steps: {result['num_steps']}\n")
                f.write(f"Guidance Scale: {result['guidance_scale']}\n")
                f.write(f"Seed: {result['seed']}\n")
            
            print(f"📝 Metadata saved to: {metadata_path}")
        else:
            # Save raw latents
            latent_path = output_path.replace('.png', '_latents.npy')
            np.save(latent_path, result["latent"])
            print(f"💾 Raw latents saved to: {latent_path}")


def main():
    parser = argparse.ArgumentParser(description="FLUX Image Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--checkpoint", type=str, 
                       default="/data0/teja_works/diffusion_training/nvidia_tools_training/mosicml_code/FlowModelTraining/train/output/checkpoints/ep25-ba8925-rank0.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Output image path")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--steps", type=int, default=100, help="Number of denoising steps")
    parser.add_argument("--guidance", type=float, default=4, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Model dtype")
    
    args = parser.parse_args()
    
    try:
        # Initialize inference engine
        engine = FluxInferenceEngine(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
            dtype=args.dtype
        )
        
        # Generate image
        result = engine.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )
        
        # Save result
        engine.save_result(result, args.output)
        
        print("🎉 Generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
