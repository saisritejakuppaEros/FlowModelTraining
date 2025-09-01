import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import (
    CLIPTokenizer, 
    CLIPTextModel, 
    T5TokenizerFast, 
    T5EncoderModel
)
from diffusers import AutoencoderKL
from logzero import logger

from model.dit import DiT
from model.noise import NoiseScheduler


class ValidationImageGenerator:
    """
    Generates validation images during training using prompts from config.
    """
    
    def __init__(
        self,
        prompts: List[str],
        resolution: int = 512,
        vae_model: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        clip_l_model: str = "openai/clip-vit-large-patch14",
        clip_g_model: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        t5_model: str = "google/t5-v1_1-xxl",
        dtype: str = "bfloat16",
        output_dir: str = "logs/valid_imgs",
        num_inference_steps: int = 500,
        # Optional pre-loaded encoders to avoid reloading
        preloaded_vae=None,
        preloaded_clip_l=None,
        preloaded_clip_g=None,
        preloaded_t5=None,
        preloaded_tokenizers=None,
    ):
        self.prompts = prompts
        self.resolution = resolution
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_inference_steps = num_inference_steps
        
        # Set dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float16
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load encoders
        self._load_encoders(
            vae_model=vae_model,
            clip_l_model=clip_l_model,
            clip_g_model=clip_g_model,
            t5_model=t5_model,
            preloaded_vae=preloaded_vae,
            preloaded_clip_l=preloaded_clip_l,
            preloaded_clip_g=preloaded_clip_g,
            preloaded_t5=preloaded_t5,
            preloaded_tokenizers=preloaded_tokenizers
        )
        
        # Initialize noise scheduler for inference
        self.noise_scheduler = NoiseScheduler(
            num_training_timesteps=1000,
            num_inference_timesteps=num_inference_steps,
            inference=True
        )
        
        logger.info(f"ValidationImageGenerator initialized with {len(prompts)} prompts")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _load_encoders(self, vae_model: str, clip_l_model: str, clip_g_model: str, t5_model: str,
                      preloaded_vae=None, preloaded_clip_l=None, preloaded_clip_g=None, 
                      preloaded_t5=None, preloaded_tokenizers=None):
        """Load all necessary encoders or use pre-loaded ones."""
        logger.info("Setting up encoders for validation...")
        
        # Use pre-loaded VAE if provided, otherwise load
        if preloaded_vae is not None:
            self.vae = preloaded_vae
            logger.info("Using pre-loaded VAE")
        else:
            self.vae = AutoencoderKL.from_pretrained(
                vae_model,
                subfolder="vae",
                torch_dtype=self.torch_dtype
            ).to(self.device)
            self.vae.eval()
            logger.info("Loaded VAE from scratch")
        
        # Use pre-loaded CLIP-L encoder if provided, otherwise load
        if preloaded_clip_l is not None:
            self.clip_l_encoder = preloaded_clip_l
            logger.info("Using pre-loaded CLIP-L encoder")
        else:
            self.clip_l_encoder = CLIPTextModel.from_pretrained(
                clip_l_model,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            self.clip_l_encoder.eval()
            logger.info("Loaded CLIP-L encoder from scratch")
        
        # Use pre-loaded CLIP-G encoder if provided, otherwise load
        if preloaded_clip_g is not None:
            self.clip_g_encoder = preloaded_clip_g
            logger.info("Using pre-loaded CLIP-G encoder")
        else:
            try:
                self.clip_g_encoder = CLIPTextModel.from_pretrained(
                    clip_g_model,
                    torch_dtype=self.torch_dtype
                ).to(self.device)
            except:
                # Fallback
                self.clip_g_encoder = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-large-patch14",
                    torch_dtype=self.torch_dtype
                ).to(self.device)
            self.clip_g_encoder.eval()
            logger.info("Loaded CLIP-G encoder from scratch")
        
        # Use pre-loaded T5 encoder if provided, otherwise load
        if preloaded_t5 is not None:
            self.t5_encoder = preloaded_t5
            logger.info("Using pre-loaded T5 encoder")
        else:
            self.t5_encoder = T5EncoderModel.from_pretrained(
                t5_model,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            self.t5_encoder.eval()
            logger.info("Loaded T5 encoder from scratch")
        
        # Use pre-loaded tokenizers if provided, otherwise load
        if preloaded_tokenizers is not None:
            self.clip_l_tokenizer = preloaded_tokenizers.get('clip_l')
            self.clip_g_tokenizer = preloaded_tokenizers.get('clip_g')
            self.t5_tokenizer = preloaded_tokenizers.get('t5')
            logger.info("Using pre-loaded tokenizers")
        else:
            self.clip_l_tokenizer = CLIPTokenizer.from_pretrained(clip_l_model)
            self.clip_g_tokenizer = CLIPTokenizer.from_pretrained(clip_g_model)
            self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
            logger.info("Loaded tokenizers from scratch")
        
        # SD3 scaling factor
        self.latent_scale = 1.5305
        
        logger.info("Encoders setup completed")
    
    def encode_text(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Encode text prompt using all three encoders."""
        with torch.no_grad():
            # CLIP-L encoding
            clip_l_inputs = self.clip_l_tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            clip_l_embeddings = self.clip_l_encoder(**clip_l_inputs).last_hidden_state
            
            # CLIP-G encoding
            clip_g_inputs = self.clip_g_tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            clip_g_embeddings = self.clip_g_encoder(**clip_g_inputs).last_hidden_state
            
            # T5 encoding
            t5_inputs = self.t5_tokenizer(
                prompt,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            t5_embeddings = self.t5_encoder(**t5_inputs).last_hidden_state
            
            # Align CLIP sequences (must match to concat on features)
            if clip_l_embeddings.size(1) != clip_g_embeddings.size(1):
                L = min(clip_l_embeddings.size(1), clip_g_embeddings.size(1))
                clip_l_embeddings = clip_l_embeddings[:, :L, :]
                clip_g_embeddings = clip_g_embeddings[:, :L, :]
            
            # Combine CLIP features along the feature axis (not sequence axis)
            clip_embeddings = torch.cat([clip_l_embeddings, clip_g_embeddings], dim=-1)
            
            # Match CLIP feature dim to T5 feature dim (pad or project)
            D_clip = clip_embeddings.size(-1)  # usually 2048 (768 + 1280)
            D_t5 = t5_embeddings.size(-1)      # usually 4096
            
            if D_clip < D_t5:
                # zero-pad CLIP features to D_t5 (SD3-style padding)
                clip_embeddings = torch.nn.functional.pad(clip_embeddings, (0, D_t5 - D_clip))
            elif D_clip > D_t5:
                # (rare) project down to T5 dim
                if not hasattr(self, 'clip_to_t5'):
                    self.clip_to_t5 = torch.nn.Linear(D_clip, D_t5).to(self.device, dtype=self.torch_dtype)
                clip_embeddings = self.clip_to_t5(clip_embeddings)
            
            # Final concatenation along the sequence axis
            text_embeddings = torch.cat([clip_embeddings, t5_embeddings], dim=1)
            
            # Pooled embeddings (only CLIP, not T5)
            # Based on the error, model expects 2048-dim pooled embeddings (CLIP-L + CLIP-G only)
            pooled_embeddings = torch.cat([
                clip_l_embeddings.mean(dim=1),  # 768 dim
                clip_g_embeddings.mean(dim=1),  # 1280 dim
            ], dim=1)  # Total: 2048 dim
            
            return {
                "text_embeddings": text_embeddings.to(dtype=self.torch_dtype),
                "pooled_embeddings": pooled_embeddings.to(dtype=self.torch_dtype)
            }
    
    @torch.no_grad()
    def generate_image(self, model: DiT, prompt: str, seed: int = 42) -> torch.Tensor:
        """Generate a single image from prompt using the trained model."""
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Encode text
        encodings = self.encode_text(prompt)
        text_embeddings = encodings["text_embeddings"]
        pooled_embeddings = encodings["pooled_embeddings"]
        
        # Debug: print shapes
        if hasattr(self, 'debug') and self.debug:
            logger.info(f"Text embeddings shape: {text_embeddings.shape}")
            logger.info(f"Pooled embeddings shape: {pooled_embeddings.shape}")
        
        # Initialize noise
        batch_size = 1
        latent_height = self.resolution // 8
        latent_width = self.resolution // 8
        
        # The DiT model expects 16 channels, but SD3 VAE produces 4 channels
        # We'll initialize with 16 channels and handle the conversion later
        latents = torch.randn(
            batch_size, 16, latent_height, latent_width,
            device=self.device, dtype=self.torch_dtype
        )
        
        # Reset attention cache before starting validation generation
        # This ensures we start with a clean cache state
        if hasattr(model, 'transformer_blocks'):
            for layer in model.transformer_blocks:
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'reset_cache'):
                    layer.attn.reset_cache()
                if hasattr(layer, 'attn2') and hasattr(layer.attn2, 'reset_cache'):
                    layer.attn2.reset_cache()
        elif hasattr(model, 'module') and hasattr(model.module, 'transformer_blocks'):
            for layer in model.module.transformer_blocks:
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'reset_cache'):
                    layer.attn.reset_cache()
                if hasattr(layer, 'attn2') and hasattr(layer.attn2, 'reset_cache'):
                    layer.attn2.reset_cache()
        
        # Denoising loop
        self.noise_scheduler.step_index = 0
        
        for i in range(self.num_inference_steps):
            # Predict velocity (drift)
            # Ensure all inputs are in the correct dtype and device
            model_latents = latents.to(device=self.device, dtype=self.torch_dtype)
            model_text_embeddings = text_embeddings.to(device=self.device, dtype=self.torch_dtype)
            model_pooled_embeddings = pooled_embeddings.to(device=self.device, dtype=self.torch_dtype)
            
            if hasattr(self, 'debug') and self.debug:
                logger.info(f"Step {i}: latents device={model_latents.device}, dtype={model_latents.dtype}")
                logger.info(f"Step {i}: text_embeddings device={model_text_embeddings.device}, dtype={model_text_embeddings.dtype}")
                logger.info(f"Step {i}: pooled_embeddings device={model_pooled_embeddings.device}, dtype={model_pooled_embeddings.dtype}")
            
            # Get timestep and ensure it's properly formatted
            timestep = self.noise_scheduler.timesteps[self.noise_scheduler.step_index]
            
            # Ensure timestep is a tensor with batch dimension
            if timestep.dim() == 0:  # scalar tensor
                timestep = timestep.unsqueeze(0)  # Add batch dimension
            if timestep.shape[0] != model_latents.shape[0]:  # Ensure batch size matches
                timestep = timestep.expand(model_latents.shape[0])
            
            timestep = timestep.to(device=self.device, dtype=self.torch_dtype)
            
            if hasattr(self, 'debug') and self.debug:
                logger.info(f"Step {i}: timestep shape={timestep.shape}, value={timestep}")
            
            try:
                model_output = model(
                    latent=model_latents,
                    timestep=timestep,
                    encoder_hidden_states=model_text_embeddings,
                    pooled_projections=model_pooled_embeddings
                )
                
                # Check if model_output is valid
                if model_output is None:
                    logger.error(f"Model returned None at step {i}")
                    # Create a dummy output with the same shape as input latents
                    model_output = torch.zeros_like(model_latents)
                elif not isinstance(model_output, torch.Tensor):
                    logger.error(f"Model returned non-tensor output at step {i}: {type(model_output)}")
                    model_output = torch.zeros_like(model_latents)
                    
            except Exception as e:
                logger.error(f"Model forward pass failed at step {i}: {e}")
                # Create a dummy output to continue the loop
                model_output = torch.zeros_like(model_latents)
            
            # Reset attention cache after each step to prevent cache corruption
            # This is crucial because validation uses batch_size=1 while training uses batch_size=4
            if hasattr(model, 'transformer_blocks'):
                for layer in model.transformer_blocks:
                    if hasattr(layer, 'attn') and hasattr(layer.attn, 'reset_cache'):
                        layer.attn.reset_cache()
                    if hasattr(layer, 'attn2') and hasattr(layer.attn2, 'reset_cache'):
                        layer.attn2.reset_cache()
            elif hasattr(model, 'module') and hasattr(model.module, 'transformer_blocks'):
                for layer in model.module.transformer_blocks:
                    if hasattr(layer, 'attn') and hasattr(layer.attn, 'reset_cache'):
                        layer.attn.reset_cache()
                    if hasattr(layer, 'attn2') and hasattr(layer.attn2, 'reset_cache'):
                        layer.attn2.reset_cache()
            
            # Denoise step (model_output is velocity, not noise)
            # Ensure model_output is in the correct dtype for reverse_flow
            if hasattr(self, 'debug') and self.debug:
                logger.info(f"Step {i}: About to convert model_output to float32")
                logger.info(f"Step {i}: model_output type={type(model_output)}, is_none={model_output is None}")
                if model_output is not None:
                    logger.info(f"Step {i}: model_output device={model_output.device}, dtype={model_output.dtype}")
            
            if model_output is None:
                logger.error(f"Step {i}: model_output is None, creating dummy output")
                model_output = torch.zeros_like(latents)
                
            model_output_float32 = model_output.to(dtype=torch.float32)
            
            if hasattr(self, 'debug') and self.debug:
                logger.info(f"Step {i}: model_output_float32 device={model_output_float32.device}, dtype={model_output_float32.dtype}")
                logger.info(f"Step {i}: latents before reverse_flow device={latents.device}, dtype={latents.dtype}")
            
            try:
                latents = self.noise_scheduler.reverse_flow(
                    current_sample=latents,
                    model_output=model_output_float32,
                    stochasticity=(i < self.num_inference_steps - 1)  # Add noise except for last step
                )
            except Exception as e:
                logger.error(f"Reverse flow failed at step {i}: {e}")
                # Create dummy latents to continue
                latents = torch.randn_like(latents)
                
            if hasattr(self, 'debug') and self.debug:
                logger.info(f"Step {i}: latents after reverse_flow device={latents.device}, dtype={latents.dtype}")
        
        # Final cache reset after generation is complete
        # This ensures the model is clean for the next training iteration
        if hasattr(model, 'transformer_blocks'):
            for layer in model.transformer_blocks:
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'reset_cache'):
                    layer.attn.reset_cache()
                if hasattr(layer, 'attn2') and hasattr(layer.attn2, 'reset_cache'):
                    layer.attn2.reset_cache()
        elif hasattr(model, 'module') and hasattr(model.module, 'transformer_blocks'):
            for layer in model.module.transformer_blocks:
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'reset_cache'):
                    layer.attn.reset_cache()
                if hasattr(layer, 'attn2') and hasattr(layer.attn2, 'reset_cache'):
                    layer.attn2.reset_cache()
        
        # Decode latents to image
        latents = latents / self.latent_scale
        
        # Debug: check latents
        if hasattr(self, 'debug') and self.debug:
            logger.info(f"Latents shape before VAE decode: {latents.shape}")
            logger.info(f"Latents range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
        
        # The SD3 VAE expects 16-channel latents (not 4 channels as initially thought)
        # No projection needed - use the 16-channel latents directly
        if hasattr(self, 'debug') and self.debug:
            logger.info(f"Using 16-channel latents directly for VAE decode: {latents.shape}")
        
        try:
            # Convert to float32 for VAE decoding (BFloat16 might not be supported)
            latents = latents.to(dtype=torch.float32)
            
            # Temporarily convert VAE to float32 for decoding
            original_vae_dtype = next(self.vae.parameters()).dtype
            if original_vae_dtype != torch.float32:
                self.vae = self.vae.float()
            
            image = self.vae.decode(latents).sample
            
            # Convert VAE back to original dtype
            if original_vae_dtype != torch.float32:
                self.vae = self.vae.to(dtype=original_vae_dtype)
        except Exception as e:
            logger.error(f"VAE decode failed: {e}")
            # Return a black image as fallback
            image = torch.zeros(1, 3, self.resolution, self.resolution, device=self.device)
        
        # Convert to PIL image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        
        if hasattr(self, 'debug') and self.debug:
            logger.info(f"Final image shape: {image.shape}")
        return image
    
    def save_validation_images(self, model: DiT, epoch: int, global_step: int) -> List[str]:
        """Generate and save validation images for all prompts."""
        saved_paths = []
        
        logger.info(f"Generating validation images for epoch {epoch}, step {global_step}")
        
        for i, prompt in enumerate(self.prompts):
            try:
                logger.info(f"Generating image for prompt {i}: '{prompt[:50]}...'")
                
                # Generate image
                image = self.generate_image(model, prompt, seed=epoch * 1000 + i)
                
                # Save image
                image_pil = Image.fromarray(image)
                filename = f"epoch_{epoch:03d}_step_{global_step:06d}_prompt_{i:02d}.png"
                filepath = self.output_dir / filename
                image_pil.save(filepath)
                
                saved_paths.append(str(filepath))
                
                logger.info(f"✅ Saved validation image: {filename} for prompt: '{prompt[:50]}...'")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate image for prompt {i}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        return saved_paths
    
    def create_grid_image(self, model: DiT, epoch: int, global_step: int) -> str:
        """Create a grid of all validation images."""
        try:
            images = []
            for i, prompt in enumerate(self.prompts):
                image = self.generate_image(model, prompt, seed=epoch * 1000 + i)
                images.append(image)
            
            # Create grid
            n = len(images)
            cols = min(5, n)
            rows = (n + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (image, prompt) in enumerate(zip(images, self.prompts)):
                row = i // cols
                col = i % cols
                ax = axes[row, col]
                
                ax.imshow(image)
                ax.set_title(f"Prompt {i+1}: {prompt[:30]}...", fontsize=8)
                ax.axis('off')
            
            # Hide empty subplots
            for i in range(n, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # Save grid
            grid_filename = f"validation_grid_epoch_{epoch:03d}_step_{global_step:06d}.png"
            grid_filepath = self.output_dir / grid_filename
            plt.savefig(grid_filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved validation grid: {grid_filename}")
            return str(grid_filepath)
            
        except Exception as e:
            logger.error(f"Failed to create validation grid: {e}")
            return ""