import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import os
import math
import gc
from typing import Callable, List, Dict, Any
from tqdm import tqdm
import sys
from data_cache import load_ae
from memory_utils import print_gpu_memory_usage, cleanup_gpu_memory



def time_shift(mu: float, sigma: float, t: torch.Tensor):
    """Time shifting function for timestep scheduling"""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    """Linear function for timestep scheduling"""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> List[float]:
    """Generate timestep schedule for sampling"""
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model,  # Flux model
    # model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    neg_txt: torch.Tensor,
    neg_txt_ids: torch.Tensor,
    neg_vec: torch.Tensor,
    # sampling parameters
    timesteps: List[float],
    guidance: float = 4.0,
    true_gs: float = 1.0,
    timestep_to_start_cfg: int = 0,
    # ip-adapter parameters
    image_proj: torch.Tensor = None, 
    neg_image_proj: torch.Tensor = None, 
    ip_scale: torch.Tensor | float = 1.0,
    neg_ip_scale: torch.Tensor | float = 1.0
):
    """Denoising process for flow matching sampling"""
    i = 0
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    
    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps)-1, desc="Sampling"):
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
        
        # Flow matching update
        img = img + (t_prev - t_curr) * pred
        i += 1
        
        # Clean up predictions to prevent memory accumulation
        del pred
        if i >= timestep_to_start_cfg and 'neg_pred' in locals():
            del neg_pred
    
    # Final cleanup
    del guidance_vec, t_vec
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return img


class TrainingInference:
    """
    Simplified inference class for use during training.
    Generates samples to monitor training progress using validation datastreamer.
    """
    
    def __init__(self, model, config: Dict[str, Any], device: str = "cuda", eval_dataloader=None):
        self.model = model
        self.config = config
        self.device = device
        self.save_dir = config['inference']['save_dir']
        self.eval_dataloader = eval_dataloader
        self.autoencoder = None
        self.original_h = None
        self.original_w = None
        
        # Create save directory
        # print(f"🔍 DEBUG: Creating inference save directory: {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=True)
        # print(f"🔍 DEBUG: Inference save directory created successfully")
        
        # Load autoencoder for VAE decoding
        self._load_autoencoder()
    
    def _load_autoencoder(self):
        """Load the autoencoder for VAE decoding"""
        try:
            print("Loading autoencoder for VAE decoding...")
            self.autoencoder = load_ae("flux-schnell", device=self.device)
            self.autoencoder.eval()
            print(f"Autoencoder loaded successfully on {self.device}")
        except Exception as e:
            print(f"Warning: Could not load autoencoder: {e}")
            print("Will save raw latents without VAE decoding")
            self.autoencoder = None
        
    def generate_sample_from_batch(self, batch: Dict[str, torch.Tensor], epoch: int, step: int = 0):
        """
        Generate a sample from a training batch for monitoring progress.
        Uses the first sample in the batch.
        """
        # Extract first sample from batch
        img_latents = batch['img_latents'][:1]  # [1, seq_len, features]
        img_ids = batch['img_ids'][:1]          # [1, seq_len, 3]
        txt_embeds = batch['txt_embeds'][:1]     # [1, seq_len, features]
        txt_ids = batch['txt_ids'][:1]           # [1, seq_len, 3]
        vec_embeds = batch['vec_embeds'][:1]     # [1, features]
        
        # Get the caption text if available
        caption_text = "Unknown caption"
        if 'caption_text' in batch and len(batch['caption_text']) > 0:
            caption_text = batch['caption_text'][0] if isinstance(batch['caption_text'], list) else batch['caption_text']
        
        # print(f"🔍 DEBUG: Using caption for inference: '{caption_text}'")
        
        # Move to device and ensure correct dtype
        # Use the same dtype as the model parameters
        model_dtype = next(self.model.parameters()).dtype
        # print(f"🔍 DEBUG: Model dtype: {model_dtype}")
        # print(f"🔍 DEBUG: Input img_latents dtype: {img_latents.dtype}")
        
        img_latents = img_latents.to(device=self.device, dtype=model_dtype)
        img_ids = img_ids.to(device=self.device, dtype=model_dtype)
        txt_embeds = txt_embeds.to(device=self.device, dtype=model_dtype)
        txt_ids = txt_ids.to(device=self.device, dtype=model_dtype)
        vec_embeds = vec_embeds.to(device=self.device, dtype=model_dtype)
        
        # print(f"🔍 DEBUG: After conversion - img_latents dtype: {img_latents.dtype}")
        
        # Generate timesteps
        image_seq_len = img_latents.shape[1]
        timesteps = get_schedule(
            self.config['inference']['num_steps'],
            image_seq_len,
            shift=True,
        )
        # Convert timesteps to the correct dtype
        timesteps = [float(t) for t in timesteps]
        
        # Prepare negative prompts (empty) with correct dtype
        neg_txt = torch.zeros_like(txt_embeds, dtype=model_dtype)
        neg_txt_ids = txt_ids
        neg_vec = torch.zeros_like(vec_embeds, dtype=model_dtype)

        noise = torch.randn_like(img_latents, dtype=model_dtype)
        
        # Generate sample
        with torch.no_grad():
            generated_latent = denoise(
                model=self.model,
                img=noise,
                img_ids=img_ids,
                txt=txt_embeds,
                txt_ids=txt_ids,
                vec=vec_embeds,
                neg_txt=neg_txt,
                neg_txt_ids=neg_txt_ids,
                neg_vec=neg_vec,
                timesteps=timesteps,
                guidance=self.config['inference']['guidance_scale'],
                true_gs=1.0,
                timestep_to_start_cfg=0,
                image_proj=None,
                neg_image_proj=None,
                ip_scale=1.0,
                neg_ip_scale=1.0
            )
            
            # Clean up intermediate tensors
            del noise, neg_txt, neg_vec
            torch.cuda.empty_cache()
            
            # Clean up intermediate tensors
            del noise, neg_txt, neg_vec
            torch.cuda.empty_cache()
        
        # Add timestamp to make filenames unique
        import time
        timestamp = int(time.time() * 1000) % 100000  # Last 5 digits of timestamp
        
        # Save raw latents
        # latent_save_path = os.path.join(
        #     self.save_dir, 
        #     f"epoch_{epoch:03d}_step_{step:06d}_{timestamp}_generated_latent.npy"
        # )
        # np.save(latent_save_path, generated_latent.cpu().numpy())
        
        # original_latent_path = os.path.join(
        #     self.save_dir, 
        #     f"epoch_{epoch:03d}_step_{step:06d}_{timestamp}_original_latent.npy"
        # )
        # np.save(original_latent_path, img_latents.cpu().numpy())
        
        # print(f"Generated latent saved to: {latent_save_path}")
        # print(f"Original latent saved to: {original_latent_path}")
        
        # Try to decode with VAE if available
        if self.autoencoder is not None:
            try:
                # print(f"🔍 DEBUG: About to decode - generated_latent shape: {generated_latent.shape}")
                # print(f"🔍 DEBUG: Generated latent stats - min: {generated_latent.min().item():.6f}, max: {generated_latent.max().item():.6f}")
                # print(f"🔍 DEBUG: Generated latent stats - mean: {generated_latent.mean().item():.6f}, std: {generated_latent.std().item():.6f}")
                
                # print(f"🔍 DEBUG: About to decode - original img_latents shape: {img_latents.shape}")
                # print(f"🔍 DEBUG: Original latent stats - min: {img_latents.min().item():.6f}, max: {img_latents.max().item():.6f}")
                # print(f"🔍 DEBUG: Original latent stats - mean: {img_latents.mean().item():.6f}, std: {img_latents.std().item():.6f}")
                
                # Check if we have raw_img_latents available (better for decoding)
                if 'raw_img_latents' in batch:
                    # print("🔍 DEBUG: Using raw_img_latents for original image decoding")
                    raw_img_latents = batch['raw_img_latents'][:1].to(device=self.device, dtype=model_dtype)
                    # print(f"🔍 DEBUG: Raw img latents shape: {raw_img_latents.shape}")
                    # print(f"🔍 DEBUG: Raw img latents stats - min: {raw_img_latents.min().item():.6f}, max: {raw_img_latents.max().item():.6f}")
                    original_image = self._decode_raw_latent_to_image(raw_img_latents)
                else:
                    # print("🔍 DEBUG: Using FLUX-format img_latents for original image decoding")
                    original_image = self._decode_latent_to_image(img_latents)
                
                # Decode generated latent to image
                generated_image = self._decode_latent_to_image(generated_latent)
                
                # Save images with caption in filename
                safe_caption = caption_text.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]  # Limit length
                image_save_path = os.path.join(
                    self.save_dir, 
                    f"epoch_{epoch:03d}_step_{step:06d}_{timestamp}_generated_{safe_caption}.png"
                )
                original_image_path = os.path.join(
                    self.save_dir, 
                    f"epoch_{epoch:03d}_step_{step:06d}_{timestamp}_original_{safe_caption}.png"
                )
                
                # Save images
                plt.imsave(image_save_path, generated_image)
                plt.imsave(original_image_path, original_image)
                
                print(f"Generated image saved to: {image_save_path}")
                print(f"Original image saved to: {original_image_path}")
                
                # Save caption text
                caption_save_path = os.path.join(
                    self.save_dir, 
                    f"epoch_{epoch:03d}_step_{step:06d}_{timestamp}_caption.txt"
                )
                with open(caption_save_path, 'w', encoding='utf-8') as f:
                    f.write(caption_text)
                print(f"Caption saved to: {caption_save_path}")
                
            except Exception as e:
                print(f"Warning: Could not decode latents to images: {e}")
        else:
            print("No autoencoder available - only saving raw latents")
        
        # Clean up remaining tensors
        del img_latents, img_ids, txt_embeds, txt_ids, vec_embeds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return generated_latent
    
    def generate_sample_from_validation(self, epoch: int, step: int = 0):
        """
        Generate samples from the validation datastreamer for monitoring progress.
        Uses configurable number of samples from the validation dataset consistently.
        """
        if self.eval_dataloader is None:
            print("Warning: No validation dataloader provided, skipping validation inference")
            return None
        
        # Check if we should save images for this epoch
        max_epochs_to_save = self.config['inference'].get('max_epochs_to_save', 0)
        if max_epochs_to_save > 0 and epoch > max_epochs_to_save:
            print(f"Skipping validation inference for epoch {epoch} (max_epochs_to_save={max_epochs_to_save})")
            return None
        
        # Check if we should run inference this epoch based on interval
        inference_interval = self.config['inference'].get('inference_interval_epochs', 1)
        if epoch % inference_interval != 0:
            print(f"Skipping validation inference for epoch {epoch} (inference_interval_epochs={inference_interval})")
            return None
            
        try:
            # Get a batch from validation dataloader
            val_batch = next(iter(self.eval_dataloader))
            
            # Get number of samples from config
            num_samples_config = self.config['inference'].get('num_validation_samples', 4)
            num_samples = min(num_samples_config, val_batch['img_latents'].shape[0])
            img_latents = val_batch['img_latents'][:num_samples]  # [4, seq_len, features]
            img_ids = val_batch['img_ids'][:num_samples]          # [4, seq_len, 3]
            txt_embeds = val_batch['txt_embeds'][:num_samples]     # [4, seq_len, features]
            txt_ids = val_batch['txt_ids'][:num_samples]           # [4, seq_len, 3]
            vec_embeds = val_batch['vec_embeds'][:num_samples]     # [4, features]
            
            # Get the caption texts if available
            caption_texts = [f"Validation sample {i+1}" for i in range(num_samples)]
            if 'caption_text' in val_batch and len(val_batch['caption_text']) > 0:
                if isinstance(val_batch['caption_text'], list):
                    caption_texts = val_batch['caption_text'][:num_samples]
                else:
                    caption_texts = [val_batch['caption_text']] * num_samples
            
            print(f"🔍 DEBUG: Using {num_samples} validation samples for inference (config: {num_samples_config})")
            
            # Move to device and ensure correct dtype
            model_dtype = next(self.model.parameters()).dtype
            
            img_latents = img_latents.to(device=self.device, dtype=model_dtype)
            img_ids = img_ids.to(device=self.device, dtype=model_dtype)
            txt_embeds = txt_embeds.to(device=self.device, dtype=model_dtype)
            txt_ids = txt_ids.to(device=self.device, dtype=model_dtype)
            vec_embeds = vec_embeds.to(device=self.device, dtype=model_dtype)
            
            # print(f"🔍 DEBUG: After conversion - img_latents dtype: {img_latents.dtype}")
            
            # Generate timesteps
            image_seq_len = img_latents.shape[1]
            timesteps = get_schedule(
                self.config['inference']['num_steps'],
                image_seq_len,
                shift=True,
            )
            timesteps = [float(t) for t in timesteps]
            
            # Prepare negative prompts (empty) with correct dtype
            neg_txt = torch.zeros_like(txt_embeds, dtype=model_dtype)
            neg_txt_ids = txt_ids
            neg_vec = torch.zeros_like(vec_embeds, dtype=model_dtype)

            noise = torch.randn_like(img_latents, dtype=model_dtype)
            
            # Initialize generated_latents to None to avoid UnboundLocalError
            generated_latents = None
            
            # Generate samples for all 4 samples
            with torch.no_grad():
                generated_latents = denoise(
                    model=self.model,
                    img=noise,
                    img_ids=img_ids,
                    txt=txt_embeds,
                    txt_ids=txt_ids,
                    vec=vec_embeds,
                    neg_txt=neg_txt,
                    neg_txt_ids=neg_txt_ids,
                    neg_vec=neg_vec,
                    timesteps=timesteps,
                    guidance=self.config['inference']['guidance_scale'],
                    true_gs=1.0,
                    timestep_to_start_cfg=0,
                    image_proj=None,
                    neg_image_proj=None,
                    ip_scale=1.0,
                    neg_ip_scale=1.0
                )
            
            # Check if generation was successful
            if generated_latents is None:
                print("Warning: Generation failed, returning None")
                return None
            
            # Add timestamp to make filenames unique
            import time
            timestamp = int(time.time() * 1000) % 100000  # Last 5 digits of timestamp
            
            # Save raw latents
            # latent_save_path = os.path.join(
            #     self.save_dir, 
            #     f"epoch_{epoch:03d}_step_{step:06d}_{timestamp}_val_generated_latent.npy"
            # )
            # np.save(latent_save_path, generated_latent.cpu().numpy())
            
            # original_latent_path = os.path.join(
            #     self.save_dir, 
            #     f"epoch_{epoch:03d}_step_{step:06d}_{timestamp}_val_original_latent.npy"
            # )
            # np.save(original_latent_path, img_latents.cpu().numpy())
            
            # print(f"Validation generated latent saved to: {latent_save_path}")
            # print(f"Validation original latent saved to: {original_latent_path}")
            
            # Try to decode with VAE if available
            if self.autoencoder is not None:
                try:
                    # Process each sample individually
                    for i in range(num_samples):
                        # Get individual sample
                        single_generated_latent = generated_latents[i:i+1]  # [1, seq_len, features]
                        single_img_latents = img_latents[i:i+1]  # [1, seq_len, features]
                        single_caption = caption_texts[i]
                        
                        # Check if we have raw_img_latents available (better for decoding)
                        if 'raw_img_latents' in val_batch:
                            raw_img_latents = val_batch['raw_img_latents'][i:i+1].to(device=self.device, dtype=model_dtype)
                            original_image = self._decode_raw_latent_to_image(raw_img_latents)
                        else:
                            original_image = self._decode_latent_to_image(single_img_latents)
                        
                        # Decode generated latent to image
                        generated_image = self._decode_latent_to_image(single_generated_latent)
                        
                        # Save images with validation prefix
                        safe_caption = single_caption.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]
                        image_save_path = os.path.join(
                            self.save_dir, 
                            f"epoch_{epoch:03d}_step_{step:06d}_{timestamp}_val_{i+1}_generated_{safe_caption}.png"
                        )
                        original_image_path = os.path.join(
                            self.save_dir, 
                            f"epoch_{epoch:03d}_step_{step:06d}_{timestamp}_val_{i+1}_original_{safe_caption}.png"
                        )
                        
                        # Save images
                        plt.imsave(image_save_path, generated_image)
                        plt.imsave(original_image_path, original_image)
                        
                        print(f"Validation sample {i+1} generated image saved to: {image_save_path}")
                        print(f"Validation sample {i+1} original image saved to: {original_image_path}")
                        
                        # Save caption text for each sample
                        caption_save_path = os.path.join(
                            self.save_dir, 
                            f"epoch_{epoch:03d}_step_{step:06d}_{timestamp}_val_{i+1}_caption.txt"
                        )
                        with open(caption_save_path, 'w', encoding='utf-8') as f:
                            f.write(single_caption)
                        print(f"Validation sample {i+1} caption saved to: {caption_save_path}")
                    
                except Exception as e:
                    print(f"Warning: Could not decode latents to images: {e}")
            else:
                print("No autoencoder available - only saving raw latents")
            
            # Store the result before cleanup
            result = generated_latents.clone() if generated_latents is not None else None
            
            # Clean up remaining tensors
            del img_latents, img_ids, txt_embeds, txt_ids, vec_embeds, val_batch
            if generated_latents is not None:
                del generated_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            return result
                
        except Exception as e:
            print(f"Error generating validation sample: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on error
            if 'generated_latents' in locals() and generated_latents is not None:
                del generated_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            return None
    
    def _decode_latent_to_image(self, latents: torch.Tensor):
        """Decode latents to images using the autoencoder"""
        if self.autoencoder is None:
            raise ValueError("Autoencoder not loaded")
        
        # Store original dimensions if not set
        if self.original_h is None or self.original_w is None:
            # Estimate dimensions from latent shape
            # latents shape: [batch, seq_len, features] where seq_len = h*w
            seq_len = latents.shape[1]
            # Assuming square images, estimate h and w
            h = w = int(np.sqrt(seq_len))
            self.original_h = h
            self.original_w = w
        
        # print(f"🔍 DEBUG: Rearranging latents from FLUX format:")
        # print(f"🔍 DEBUG: Input latents shape: {latents.shape}")
        # print(f"🔍 DEBUG: Using h={self.original_h}, w={self.original_w}, ph=2, pw=2")
        # print(f"🔍 DEBUG: Expected seq_len = h*w = {self.original_h * self.original_w}, actual seq_len = {latents.shape[1]}")
        # print(f"🔍 DEBUG: Expected features = c*ph*pw = 16*2*2 = 64, actual features = {latents.shape[2]}")
        
        # Rearrange latents back to image format
        latents_reshaped = rearrange(latents, "b (h w) (c ph pw) -> b c (h ph) (w pw)", 
                                   ph=2, pw=2, h=self.original_h, w=self.original_w)
        
        # print(f"🔍 DEBUG: Reshaped latents shape: {latents_reshaped.shape}")
        # print(f"🔍 DEBUG: Expected final shape: [1, 16, {self.original_h * 2}, {self.original_w * 2}]")
        
        # Convert latents to float32 to match autoencoder dtype
        latents_reshaped = latents_reshaped.float()
        
        # NOTE: Do NOT apply manual scaling here!
        # The autoencoder.decode() method automatically applies: z = z / self.scale_factor + self.shift_factor
        # Manual scaling would cause double scaling and incorrect latent values
        # The latents from training/generation are already in the correct format for decode()
        
        # Check if we need to apply any additional scaling/normalization
        # Compare with what the autoencoder expects
        if hasattr(self.autoencoder, 'ae_params'):
            # print(f"🔍 DEBUG: AutoEncoder scale_factor: {self.autoencoder.ae_params.scale_factor}")
            # print(f"🔍 DEBUG: AutoEncoder shift_factor: {self.autoencoder.ae_params.shift_factor}")
            pass
        else:
            # print("🔍 DEBUG: No ae_params found in autoencoder")
            pass
        
        # Debug: Print latent statistics before decoding
        # print(f"🔍 DEBUG: Latents before decode - shape: {latents_reshaped.shape}")
        # print(f"🔍 DEBUG: Latents before decode - min: {latents_reshaped.min().item():.6f}, max: {latents_reshaped.max().item():.6f}")
        # print(f"🔍 DEBUG: Latents before decode - mean: {latents_reshaped.mean().item():.6f}, std: {latents_reshaped.std().item():.6f}")
        
        # Decode to image
        with torch.no_grad():
            images = self.autoencoder.decode(latents_reshaped)
        
        # Debug: Print decoded image statistics
        # print(f"🔍 DEBUG: Decoded images - shape: {images.shape}")
        # print(f"🔍 DEBUG: Decoded images - min: {images.min().item():.6f}, max: {images.max().item():.6f}")
        # print(f"🔍 DEBUG: Decoded images - mean: {images.mean().item():.6f}, std: {images.std().item():.6f}")
        
        # Convert to numpy and normalize to [0, 1]
        images = images.cpu().float().numpy()
        # FLUX VAE outputs in [0, 1] range directly, no normalization needed
        images = np.clip(images, 0, 1)
        
        # print(f"🔍 DEBUG: Final numpy images - min: {images.min():.6f}, max: {images.max():.6f}")
        # print(f"🔍 DEBUG: Final numpy images - mean: {images.mean():.6f}, std: {images.std():.6f}")
        
        # Clean up GPU tensors immediately after use
        del latents_reshaped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return first image in batch
        return images[0].transpose(1, 2, 0)  # Convert from CHW to HWC
    
    def _decode_raw_latent_to_image(self, raw_latents: torch.Tensor):
        """Decode raw latents (not FLUX-patchified) to images using the autoencoder"""
        if self.autoencoder is None:
            raise ValueError("Autoencoder not loaded")
        
        # print(f"🔍 DEBUG RAW: Decoding raw latents directly (no rearrangement needed)")
        # print(f"🔍 DEBUG RAW: Raw latents shape: {raw_latents.shape}")
        # print(f"🔍 DEBUG RAW: Raw latents stats - min: {raw_latents.min().item():.6f}, max: {raw_latents.max().item():.6f}")
        # print(f"🔍 DEBUG RAW: Raw latents stats - mean: {raw_latents.mean().item():.6f}, std: {raw_latents.std().item():.6f}")
        
        # Convert latents to float32 to match autoencoder dtype
        raw_latents = raw_latents.float()
        
        # Check autoencoder parameters
        if hasattr(self.autoencoder, 'ae_params'):
            # print(f"🔍 DEBUG RAW: AutoEncoder scale_factor: {self.autoencoder.ae_params.scale_factor}")
            # print(f"🔍 DEBUG RAW: AutoEncoder shift_factor: {self.autoencoder.ae_params.shift_factor}")
            pass
        
        # Decode to image directly (raw latents are already in correct format)
        with torch.no_grad():
            images = self.autoencoder.decode(raw_latents)
        
        # Debug: Print decoded image statistics
        # print(f"🔍 DEBUG RAW: Decoded images - shape: {images.shape}")
        # print(f"🔍 DEBUG RAW: Decoded images - min: {images.min().item():.6f}, max: {images.max().item():.6f}")
        # print(f"🔍 DEBUG RAW: Decoded images - mean: {images.mean().item():.6f}, std: {images.std().item():.6f}")
        
        # Convert to numpy and normalize to [0, 1]
        images = images.cpu().float().numpy()
        # FLUX VAE outputs in [0, 1] range directly, no normalization needed
        images = np.clip(images, 0, 1)
        
        # print(f"🔍 DEBUG RAW: Final numpy images - min: {images.min():.6f}, max: {images.max():.6f}")
        # print(f"🔍 DEBUG RAW: Final numpy images - mean: {images.mean():.6f}, std: {images.std():.6f}")
        
        # Clean up GPU tensors immediately after use
        del raw_latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return first image in batch
        return images[0].transpose(1, 2, 0)  # Convert from CHW to HWC
    
    def generate_from_prompts(self, prompts: List[str], epoch: int):
        """
        Generate samples from text prompts (placeholder for future implementation).
        For now, this is a placeholder that would require text encoding.
        """
        print(f"Prompt-based generation not yet implemented for epoch {epoch}")
        print(f"Prompts: {prompts}")
        return None


def create_inference_callback(config: Dict[str, Any], eval_dataloader=None):
    """
    Create a Composer callback for running inference during training.
    """
    from composer import Callback, Logger, State
    from composer.core import Event
    
    class InferenceCallback(Callback):
        def __init__(self, config: Dict[str, Any], eval_dataloader=None):
            self.config = config
            self.eval_dataloader = eval_dataloader
            self.inference = None
            self.last_inference_epoch = -1
            # print("🔍 DEBUG: InferenceCallback initialized with validation dataloader")
            
        def fit_start(self, state: State, logger: Logger) -> None:
            """Called at the start of training"""
            # print("🔍 DEBUG: fit_start called - training started")
            
        def epoch_start(self, state: State, logger: Logger) -> None:
            """Called at the start of each epoch"""
            # print(f"🔍 DEBUG: epoch_start called - epoch: {state.timestamp.epoch.value}")
            
        def batch_start(self, state: State, logger: Logger) -> None:
            """Called at the start of each batch"""
            if state.timestamp.batch.value % 100 == 0:  # Print every 100 batches
                pass
            
        def epoch_end(self, state: State, logger: Logger) -> None:
            """Run inference at the end of each epoch if enabled"""
            # print(f"🔍 DEBUG: epoch_end called - epoch: {state.timestamp.epoch.value}")
            
            if not self.config['inference']['enabled']:
                # print("🔍 DEBUG: Inference disabled in config")
                return
                
            current_epoch = int(state.timestamp.epoch.value)
            # print(f"🔍 DEBUG: Current epoch: {current_epoch}, Last inference epoch: {self.last_inference_epoch}")
            
            # Check if we should run inference based on config interval
            # Parse interval from config (e.g., "1ep", "2ep")
            interval_epochs = 1  # Default to every epoch
            if 'interval' in self.config['inference']:
                interval_str = self.config['inference']['interval']
                if interval_str.endswith('ep'):
                    interval_epochs = int(interval_str[:-2])
            
            # print(f"🔍 DEBUG: Checking inference condition:")
            # print(f"🔍 DEBUG: current_epoch > self.last_inference_epoch: {current_epoch > self.last_inference_epoch}")
            # print(f"🔍 DEBUG: current_epoch % interval_epochs == 0: {current_epoch % interval_epochs == 0}")
            # print(f"🔍 DEBUG: current_epoch: {current_epoch}, interval_epochs: {interval_epochs}")
            
            # Add debug print to track epoch numbers
            # print(f"DEBUG: Checking inference - Current epoch: {current_epoch}, Last inference: {self.last_inference_epoch}, Interval: {interval_epochs}")
            
            if current_epoch > self.last_inference_epoch and current_epoch % interval_epochs == 0:
                print(f"DEBUG: Running inference for epoch {current_epoch}")
                
                # Update last inference epoch immediately to prevent multiple calls
                self.last_inference_epoch = current_epoch
                if self.inference is None:
                    # Initialize inference with the model
                    # Get the model object if it has been wrapped by DDP
                    from torch.nn.parallel import DistributedDataParallel
                    model = state.model.module if isinstance(state.model, DistributedDataParallel) else state.model
                    
                    self.inference = TrainingInference(
                        model=model.flux_model,
                        config=self.config,
                        device=next(model.flux_model.parameters()).device,
                        eval_dataloader=self.eval_dataloader
                    )
                
                # Use validation dataloader for inference instead of training batch
                try:
                    # Print memory usage before inference
                    # print_gpu_memory_usage(f"🔍 Before inference (epoch {current_epoch}):")
                    
                    # Ensure epoch number is correct
                    epoch_num = int(state.timestamp.epoch.value)
                    print(f"DEBUG: Generating validation sample for epoch {epoch_num} (raw: {state.timestamp.epoch.value})")
                    
                    self.inference.generate_sample_from_validation(
                        epoch=epoch_num,
                        step=int(state.timestamp.batch.value)
                    )
                    
                    # Clean up GPU memory after inference
                    cleanup_gpu_memory(verbose=False)  # Disabled verbose output
                    # print_gpu_memory_usage(f"🔍 After inference (epoch {current_epoch}):")
                    
                    print(f"Completed validation inference for epoch {current_epoch}")
                except Exception as e:
                    print(f"Error during validation inference: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Emergency memory cleanup on error
                    print("🚨 Emergency cleanup after inference error:")
                    cleanup_gpu_memory(verbose=True)
            else:
                # print(f"🔍 DEBUG: Inference not triggered - current_epoch: {current_epoch}, last_inference_epoch: {self.last_inference_epoch}, interval_epochs: {interval_epochs}")
                pass
    
    return InferenceCallback(config, eval_dataloader)


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
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        image_array = np.array(image)
        
        # Convert to tensor and normalize to [0, 1] (same as stream_flux.py)
        image_tensor = torch.from_numpy(image_array).float() / 255.0
        # Keep in [0, 1] range - FLUX VAE expects this range
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
        # FLUX VAE outputs in [0, 1] range directly, no normalization needed
        decoded_image_np = np.clip(decoded_image_np, 0, 1)
        
        # Convert from CHW to HWC
        decoded_image_np = decoded_image_np[0].transpose(1, 2, 0)
        
        # Save the original and decoded images
        original_save_path = os.path.join(save_dir, "original_image.png")
        decoded_save_path = os.path.join(save_dir, "decoded_image.png")
        
        # Convert original image back to [0, 1] for saving (already in [0, 1] range)
        original_image_np = image_tensor.cpu().numpy()[0].transpose(1, 2, 0)
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


def test_latent_scaling_issue(image_path: str, save_dir: str = "/tmp", device: str = "cuda"):
    """
    Test function to verify if the issue is with latent scaling during dataset generation vs inference.
    This will help identify if the original latents are being processed incorrectly.
    """
    import torch
    import numpy as np
    from PIL import Image
    import os
    from einops import rearrange
    
    print(f"🔍 Testing latent scaling issue with image: {image_path}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Load the autoencoder
        print("Loading autoencoder...")
        autoencoder = load_ae("flux-schnell", device=device)
        autoencoder.eval()
        print(f"Autoencoder loaded successfully on {device}")
        
        # Load and preprocess the image (same as dataset generation)
        print("Loading and preprocessing image...")
        image = Image.open(image_path).convert('RGB')
        
        # Resize image to 512x512 (same as your test)
        print("Resizing image to 512x512...")
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        image_array = np.array(image)
        
        # Convert to tensor and normalize to [0, 1] (same as stream_flux.py)
        image_tensor = torch.from_numpy(image_array).float() / 255.0
        # Keep in [0, 1] range - FLUX VAE expects this range
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        image_tensor = image_tensor.to(device)
        
        print(f"Original image shape: {image_tensor.shape}")
        
        # Encode the image to latents (same as dataset generation)
        print("Encoding image to latents...")
        with torch.no_grad():
            img_latents = autoencoder.encode(image_tensor)
        
        print(f"Encoded latents shape: {img_latents.shape}")
        
        # Check the scaling factors from the autoencoder
        ae_params = autoencoder.ae_params if hasattr(autoencoder, 'ae_params') else None
        if ae_params:
            print(f"Autoencoder scale_factor: {ae_params.scale_factor}")
            print(f"Autoencoder shift_factor: {ae_params.shift_factor}")
        
        # Test 1: Decode latents directly (as they come from VAE encode)
        print("Test 1: Decoding latents directly from VAE encode...")
        with torch.no_grad():
            decoded_direct = autoencoder.decode(img_latents)
        
        # Test 2: Apply the same preprocessing as in dataset generation
        print("Test 2: Applying dataset preprocessing (rearrange to FLUX format)...")
        # Rearrange to FLUX format (same as in stream_flux.py)
        h, w = img_latents.shape[2], img_latents.shape[3]
        flux_latents = rearrange(img_latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        print(f"FLUX format latents shape: {flux_latents.shape}")
        
        # Rearrange back to image format
        flux_latents_back = rearrange(flux_latents, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=h//2, w=w//2)
        print(f"FLUX format back to image shape: {flux_latents_back.shape}")
        
        # Decode the rearranged latents
        with torch.no_grad():
            decoded_rearranged = autoencoder.decode(flux_latents_back)
        
        # Test 3: Check if there are any scaling issues
        print("Test 3: Checking for scaling differences...")
        print(f"Original latents mean: {img_latents.mean():.6f}, std: {img_latents.std():.6f}")
        print(f"FLUX latents mean: {flux_latents.mean():.6f}, std: {flux_latents.std():.6f}")
        print(f"FLUX back latents mean: {flux_latents_back.mean():.6f}, std: {flux_latents_back.std():.6f}")
        
        # Convert to numpy and save
        def to_numpy(tensor):
            return tensor.detach().cpu().float().numpy()
        
        # Save all versions
        direct_save_path = os.path.join(save_dir, "decoded_direct.png")
        rearranged_save_path = os.path.join(save_dir, "decoded_rearranged.png")
        original_save_path = os.path.join(save_dir, "original_image.png")
        
        # Convert to [0, 1] range and save (already in [0, 1] range)
        def save_image(tensor, path):
            img_np = tensor.cpu().float().numpy()
            # Images are already in [0, 1] range, no normalization needed
            img_np = np.clip(img_np, 0, 1)
            img_np = img_np[0].transpose(1, 2, 0)  # Convert from CHW to HWC
            plt.imsave(path, img_np)
        
        save_image(decoded_direct, direct_save_path)
        save_image(decoded_rearranged, rearranged_save_path)
        save_image(image_tensor, original_save_path)
        
        print(f"Direct decode saved to: {direct_save_path}")
        print(f"Rearranged decode saved to: {rearranged_save_path}")
        print(f"Original image saved to: {original_save_path}")
        
        # Calculate differences
        direct_np = decoded_direct.cpu().float().numpy()
        rearranged_np = decoded_rearranged.cpu().float().numpy()
        original_np = image_tensor.cpu().float().numpy()
        
        mse_direct = np.mean((original_np - direct_np) ** 2)
        mse_rearranged = np.mean((original_np - rearranged_np) ** 2)
        mse_between = np.mean((direct_np - rearranged_np) ** 2)
        
        print(f"MSE Original vs Direct: {mse_direct:.6f}")
        print(f"MSE Original vs Rearranged: {mse_rearranged:.6f}")
        print(f"MSE Direct vs Rearranged: {mse_between:.6f}")
        
        if mse_between > 1e-6:
            print("⚠️  WARNING: There's a difference between direct and rearranged decoding!")
            print("This suggests the FLUX format rearrangement is causing issues.")
        else:
            print("✅ Direct and rearranged decoding are identical.")
        
        print("✅ Latent scaling test completed!")
        
        return {
            'direct_decoded': decoded_direct,
            'rearranged_decoded': decoded_rearranged,
            'original': image_tensor,
            'mse_direct': mse_direct,
            'mse_rearranged': mse_rearranged,
            'mse_between': mse_between
        }
        
    except Exception as e:
        print(f"❌ Error during latent scaling test: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test the latent scaling issue with the specified image
    test_image_path = "/data0/teja_works/teja_ss/road_crossing.png"
    test_save_dir = "./test_latent_scaling_results"
    
    print("🚀 Starting latent scaling test...")
    result = test_latent_scaling_issue(
        image_path=test_image_path,
        save_dir=test_save_dir,
        device="cuda"
    )
    
    if result is not None:
        print("🎉 Test completed successfully!")
    else:
        print("💥 Test failed!")
