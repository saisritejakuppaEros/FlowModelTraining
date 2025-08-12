from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
import torch
import torch.nn.functional as F
import diffusers
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
import copy
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration, DistributedDataParallelKwargs
import os
import gc

# Memory optimization: Enable memory efficient attention
torch.backends.cuda.enable_flash_sdp(True)

# Configure distributed training properly
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
project_config = ProjectConfiguration(project_dir="./logs", logging_dir="./logs")

# Initialize accelerator with proper distributed training setup
accelerator = Accelerator(
    gradient_accumulation_steps=4,  # Increased for memory efficiency
    mixed_precision="bf16",  # bf16 is generally more stable than fp16
    log_with="tensorboard",
    project_config=project_config,
    kwargs_handlers=[kwargs],
    # Remove split_batches to allow proper distribution across GPUs
    split_batches=False,
)

logger = get_logger(__name__)
set_seed(42)

pretrained_model_name_or_path = "Qwen/Qwen-Image"

weight_dtype = torch.bfloat16
device = accelerator.device

# Training parameters - optimized for multi-GPU
learning_rate = 1e-5
num_epochs = 10
gradient_accumulation_steps = 4  # Process smaller batches, accumulate gradients
max_grad_norm = 1.0
resolution = 512
weighting_scheme = "none"
logit_mean = 0.0
logit_std = 1.0
mode_scale = 1.29
max_sequence_length = 512

# Memory optimization: Enable gradient checkpointing
enable_gradient_checkpointing = True

# Print distributed training info
if accelerator.is_local_main_process:
    print(f"Number of processes: {accelerator.num_processes}")
    print(f"Distributed type: {accelerator.distributed_type}")
    print(f"Local rank: {accelerator.local_process_index}")
    print(f"Device: {device}")

print("Loading models...")

# Load tokenizer
tokenizer = Qwen2Tokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)

# Load VAE with memory optimization
vae = AutoencoderKLQwenImage.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="vae",
    torch_dtype=weight_dtype,
)

# Fix potential typo: temperal -> temporal
vae_scale_factor = 2 ** len(getattr(vae, 'temporal_downsample', getattr(vae, 'temperal_downsample', [0, 0])))
latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype=weight_dtype)
latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype=weight_dtype)

# Load text encoder with CPU offload option
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="text_encoder", 
    torch_dtype=weight_dtype,
    low_cpu_mem_usage=True,
)

# Load transformer with optimizations
transformer = QwenImageTransformer2DModel.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="transformer",
    torch_dtype=weight_dtype,
    low_cpu_mem_usage=True,
)

# Enable gradient checkpointing for transformer
if enable_gradient_checkpointing and hasattr(transformer, 'enable_gradient_checkpointing'):
    transformer.enable_gradient_checkpointing()

# Load scheduler
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="scheduler", 
    shift=3.0
)
noise_scheduler_copy = copy.deepcopy(noise_scheduler)

# Set requires_grad
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
transformer.requires_grad_(True)

# Move models to device with memory optimization
print("Moving models to device...")
vae.to(device, dtype=weight_dtype)
text_encoder.to(device, dtype=weight_dtype)
transformer.to(device, dtype=weight_dtype)

# Set evaluation mode for frozen models
transformer.train()
vae.eval()
text_encoder.eval()

# Memory cleanup
gc.collect()
torch.cuda.empty_cache()

# Initialize text encoding pipeline
text_encoding_pipeline = QwenImagePipeline.from_pretrained(
    pretrained_model_name_or_path,
    vae=None,
    transformer=None,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=None,
    torch_dtype=weight_dtype,
)

def compute_text_embeddings(prompt, text_encoding_pipeline):
    """Compute text embeddings with memory optimization."""
    with torch.no_grad():
        # Use autocast for mixed precision
        with accelerator.autocast():
            prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                prompt=prompt,
                max_sequence_length=max_sequence_length,
            )
    return prompt_embeds, prompt_embeds_mask

def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    """Get sigma values for flow matching."""
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

# Setup optimizer with memory-efficient settings
import bitsandbytes as bnb
optimizer_class = bnb.optim.AdamW8bit
optimizer = optimizer_class(
    transformer.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    eps=1e-8,
    # Memory optimization: Enable fused optimizer
)

# Setup learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Your dataloader setup
from dataloading import create_dataloader
# Increase batch size for better GPU utilization across 8 GPUs
# Each GPU will get batch_size // num_processes samples
total_batch_size = 16  # This will be split across GPUs
batch_size_per_gpu = max(1, total_batch_size // accelerator.num_processes)
dataloader = create_dataloader("config.yaml", batch_size=total_batch_size, num_workers=8)

# Prepare with accelerator
transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    transformer, optimizer, dataloader, lr_scheduler
)

unwrapped_transformer = accelerator.unwrap_model(transformer)

print("Starting optimized fine-tuning...")
print(f"Total batch size: {total_batch_size}")
print(f"Batch size per GPU: {batch_size_per_gpu}")
print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
print(f"Effective batch size: {total_batch_size * gradient_accumulation_steps}")
print(f"Mixed precision: {accelerator.mixed_precision}")
print(f"Number of GPUs: {accelerator.num_processes}")

# Training loop with memory optimizations
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch+1}/{num_epochs}",
        disable=not accelerator.is_local_main_process
    )
    
    for step, batch in enumerate(progress_bar):
        with accelerator.accumulate(transformer):
            # Prepare prompts
            prompts = batch['prompts']
            
            # Get text embeddings with memory optimization
            with torch.no_grad():
                prompt_embeds, prompt_embeds_mask = compute_text_embeddings(prompts, text_encoding_pipeline)
            
            # Prepare pixel values
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            
            # Encode images to latents with autocast
            with torch.no_grad():
                with accelerator.autocast():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (latents - latents_mean) * latents_std
                    model_input = model_input.to(dtype=weight_dtype)
            
            # Clear pixel_values from memory
            del pixel_values
            
            # Sample noise
            noise = torch.randn_like(model_input, dtype=weight_dtype)
            bsz = model_input.shape[0]
            
            # Sample timesteps
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=bsz,
                logit_mean=logit_mean,
                logit_std=logit_std,
                mode_scale=mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
            
            # Add noise according to flow matching
            sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
            
            # Prepare inputs for transformer
            img_shapes = [
                (1, resolution // vae_scale_factor // 2, resolution // vae_scale_factor // 2)
            ] * bsz
            
            # Transpose dimensions and pack latents
            noisy_model_input = noisy_model_input.permute(0, 2, 1, 3, 4)
            packed_noisy_model_input = QwenImagePipeline._pack_latents(
                noisy_model_input,
                batch_size=model_input.shape[0],
                num_channels_latents=model_input.shape[1],
                height=model_input.shape[3],
                width=model_input.shape[4],
            )
            
            # Forward pass with mixed precision
            with accelerator.autocast():
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    timestep=timesteps / 1000,
                    img_shapes=img_shapes,
                    txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
                    return_dict=False,
                )[0]
            
            # Unpack latents
            model_pred = QwenImagePipeline._unpack_latents(
                model_pred, resolution, resolution, vae_scale_factor
            )
            
            # Compute loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
            target = noise - model_input
            
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Memory cleanup periodically
            if step % 10 == 0:
                torch.cuda.empty_cache()
        
        # Update progress
        if accelerator.sync_gradients:
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log metrics
            accelerator.log({
                "train_loss": loss.item(),
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": step,
            })
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/num_batches:.4f}',
                'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'
            })
    
    # Wait for all processes
    accelerator.wait_for_everyone()
    
    # Memory cleanup at end of epoch
    gc.collect()
    torch.cuda.empty_cache()
    
    if accelerator.is_local_main_process:
        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/num_batches:.4f}")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_dir = f"checkpoint_epoch_{epoch+1}"
            unwrapped_transformer.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved to {checkpoint_dir}")

print("Training completed!")

# Final model saving
if accelerator.is_local_main_process:
    output_dir = "fine_tuned_qwen_image"
    unwrapped_transformer.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

    # Save complete pipeline
    try:
        pipeline = QwenImagePipeline.from_pretrained(
            pretrained_model_name_or_path,
            transformer=unwrapped_transformer,
            torch_dtype=weight_dtype
        )
        pipeline.save_pretrained(f"{output_dir}_pipeline")
        print(f"Complete pipeline saved to {output_dir}_pipeline")
    except Exception as e:
        print(f"Could not save complete pipeline: {e}")

accelerator.end_training()