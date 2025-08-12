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

pretrained_model_name_or_path = "Qwen/Qwen-Image"

weight_dtype = torch.bfloat16
device = "cuda"

# Training parameters
learning_rate = 1e-5
num_epochs = 10
gradient_accumulation_steps = 1
max_grad_norm = 1.0
resolution = 512
weighting_scheme = "none"
logit_mean = 0.0
logit_std = 1.0
mode_scale = 1.29

# Load the tokenizers
tokenizer = Qwen2Tokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)

# Load VAE
vae = AutoencoderKLQwenImage.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="vae",
)
vae_scale_factor = 2 ** len(vae.temperal_downsample)
latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(device)
latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device)

# Load text encoder
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="text_encoder", 
    torch_dtype=weight_dtype
)

# Load transformer (this will be fine-tuned)
transformer = QwenImageTransformer2DModel.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="transformer",
    torch_dtype=weight_dtype
)

# Load scheduler
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="scheduler", 
    shift=3.0
)
noise_scheduler_copy = copy.deepcopy(noise_scheduler)

# Set requires_grad for full fine-tuning
vae.requires_grad_(False)  # VAE stays frozen
text_encoder.requires_grad_(False)  # Text encoder stays frozen
transformer.requires_grad_(True)  # Transformer will be fine-tuned

# Move models to device
to_kwargs = {"dtype": weight_dtype, "device": device}
vae.to(**to_kwargs)
text_encoder.to(**to_kwargs)
transformer.to(**to_kwargs)

# Set models to training mode
transformer.train()
vae.eval()
text_encoder.eval()

max_sequence_length = 512

# Initialize a text encoding pipeline
text_encoding_pipeline = QwenImagePipeline.from_pretrained(
    pretrained_model_name_or_path,
    vae=None,
    transformer=None,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=None,
)

def compute_text_embeddings(prompt, text_encoding_pipeline):
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
            prompt=prompt,                         # can be str or List[str]
            max_sequence_length=max_sequence_length,
            padding="max_length",                 # ensure fixed length
            truncation=True
        )
    return prompt_embeds, prompt_embeds_mask


def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

# Setup optimizer
optimizer = torch.optim.AdamW(
    transformer.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    eps=1e-8
)

# Your existing dataloader setup
from dataloading import create_dataloader
dataloader = create_dataloader("config.yaml")

print("Starting full fine-tuning...")

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for step, batch in enumerate(progress_bar):
        # Prepare prompts
        prompts = batch['prompts']
        
        # Get text embeddings
        with torch.no_grad():
            if isinstance(prompts, list):
                # Handle batch of prompts
                prompt_embeds_list = []
                prompt_embeds_mask_list = []
                for prompt in prompts:
                    pe, pem = compute_text_embeddings(prompt, text_encoding_pipeline)
                    prompt_embeds_list.append(pe)
                    prompt_embeds_mask_list.append(pem)
                prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
                prompt_embeds_mask = torch.cat(prompt_embeds_mask_list, dim=0)
            else:
                prompt_embeds, prompt_embeds_mask = compute_text_embeddings(prompts, text_encoding_pipeline)
        
        # Prepare pixel values
        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        
        # Encode images to latents
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            model_input = (latents - latents_mean) * latents_std
            model_input = model_input.to(dtype=weight_dtype)
        
        # Sample noise
        noise = torch.randn_like(model_input)
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
        
        # Forward pass through transformer
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
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Update progress
        epoch_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{epoch_loss/num_batches:.4f}'
        })
    
    print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/num_batches:.4f}")

print("Training completed!")

# Save the fine-tuned transformer
output_dir = "fine_tuned_qwen_image"
transformer.save_pretrained(output_dir)
print(f"Fine-tuned model saved to {output_dir}")

# Optional: Save a complete pipeline
pipeline = QwenImagePipeline.from_pretrained(
    pretrained_model_name_or_path,
    transformer=transformer,
    torch_dtype=weight_dtype
)
pipeline.save_pretrained(f"{output_dir}_pipeline")
print(f"Complete pipeline saved to {output_dir}_pipeline")