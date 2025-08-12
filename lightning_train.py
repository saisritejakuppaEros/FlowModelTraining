import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
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
import os
import gc
from lightning_dataloading import QwenImageDataModule
from typing import Optional, Dict, Any
import bitsandbytes as bnb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from lightning.pytorch.utilities import rank_zero_only


class QwenImageLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Qwen Image fine-tuning
    Optimized for distributed training with DeepSpeed support
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Qwen/Qwen-Image",
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        resolution: int = 512,
        weighting_scheme: str = "none",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
        max_sequence_length: int = 512,
        enable_gradient_checkpointing: bool = True,
        use_8bit_optimizer: bool = True,
        scheduler_type: str = "cosine",
        warmup_steps: int = 100,
        compile_model: bool = False,
        text_encoder_cpu: bool = True,
        use_deepspeed: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Training parameters
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.resolution = resolution
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale
        self.max_sequence_length = max_sequence_length
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.use_8bit_optimizer = use_8bit_optimizer
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.compile_model = compile_model
        self.text_encoder_cpu = text_encoder_cpu
        self.use_deepspeed = use_deepspeed
        
        # Model components
        self.tokenizer = None
        self.vae = None
        self.text_encoder = None
        self.transformer = None
        self.noise_scheduler = None
        self.noise_scheduler_copy = None
        self.text_encoding_pipeline = None
        
        # Model properties
        self.weight_dtype = torch.bfloat16
        self.latents_mean = None
        self.latents_std = None
        self.vae_scale_factor = None
        
        # Initialize models
        self._setup_models()
    
    def _setup_models(self):
        """Initialize all model components"""
        print("Loading models...")
        
        # Load tokenizer
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        
        # Load VAE with memory optimization
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=self.weight_dtype,
        )
        
        # Fix potential typo: temperal -> temporal
        self.vae_scale_factor = 2 ** len(getattr(self.vae, 'temporal_downsample', getattr(self.vae, 'temperal_downsample', [0, 0])))
        
        # Load text encoder (will be kept on CPU to save GPU memory)
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            torch_dtype=self.weight_dtype,
            low_cpu_mem_usage=True,
        )
        
        # Load transformer
        self.transformer = QwenImageTransformer2DModel.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
            low_cpu_mem_usage=True,
        )
        
        # Enable gradient checkpointing
        if self.enable_gradient_checkpointing and hasattr(self.transformer, 'enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()
        
        # Load scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="scheduler", 
            shift=3.0
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        
        # Set requires_grad
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(True)
        
        # Set evaluation mode for frozen models
        self.vae.eval()
        self.text_encoder.eval()
        
        # Optionally keep text encoder on CPU to save GPU memory
        if self.text_encoder_cpu:
            self.text_encoder.to('cpu')
            print("Text encoder moved to CPU to save GPU memory")
        else:
            print("Text encoder will be moved to GPU with other models")
        
        # Compile model if requested (PyTorch 2.0+)
        if self.compile_model and hasattr(torch, 'compile'):
            self.transformer = torch.compile(self.transformer)
            print("Model compiled with torch.compile")
    
    def setup(self, stage: str):
        """Setup method called by Lightning"""
        if stage == "fit":
            # Move latents mean/std to device
            self.latents_mean = (torch.tensor(self.vae.config.latents_mean)
                                .view(1, self.vae.config.z_dim, 1, 1, 1)
                                .to(self.device, dtype=self.weight_dtype))
            self.latents_std = (1.0 / torch.tensor(self.vae.config.latents_std)
                               .view(1, self.vae.config.z_dim, 1, 1, 1)
                               .to(self.device, dtype=self.weight_dtype))
            
            # Initialize text encoding pipeline
            # Text encoder device placement depends on text_encoder_cpu setting
            self.text_encoding_pipeline = QwenImagePipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                vae=None,
                transformer=None,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                scheduler=None,
                torch_dtype=self.weight_dtype,
            )
            
            # Ensure text encoder is on correct device after pipeline creation
            if self.text_encoder_cpu:
                self.text_encoder.to('cpu')
            else:
                self.text_encoder.to(self.device)
    
    def compute_text_embeddings(self, prompt):
        """Compute text embeddings with memory optimization"""
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = self.text_encoding_pipeline.encode_prompt(
                prompt=prompt,
                max_sequence_length=self.max_sequence_length,
            )
            # Move embeddings to GPU for training if text encoder is on CPU
            if self.text_encoder_cpu:
                prompt_embeds = prompt_embeds.to(self.device, dtype=self.weight_dtype)
                prompt_embeds_mask = prompt_embeds_mask.to(self.device)
            else:
                # Ensure correct dtype if text encoder is on GPU
                prompt_embeds = prompt_embeds.to(dtype=self.weight_dtype)
        return prompt_embeds, prompt_embeds_mask
    
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        """Get sigma values for flow matching"""
        sigmas = self.noise_scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Prepare prompts
        prompts = batch['prompts']
        
        # Get text embeddings
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = self.compute_text_embeddings(prompts)
        
        # Prepare pixel values
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.weight_dtype)
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            model_input = (latents - self.latents_mean) * self.latents_std
            model_input = model_input.to(dtype=self.weight_dtype)
        
        # Clear pixel_values from memory
        del pixel_values
        
        # Sample noise
        noise = torch.randn_like(model_input, dtype=self.weight_dtype)
        bsz = model_input.shape[0]
        
        # Sample timesteps
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
        
        # Add noise according to flow matching
        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        
        # Prepare inputs for transformer
        img_shapes = [
            (1, self.resolution // self.vae_scale_factor // 2, self.resolution // self.vae_scale_factor // 2)
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
        
        # Forward pass
        model_pred = self.transformer(
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
            model_pred, self.resolution, self.resolution, self.vae_scale_factor
        )
        
        # Compute loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)
        target = noise - model_input
        
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], on_step=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # When using DeepSpeed with optimizer config, let DeepSpeed handle the optimizer
        # This avoids conflicts with ZeRO-Offload and 8-bit optimizers
        if self.use_deepspeed:
            return None
        
        # For non-DeepSpeed training, configure optimizer normally
        if self.use_8bit_optimizer:
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        
        optimizer = optimizer_class(
            self.transformer.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
            eps=1e-8,
        )
        
        # Setup scheduler
        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs,
                eta_min=self.learning_rate * 0.1
            )
        elif self.scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.warmup_steps
            )
        else:
            return optimizer
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping before optimizer step"""
        if self.max_grad_norm > 0:
            self.clip_gradients(optimizer, gradient_clip_val=self.max_grad_norm, gradient_clip_algorithm="norm")
    
    @rank_zero_only
    def on_train_epoch_end(self):
        """Clean up memory at the end of each epoch"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save additional model components"""
        # Only save transformer state dict to reduce checkpoint size
        checkpoint['transformer_state_dict'] = self.transformer.state_dict()
    
    def load_from_checkpoint(self, checkpoint_path: str, **kwargs):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load transformer state dict if available
        if 'transformer_state_dict' in checkpoint:
            self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        
        return super().load_from_checkpoint(checkpoint_path, **kwargs)


def main():
    """Main training function"""
    import argparse
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.strategies import DeepSpeedStrategy
    
    parser = argparse.ArgumentParser(description="Train Qwen Image with PyTorch Lightning")
    
    # Model arguments
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--pretrained_model", type=str, default="Qwen/Qwen-Image", help="Pretrained model path")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Training arguments
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--use_8bit_optimizer", action="store_true", default=True, help="Use 8-bit optimizer")
    parser.add_argument("--compile_model", action="store_true", help="Compile model with torch.compile")
    parser.add_argument("--precision", type=str, default="bf16-mixed", choices=["16-mixed", "bf16-mixed", "32"], help="Training precision")
    
    # DeepSpeed arguments
    parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed strategy")
    parser.add_argument("--deepspeed_config", type=str, help="DeepSpeed config file path")
    
    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="qwen_image_finetune", help="Experiment name")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="qwen-image-training", help="W&B project name")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup data module
    data_module = QwenImageDataModule(
        config_path=args.config_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        size=args.resolution,
    )
    
    # Setup model
    model = QwenImageLightningModule(
        pretrained_model_name_or_path=args.pretrained_model,
        learning_rate=args.learning_rate,
        resolution=args.resolution,
        enable_gradient_checkpointing=args.gradient_checkpointing,
        use_8bit_optimizer=args.use_8bit_optimizer,
        compile_model=args.compile_model,
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="qwen-image-{epoch:02d}-{val/loss:.2f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_epochs=1,
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val/loss",
            patience=3,
            mode="min",
            verbose=True,
        ),
    ]
    
    # Setup logger
    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.experiment_name,
            save_dir=args.output_dir,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name=args.experiment_name,
        )
    
    # Setup strategy
    strategy = "auto"
    if args.use_deepspeed:
        deepspeed_config = args.deepspeed_config or {
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "wall_clock_breakdown": False,
        }
        strategy = DeepSpeedStrategy(config=deepspeed_config)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        strategy=strategy,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        log_every_n_steps=10,
        val_check_interval=0.5,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Save final model
    if trainer.is_global_zero:
        final_output_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Save transformer
        model.transformer.save_pretrained(final_output_dir)
        model.tokenizer.save_pretrained(final_output_dir)
        
        print(f"Final model saved to {final_output_dir}")


if __name__ == "__main__":
    main() 