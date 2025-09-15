import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Any, Dict, Optional, Tuple
from einops import rearrange

# MosaicML Composer imports
import composer
from composer import Trainer
from composer.core import Precision
from composer.models import ComposerModel
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer.algorithms import GradientClipping
from composer.loggers import WandBLogger
from composer.utils import dist
from streaming import StreamingDataset

import os

# Your FLUX model import
from model_utils.dit import load_flow_model2

# Let Composer handle distributed setup - don't manually initialize torch.distributed


class FluxComposerModel(ComposerModel):
    """Composer wrapper for FLUX flow matching model"""
    
    def __init__(self, model_name: str = "flux-schnell"):
        super().__init__()
        
        # Load the FLUX model
        print(f"Loading FLUX model: {model_name}")
        self.flux_model = load_flow_model2(model_name)
        self.model_name = model_name
        
        # Flow matching parameters
        self.sigma = 1.0  # Noise scale
        
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for training"""
        
        # Extract batch components
        img_latents = batch['img_latents']  # Shape: [B, H*W, C*Ph*Pw]
        img_ids = batch['img_ids']          # Shape: [B, H*W, 3]
        txt_embeds = batch['txt_embeds']    # Shape: [B, seq_len, dim]
        txt_ids = batch['txt_ids']          # Shape: [B, seq_len, 3] 
        vec_embeds = batch['vec_embeds']    # Shape: [B, dim]
        
        batch_size = img_latents.shape[0]
        device = img_latents.device
        
        # Sample random timesteps
        t = torch.rand(batch_size, device=device, dtype=img_latents.dtype)
        
        # Create noise
        noise = torch.randn_like(img_latents)
        
        # Flow matching interpolation: x_t = (1-t) * x_1 + t * noise
        t_expanded = t.view(batch_size, 1, 1)
        x_t = (1 - t_expanded) * img_latents + t_expanded * noise
        
        # Target velocity field: v_t = noise - x_1 (from x_t to noise)
        target_velocity = noise - img_latents
        
        # Guidance scale (typical values 3-7 for FLUX)
        guidance = torch.full((batch_size,), 4.0, device=device, dtype=img_latents.dtype)
        
        # Forward through FLUX model
        predicted_velocity = self.flux_model(
            img=x_t,
            img_ids=img_ids,
            txt=txt_embeds,
            txt_ids=txt_ids,
            timesteps=t,
            y=vec_embeds,
            guidance=guidance
        )
        
        return predicted_velocity, target_velocity, t
    
    def loss(self, outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
             batch: Dict[str, Any]) -> torch.Tensor:
        """Compute flow matching loss"""
        predicted_velocity, target_velocity, timesteps = outputs
        
        # Simple MSE loss
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        return loss
    
    def metrics(self, train: bool = False) -> Dict[str, Any]:
        """Define metrics to track"""
        return {}

def create_flux_dataloader(
    dataset_path: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    max_samples: Optional[int] = None
) -> DataLoader:
    """Create dataloader from MosaicML streaming dataset"""
    
    print(f"Loading streaming dataset from: {dataset_path}")
    
    # Create streaming dataset
    dataset = StreamingDataset(
        local=dataset_path,
        shuffle=shuffle,
    )
    
    if max_samples:
        # Limit dataset size for testing
        dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
        print(f"Limited dataset to {len(dataset)} samples")
    
    # Handle distributed training using Composer's utilities
    sampler = None
    if dist.get_world_size() > 1:
        sampler = dist.get_sampler(
            dataset,
            shuffle=shuffle,
            drop_last=True  # Ensure all ranks have same number of batches
        )
        shuffle = False  # Don't shuffle in DataLoader when using DistributedSampler
    
    # The batch_size parameter should be per-device batch size, not global
    # Following the working_code.py pattern where batch_size is already divided by world_size
    per_device_batch_size = max(1, batch_size // dist.get_world_size())
    
    print(f"World size: {dist.get_world_size()}, Global batch size: {batch_size}, Per-device batch size: {per_device_batch_size}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        drop_last=True,  # Important for distributed training consistency
    )
    
    print(f"Created dataloader with {len(dataloader)} batches")
    return dataloader

def train_flux_with_composer(
    dataset_path: str = "./flux_mds_dataset",
    output_dir: str = "./flux_composer_checkpoints",
    model_name: str = "flux-schnell",
    batch_size: int = 4,
    max_duration: str = "50ep",  # 50 epochs
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_duration: str = "100ba",  # 100 batches warmup
    save_interval: str = "10ep",  # Save every 10 epochs
    eval_interval: str = "5ep",   # Evaluate every 5 epochs
    precision: str = "amp_bf16",  # Use bfloat16 mixed precision
    grad_clip_norm: float = 1.0,
    num_workers: int = 4,
    log_to_wandb: bool = False,
    wandb_project: str = "flux-training",
    max_samples: Optional[int] = None,  # Limit samples for testing
    seed: int = 42,
):
    """Train FLUX model using MosaicML Composer"""
    
    # Set seed for reproducibility
    composer.utils.reproducibility.seed_all(seed)
    
    # Create model
    print("Initializing FLUX Composer model...")
    model = FluxComposerModel(model_name=model_name)
    
    # Create dataloaders
    train_dataloader = create_flux_dataloader(
        dataset_path=dataset_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_samples=max_samples
    )
    
    # For now, use same dataset for eval (in practice you'd want a separate eval set)
    eval_dataloader = create_flux_dataloader(
        dataset_path=dataset_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_samples=min(1000, max_samples) if max_samples else 1000  # Small eval set
    )
    
    # Create optimizer
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
    )
    
    # Create scheduler with warmup
    scheduler = CosineAnnealingWithWarmupScheduler(
        t_warmup=warmup_duration,
        t_max=max_duration,  # Duration for cosine annealing
        alpha_f=0.1,  # Final learning rate multiplier
    )
    
    # Setup gradient clipping
    gradient_clipping = GradientClipping(
        clipping_type='norm',
        clipping_threshold=grad_clip_norm
    )
    
    # Setup loggers
    loggers = []
    if log_to_wandb:
        loggers.append(WandBLogger(project=wandb_project))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration=max_duration,
        optimizers=optimizer,
        schedulers=scheduler,
        algorithms=[gradient_clipping],
        device="gpu" if torch.cuda.is_available() else "cpu",
        precision=getattr(Precision, precision.upper()),
        save_folder=output_dir,
        save_interval=save_interval,
        eval_interval=eval_interval,
        loggers=loggers,
        seed=seed,
        
    )
    
    # Print training info
    print(f"Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max duration: {max_duration}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Precision: {precision}")
    print(f"  Device: {trainer.state.device}")
    print(f"  Output directory: {output_dir}")
    
    # Start training
    print("Starting training...")
    trainer.fit()
    
    print("Training completed!")
    return trainer

# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    config = {
        "dataset_path": "/data0/teja_works/diffusion_training/flux_datastreaming/data_preparation/flux_mds_dataset",
        "output_dir": "./flux_composer_checkpoints", 
        "model_name": "flux-schnell",
        "batch_size": 8,  # Global batch size that divides evenly by 8 GPUs (1 per GPU)
        "max_duration": "50ep",
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_duration": "100ba", 
        "save_interval": "10ep",
        "eval_interval": "5ep",
        "precision": "amp_bf16",
        "grad_clip_norm": 1.0,
        "num_workers": 4,
        "log_to_wandb": False,  # Set to True to log to Weights & Biases
        "wandb_project": "flux-flow-matching",
        "max_samples": 5000,  # Limit for testing, set to None for full dataset
        "seed": 42,
    }
    
    # Run training
    trainer = train_flux_with_composer(**config)
    
    # Optional: Save final model state dict
    final_model_path = os.path.join(config["output_dir"], "final_model.pt")
    torch.save(trainer.state.model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

# Additional utility for resuming training
def resume_training(checkpoint_path: str, **new_config):
    """Resume training from a checkpoint"""
    
    # Load trainer state
    trainer = Trainer.load_from_checkpoint(checkpoint_path)
    
    # Update configuration if provided
    if new_config:
        for key, value in new_config.items():
            if hasattr(trainer, key):
                setattr(trainer, key, value)
    
    # Continue training
    trainer.fit()
    return trainer