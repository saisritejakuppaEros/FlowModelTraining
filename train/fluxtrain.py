import os
import yaml
import gc
from pathlib import Path

# Load configuration from YAML file
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# Load configuration
cfg = load_config()

# Set NCCL environment variables from config before importing PyTorch
nccl_config = cfg.get('nccl', {})
for key, value in nccl_config.items():
    os.environ[key] = str(value)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Any, Dict, Optional, Tuple, List, Union
from einops import rearrange

# MosaicML Composer imports
import composer
from composer import Trainer
from composer.core import Precision
from composer.models import ComposerModel
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer.algorithms import GradientClipping
from composer.loggers import WandBLogger
from composer.utils import dist, reproducibility
from streaming import Stream, StreamingDataset
from composer.loggers import TensorboardLogger
from dataloading_ops import build_flux_streaming_dataloader



import os
import time

# Your FLUX model import
from model_utils.dit import load_flow_model2

# Import inference utilities
from inference_utils import create_inference_callback
from memory_utils import print_gpu_memory_usage, cleanup_gpu_memory, MemoryManager

torch.backends.cudnn.benchmark = True  # 3-5% speedup

# Set memory management environment variables
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class FluxComposerModel(ComposerModel):
    """Composer wrapper for FLUX flow matching model"""
    
    def __init__(self, model_name: str = None):
        super().__init__()
        
        # Load the FLUX model
        print(f"Loading FLUX model: {model_name}")
        
        self.flux_model = load_flow_model2(model_name)
        self.model_name = model_name
        print("FLUX model loaded successfully")
        
        
        # Flow matching parameters from config
        self.sigma = cfg['flow_matching']['sigma']  # Noise scale
        
        # Initialize metrics tracking
        self.train_loss = torch.tensor(0.0)
        self.eval_loss = torch.tensor(0.0)
        
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
        
        # Guidance scale from config
        guidance = torch.full((batch_size,), cfg['flow_matching']['guidance_scale'], device=device, dtype=img_latents.dtype)
        
        # Validate expected dimensions from config
        expected_img_shape = cfg['data_preprocessing']['img_latents_shape']
        expected_txt_shape = cfg['data_preprocessing']['txt_embeds_shape']
        
        if x_t.shape[1] != expected_img_shape[0] or x_t.shape[2] != expected_img_shape[1]:
            print(f"ERROR: Expected x_t shape [B, {expected_img_shape[0]}, {expected_img_shape[1]}], got {x_t.shape}")
            raise ValueError(f"Invalid x_t shape: {x_t.shape}")
        
        if txt_embeds.shape[1] != expected_txt_shape[0] or txt_embeds.shape[2] != expected_txt_shape[1]:
            print(f"ERROR: Expected txt_embeds shape [B, {expected_txt_shape[0]}, {expected_txt_shape[1]}], got {txt_embeds.shape}")
            raise ValueError(f"Invalid txt_embeds shape: {txt_embeds.shape}")
        
        
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
        
        # Clean up intermediate tensors to prevent memory leaks
        del predicted_velocity, target_velocity, timesteps
        
        return loss
    
    def metrics(self, train: bool = False) -> Dict[str, Any]:
        """Define metrics to track - Composer will automatically log these to TensorBoard"""
        if train:
            return {
                'train/loss': self.train_loss,
            }
        else:
            return {
                'eval/loss': self.eval_loss,
            }
    
    def update_metric(self, batch: Dict[str, Any], outputs: Any, metric_name: str, metric_value: torch.Tensor) -> None:
        """Update metrics during training/evaluation"""
        if metric_name == 'loss':
            if hasattr(self, 'train_loss'):
                self.train_loss = metric_value.detach().cpu()
            if hasattr(self, 'eval_loss'):
                self.eval_loss = metric_value.detach().cpu()
                
        # Clean up GPU memory periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()




def train():
    """Train FLUX model using configuration from YAML file"""
    
    # Configuration is already loaded from YAML file at module level
    if not cfg:
        raise ValueError('Config not specified.')
    
    reproducibility.seed_all(cfg['seed'])

    # Create model
    print("Initializing FLUX Composer model...")

    model = FluxComposerModel(model_name=cfg['model']['name'])

    # Set up optimizer - using micro_diffusion pattern
    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        lr=float(cfg['optimizer']['lr']),
        weight_decay=float(cfg['optimizer']['weight_decay']),
        betas=cfg['optimizer']['betas'],
        eps=float(cfg['optimizer']['eps'])
    )

    # Convert ListConfig betas to native list to avoid ValueError when saving optimizer state
    for p in optimizer.param_groups:
        p['betas'] = list(p['betas'])

    # Set up data loaders with separate paths
    print("Dataset configuration:")
    print(f"  - Train data dir: {cfg['dataset']['train_datadir']}")
    print(f"  - Eval data dir: {cfg['dataset']['eval_datadir']}")
    print(f"  - Test data dir: {cfg['dataset']['test_datadir']}")
    print(f"  - Eval dataset: Full dataset (no limitation)")
    print(f"  - Create test loader: {cfg['dataset'].get('create_test_loader', False)}")
    print()
    
    print("Creating training dataloader...")
    
    train_loader = build_flux_streaming_dataloader(
        datadir=cfg['dataset']['train_datadir'],
        batch_size=cfg['dataset']['train_batch_size'] // dist.get_world_size(),
        shuffle=True,
        drop_last=True,
        num_workers=cfg['dataset']['num_workers'],
        persistent_workers=cfg['hardware']['persistent_workers'] if cfg['dataset']['num_workers'] > 0 else False,
        pin_memory=cfg['hardware']['pin_memory']
    )
    print(f"Found {len(train_loader.dataset)*dist.get_world_size()} samples in the training dataset")
    
    
    time.sleep(3)

    print("Creating evaluation dataloader...")
    eval_loader = build_flux_streaming_dataloader(
        datadir=cfg['dataset']['eval_datadir'],
        batch_size=cfg['dataset']['eval_batch_size'] // dist.get_world_size(),
        shuffle=False,
        drop_last=True,
        num_workers=cfg['dataset']['num_workers'],
        persistent_workers=True if cfg['dataset']['num_workers'] > 0 else False,
        pin_memory=True
    )
    
    # Note: Dataset limitation removed to avoid DataLoader issues
    print(f"Eval dataset size: {len(eval_loader.dataset)} samples")
    
    print(f"Found {len(eval_loader.dataset)*dist.get_world_size()} samples in the eval dataset")
    time.sleep(3)

    # Optional: Create test dataloader (for future use)
    if cfg['dataset'].get('create_test_loader', False) and 'test_datadir' in cfg['dataset'] and cfg['dataset']['test_datadir']:
        print("Creating test dataloader...")
        test_loader = build_flux_streaming_dataloader(
            datadir=cfg['dataset']['test_datadir'],
            batch_size=cfg['dataset']['eval_batch_size'] // dist.get_world_size(),
            shuffle=False,
            drop_last=True,
            num_workers=cfg['dataset']['num_workers'],
            persistent_workers=True if cfg['dataset']['num_workers'] > 0 else False,
            pin_memory=True
        )
        print(f"Found {len(test_loader.dataset)*dist.get_world_size()} samples in the test dataset")
        time.sleep(3)
    else:
        test_loader = None
        print("Test dataloader creation disabled or no test_datadir specified")

    # Initialize TensorBoard logger
    tb_logger = TensorboardLogger(log_dir=cfg['logging']['tensorboard']['log_dir'])
    
    # Initialize training components
    logger, callbacks, algorithms = [tb_logger], [], []
    
    # Add inference callback if enabled
    if cfg['inference']['enabled']:
        print("Adding inference callback for monitoring training progress...")
        print(f"Inference configuration:")
        print(f"  - Interval: {cfg['inference']['interval']}")
        print(f"  - Steps: {cfg['inference']['num_steps']}")
        print(f"  - Guidance: {cfg['inference']['guidance_scale']}")
        print(f"  - Save dir: {cfg['inference']['save_dir']}")
        print(f"  - Using validation datastreamer for inference")
        print("  - Note: Inference will be memory-intensive, consider increasing interval if OOM occurs")
        inference_callback = create_inference_callback(cfg, eval_loader)
        callbacks.append(inference_callback)

    # Configure algorithms
    if 'algorithms' in cfg:
        for alg_name, alg_conf in cfg['algorithms'].items():
            if alg_name == 'gradient_clipping':
                algorithms.append(GradientClipping(clipping_type='norm', clipping_threshold=alg_conf['clip_norm']))
            else:
                print(f'Algorithm {alg_name} not supported.')

    scheduler = CosineAnnealingWithWarmupScheduler(
        t_warmup=cfg['scheduler']['warmup_duration'],
        t_max=cfg['scheduler']['max_duration'],
        alpha_f=cfg['scheduler']['alpha_f']
    )

    # disable online evals if using torch.compile
    if cfg['misc']['compile']:
        cfg['trainer']['eval_interval'] = 0
        
    trainer = Trainer(
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        max_duration=cfg['trainer']['max_duration'],
        optimizers=optimizer,
        model=model,
        loggers=logger,
        algorithms=algorithms,
        schedulers=scheduler,
        callbacks=callbacks,
        precision='amp_bf16' if cfg['model']['dtype'] == 'bfloat16' else 'amp_fp16',
        python_log_level='debug',
        compile_config={} if cfg['misc']['compile'] else None,
        save_folder=cfg['trainer']['save_folder'],
        save_interval=cfg['trainer']['save_interval'],
        eval_interval=cfg['trainer']['eval_interval'],
        run_name=cfg['trainer']['run_name'],
        autoresume=cfg['trainer']['autoresume'],
        device="gpu" if torch.cuda.is_available() else "cpu",
        seed=cfg['seed'],
        save_overwrite=True
    )

    # Ensure models are on correct device
    device = next(model.flux_model.parameters()).device
    print(f"Training on device: {device}")
    
    # Print initial memory state
    print("\n🔍 Initial GPU Memory State:")
    print_gpu_memory_usage("Before training:")
    
    # Print inference information
    if cfg['inference']['enabled']:
        print(f"\nInference samples will be saved to: {cfg['inference']['save_dir']}")
        print("Inference will run every epoch to monitor training progress")
        print("⚠️  Note: If you encounter OOM errors, consider:")
        print("   - Increasing inference interval (e.g., '2ep' or '3ep')")
        print("   - Disabling inference during training")
        print("   - Reducing batch size")

    
    # Clean up memory before training starts
    cleanup_gpu_memory(verbose=True)
    
    try:
        result = trainer.fit()
        print("\n✅ Training completed successfully!")
        return result
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ CUDA Out of Memory Error: {e}")
        print("🔧 Troubleshooting suggestions:")
        print("1. Reduce batch size in config.yaml")
        print("2. Disable or reduce inference frequency")
        print("3. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("4. Use gradient checkpointing if available")
        cleanup_gpu_memory(verbose=True)
        raise
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        cleanup_gpu_memory(verbose=True)
        raise


if __name__ == '__main__':
    print("Starting FLUX training with TensorBoard logging...")
    print("To view training logs, run: tensorboard --logdir=./my_tensorboard_logs")
    print("Then open http://localhost:6006 in your browser")
    train()