import os

# Set NCCL environment variables before importing PyTorch
os.environ['NCCL_NVLS_ENABLE'] = '0'          # Disable NVLS transport
os.environ['NCCL_TREE_THRESHOLD'] = '0'       # Force ring algorithms
os.environ['NCCL_NET_GDR_LEVEL'] = '0'        # Disable GPU Direct RDMA
os.environ['NCCL_P2P_LEVEL'] = 'SYS'          # Enable system-level P2P
os.environ['NCCL_SHM_DISABLE'] = '0'          # Ensure shared memory enabled
os.environ['NCCL_ALGO'] = 'Ring'              # Force ring algorithm
os.environ['NCCL_TIMEOUT'] = '1800'           # 30 minute timeout
os.environ['NCCL_DEBUG'] = 'WARN'             # Reduce debug output

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
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer.algorithms import GradientClipping
from composer.loggers import WandBLogger
from composer.utils import dist, reproducibility
from streaming import Stream, StreamingDataset

import os
import time

# Your FLUX model import
from model_utils.dit import load_flow_model2

torch.backends.cudnn.benchmark = True  # 3-5% speedup


class FluxStreamingDataset(StreamingDataset):
    """Dataset class for loading FLUX precomputed data from mds format."""

    def __init__(
        self,
        streams: Optional[List[Stream]] = None,
        shuffle: bool = False,
        batch_size: int = 8,
        **kwargs
    ) -> None:
        # Remove batch_size parameter to avoid distributed issues
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        out = {}

        # Convert numpy arrays to tensors with correct shapes
        # Expected shapes from dataset preparation:
        # - img_latents: [256, 64] (256 tokens, 64 features each)
        # - txt_embeds: [512, 4096] (512 tokens, 4096 features each)
        # - vec_embeds: [768] (CLIP embedding)
        
        if 'img_latents' in sample:
            # Reshape from flattened to [256, 64]
            img_data = sample['img_latents'].astype(np.float16)
            out['img_latents'] = torch.from_numpy(img_data.reshape(256, 64))
        
        if 'img_ids' in sample:
            # Reshape from flattened to [256, 3]
            img_ids_data = sample['img_ids'].astype(np.float32)
            out['img_ids'] = torch.from_numpy(img_ids_data.reshape(256, 3))
            
        if 'txt_embeds' in sample:
            # Reshape from flattened to [512, 4096]
            txt_data = sample['txt_embeds'].astype(np.float16)
            out['txt_embeds'] = torch.from_numpy(txt_data.reshape(512, 4096))
            
        if 'txt_ids' in sample:
            # Reshape from flattened to [512, 3]
            txt_ids_data = sample['txt_ids'].astype(np.float32)
            out['txt_ids'] = torch.from_numpy(txt_ids_data.reshape(512, 3))
            
        if 'vec_embeds' in sample:
            # Keep as [768] - no reshaping needed
            vec_data = sample['vec_embeds'].astype(np.float16)
            out['vec_embeds'] = torch.from_numpy(vec_data)

        return out


class FluxComposerModel(ComposerModel):
    """Composer wrapper for FLUX flow matching model"""
    
    def __init__(self, model_name: str = "flux-schnell"):
        super().__init__()
        
        # Load the FLUX model
        print(f"Loading FLUX model: {model_name}")
        self.flux_model = load_flow_model2(model_name)
        self.model_name = model_name
        print("FLUX model loaded successfully")
        
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
        
        # Ensure tensors have the correct 3D shape for FLUX model
        # FLUX expects: img [B, seq_len, dim] and txt [B, seq_len, dim]
        
        # Debug: Print tensor shapes before reshaping
        # print(f"Original x_t shape: {x_t.shape}")
        # print(f"Original txt_embeds shape: {txt_embeds.shape}")
        
        # The tensors should already be in the correct 3D format from dataset preparation
        # Expected shapes:
        # - x_t: [B, 256, 64] (256 tokens, 64 features each)
        # - txt_embeds: [B, 512, 4096] (512 tokens, 4096 features each)
        
        # Debug: Print tensor shapes after reshaping
        # print(f"Final x_t shape: {x_t.shape}")
        # print(f"Final txt_embeds shape: {txt_embeds.shape}")
        
        # Validate expected dimensions
        if x_t.shape[1] != 256 or x_t.shape[2] != 64:
            print(f"ERROR: Expected x_t shape [B, 256, 64], got {x_t.shape}")
            raise ValueError(f"Invalid x_t shape: {x_t.shape}")
        
        if txt_embeds.shape[1] != 512 or txt_embeds.shape[2] != 4096:
            print(f"ERROR: Expected txt_embeds shape [B, 512, 4096], got {txt_embeds.shape}")
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
        
        return loss
    
    def metrics(self, train: bool = False) -> Dict[str, Any]:
        """Define metrics to track"""
        return {}


def build_flux_streaming_dataloader(
    datadir: Union[str, List[str]],
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    **dataloader_kwargs
) -> DataLoader:
    """Creates a DataLoader for FLUX streaming dataset - exact copy of working pattern."""
    
    if isinstance(datadir, str):
        datadir = [datadir]

    streams = [Stream(remote=None, local=d) for d in datadir]

    dataset = FluxStreamingDataset(
        streams=streams,
        shuffle=shuffle,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,  # Important: let dataset handle everything
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader


def train():
    """Train FLUX model - following exact working pattern"""
    
    # Configuration
    cfg = {
        'seed': 42,
        'dataset': {
            'train_batch_size': 8,
            'eval_batch_size': 8,
            'datadir': "/data0/teja_works/diffusion_training/nvidia_tools_training/mosicml_code/FlowModelTraining/data_gen/flux_mds_dataset",
            'num_workers': 16,
        },
        'model': {
            'name': "flux-schnell",
            'dtype': 'bfloat16'
        },
        'optimizer': {
            'lr': 1e-5,
            'weight_decay': 0.01,
            'betas': [0.9, 0.999]
        },
        'scheduler': {
            'warmup_duration': "100ba",
            'max_duration': "50ep",
        },
        'trainer': {
            'max_duration': "50ep",
            'save_interval': "10ep",
            'eval_interval': "5ep",
            'save_folder': "./flux_composer_checkpoints",
        },
        'algorithms': {
            'gradient_clipping': {'clip_norm': 1.0}
        },
        'misc': {
            'compile': False
        }
    }
    
    if not cfg:
        raise ValueError('Config not specified.')
    
    reproducibility.seed_all(cfg['seed'])

    # Create model
    print("Initializing FLUX Composer model...")
    model = FluxComposerModel(model_name=cfg['model']['name'])

    # Set up optimizer - exact pattern from working code
    optimizer = DecoupledAdamW(
        params=model.parameters(), 
        lr=cfg['optimizer']['lr'],
        weight_decay=cfg['optimizer']['weight_decay'],
        betas=cfg['optimizer']['betas'],
        eps=1e-8
    )

    # Convert ListConfig betas to native list to avoid ValueError when saving optimizer state
    for p in optimizer.param_groups:
        p['betas'] = list(p['betas'])

    # Set up data loaders - EXACTLY like working code
    print("Creating training dataloader...")
    train_loader = build_flux_streaming_dataloader(
        datadir=cfg['dataset']['datadir'],
        batch_size=cfg['dataset']['train_batch_size'] // dist.get_world_size(),
        shuffle=True,
        drop_last=True,
        num_workers=cfg['dataset']['num_workers'],
        persistent_workers=True if cfg['dataset']['num_workers'] > 0 else False,
        pin_memory=True
    )
    print(f"Found {len(train_loader.dataset)*dist.get_world_size()} samples in the training dataset")
    time.sleep(3)

    eval_loader = build_flux_streaming_dataloader(
        datadir=cfg['dataset']['datadir'],
        batch_size=cfg['dataset']['eval_batch_size'] // dist.get_world_size(),
        shuffle=False,
        drop_last=True,
        num_workers=cfg['dataset']['num_workers'],
        persistent_workers=True if cfg['dataset']['num_workers'] > 0 else False,
        pin_memory=True
    )
    print(f"Found {len(eval_loader.dataset)*dist.get_world_size()} samples in the eval dataset")
    time.sleep(3)

    # Initialize training components
    logger, callbacks, algorithms = [], [], []

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
        alpha_f=0.1
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
        device="gpu" if torch.cuda.is_available() else "cpu",
        seed=cfg['seed'],
    )

    # Ensure models are on correct device
    device = next(model.flux_model.parameters()).device
    print(f"Training on device: {device}")

    return trainer.fit()


if __name__ == '__main__':
    train()