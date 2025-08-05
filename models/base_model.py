"""
Base model class for Flow Model Training
Handles common operations for PyTorch Lightning models
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any, Optional, Tuple
import math


class BaseFlowModel(pl.LightningModule):
    """
    Base class for flow-based models using PyTorch Lightning
    Handles common operations like optimizer setup, training loop, and logging
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        gradient_clip_val: float = 1.0,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        **kwargs
    ):
        """
        Initialize base model
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            max_steps: Maximum number of training steps
            gradient_clip_val: Gradient clipping value
            use_ema: Whether to use exponential moving average
            ema_decay: EMA decay rate
        """
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_clip_val = gradient_clip_val
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['model'])
        
        # Initialize EMA if enabled
        if self.use_ema:
            self.ema_model = None
            self.ema_decay = ema_decay
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler
        """
        # Get all parameters that require gradients
        param_groups = self.get_param_groups()
        
        # Create optimizer
        optimizer = AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Create learning rate scheduler
        scheduler = self.create_lr_scheduler(optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def get_param_groups(self) -> list:
        """
        Get parameter groups for optimizer
        Can be overridden by subclasses to customize parameter grouping
        """
        return [
            {
                "params": [p for n, p in self.named_parameters() if p.requires_grad],
                "lr": self.learning_rate
            }
        ]
    
    def create_lr_scheduler(self, optimizer) -> LambdaLR:
        """
        Create learning rate scheduler with warmup
        """
        def lr_lambda(step):
            # Warmup phase
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            
            # Cosine decay phase
            progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    def training_step(self, batch, batch_idx):
        """
        Training step - must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement training_step")
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step - must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement validation_step")
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Called at the end of each training batch
        Update EMA if enabled
        """
        if self.use_ema and self.ema_model is not None:
            self.update_ema()
    
    def update_ema(self):
        """
        Update exponential moving average of model parameters
        """
        if self.ema_model is None:
            # Initialize EMA model
            self.ema_model = type(self.model)(*self.model.args, **self.model.kwargs)
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_model.to(self.device)
            self.ema_model.eval()
        
        # Update EMA parameters
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def get_ema_model(self):
        """
        Get the EMA model if available
        """
        return self.ema_model if self.use_ema else self.model
    
    def on_save_checkpoint(self, checkpoint):
        """
        Save additional information to checkpoint
        """
        if self.use_ema and self.ema_model is not None:
            checkpoint['ema_model'] = self.ema_model.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        """
        Load additional information from checkpoint
        """
        if self.use_ema and 'ema_model' in checkpoint:
            if self.ema_model is None:
                self.ema_model = type(self.model)(*self.model.args, **self.model.kwargs)
                self.ema_model.to(self.device)
            self.ema_model.load_state_dict(checkpoint['ema_model'])
    
    def log_losses(self, losses: Dict[str, torch.Tensor], step_type: str = "train"):
        """
        Log losses to tensorboard/wandb
        
        Args:
            losses: Dictionary of loss values
            step_type: Type of step ('train', 'val', 'test')
        """
        for name, value in losses.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.log(f"{step_type}/{name}", value, prog_bar=True, sync_dist=True)
    
    def log_images(self, images: torch.Tensor, step_type: str = "train", max_images: int = 8):
        """
        Log images to tensorboard/wandb
        
        Args:
            images: Tensor of images [B, C, H, W]
            step_type: Type of step ('train', 'val', 'test')
            max_images: Maximum number of images to log
        """
        if images.dim() == 4:
            # Limit number of images
            images = images[:max_images]
            
            # Convert from [-1, 1] to [0, 1] range for logging
            if images.min() < 0:
                images = (images + 1) / 2
            
            # Log images
            self.logger.experiment.add_images(
                f"{step_type}/images",
                images,
                self.global_step,
                dataformats='NCHW'
            )
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        """
        Configure gradient clipping
        """
        if gradient_clip_val is None:
            gradient_clip_val = self.gradient_clip_val
        
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm or "norm"
        )
    
    def get_learning_rate(self):
        """
        Get current learning rate
        """
        optimizer = self.optimizers()
        if optimizer is not None:
            return optimizer.param_groups[0]['lr']
        return self.learning_rate
    
    def on_fit_start(self):
        """
        Called when fit begins
        """
        # Log model parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.log("model/total_params", total_params)
        self.log("model/trainable_params", trainable_params)
        
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def on_train_epoch_start(self):
        """
        Called at the beginning of each training epoch
        """
        # Log learning rate
        lr = self.get_learning_rate()
        self.log("train/learning_rate", lr, prog_bar=True)
    
    def on_validation_epoch_start(self):
        """
        Called at the beginning of each validation epoch
        """
        pass
    
    def on_test_epoch_start(self):
        """
        Called at the beginning of each test epoch
        """
        pass 