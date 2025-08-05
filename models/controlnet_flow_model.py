"""
ControlNet Flow Model for PyTorch Lightning Training
Based on x-flux repository ControlNet implementation
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import math
import sys
from pathlib import Path

# Add x-flux to path for imports
sys.path.append(str(Path(__file__).parent.parent / "x-flux" / "src"))

from flux.model import Flux, FluxParams
from flux.controlnet import ControlNetFlux
from flux.util import load_flow_model2, load_ae, load_clip, load_t5
from flux.sampling import denoise_controlnet, get_noise, get_schedule, prepare, unpack
from base_model import BaseFlowModel


class ControlNetFlowModel(BaseFlowModel):
    """
    ControlNet Flow-based model for PyTorch Lightning training
    Based on x-flux repository ControlNet implementation
    """
    
    def __init__(
        self,
        model_name: str = "flux-dev",
        learning_rate: float = 2e-5,  # Higher learning rate for ControlNet
        weight_decay: float = 0.01,
        warmup_steps: int = 10,
        max_steps: int = 100000,
        gradient_clip_val: float = 1.0,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        snr_gamma: float = 5.0,
        guidance_scale: float = 7.5,
        controlnet_guidance_scale: float = 0.7,
        sample_every_n_steps: int = 1000,
        sample_prompts: List[str] = None,
        controlnet_depth: int = 2,
        **kwargs
    ):
        """
        Initialize ControlNet Flow Model
        
        Args:
            model_name: Name of the flux model to use
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps
            max_steps: Maximum number of training steps
            gradient_clip_val: Gradient clipping value
            use_ema: Whether to use exponential moving average
            ema_decay: EMA decay rate
            snr_gamma: SNR weighting for loss calculation
            guidance_scale: Guidance scale for sampling
            controlnet_guidance_scale: ControlNet guidance scale
            sample_every_n_steps: How often to generate samples
            sample_prompts: Prompts to use for sampling
            controlnet_depth: Depth of ControlNet blocks
        """
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            gradient_clip_val=gradient_clip_val,
            use_ema=use_ema,
            ema_decay=ema_decay,
            **kwargs
        )
        
        self.model_name = model_name
        self.snr_gamma = snr_gamma
        self.guidance_scale = guidance_scale
        self.controlnet_guidance_scale = controlnet_guidance_scale
        self.sample_every_n_steps = sample_every_n_steps
        self.controlnet_depth = controlnet_depth
        
        # Default sample prompts
        if sample_prompts is None:
            sample_prompts = [
                "a beautiful landscape with mountains and trees",
                "a portrait of a person in a garden"
            ]
        self.sample_prompts = sample_prompts
        
        # Initialize models (will be loaded in setup)
        self.flow_model = None
        self.controlnet = None
        self.vae = None
        self.t5 = None
        self.clip = None
        
        # Training state
        self.is_setup = False
    
    def setup(self, stage: str = None):
        """
        Setup models and components
        """
        if self.is_setup:
            return
        
        print(f"Setting up ControlNet Flow Model: {self.model_name}")
        
        # Load models
        device = self.device
        
        # Load flow model
        self.flow_model = load_flow_model2(self.model_name, device="cpu")
        self.flow_model.to(device)
        
        # Load ControlNet
        self.controlnet = self.create_controlnet()
        self.controlnet.to(device)
        
        # Load VAE
        self.vae = load_ae(self.model_name, device="cpu")
        self.vae.to(device)
        
        # Load text encoders
        is_schnell = self.model_name == "flux-schnell"
        self.t5 = load_t5(device, max_length=256 if is_schnell else 512)
        self.clip = load_clip(device)
        
        # Freeze text encoders and VAE
        self.t5.requires_grad_(False)
        self.clip.requires_grad_(False)
        self.vae.requires_grad_(False)
        
        self.is_setup = True
        print("ControlNet Flow Model setup complete")
    
    def create_controlnet(self):
        """
        Create ControlNet model
        """
        # Get model parameters from the main flow model
        if self.flow_model is None:
            # Load temporarily to get params
            temp_model = load_flow_model2(self.model_name, device="cpu")
            params = temp_model.params
            del temp_model
        else:
            params = self.flow_model.params
        
        # Create ControlNet with same parameters
        controlnet = ControlNetFlux(params, controlnet_depth=self.controlnet_depth)
        return controlnet
    
    def forward(self, batch):
        """
        Forward pass through the ControlNet flow model
        """
        if not self.is_setup:
            self.setup()
        
        images, control_signals, captions = batch
        
        # Prepare inputs
        inputs = prepare(self.t5, self.clip, images, captions)
        
        # Get timesteps
        batch_size = images.shape[0]
        timesteps = torch.rand(batch_size, device=self.device)
        
        # Add noise to images
        noise = torch.randn_like(images)
        noisy_images = images + noise * timesteps.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Forward pass through flow model with ControlNet
        predicted_noise = self.flow_model(
            img=noisy_images,
            img_ids=inputs["img_ids"],
            txt=inputs["txt"],
            txt_ids=inputs["txt_ids"],
            timesteps=timesteps,
            y=noise,
            block_controlnet_hidden_states=self.controlnet(control_signals)
        )
        
        return predicted_noise, noise, timesteps
    
    def compute_loss(self, predicted_noise, target_noise, timesteps):
        """
        Compute loss with SNR weighting
        """
        # Compute SNR
        snr = (1 - timesteps) / timesteps
        
        # SNR weighting
        snr_weight = torch.where(snr > 0, snr ** self.snr_gamma, torch.ones_like(snr))
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, target_noise, reduction='none')
        loss = loss.mean(dim=[1, 2, 3])  # Mean over spatial dimensions
        
        # Apply SNR weighting
        weighted_loss = loss * snr_weight
        
        return weighted_loss.mean()
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        if not self.is_setup:
            self.setup()
        
        # Forward pass
        predicted_noise, target_noise, timesteps = self.forward(batch)
        
        # Compute loss
        loss = self.compute_loss(predicted_noise, target_noise, timesteps)
        
        # Log losses
        self.log_losses({
            "loss": loss,
            "timestep_mean": timesteps.mean(),
            "timestep_std": timesteps.std()
        }, "train")
        
        # Generate samples periodically
        if self.global_step % self.sample_every_n_steps == 0:
            self.generate_and_log_samples()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        if not self.is_setup:
            self.setup()
        
        # Forward pass
        predicted_noise, target_noise, timesteps = self.forward(batch)
        
        # Compute loss
        loss = self.compute_loss(predicted_noise, target_noise, timesteps)
        
        # Log losses
        self.log_losses({
            "loss": loss,
            "timestep_mean": timesteps.mean(),
            "timestep_std": timesteps.std()
        }, "val")
        
        # Log sample images from validation
        if batch_idx == 0:
            images, control_signals, captions = batch
            self.log_images(images, "val")
            self.log_images(control_signals, "val_control")
        
        return loss
    
    def generate_and_log_samples(self):
        """
        Generate samples and log them
        """
        if not self.is_setup:
            return
        
        try:
            # Generate samples
            samples = self.generate_samples(self.sample_prompts)
            
            # Log samples
            if samples is not None:
                self.log_images(samples, "samples")
                
        except Exception as e:
            print(f"Error generating samples: {e}")
    
    def generate_samples(self, prompts: List[str], num_steps: int = 20, height: int = 512, width: int = 512):
        """
        Generate samples from prompts with ControlNet
        
        Args:
            prompts: List of text prompts
            num_steps: Number of denoising steps
            height: Image height
            width: Image width
            
        Returns:
            Generated images tensor
        """
        if not self.is_setup:
            return None
        
        # Get sampling schedule
        timesteps = get_schedule(num_steps, height * width)
        
        # Generate noise
        noise = get_noise(
            num_samples=len(prompts),
            height=height,
            width=width,
            device=self.device,
            dtype=self.flow_model.dtype,
            seed=42
        )
        
        # Generate control signals (for demo, use random control signals)
        control_signals = torch.randn_like(noise)
        
        # Prepare inputs
        inputs = prepare(self.t5, self.clip, noise, prompts)
        
        # Denoise with ControlNet
        with torch.no_grad():
            denoised = denoise_controlnet(
                model=self.flow_model,
                controlnet=self.controlnet,
                img=noise,
                img_ids=inputs["img_ids"],
                txt=inputs["txt"],
                txt_ids=inputs["txt_ids"],
                vec=inputs["vec"],
                neg_txt=inputs["txt"],  # Use same text for negative
                neg_txt_ids=inputs["txt_ids"],
                neg_vec=inputs["vec"],
                controlnet_cond=control_signals,
                timesteps=timesteps,
                guidance=self.guidance_scale,
                controlnet_gs=self.controlnet_guidance_scale
            )
        
        # Unpack to image format
        images = unpack(denoised, height, width)
        
        # Decode with VAE
        with torch.no_grad():
            latents = self.vae.encode(images)
            decoded_images = self.vae.decode(latents)
        
        return decoded_images
    
    def on_save_checkpoint(self, checkpoint):
        """
        Save additional information to checkpoint
        """
        super().on_save_checkpoint(checkpoint)
        
        # Save model states
        if self.flow_model is not None:
            checkpoint['flow_model'] = self.flow_model.state_dict()
        if self.controlnet is not None:
            checkpoint['controlnet'] = self.controlnet.state_dict()
        if self.vae is not None:
            checkpoint['vae'] = self.vae.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        """
        Load additional information from checkpoint
        """
        super().on_load_checkpoint(checkpoint)
        
        # Load model states
        if 'flow_model' in checkpoint and self.flow_model is not None:
            self.flow_model.load_state_dict(checkpoint['flow_model'])
        if 'controlnet' in checkpoint and self.controlnet is not None:
            self.controlnet.load_state_dict(checkpoint['controlnet'])
        if 'vae' in checkpoint and self.vae is not None:
            self.vae.load_state_dict(checkpoint['vae'])
    
    def get_param_groups(self) -> list:
        """
        Get parameter groups for optimizer
        Optimize both flow model and ControlNet parameters
        """
        params = []
        
        # Flow model parameters
        if self.flow_model is not None:
            params.append({
                "params": [p for n, p in self.flow_model.named_parameters() if p.requires_grad],
                "lr": self.learning_rate
            })
        
        # ControlNet parameters
        if self.controlnet is not None:
            params.append({
                "params": [p for n, p in self.controlnet.named_parameters() if p.requires_grad],
                "lr": self.learning_rate
            })
        
        return params
    
    def on_fit_start(self):
        """
        Called when fit begins
        """
        # Setup models if not already done
        if not self.is_setup:
            self.setup()
        
        # Call parent method
        super().on_fit_start()
        
        # Log model info
        total_params = 0
        trainable_params = 0
        
        if self.flow_model is not None:
            flow_params = sum(p.numel() for p in self.flow_model.parameters())
            flow_trainable = sum(p.numel() for p in self.flow_model.parameters() if p.requires_grad)
            total_params += flow_params
            trainable_params += flow_trainable
            print(f"Flow Model parameters: {flow_params:,} total, {flow_trainable:,} trainable")
        
        if self.controlnet is not None:
            controlnet_params = sum(p.numel() for p in self.controlnet.parameters())
            controlnet_trainable = sum(p.numel() for p in self.controlnet.parameters() if p.requires_grad)
            total_params += controlnet_params
            trainable_params += controlnet_trainable
            print(f"ControlNet parameters: {controlnet_params:,} total, {controlnet_trainable:,} trainable")
        
        print(f"ControlNet Flow Model: {self.model_name}")
        print(f"Total parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def predict_step(self, batch, batch_idx):
        """
        Prediction step
        """
        if not self.is_setup:
            self.setup()
        
        # Generate samples from prompts in batch
        images, control_signals, captions = batch
        
        # Generate samples
        samples = self.generate_samples(captions)
        
        return {
            'prompts': captions,
            'control_signals': control_signals,
            'samples': samples
        } 