"""
Configuration loader for Flow Model Training
Based on x-flux repository configuration structure
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    model_name: str
    train_batch_size: int
    max_train_steps: int
    learning_rate: float
    lr_scheduler: str
    lr_warmup_steps: int
    gradient_accumulation_steps: int
    mixed_precision: str
    output_dir: str
    checkpointing_steps: int
    checkpoints_total_limit: int
    resume_from_checkpoint: str
    logging_dir: str
    report_to: str
    tracker_project_name: str


@dataclass
class DataConfig:
    """Data configuration dataclass"""
    train_batch_size: int
    num_workers: int
    img_size: int
    img_dir: str
    random_ratio: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer configuration dataclass"""
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: float


@dataclass
class FluxParams:
    """Flux model parameters dataclass"""
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class ConfigLoader:
    """Configuration loader for Flow Model Training"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_name: str) -> DictConfig:
        """
        Load configuration from YAML file
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            
        Returns:
            OmegaConf DictConfig object
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        config = OmegaConf.load(config_path)
        return config
    
    def load_model_config(self) -> DictConfig:
        """Load model configuration"""
        return self.load_config("model_config")
    
    def load_training_config(self, config_name: str = "config") -> DictConfig:
        """Load training configuration"""
        return self.load_config(config_name)
    
    def validate_config(self, config: DictConfig) -> bool:
        """
        Validate configuration parameters
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Validate model name
        valid_models = ["flux-dev", "flux-dev-fp8", "flux-schnell"]
        if config.model_name not in valid_models:
            raise ValueError(f"Invalid model_name: {config.model_name}. Must be one of {valid_models}")
        
        # Validate learning rate
        if config.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive: {config.learning_rate}")
        
        # Validate batch sizes
        if config.train_batch_size <= 0:
            raise ValueError(f"Train batch size must be positive: {config.train_batch_size}")
        
        if config.data_config.train_batch_size <= 0:
            raise ValueError(f"Data batch size must be positive: {config.data_config.train_batch_size}")
        
        # Validate image size
        if config.data_config.img_size <= 0:
            raise ValueError(f"Image size must be positive: {config.data_config.img_size}")
        
        # Validate mixed precision
        valid_precision = ["fp16", "bf16", "no"]
        if config.mixed_precision not in valid_precision:
            raise ValueError(f"Invalid mixed_precision: {config.mixed_precision}. Must be one of {valid_precision}")
        
        # Validate scheduler
        valid_schedulers = ["constant", "cosine", "linear"]
        if config.lr_scheduler not in valid_schedulers:
            raise ValueError(f"Invalid lr_scheduler: {config.lr_scheduler}. Must be one of {valid_schedulers}")
        
        return True
    
    def merge_configs(self, base_config: str, override_config: str) -> DictConfig:
        """
        Merge two configurations, with override_config taking precedence
        
        Args:
            base_config: Base configuration name
            override_config: Override configuration name
            
        Returns:
            Merged configuration
        """
        base = self.load_config(base_config)
        override = self.load_config(override_config)
        
        merged = OmegaConf.merge(base, override)
        return merged
    
    def save_config(self, config: DictConfig, output_path: str):
        """
        Save configuration to file
        
        Args:
            config: Configuration to save
            output_path: Output file path
        """
        OmegaConf.save(config, output_path)
    
    def get_model_params(self, model_name: str) -> FluxParams:
        """
        Get model parameters for a specific model variant
        
        Args:
            model_name: Name of the model variant
            
        Returns:
            FluxParams object
        """
        model_config = self.load_model_config()
        
        if model_name not in model_config.model_variants:
            raise ValueError(f"Model variant {model_name} not found in configuration")
        
        params = model_config.model_variants[model_name].params
        return FluxParams(**params)
    
    def create_training_config(self, 
                             model_name: str = "flux-dev",
                             config_type: str = "base") -> DictConfig:
        """
        Create a complete training configuration
        
        Args:
            model_name: Name of the model to use
            config_type: Type of configuration (base, lora, controlnet)
            
        Returns:
            Complete training configuration
        """
        # Load base config
        if config_type == "lora":
            config = self.load_config("lora_config")
        elif config_type == "controlnet":
            config = self.load_config("controlnet_config")
        else:
            config = self.load_config("config")
        
        # Override model name if specified
        config.model_name = model_name
        
        # Validate configuration
        self.validate_config(config)
        
        return config


def load_config(config_name: str, config_dir: str = "configs") -> DictConfig:
    """
    Convenience function to load configuration
    
    Args:
        config_name: Name of the config file
        config_dir: Directory containing config files
        
    Returns:
        Configuration object
    """
    loader = ConfigLoader(config_dir)
    return loader.load_config(config_name)


def validate_and_load_config(config_name: str, config_dir: str = "configs") -> DictConfig:
    """
    Load and validate configuration
    
    Args:
        config_name: Name of the config file
        config_dir: Directory containing config files
        
    Returns:
        Validated configuration object
    """
    loader = ConfigLoader(config_dir)
    config = loader.load_config(config_name)
    loader.validate_config(config)
    return config


if __name__ == "__main__":
    # Example usage
    loader = ConfigLoader()
    
    # Load base configuration
    config = loader.load_config("config")
    print("Base config loaded successfully")
    
    # Load model configuration
    model_config = loader.load_model_config()
    print("Model config loaded successfully")
    
    # Validate configuration
    try:
        loader.validate_config(config)
        print("Configuration validation passed")
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
    
    # Get model parameters
    try:
        params = loader.get_model_params("flux-dev")
        print(f"Model parameters loaded: hidden_size={params.hidden_size}")
    except ValueError as e:
        print(f"Failed to load model parameters: {e}") 