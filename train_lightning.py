#!/usr/bin/env python3
"""
Simple training script for Qwen Image with PyTorch Lightning
Usage examples:

# Basic training
python train_lightning.py --config_path config.yaml --batch_size 4 --max_epochs 10

# With DeepSpeed
python train_lightning.py --config_path config.yaml --use_deepspeed --batch_size 8

# With W&B logging
python train_lightning.py --config_path config.yaml --use_wandb --wandb_project my-project

# Multi-GPU training
python train_lightning.py --config_path config.yaml --batch_size 2 --max_epochs 20
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy, FSDPStrategy
import argparse

from lightning_train import QwenImageLightningModule
from lightning_dataloading import QwenImageDataModule


def get_strategy(args):
    """Get the training strategy based on arguments"""
    if args.use_deepspeed:
        # Use provided config or default DeepSpeed config
        if args.deepspeed_config and os.path.exists(args.deepspeed_config):
            strategy = DeepSpeedStrategy(config=args.deepspeed_config)
        else:
            # Default DeepSpeed configuration
            deepspeed_config = {
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "reduce_bucket_size": 2e8,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                },
                                 "gradient_clipping": args.max_grad_norm,
                 "wall_clock_breakdown": False,
            }
            strategy = DeepSpeedStrategy(config=deepspeed_config)
        print("Using DeepSpeed strategy")
    elif torch.cuda.device_count() > 1:
        # strategy = DDPStrategy(find_unused_parameters=True)
        
        # use fsdp strategy
        strategy = FSDPStrategy()
        print(f"Using DDP strategy with {torch.cuda.device_count()} GPUs")
    else:
        strategy = "auto"
        print("Using automatic strategy selection")
    
    return strategy


def get_logger(args):
    """Get the logger based on arguments"""
    if args.use_wandb:
        try:
            import wandb
            logger = WandbLogger(
                project=args.wandb_project,
                name=args.experiment_name,
                save_dir=args.output_dir,
                tags=["qwen-image", "fine-tuning"],
            )
            print(f"Using W&B logger: {args.wandb_project}/{args.experiment_name}")
        except ImportError:
            print("W&B not available, falling back to TensorBoard")
            logger = TensorBoardLogger(
                save_dir=args.output_dir,
                name=args.experiment_name,
            )
    else:
        logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name=args.experiment_name,
        )
        print(f"Using TensorBoard logger: {args.output_dir}/{args.experiment_name}")
    
    return logger


def get_callbacks(args):
    """Get training callbacks"""
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="qwen-image-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=args.save_top_k,
            save_last=True,
            every_n_epochs=args.checkpoint_every_n_epochs,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=args.early_stopping_patience,
                mode="min",
                verbose=True,
            )
        )
    
    return callbacks


def main():
    parser = argparse.ArgumentParser(description="Train Qwen Image with PyTorch Lightning")
    
    # Data arguments
    parser.add_argument("--config_path", type=str, default="config.yaml", 
                       help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="Batch size per GPU")
    parser.add_argument("--val_batch_size", type=int, default=None,
                       help="Validation batch size (defaults to batch_size)")
    parser.add_argument("--num_workers", type=int, default=4, 
                       help="Number of data loader workers")
    parser.add_argument("--resolution", type=int, default=512, 
                       help="Image resolution")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, default="Qwen/Qwen-Image", 
                       help="Pretrained model path")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    
    # Training arguments
    parser.add_argument("--max_epochs", type=int, default=10, 
                       help="Maximum number of epochs")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--val_check_interval", type=float, default=0.5,
                       help="Validation check interval")
    parser.add_argument("--precision", type=str, default="bf16-mixed", 
                       choices=["16-mixed", "bf16-mixed", "32"], 
                       help="Training precision")
    
    # Optimization arguments
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                       help="Enable gradient checkpointing")
    parser.add_argument("--use_8bit_optimizer", action="store_true", default=True,
                       help="Use 8-bit optimizer")
    parser.add_argument("--compile_model", action="store_true", 
                       help="Compile model with torch.compile")
    parser.add_argument("--text_encoder_cpu", action="store_true", default=True,
                       help="Keep text encoder on CPU to save GPU memory")
    parser.add_argument("--scheduler_type", type=str, default="cosine",
                       choices=["cosine", "linear", "none"],
                       help="Learning rate scheduler type")
    
    # DeepSpeed arguments
    parser.add_argument("--use_deepspeed", action="store_true", 
                       help="Use DeepSpeed strategy")
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json",
                       help="DeepSpeed config file path")
    
    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                       help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="qwen_image_finetune", 
                       help="Experiment name")
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="qwen-image-training", 
                       help="W&B project name")
    
    # Checkpoint arguments
    parser.add_argument("--save_top_k", type=int, default=3,
                       help="Number of best checkpoints to save")
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=1,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    
    # Early stopping
    parser.add_argument("--early_stopping", action="store_true",
                       help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Early stopping patience")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic algorithms")
    
    args = parser.parse_args()
    
    # Set seed
    pl.seed_everything(args.seed, workers=True)
    
    # Optimize tensor core usage for H200/A100
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 50)
    print("Training Configuration:")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Setup data module
    data_module = QwenImageDataModule(
        config_path=args.config_path,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        size=args.resolution,
    )
    
    # Setup model
    model = QwenImageLightningModule(
        pretrained_model_name_or_path=args.pretrained_model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        resolution=args.resolution,
        enable_gradient_checkpointing=args.gradient_checkpointing,
        use_8bit_optimizer=args.use_8bit_optimizer and not args.use_deepspeed,  # Disable 8bit optimizer with DeepSpeed
        compile_model=args.compile_model,
        scheduler_type=args.scheduler_type,
        text_encoder_cpu=args.text_encoder_cpu,
        use_deepspeed=args.use_deepspeed,
    )
    
    # Setup strategy, logger, and callbacks
    strategy = get_strategy(args)
    logger = get_logger(args)
    callbacks = get_callbacks(args)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=strategy,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.max_grad_norm,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=args.deterministic,
        benchmark=not args.deterministic,  # Disable benchmarking if deterministic
        # devices = [0,3,4,5,6]
    )
    
    # Print training info
    print(f"\nStarting training with:")
    print(f"  - Strategy: {type(strategy).__name__}")
    print(f"  - Precision: {args.precision}")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Batch size per GPU: {args.batch_size}")
    print(f"  - Gradient accumulation: {args.accumulate_grad_batches}")
    if torch.cuda.is_available():
        print(f"  - Available GPUs: {torch.cuda.device_count()}")
    print()
    
    # Train model
    trainer.fit(
        model, 
        data_module, 
        ckpt_path=args.resume_from_checkpoint
    )
    
    # Save final model
    if trainer.is_global_zero:
        final_output_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Save transformer and tokenizer
        model.transformer.save_pretrained(final_output_dir)
        model.tokenizer.save_pretrained(final_output_dir)
        
        print(f"\nTraining completed!")
        print(f"Final model saved to: {final_output_dir}")
        print(f"Checkpoints saved to: {os.path.join(args.output_dir, 'checkpoints')}")
        
        # Log final metrics
        if hasattr(trainer, 'callback_metrics'):
            print(f"Final validation loss: {trainer.callback_metrics.get('val/loss', 'N/A')}")


if __name__ == "__main__":
    main() 