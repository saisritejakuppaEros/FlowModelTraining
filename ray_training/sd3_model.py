from __future__ import annotations
import os, argparse, tempfile
from typing import Dict, Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset, DataLoader

import ray
import ray.data as rd
import ray.train as train
from ray.train import Checkpoint, RunConfig, ScalingConfig, FailureConfig
from ray.train.torch import TorchTrainer
import ray.train.torch

# TensorBoard imports
from torch.utils.tensorboard import SummaryWriter
import datetime

# ----- your project imports -----
from model.dit import DiT
from common import loss_fn
from model.noise import NoiseScheduler
from data_stream import load_files_from_paths, SD3Transform, SD3Encoder
from validation_generator import ValidationImageGenerator

from pathlib import Path
from logzero import logger

import torch
torch.set_float32_matmul_precision("high")

# =========================
# Helper function for dtype conversion
# =========================
def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]

def get_numpy_dtype(dtype_str: str) -> np.dtype:
    """Convert dtype string to numpy dtype."""
    dtype_map = {
        "float16": np.float16,
        "bfloat16": np.float16,  # numpy doesn't have bfloat16, use float16
        "float32": np.float32,
        "float64": np.float64,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]

# =========================
# Ray Data: encoder pipeline
# =========================
def _to_fp16(batch: Dict[str, np.ndarray], dtype_str: str = "float16") -> Dict[str, np.ndarray]:
    out = {}
    target_dtype = get_numpy_dtype(dtype_str)
    for k, v in batch.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
            out[k] = v.astype(target_dtype, copy=False)
        else:
            out[k] = v
    return out


def build_encoded_dataset(
    csv_path: str,
    resolution: int = 512,
    batch_size: int = 32,
    parallelism: int = 8,
    min_aesthetic: float | None = None,
    enc_vae_model: str = "stabilityai/stable-diffusion-3-medium",
    enc_clip_l_model: str = "openai/clip-vit-large-patch14",
    enc_clip_g_model: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    enc_t5_model: str = "google/t5-v1_1-xxl",
    enc_dtype: str = "float16",
    enc_num_actors: int = 1,
    keep_intermediate_tensors: bool = False,
) -> rd.Dataset:
    ds = rd.read_csv(csv_path, parallelism=parallelism)

    ds = ds.map_batches(
        load_files_from_paths, batch_size=batch_size, concurrency=parallelism
    )

    ds = ds.map_batches(
        SD3Transform,
        fn_constructor_kwargs=dict(
            resolution=resolution,
            clip_l_model=enc_clip_l_model,
            clip_g_model=enc_clip_g_model,
            t5_model=enc_t5_model,
            max_length_clip=77,
            max_length_t5=256,
            min_aesthetic=min_aesthetic,
            drop_invalid=True,
        ),
        batch_size=batch_size,
        num_cpus=30,
        concurrency=2,
    )

    ds = ds.map_batches(
        SD3Encoder,
        fn_constructor_kwargs=dict(
            resolution=resolution,
            vae_model=enc_vae_model,
            clip_l_model=enc_clip_l_model,
            clip_g_model=enc_clip_g_model,
            t5_model=enc_t5_model,
            dtype=enc_dtype,
            keep_intermediate_tensors=keep_intermediate_tensors,
        ),
        batch_size=max(1, batch_size // 2),
        num_gpus=1,
        concurrency=2,
    )

    ds = ds.map_batches(
        lambda batch: _to_fp16(batch, enc_dtype), 
        batch_size=None, 
        concurrency=max(1, parallelism // 2)
    )
    return ds


# =========================
# Ray → PyTorch DataLoader
# =========================
class RayIterableDataset(IterableDataset):
    def __init__(self, name: str, batch_size: int, prefetch_batches: int):
        self.name = name
        self.bs = batch_size
        self.prefetch = prefetch_batches

    def __iter__(self) -> Iterable[Dict[str, torch.Tensor]]:
        ds = train.get_dataset_shard(self.name)
        # Returns Dict[str, torch.Tensor] on CPU (Ray Train moves to device)
        yield from ds.iter_torch_batches(
            batch_size=self.bs, prefetch_batches=self.prefetch, drop_last=True
        )


def make_dl(name: str, bs: int, prefetch: int) -> DataLoader:
    ds = RayIterableDataset(name, bs, prefetch)
    # dataset yields already-batched dicts; unwrap the single element from DataLoader
    return DataLoader(ds, batch_size=1, num_workers=0, collate_fn=lambda x: x[0])


# =========================
# Training function (Plain PyTorch)
# =========================
def train_func(cfg: Dict[str, Any]) -> None:
    # Seed everything
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    
    # Initialize model, noise scheduler
    model = DiT().to(dtype=get_torch_dtype(cfg["dtype"])) 
    noise_sched = NoiseScheduler()
    
    # Initialize validation image generator if prompts are provided
    validation_generator = None
    if "validation_prompts" in cfg and cfg["validation_prompts"]:
        output_dir = cfg.get("validation_output_dir", "/data0/teja_codes/ImmersoAiResearch/FlowModelTraining/miniDiffusion/ray_training/logs/valid_imgs")
        validation_generator = ValidationImageGenerator(
            prompts=cfg["validation_prompts"],
            resolution=cfg["resolution"],
            vae_model=cfg["enc_vae_model"],
            clip_l_model=cfg["enc_clip_l_model"],
            clip_g_model=cfg["enc_clip_g_model"],
            t5_model=cfg["enc_t5_model"],
            dtype=cfg["dtype"],
            output_dir=output_dir,
            num_inference_steps=cfg.get("validation_inference_steps", 50)
        )
        # Set debug flag if enabled in config
        if cfg.get("validation_debug", False):
            validation_generator.debug = True
        logger.info(f"🎨 Initialized validation generator with {len(cfg['validation_prompts'])} prompts")
        logger.info(f"📁 Validation images will be saved to: {os.path.abspath(output_dir)}")
    
    # Initialize TensorBoard writer (only on rank 0 to avoid conflicts)
    tensorboard_writer = None
    if cfg.get("tensorboard_enabled", True) and ray.train.get_context().get_world_rank() == 0:
        # Create unique log directory with timestamp and experiment name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = cfg.get("experiment_name", "sd3_training")
        log_dir = os.path.join(
            cfg.get("tensorboard_log_dir", "./tensorboard_logs"),
            f"{experiment_name}_{timestamp}"
        )
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_writer = SummaryWriter(log_dir)
        logger.info(f"📊 Initialized TensorBoard logging to: {os.path.abspath(log_dir)}")
        logger.info(f"🖥️ View logs with: tensorboard --logdir {os.path.abspath(log_dir)}")
    
    # [1] Prepare model for distributed training
    model = ray.train.torch.prepare_model(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        fused=torch.cuda.is_available(),
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(step: int):
        if step < cfg["warmup_steps"]:
            return float(step) / float(max(1, cfg["warmup_steps"]))
        return 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # [2] Prepare dataloaders
    train_loader = make_dl("train", cfg["batch_size_per_worker"], cfg["prefetch_batches"])
    val_loader = make_dl("validation", cfg["batch_size_per_worker"], max(1, cfg["prefetch_batches"] // 2))
    
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader = ray.train.torch.prepare_data_loader(val_loader)
    
    # Helper function for noise triplet
    def noise_triplet(x0: torch.Tensor):
        """Best-effort normalize different add_noise signatures."""
        x_t, eps, t = noise_sched.add_noise(x0)
        # Ensure all tensors are in the configured dtype to match model dtype
        # Only convert dtype, don't detach as we need gradients
        target_dtype = get_torch_dtype(cfg["dtype"])
        return x_t.to(dtype=target_dtype), eps.to(dtype=target_dtype), t
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(cfg["num_epochs"]):
        model.train()
        
        # Set epoch for distributed sampler
        # if ray.train.get_context().get_world_size() > 1:
        #     train_loader.sampler.set_epoch(epoch)
        
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Forward pass - convert to configured dtype to match model dtype
            target_dtype = get_torch_dtype(cfg["dtype"])
            x0 = batch[f"image_latents_{cfg['resolution']}"].to(dtype=target_dtype)
            e_txt = batch["text_embeddings"].to(dtype=target_dtype)
            e_pool = batch["pooled_embeddings"].to(dtype=target_dtype)
            
            x_t, eps, t = noise_triplet(x0)
            
            # Model predicts velocity field (drift)
            pred = model(
                latent=x_t, 
                timestep=t, 
                encoder_hidden_states=e_txt, 
                pooled_projections=e_pool
            )
            
            loss = loss_fn(pred, eps, t)
            
            # Store loss value before backward to avoid graph retention
            loss_item = loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            try:
                loss.backward()
            except RuntimeError as e:
                if "backward through the graph a second time" in str(e):
                    logger.error("❌ Backward pass error: Trying to backward through graph twice")
                    logger.error("This usually happens with gradient accumulation or memory issues")
                    continue
                else:
                    raise e
            
            # Gradient clipping
            if cfg["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            
            optimizer.step()
            scheduler.step()
            
            # Reset KV cache after optimizer step to prevent memory issues
            for layer in model.module.transformer_blocks if hasattr(model, 'module') else model.transformer_blocks:
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'reset_cache'):
                    layer.attn.reset_cache()
                if hasattr(layer, 'attn2') and hasattr(layer.attn2, 'reset_cache'):
                    layer.attn2.reset_cache()
            
            # Clear any remaining references
            del loss, pred, x_t, eps, t
            
            epoch_train_loss += loss_item
            num_train_batches += 1
            global_step += 1
            
            # Log training loss periodically
            log_interval = cfg.get("tensorboard_log_interval", 25)
            if batch_idx % log_interval == 0 and ray.train.get_context().get_world_rank() == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Train Loss: {loss_item:.6f}")
                
                # TensorBoard logging
                if tensorboard_writer is not None:
                    # Log training loss
                    tensorboard_writer.add_scalar('Training/Loss', loss_item, global_step)
                    
                    # Log learning rate
                    current_lr = scheduler.get_last_lr()[0]
                    tensorboard_writer.add_scalar('Training/Learning_Rate', current_lr, global_step)
                    
                    # Log gradient norms if enabled
                    if cfg.get("tensorboard_log_gradients", False):
                        total_norm = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                        tensorboard_writer.add_scalar('Training/Gradient_Norm', total_norm, global_step)
                    
                    # Log weight histograms if enabled (can be expensive)
                    if cfg.get("tensorboard_log_weights", False):
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                tensorboard_writer.add_histogram(f'Weights/{name}', param.data, global_step)
                                if param.grad is not None:
                                    tensorboard_writer.add_histogram(f'Gradients/{name}', param.grad.data, global_step)
            
            # Early stopping if max_steps reached
            if cfg["max_steps"] > 0 and global_step >= cfg["max_steps"]:
                break
        
        avg_train_loss = epoch_train_loss / max(num_train_batches, 1)
        
        # Validation loop
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        
        # Check if validation dataset is empty
        val_dataset_empty = True
        
        with torch.no_grad():
            for batch in val_loader:
                val_dataset_empty = False
                target_dtype = get_torch_dtype(cfg["dtype"])
                x0 = batch[f"image_latents_{cfg['resolution']}"].to(dtype=target_dtype)
                e_txt = batch["text_embeddings"].to(dtype=target_dtype)
                e_pool = batch["pooled_embeddings"].to(dtype=target_dtype)
                
                x_t, eps, t = noise_triplet(x0)
                
                # Model predicts velocity field (drift)
                pred = model(
                    latent=x_t,
                    timestep=t,
                    encoder_hidden_states=e_txt,
                    pooled_projections=e_pool
                )
                
                val_loss = loss_fn(pred, eps, t)
                val_loss_item = val_loss.item()
                epoch_val_loss += val_loss_item
                num_val_batches += 1
                
                # Reset KV cache during validation as well
                for layer in model.module.transformer_blocks if hasattr(model, 'module') else model.transformer_blocks:
                    if hasattr(layer, 'attn') and hasattr(layer.attn, 'reset_cache'):
                        layer.attn.reset_cache()
                    if hasattr(layer, 'attn2') and hasattr(layer.attn2, 'reset_cache'):
                        layer.attn2.reset_cache()
                
                # Clear validation tensors to prevent graph retention
                del val_loss, pred, x_t, eps, t
        
        # Handle empty validation dataset
        if val_dataset_empty:
            avg_val_loss = float('inf')  # Use infinity to indicate no validation data
            if ray.train.get_context().get_world_rank() == 0:
                logger.warning("⚠️ Validation dataset is empty! Check your data splitting configuration.")
        else:
            avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
        
        # [3] Generate validation images if generator is available
        validation_images_generated = []
        validation_frequency = cfg.get("validation_frequency", 1)
        if (validation_generator is not None and 
            ray.train.get_context().get_world_rank() == 0 and
            validation_frequency > 0 and
            epoch % validation_frequency == 0):
            try:
                # Log where images will be saved
                output_dir = cfg.get("validation_output_dir", "/data0/teja_codes/ImmersoAiResearch/FlowModelTraining/miniDiffusion/ray_training/logs/valid_imgs")
                logger.info(f"🖼️ Generating validation images for epoch {epoch}...")
                logger.info(f"📁 Images will be saved to: {os.path.abspath(output_dir)}")
                
                # Generate individual images
                saved_paths = validation_generator.save_validation_images(
                    model=model.module if hasattr(model, 'module') else model,
                    epoch=epoch,
                    global_step=global_step
                )
                validation_images_generated = saved_paths
                
                # Create grid image
                grid_path = validation_generator.create_grid_image(
                    model=model.module if hasattr(model, 'module') else model,
                    epoch=epoch,
                    global_step=global_step
                )
                if grid_path:
                    validation_images_generated.append(grid_path)
                
                logger.info(f"✅ Generated {len(validation_images_generated)} validation images for epoch {epoch}")
                logger.info(f"📂 All images saved in: {os.path.abspath(output_dir)}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate validation images: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        # [4] Report metrics and checkpoint
        metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "global_step": global_step,
            "validation_images_generated": len(validation_images_generated)
        }
        
        # TensorBoard logging for epoch metrics
        if tensorboard_writer is not None and ray.train.get_context().get_world_rank() == 0:
            tensorboard_writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch)
            tensorboard_writer.add_scalar('Epoch/Validation_Loss', avg_val_loss, epoch)
            tensorboard_writer.add_scalar('Epoch/Learning_Rate', scheduler.get_last_lr()[0], epoch)
            tensorboard_writer.add_scalar('Epoch/Global_Step', global_step, epoch)
            
            # Log validation images to TensorBoard if enabled and images were generated
            if (cfg.get("tensorboard_log_images", True) and 
                len(validation_images_generated) > 0 and 
                validation_generator is not None):
                try:
                    # Log first few validation images (limit to 4 to save space)
                    for idx, img_path in enumerate(validation_images_generated[:4]):
                        if os.path.exists(img_path):
                            # Read image and convert to tensor for TensorBoard
                            from PIL import Image
                            import torchvision.transforms as transforms
                            
                            img = Image.open(img_path).convert('RGB')
                            transform = transforms.ToTensor()
                            img_tensor = transform(img)
                            
                            # Log image with prompt as caption
                            prompt = cfg["validation_prompts"][idx] if idx < len(cfg["validation_prompts"]) else f"Image_{idx}"
                            tensorboard_writer.add_image(f'Validation_Images/Epoch_{epoch}_Image_{idx}', 
                                                       img_tensor, epoch)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to log validation images to TensorBoard: {e}")
            
            # Add scalars comparing train vs validation loss
            tensorboard_writer.add_scalars('Loss_Comparison', {
                'Train': avg_train_loss,
                'Validation': avg_val_loss
            }, epoch)
        
        # Always save checkpoints - Ray will handle checkpoint frequency and retention
        # This ensures all workers report the same structure
        
        # Update best validation loss
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        # Always create checkpoint directory and save checkpoint data
        # This ensures all workers report the same structure to Ray
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            # Save model state dict (unwrap from DDP if needed)
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            checkpoint_data = {
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'best_val_loss': best_val_loss,
                'config': cfg
            }
            
            # Save regular checkpoint
            checkpoint_path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(checkpoint_data, checkpoint_path)
            
            # Save best model separately if it's the best
            if is_best:
                best_checkpoint_path = os.path.join(temp_checkpoint_dir, "best_model.pt")
                torch.save(checkpoint_data, best_checkpoint_path)
                if ray.train.get_context().get_world_rank() == 0:
                    logger.info(f"🏆 New best validation loss: {best_val_loss:.6f} (epoch {epoch})")
                    logger.info(f"💾 Saved best model checkpoint (epoch {epoch})")
            
            # Always report with checkpoint to maintain consistency across workers
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
            ray.train.report(metrics=metrics, checkpoint=checkpoint)
            
            if ray.train.get_context().get_world_rank() == 0:
                logger.info(f"📊 Reported metrics and checkpoint at epoch {epoch}")
        
        if ray.train.get_context().get_world_rank() == 0:
            logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping if max_steps reached
        if cfg["max_steps"] > 0 and global_step >= cfg["max_steps"]:
            break
    
    # Cleanup TensorBoard writer
    if tensorboard_writer is not None and ray.train.get_context().get_world_rank() == 0:
        tensorboard_writer.close()
        logger.info("📊 TensorBoard writer closed")


# =========================
# Driver: build ds, split, launch Ray
# =========================
def main(cfg: Dict[str, Any]) -> None:
    ray.init(ignore_reinit_error=True)

    # 1) Pre-encode SD3 inputs with Ray Data
    ds = build_encoded_dataset(
        csv_path=cfg["csv"],
        resolution=cfg["resolution"],
        batch_size=cfg["batch_size"],
        parallelism=cfg["parallelism"],
        min_aesthetic=cfg.get("min_aesthetic"),
        enc_vae_model=cfg["enc_vae_model"],
        enc_clip_l_model=cfg["enc_clip_l_model"],
        enc_clip_g_model=cfg["enc_clip_g_model"],
        enc_t5_model=cfg["enc_t5_model"],
        enc_dtype=cfg["enc_dtype"],
        enc_num_actors=cfg["enc_num_actors"],
        keep_intermediate_tensors=cfg.get("keep_intermediate_tensors", False),
    )

    # 2) Shuffle + split into train/val
    n = ds.count()
    ds = ds.random_shuffle(seed=cfg["seed"])
    n_train = int(n * float(cfg.get("train_split", 0.98)))
    
    # Ensure we have at least some validation data
    if n_train >= n:
        n_train = max(1, n - 10)  # Leave at least 10 samples for validation
    
    train_ds, val_ds = ds.split_at_indices([n_train])
    
    # Log dataset sizes and checkpoint configuration
    checkpoint_frequency = cfg.get("checkpoint_frequency", 1)
    if ray.train.get_context().get_world_rank() == 0:
        logger.info(f"Total dataset size: {n}")
        logger.info(f"Training samples: {n_train}")
        logger.info(f"Validation samples: {n - n_train}")
        logger.info(f"Checkpoint frequency: {checkpoint_frequency} epochs")
        
        # Check if validation dataset is too small
        if n - n_train < 5:
            logger.warning(f"⚠️ Validation dataset is very small ({n - n_train} samples). Consider reducing train_split.")

    # 3) Configure scaling and resources
    scaling_config = ScalingConfig(
        num_workers=cfg["num_training_workers"], 
        use_gpu=True
    )
    
    # 4) Configure persistent storage and checkpoint management
    checkpoint_frequency = cfg.get("checkpoint_frequency", 1)
    run_config = RunConfig(
        name=cfg["experiment_name"],
        storage_path=Path(cfg["storage_path"]).resolve().as_uri(),
        failure_config=FailureConfig(max_failures=cfg["max_failures"]),
        checkpoint_config=ray.train.CheckpointConfig(
            num_to_keep=5,  # Keep last 5 checkpoints
        ),
    )

    # 5) Launch PyTorch training under Ray Train
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "validation": val_ds},
        train_loop_config=cfg,
    )
    
    result = trainer.fit()
    print("Finished. Latest metrics:", result.metrics)
    
    # 6) Optionally load the trained model for inference
    if result.checkpoint:
        print("Loading final checkpoint...")
        with result.checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
                print(f"Final epoch: {checkpoint_data['epoch']}")
                print(f"Final validation loss: {checkpoint_data.get('best_val_loss', 'N/A')}")


if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config_sd3.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)