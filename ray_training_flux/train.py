import ray
import ray.data as rd
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
import torch
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import os
import json
from model_utils.dit import load_flow_model2
from torch import Tensor
import math
from typing import Callable

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()

def flow_matching_loss(model_output, target, t):
    """Flow matching loss function."""
    loss = F.mse_loss(model_output, target, reduction='mean')
    return loss

def preprocess_batch(batch, device, dtype):
    """Preprocess a batch from Ray dataset"""
    processed_batch = {}
    
    # List of columns to skip tensor conversion (keep as strings/objects)
    string_columns = {"caption_text"}
    
    for key, value in batch.items():
        if key in string_columns:
            # Keep string columns as-is
            processed_batch[key] = value
        elif isinstance(value, np.ndarray):
            # Convert numeric arrays to tensors, move to device and convert dtype
            tensor = torch.from_numpy(value).to(device)
            # Convert to proper dtype if it's a floating point tensor
            if tensor.dtype.is_floating_point:
                tensor = tensor.to(dtype)
            processed_batch[key] = tensor
        else:
            processed_batch[key] = value
    
    return processed_batch

def train_func(config):
    """Training function that runs on each Ray worker"""
    
    # Get training parameters from config
    num_epochs = config.get("num_epochs", 100)
    learning_rate = config.get("learning_rate", 1e-5)
    batch_size = config.get("batch_size", 1)
    parquet_path = config.get("parquet_path", "./processed_flux_dataset")
    
    # Load model
    print("Loading model...")
    model = load_flow_model2("flux-schnell")
    
    # Prepare model for distributed training
    model = ray.train.torch.prepare_model(model)
    model.train()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Load Ray dataset
    print(f"Loading Ray dataset from {parquet_path}...")
    ds = rd.read_parquet(parquet_path)
    dataset_size = ds.count()
    print(f"Dataset size: {dataset_size}")
    
    total_samples = dataset_size
    print(f"Total training samples: {total_samples}")
    
    # Setup scheduler
    steps_per_epoch = max(1, total_samples // batch_size)
    total_steps = steps_per_epoch * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    losses = []
    epoch_losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        num_batches = 0
        
        # Create batched iterator for this epoch - exclude string columns
        batch_iterator = ds.iter_batches(
            batch_size=batch_size,
            local_shuffle_seed=epoch,  # Different shuffle each epoch
            prefetch_batches=2,
            batch_format="numpy"  # Use numpy format to handle mixed types
        )
        
        # Wrap with progress bar
        batch_iterator = tqdm(
            batch_iterator, 
            desc=f"Epoch {epoch+1}/{num_epochs}",
            total=steps_per_epoch
        )
        
        for batch_idx, batch in enumerate(batch_iterator):
            try:
                # Get device and dtype from model
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
                
                # Preprocess batch
                batch = preprocess_batch(batch, device, dtype)
                
                # Extract data - adjust keys based on your actual dataset structure
                if "raw_img_latents" in batch:
                    x_1 = batch["raw_img_latents"]
                elif "img_latents" in batch:
                    x_1 = batch["img_latents"]
                else:
                    print(f"Available keys: {batch.keys()}")
                    raise KeyError("No image latents found in batch")
                
                # Rearrange if needed
                if len(x_1.shape) == 4:
                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                
                # Get embeddings
                if "t5_embeddings" in batch:
                    text_embed = batch["t5_embeddings"]
                elif "txt_embeds" in batch:
                    text_embed = batch["txt_embeds"]
                else:
                    raise KeyError("No text embeddings found in batch")
                
                clip_embed = batch["clip_embeddings"]
                
                # Get IDs and vectors
                img_ids = batch["img_ids"]
                txt_ids = batch["txt_ids"]
                if "vec_embeds" in batch:
                    vec = batch["vec_embeds"]
                elif "vec_embed" in batch:
                    vec = batch["vec_embed"]
                else:
                    # Create default vec if not available
                    vec = torch.zeros((x_1.shape[0], 768), device=x_1.device)
                
                # Sample random timesteps
                bs = x_1.shape[0]
                device = x_1.device
                t = torch.rand((bs,), device=device)
                
                # Create noise
                x_0 = torch.randn_like(x_1)
                
                # Interpolate between noise and data
                t_broadcast = t.view(-1, 1, 1)
                x_t = (1 - t_broadcast) * x_1 + t_broadcast * x_0
                
                # Target velocity
                target_velocity = x_0 - x_1
                
                # Guidance
                guidance_vec = torch.full((bs,), 4.0, device=device, dtype=x_t.dtype)
                
                # Forward pass
                model_output = model(
                    img=x_t,
                    img_ids=img_ids,
                    txt=text_embed,
                    txt_ids=txt_ids,
                    timesteps=t,
                    y=vec,
                    guidance=guidance_vec
                )
                
                # Compute loss
                loss = flow_matching_loss(model_output, target_velocity, t)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Log loss
                loss_value = loss.detach().float().item()
                losses.append(loss_value)
                epoch_loss += loss_value
                num_batches += 1
                
                # Update progress bar
                batch_iterator.set_postfix({
                    'loss': f'{loss_value:.6f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Break after enough batches for this epoch
                if batch_idx >= steps_per_epoch - 1:
                    break
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate average epoch loss
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.6f}")
            
            # Report metrics to Ray Train
            metrics = {
                "loss": avg_epoch_loss,
                "epoch": epoch + 1,
                # "learning_rate": scheduler.get_last_lr()[0]
            }
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                import tempfile
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    # Save model state dict
                    model_path = os.path.join(temp_checkpoint_dir, "model.pt")
                    if hasattr(model, 'module'):
                        torch.save(model.module.state_dict(), model_path)
                    else:
                        torch.save(model.state_dict(), model_path)
                    
                    # Save optimizer and scheduler
                    torch.save(optimizer.state_dict(), os.path.join(temp_checkpoint_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(temp_checkpoint_dir, "scheduler.pt"))
                    
                    # Save metadata
                    metadata = {
                        "epoch": epoch + 1,
                        "avg_loss": avg_epoch_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "total_epochs": num_epochs,
                        "batch_size": batch_size,
                        "num_batches": num_batches
                    }
                    
                    with open(os.path.join(temp_checkpoint_dir, "metadata.json"), 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Report to Ray Train
                    checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
                    ray.train.report(metrics=metrics, checkpoint=checkpoint)
            else:
                # Report metrics without checkpoint
                ray.train.report(metrics=metrics)
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - No valid batches processed!")
            break
    
    # Final checkpoint
    if epoch_losses:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            # Save final model
            model_path = os.path.join(temp_checkpoint_dir, "final_model.pt")
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            
            # Save final metadata
            final_metadata = {
                "epoch": num_epochs,
                "final_loss": epoch_losses[-1],
                "initial_loss": epoch_losses[0],
                "best_loss": min(epoch_losses),
                "loss_reduction_percent": ((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100),
                "total_epochs_completed": len(epoch_losses),
                "batch_size": batch_size,
                "training_completed": True
            }
            
            with open(os.path.join(temp_checkpoint_dir, "final_metadata.json"), 'w') as f:
                json.dump(final_metadata, f, indent=2)
            
            # Final report
            final_checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
            ray.train.report(
                metrics={"final_loss": epoch_losses[-1], "training_completed": True},
                checkpoint=final_checkpoint
            )
        
        print(f"\nTraining completed!")
        print(f"Final loss: {epoch_losses[-1]:.6f}")
        print(f"Loss reduction: {((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100):.2f}%")

def main():
    """Main function to launch Ray training"""
    
    # Training configuration
    train_config = {
        "num_epochs": 1000,
        "learning_rate": 1e-5,
        "batch_size": 4,
        "parquet_path": os.path.abspath("./processed_flux_dataset")
    }
    
    # Scaling configuration
    scaling_config = ScalingConfig(
        num_workers=8,  # Start with 1 worker, increase if you have multiple GPUs
        use_gpu=True,
        resources_per_worker={"CPU": 12, "GPU": 1}
    )


    storage_path = os.path.abspath("./ray_results")

    os.makedirs(storage_path, exist_ok=True)
    # Run configuration for saving results
    run_config = RunConfig(
        name="flux_flow_matching_training",
        storage_path=storage_path,  # Local storage
        checkpoint_config=ray.train.CheckpointConfig(
            num_to_keep=3,  # Keep last 3 checkpoints
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min"
        )
    )
    
    # Create and run trainer
    trainer = TorchTrainer(
        train_func,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config
    )
    
    print("Starting Ray training...")
    result = trainer.fit()
    
    print("Training completed!")
    print(f"Best checkpoint: {result.best_checkpoints}")
    print(f"Final metrics: {result.metrics}")
    
    # Plot training results
    if result.metrics_dataframe is not None and not result.metrics_dataframe.empty:
        df = result.metrics_dataframe
        print(f"Available columns in metrics dataframe: {df.columns.tolist()}")
        
        # Check if we have the required columns for plotting
        if 'epoch' in df.columns and 'loss' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['loss'])
            plt.title('Training Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('ray_training_loss.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Training loss plot saved as 'ray_training_loss.png'")
        else:
            print("Metrics dataframe does not contain required columns for plotting")
            print(f"DataFrame contents:\n{df.head()}")
    else:
        print("No metrics dataframe available or dataframe is empty")
    
    return result

if __name__ == "__main__":
    # Initialize Ray
    # ray.init(ignore_reinit_error=True)
    context = ray.init(include_dashboard=True)
    # print('--------------------------------')
    # print(context.dashboard_url)
    # quit()
    
    try:
        result = main()
    finally:
        ray.shutdown()