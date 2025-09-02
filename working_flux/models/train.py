import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import os
import json
from dit import load_flow_model2

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
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()





def flow_matching_loss(model_output, target, t):
    """
    Flow matching loss function.
    
    Args:
        model_output: Model prediction of the velocity field
        target: Target velocity (x_1 - x_0)
        t: Time step tensor
    """
    # Simple MSE loss between predicted and target velocity
    loss = F.mse_loss(model_output, target, reduction='mean')
    return loss

def create_training_data(data, batch_size=1, repeat_factor=1000):
    """Create training dataset from loaded data, repeating samples for overfitting"""
    x_1 = data["img_latent"]
    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    text_embed = data["text_embed"]
    clip_embed = data["clip_embed"]
    inp = data["inp"]
    
    print(f"Original data shapes:")
    print(f"  x_1: {x_1.shape}")
    print(f"  text_embed: {text_embed.shape}")
    print(f"  clip_embed: {clip_embed.shape}")
    print(f"  img_ids: {inp['img_ids'].shape}")
    print(f"  txt_ids: {inp['txt_ids'].shape}")
    print(f"  vec: {inp['vec'].shape}")
    
    # Repeat the data multiple times for overfitting
    x_1_repeated = x_1.repeat(repeat_factor, 1, 1)
    text_embed_repeated = text_embed.repeat(repeat_factor, 1, 1)
    clip_embed_repeated = clip_embed.repeat(repeat_factor, 1)
    img_ids_repeated = inp["img_ids"].repeat(repeat_factor, 1, 1)
    txt_ids_repeated = inp["txt_ids"].repeat(repeat_factor, 1, 1)
    vec_repeated = inp["vec"].repeat(repeat_factor, 1)
    
    print(f"\nAfter repeating {repeat_factor} times:")
    print(f"  x_1_repeated: {x_1_repeated.shape}")
    print(f"  text_embed_repeated: {text_embed_repeated.shape}")
    print(f"  clip_embed_repeated: {clip_embed_repeated.shape}")
    print(f"  img_ids_repeated: {img_ids_repeated.shape}")
    print(f"  txt_ids_repeated: {txt_ids_repeated.shape}")
    print(f"  vec_repeated: {vec_repeated.shape}")
    
    # Create dataset
    dataset = TensorDataset(
        x_1_repeated,
        text_embed_repeated,
        clip_embed_repeated,
        img_ids_repeated,
        txt_ids_repeated,
        vec_repeated
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Number of batches: {len(dataloader)}")
    
    return dataloader

def train_flow_model():
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16"  # Use bfloat16 for better performance
    )
    
    # Create model_checkpoints directory
    checkpoint_dir = "model_checkpoints"
    if accelerator.is_local_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created checkpoint directory: {checkpoint_dir}")
    
    # Load data
    print("Loading data...")
    try:
        data = torch.load("/data0/teja_codes/ImmersoAiResearch/flow_matching_v2/FlowModelTraining/dataset_preparation/data_cache/data.pt")
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create dataloader with batch size 1 and repeat the sample 1000 times
    batch_size = 1  # Set to 1 as requested
    repeat_factor = 1000  # Repeat the single sample 1000 times for overfitting
    dataloader = create_training_data(data, batch_size=batch_size, repeat_factor=repeat_factor)
    
    # Check if dataloader is empty
    if len(dataloader) == 0:
        print("ERROR: Dataloader is empty! Check your data or reduce batch_size further.")
        return
    
    # Load model
    print("Loading model...")
    try:
        model = load_flow_model2("flux-schnell")
        model.train()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,  # Conservative learning rate for flow matching
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Setup scheduler
    num_epochs = 100
    total_steps = len(dataloader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps,
        eta_min=1e-7
    )
    
    # Prepare everything with accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {accelerator.device}")
    print(f"Number of batches per epoch: {len(dataloader)}")
    
    if len(dataloader) == 0:
        print("ERROR: No batches available for training!")
        return
    
    losses = []
    epoch_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                           disable=not accelerator.is_local_main_process)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                x_1, text_embed, clip_embed, img_ids, txt_ids, vec = batch
                
                # Sample random timesteps
                bs = x_1.shape[0]
                t = torch.rand((bs,), device=accelerator.device)
                
                # Create noise
                x_0 = torch.randn_like(x_1)
                
                # Interpolate between noise and data: x_t = (1-t) * x_1 + t * x_0
                t_broadcast = t.view(-1, 1, 1)  # Shape: (bs, 1, 1)
                x_t = (1 - t_broadcast) * x_1 + t_broadcast * x_0
                
                # Target velocity: v = x_1 - x_0
                # target_velocity = x_1 - x_0
                target_velocity = x_0 - x_1

                
                # Guidance for the model
                guidance_vec = torch.full((bs,), 4.0, device=accelerator.device, dtype=x_t.dtype)
                
                with accelerator.accumulate(model):
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
                    accelerator.backward(loss)
                    
                    # Gradient clipping
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Log loss
                loss_value = loss.detach().float()
                losses.append(loss_value.item())
                epoch_loss += loss_value.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss_value.item():.6f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate average epoch loss (with safety check)
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            
            # Print epoch summary
            if accelerator.is_local_main_process:
                print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.6f}")
                
                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    try:
                        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
                        accelerator.save_state(checkpoint_path)
                        
                        # Also save just the model weights for easier loading later
                        model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model.state_dict(), model_path)
                        
                        # Save training metadata
                        metadata = {
                            "epoch": epoch + 1,
                            "avg_loss": avg_epoch_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "total_epochs": num_epochs,
                            "batch_size": batch_size,
                            "num_batches": num_batches
                        }
                        metadata_path = os.path.join(checkpoint_dir, f"metadata_epoch_{epoch+1}.json")
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        print(f"Checkpoint saved: {checkpoint_path}")
                        print(f"Model weights saved: {model_path}")
                        print(f"Metadata saved: {metadata_path}")
                    except Exception as e:
                        print(f"Error saving checkpoint: {e}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - No valid batches processed!")
            break
    
    # Save final model and plot results
    if accelerator.is_local_main_process:
        try:
            final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint")
            accelerator.save_state(final_checkpoint_path)
            
            # Also save final model weights
            final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), final_model_path)
            
            # Save final training metadata
            if epoch_losses:
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
                final_metadata_path = os.path.join(checkpoint_dir, "final_metadata.json")
                with open(final_metadata_path, 'w') as f:
                    json.dump(final_metadata, f, indent=2)
                print(f"Final metadata saved: {final_metadata_path}")
            
            print(f"Final checkpoint saved: {final_checkpoint_path}")
            print(f"Final model weights saved: {final_model_path}")
        except Exception as e:
            print(f"Error saving final checkpoint: {e}")
        
        # Only plot if we have loss data
        if epoch_losses:
            plt.figure(figsize=(12, 5))
            
            # Plot per-batch loss
            if losses:
                plt.subplot(1, 2, 1)
                plt.plot(losses)
                plt.title('Training Loss (Per Batch)')
                plt.xlabel('Batch')
                plt.ylabel('Loss')
                plt.yscale('log')
                plt.grid(True)
            
            # Plot per-epoch loss
            plt.subplot(1, 2, 2)
            plt.plot(epoch_losses)
            plt.title('Training Loss (Per Epoch)')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.yscale('log')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\nTraining completed!")
            print(f"Final loss: {epoch_losses[-1]:.6f}")
            print(f"Training loss plot saved as 'training_loss.png'")
            
            # Print loss statistics
            print(f"\nLoss Statistics:")
            print(f"  Initial loss: {epoch_losses[0]:.6f}")
            print(f"  Final loss: {epoch_losses[-1]:.6f}")
            print(f"  Best loss: {min(epoch_losses):.6f}")
            print(f"  Loss reduction: {((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100):.2f}%")
        else:
            print("No training data was processed successfully!")

if __name__ == "__main__":
    train_flow_model()