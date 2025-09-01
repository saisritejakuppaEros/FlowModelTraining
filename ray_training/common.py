import torch
import torch.nn.functional as F
from logzero import logger

def loss_fn(pred: torch.Tensor, target: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """
    Loss function for SD3 training.
    
    Args:
        pred: Model prediction (velocity field)
        target: Target noise (epsilon)
        timestep: Current timestep
    
    Returns:
        Loss tensor
    """
    # Ensure inputs are the same shape
    if pred.shape != target.shape:
        logger.error(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    # Check for NaN/Inf in inputs
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        logger.error("NaN/Inf detected in prediction")
        return torch.tensor(float('nan'), device=pred.device, dtype=pred.dtype)
    
    if torch.isnan(target).any() or torch.isinf(target).any():
        logger.error("NaN/Inf detected in target")
        return torch.tensor(float('nan'), device=target.device, dtype=target.dtype)
    
    # MSE loss with numerical stability
    loss = F.mse_loss(pred, target, reduction='mean')
    
    # Additional stability check
    if torch.isnan(loss) or torch.isinf(loss):
        logger.error(f"Loss computation resulted in NaN/Inf: {loss.item()}")
        logger.error(f"Pred range: [{pred.min().item():.6f}, {pred.max().item():.6f}]")
        logger.error(f"Target range: [{target.min().item():.6f}, {target.max().item():.6f}]")
        return torch.tensor(float('nan'), device=loss.device, dtype=loss.dtype)
    
    return loss

def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Huber loss for more robust training.
    
    Args:
        pred: Model prediction
        target: Target values
        delta: Huber loss parameter
    
    Returns:
        Loss tensor
    """
    # Check for NaN/Inf in inputs
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        logger.error("NaN/Inf detected in prediction")
        return torch.tensor(float('nan'), device=pred.device, dtype=pred.dtype)
    
    if torch.isnan(target).any() or torch.isinf(target).any():
        logger.error("NaN/Inf detected in target")
        return torch.tensor(float('nan'), device=target.device, dtype=target.dtype)
    
    # Huber loss
    loss = F.huber_loss(pred, target, delta=delta, reduction='mean')
    
    if torch.isnan(loss) or torch.isinf(loss):
        logger.error(f"Huber loss computation resulted in NaN/Inf: {loss.item()}")
        return torch.tensor(float('nan'), device=loss.device, dtype=loss.dtype)
    
    return loss 