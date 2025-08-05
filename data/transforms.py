"""
Data transforms and augmentation for Flow Model Training
Based on x-flux repository preprocessing requirements
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
import random
import numpy as np
from typing import Tuple, Optional, Union, List


class RandomHorizontalFlip:
    """Random horizontal flip with probability"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return TF.hflip(image)
        return image


class RandomRotation:
    """Random rotation with probability"""
    
    def __init__(self, degrees: float = 10.0, p: float = 0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            return TF.rotate(image, angle)
        return image


class ColorJitter:
    """Color jitter augmentation"""
    
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1, 
                 saturation: float = 0.1, hue: float = 0.1, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            # Convert to PIL for color jitter
            if image.dim() == 3:
                image_pil = TF.to_pil_image(image)
            else:
                image_pil = TF.to_pil_image(image.unsqueeze(0))
            
            # Apply color jitter
            transform = T.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )
            image_pil = transform(image_pil)
            
            # Convert back to tensor
            image = TF.to_tensor(image_pil)
            if image.dim() == 3 and image.shape[0] == 1:
                image = image.squeeze(0)
        
        return image


class RandomCrop:
    """Random crop to specific size"""
    
    def __init__(self, size: Tuple[int, int], p: float = 0.5):
        self.size = size
        self.p = p
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return TF.random_crop(image, self.size)
        return image


class Normalize:
    """Normalize image to specific range"""
    
    def __init__(self, mean: List[float] = None, std: List[float] = None, 
                 min_val: float = -1.0, max_val: float = 1.0):
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.std is not None:
            # Standard normalization
            return TF.normalize(image, self.mean, self.std)
        else:
            # Range normalization
            if image.min() < self.min_val or image.max() > self.max_val:
                image = (image - image.min()) / (image.max() - image.min())
                image = image * (self.max_val - self.min_val) + self.min_val
        return image


class ResizeToMultiple:
    """Resize image dimensions to be multiples of a number (e.g., 32 for VAE)"""
    
    def __init__(self, multiple: int = 32):
        self.multiple = multiple
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            h, w = image.shape[1], image.shape[2]
        else:
            h, w = image.shape[0], image.shape[1]
        
        new_h = (h // self.multiple) * self.multiple
        new_w = (w // self.multiple) * self.multiple
        
        if new_h != h or new_w != w:
            image = F.interpolate(
                image.unsqueeze(0) if image.dim() == 3 else image.unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )
            image = image.squeeze(0)
        
        return image


class Compose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        return image


def get_training_transforms(
    img_size: int = 512,
    random_flip: bool = True,
    random_rotation: bool = False,
    color_jitter: bool = False,
    normalize: bool = True,
    resize_to_multiple: int = 32
) -> Compose:
    """
    Get training transforms
    
    Args:
        img_size: Target image size
        random_flip: Whether to apply random horizontal flip
        random_rotation: Whether to apply random rotation
        color_jitter: Whether to apply color jitter
        normalize: Whether to normalize images
        resize_to_multiple: Resize to multiple of this number
        
    Returns:
        Composed transforms
    """
    transforms = []
    
    if random_flip:
        transforms.append(RandomHorizontalFlip(p=0.5))
    
    if random_rotation:
        transforms.append(RandomRotation(degrees=10.0, p=0.3))
    
    if color_jitter:
        transforms.append(ColorJitter(p=0.3))
    
    if resize_to_multiple > 1:
        transforms.append(ResizeToMultiple(multiple=resize_to_multiple))
    
    if normalize:
        transforms.append(Normalize(min_val=-1.0, max_val=1.0))
    
    return Compose(transforms)


def get_validation_transforms(
    img_size: int = 512,
    normalize: bool = True,
    resize_to_multiple: int = 32
) -> Compose:
    """
    Get validation transforms (minimal augmentation)
    
    Args:
        img_size: Target image size
        normalize: Whether to normalize images
        resize_to_multiple: Resize to multiple of this number
        
    Returns:
        Composed transforms
    """
    transforms = []
    
    if resize_to_multiple > 1:
        transforms.append(ResizeToMultiple(multiple=resize_to_multiple))
    
    if normalize:
        transforms.append(Normalize(min_val=-1.0, max_val=1.0))
    
    return Compose(transforms)


def get_inference_transforms(
    img_size: int = 512,
    normalize: bool = True,
    resize_to_multiple: int = 32
) -> Compose:
    """
    Get inference transforms (no augmentation)
    
    Args:
        img_size: Target image size
        normalize: Whether to normalize images
        resize_to_multiple: Resize to multiple of this number
        
    Returns:
        Composed transforms
    """
    transforms = []
    
    if resize_to_multiple > 1:
        transforms.append(ResizeToMultiple(multiple=resize_to_multiple))
    
    if normalize:
        transforms.append(Normalize(min_val=-1.0, max_val=1.0))
    
    return Compose(transforms)


# Text preprocessing functions
def tokenize_text(text: str, tokenizer, max_length: int = 77) -> torch.Tensor:
    """
    Tokenize text for CLIP/T5 models
    
    Args:
        text: Input text
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tokenized text tensor
    """
    tokens = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    return tokens


def preprocess_caption(caption: str, max_length: int = 77) -> str:
    """
    Preprocess caption text
    
    Args:
        caption: Input caption
        max_length: Maximum caption length
        
    Returns:
        Preprocessed caption
    """
    # Remove extra whitespace
    caption = " ".join(caption.split())
    
    # Truncate if too long
    if len(caption) > max_length:
        caption = caption[:max_length].rsplit(' ', 1)[0]
    
    return caption.strip()


# Utility functions for data loading
def collate_fn(batch):
    """
    Custom collate function for batching
    
    Args:
        batch: List of (image, caption) tuples
        
    Returns:
        Batched tensors
    """
    images = []
    captions = []
    
    for image, caption in batch:
        images.append(image)
        captions.append(caption)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    return images, captions


def collate_fn_controlnet(batch):
    """
    Custom collate function for ControlNet batching
    
    Args:
        batch: List of (image, control_signal, caption) tuples
        
    Returns:
        Batched tensors
    """
    images = []
    control_signals = []
    captions = []
    
    for image, control_signal, caption in batch:
        images.append(image)
        control_signals.append(control_signal)
        captions.append(caption)
    
    # Stack tensors
    images = torch.stack(images, dim=0)
    control_signals = torch.stack(control_signals, dim=0)
    
    return images, control_signals, captions 