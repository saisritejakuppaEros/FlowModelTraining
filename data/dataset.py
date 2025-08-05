"""
Dataset classes for Flow Model Training
Based on x-flux repository dataset implementation
"""

import os
import json
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import cv2
from pathlib import Path


def image_resize(img: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize image maintaining aspect ratio"""
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))


def center_crop(image: Image.Image) -> Image.Image:
    """Center crop image to square"""
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = left + new_size
    bottom = top + new_size
    return image.crop((left, top, right, bottom))


def crop_to_aspect_ratio(image: Image.Image, ratio: str = "16:9") -> Image.Image:
    """Crop image to specific aspect ratio"""
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1),
        "3:4": (3, 4),
        "9:16": (9, 16)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    return image.crop(crop_box)


def canny_processor(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """Apply Canny edge detection to image"""
    image_array = np.array(image)
    canny_image = cv2.Canny(image_array, low_threshold, high_threshold)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    return Image.fromarray(canny_image)


class FlowImageDataset(Dataset):
    """
    Base dataset class for Flow Model Training
    Supports image-text pairs with various preprocessing options
    """
    
    def __init__(
        self,
        img_dir: str,
        img_size: int = 512,
        caption_type: str = 'json',
        random_ratio: bool = False,
        center_crop: bool = False,
        normalize: bool = True,
        supported_formats: List[str] = None
    ):
        """
        Initialize dataset
        
        Args:
            img_dir: Directory containing images
            img_size: Target image size
            caption_type: Type of caption file ('json', 'txt')
            random_ratio: Whether to apply random aspect ratio cropping
            center_crop: Whether to center crop images
            normalize: Whether to normalize images to [-1, 1]
            supported_formats: List of supported image formats
        """
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.center_crop = center_crop
        self.normalize = normalize
        
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.webp']
        self.supported_formats = supported_formats
        
        # Find all image files
        self.images = []
        for format_ext in self.supported_formats:
            self.images.extend(list(self.img_dir.glob(f"*{format_ext}")))
            self.images.extend(list(self.img_dir.glob(f"*{format_ext.upper()}")))
        
        self.images.sort()
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {img_dir} with formats {supported_formats}")
        
        print(f"Found {len(self.images)} images in {img_dir}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def _load_caption(self, image_path: Path) -> str:
        """Load caption from file"""
        caption_path = image_path.with_suffix(f'.{self.caption_type}')
        
        if not caption_path.exists():
            # Try alternative caption paths
            alt_paths = [
                image_path.with_suffix('.txt'),
                image_path.with_suffix('.caption'),
                image_path.parent / f"{image_path.stem}.txt",
                image_path.parent / f"{image_path.stem}.caption"
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    caption_path = alt_path
                    break
            else:
                # Return filename as caption if no caption file found
                return image_path.stem
        
        try:
            if self.caption_type == "json":
                with open(caption_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('caption', data.get('text', image_path.stem))
            else:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
        except Exception as e:
            print(f"Error loading caption from {caption_path}: {e}")
            return image_path.stem
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for training"""
        # Apply aspect ratio cropping if enabled
        if self.random_ratio:
            ratio = random.choice(["16:9", "default", "1:1", "4:3", "3:4", "9:16"])
            if ratio != "default":
                image = crop_to_aspect_ratio(image, ratio)
        
        # Center crop if enabled
        if self.center_crop:
            image = center_crop(image)
        
        # Resize image
        image = image_resize(image, self.img_size)
        
        # Ensure dimensions are divisible by 32 (for VAE)
        w, h = image.size
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32
        image = image.resize((new_w, new_h))
        
        # Convert to tensor
        image_array = np.array(image)
        
        if self.normalize:
            # Normalize to [-1, 1]
            image_array = (image_array / 127.5) - 1
        else:
            # Normalize to [0, 1]
            image_array = image_array / 255.0
        
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        
        return image_tensor
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get item from dataset"""
        try:
            image_path = self.images[idx]
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            
            # Load caption
            caption = self._load_caption(image_path)
            
            return image_tensor, caption
            
        except Exception as e:
            print(f"Error loading item {idx} from {self.images[idx]}: {e}")
            # Return a random item as fallback
            return self.__getitem__(random.randint(0, len(self.images) - 1))


class ControlNetDataset(FlowImageDataset):
    """
    Dataset for ControlNet training with control signals (e.g., Canny edges)
    """
    
    def __init__(
        self,
        img_dir: str,
        img_size: int = 512,
        caption_type: str = 'json',
        control_type: str = 'canny',
        canny_low: int = 100,
        canny_high: int = 200,
        **kwargs
    ):
        """
        Initialize ControlNet dataset
        
        Args:
            img_dir: Directory containing images
            img_size: Target image size
            caption_type: Type of caption file
            control_type: Type of control signal ('canny', 'depth', 'pose')
            canny_low: Low threshold for Canny edge detection
            canny_high: High threshold for Canny edge detection
        """
        super().__init__(img_dir, img_size, caption_type, **kwargs)
        self.control_type = control_type
        self.canny_low = canny_low
        self.canny_high = canny_high
    
    def _generate_control_signal(self, image: Image.Image) -> torch.Tensor:
        """Generate control signal from image"""
        if self.control_type == 'canny':
            control_image = canny_processor(image, self.canny_low, self.canny_high)
        else:
            # For other control types, you can extend this
            control_image = image
        
        # Preprocess control image
        control_tensor = self._preprocess_image(control_image)
        return control_tensor
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get item from dataset (image, control_signal, caption)"""
        try:
            image_path = self.images[idx]
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            
            # Generate control signal
            control_tensor = self._generate_control_signal(image)
            
            # Load caption
            caption = self._load_caption(image_path)
            
            return image_tensor, control_tensor, caption
            
        except Exception as e:
            print(f"Error loading item {idx} from {self.images[idx]}: {e}")
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with common settings
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs
    )


def create_train_val_datasets(
    data_dir: str,
    train_split: float = 0.9,
    img_size: int = 512,
    caption_type: str = 'json',
    random_ratio: bool = False,
    center_crop: bool = False,
    **kwargs
) -> Tuple[FlowImageDataset, FlowImageDataset]:
    """
    Create training and validation datasets
    
    Args:
        data_dir: Directory containing all data
        train_split: Fraction of data to use for training
        **kwargs: Additional dataset arguments
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create full dataset
    full_dataset = FlowImageDataset(
        img_dir=data_dir,
        img_size=img_size,
        caption_type=caption_type,
        random_ratio=random_ratio,
        center_crop=center_crop,
        **kwargs
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset


# Backward compatibility functions
def loader(train_batch_size: int, num_workers: int, **kwargs) -> DataLoader:
    """Backward compatibility function for x-flux loader"""
    dataset = FlowImageDataset(**kwargs)
    return create_dataloader(dataset, train_batch_size, num_workers) 