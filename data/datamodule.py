"""
PyTorch Lightning DataModule for Flow Model Training
Integrates with the dataset and transforms modules
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os
from pathlib import Path

from dataset import (
    FlowImageDataset, 
    ControlNetDataset, 
    create_dataloader, 
    create_train_val_datasets
)
from transforms import (
    get_training_transforms,
    get_validation_transforms,
    get_inference_transforms,
    collate_fn,
    collate_fn_controlnet
)


class FlowDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Flow Model Training
    """
    
    def __init__(
        self,
        data_dir: str,
        img_size: int = 512,
        batch_size: int = 1,
        num_workers: int = 4,
        train_split: float = 0.9,
        caption_type: str = 'json',
        random_ratio: bool = False,
        center_crop: bool = False,
        random_flip: bool = True,
        random_rotation: bool = False,
        color_jitter: bool = False,
        normalize: bool = True,
        resize_to_multiple: int = 32,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle: bool = True,
        **kwargs
    ):
        """
        Initialize DataModule
        
        Args:
            data_dir: Directory containing images and captions
            img_size: Target image size
            batch_size: Batch size for training
            num_workers: Number of worker processes
            train_split: Fraction of data to use for training
            caption_type: Type of caption file ('json', 'txt')
            random_ratio: Whether to apply random aspect ratio cropping
            center_crop: Whether to center crop images
            random_flip: Whether to apply random horizontal flip
            random_rotation: Whether to apply random rotation
            color_jitter: Whether to apply color jitter
            normalize: Whether to normalize images
            resize_to_multiple: Resize to multiple of this number
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            shuffle: Whether to shuffle data
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.random_rotation = random_rotation
        self.color_jitter = color_jitter
        self.normalize = normalize
        self.resize_to_multiple = resize_to_multiple
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Store additional kwargs
        self.kwargs = kwargs
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Initialize transforms
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')
        """
        # Setup transforms
        self.train_transforms = get_training_transforms(
            img_size=self.img_size,
            random_flip=self.random_flip,
            random_rotation=self.random_rotation,
            color_jitter=self.color_jitter,
            normalize=self.normalize,
            resize_to_multiple=self.resize_to_multiple
        )
        
        self.val_transforms = get_validation_transforms(
            img_size=self.img_size,
            normalize=self.normalize,
            resize_to_multiple=self.resize_to_multiple
        )
        
        self.test_transforms = get_inference_transforms(
            img_size=self.img_size,
            normalize=self.normalize,
            resize_to_multiple=self.resize_to_multiple
        )
        
        # Setup datasets
        if stage == 'fit' or stage is None:
            # Create training and validation datasets
            train_dataset, val_dataset = create_train_val_datasets(
                data_dir=self.data_dir,
                train_split=self.train_split,
                img_size=self.img_size,
                caption_type=self.caption_type,
                random_ratio=self.random_ratio,
                center_crop=self.center_crop,
                **self.kwargs
            )
            
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            
        elif stage == 'validate':
            # Create validation dataset only
            _, val_dataset = create_train_val_datasets(
                data_dir=self.data_dir,
                train_split=self.train_split,
                img_size=self.img_size,
                caption_type=self.caption_type,
                random_ratio=self.random_ratio,
                center_crop=self.center_crop,
                **self.kwargs
            )
            self.val_dataset = val_dataset
            
        elif stage == 'test':
            # Create test dataset (use validation split as test)
            _, test_dataset = create_train_val_datasets(
                data_dir=self.data_dir,
                train_split=self.train_split,
                img_size=self.img_size,
                caption_type=self.caption_type,
                random_ratio=self.random_ratio,
                center_crop=self.center_crop,
                **self.kwargs
            )
            self.test_dataset = test_dataset
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader"""
        if self.train_dataset is None:
            self.setup(stage='fit')
        
        return create_dataloader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader"""
        if self.val_dataset is None:
            self.setup(stage='validate')
        
        return create_dataloader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # No shuffling for validation
            pin_memory=self.pin_memory,
            drop_last=False,  # Don't drop last batch for validation
            collate_fn=collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader"""
        if self.test_dataset is None:
            self.setup(stage='test')
        
        return create_dataloader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # No shuffling for testing
            pin_memory=self.pin_memory,
            drop_last=False,  # Don't drop last batch for testing
            collate_fn=collate_fn
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction DataLoader"""
        # For prediction, we might want to use the full dataset
        dataset = FlowImageDataset(
            img_dir=self.data_dir,
            img_size=self.img_size,
            caption_type=self.caption_type,
            random_ratio=False,  # No augmentation for prediction
            center_crop=self.center_crop,
            **self.kwargs
        )
        
        return create_dataloader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # No shuffling for prediction
            pin_memory=self.pin_memory,
            drop_last=False,  # Don't drop last batch for prediction
            collate_fn=collate_fn
        )


class ControlNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for ControlNet Training
    """
    
    def __init__(
        self,
        data_dir: str,
        img_size: int = 512,
        batch_size: int = 1,
        num_workers: int = 4,
        train_split: float = 0.9,
        caption_type: str = 'json',
        control_type: str = 'canny',
        canny_low: int = 100,
        canny_high: int = 200,
        center_crop: bool = True,  # Usually True for ControlNet
        normalize: bool = True,
        resize_to_multiple: int = 32,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle: bool = True,
        **kwargs
    ):
        """
        Initialize ControlNet DataModule
        
        Args:
            data_dir: Directory containing images and captions
            img_size: Target image size
            batch_size: Batch size for training
            num_workers: Number of worker processes
            train_split: Fraction of data to use for training
            caption_type: Type of caption file
            control_type: Type of control signal ('canny', 'depth', 'pose')
            canny_low: Low threshold for Canny edge detection
            canny_high: High threshold for Canny edge detection
            center_crop: Whether to center crop images
            normalize: Whether to normalize images
            resize_to_multiple: Resize to multiple of this number
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            shuffle: Whether to shuffle data
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.caption_type = caption_type
        self.control_type = control_type
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.center_crop = center_crop
        self.normalize = normalize
        self.resize_to_multiple = resize_to_multiple
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Store additional kwargs
        self.kwargs = kwargs
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')
        """
        # Create full ControlNet dataset
        full_dataset = ControlNetDataset(
            img_dir=self.data_dir,
            img_size=self.img_size,
            caption_type=self.caption_type,
            control_type=self.control_type,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            center_crop=self.center_crop,
            normalize=self.normalize,
            **self.kwargs
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = total_size - train_size
        
        if stage == 'fit' or stage is None:
            # Create training and validation datasets
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            
        elif stage == 'validate':
            # Create validation dataset only
            _, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            self.val_dataset = val_dataset
            
        elif stage == 'test':
            # Create test dataset (use validation split as test)
            _, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            self.test_dataset = test_dataset
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader"""
        if self.train_dataset is None:
            self.setup(stage='fit')
        
        return create_dataloader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=collate_fn_controlnet
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader"""
        if self.val_dataset is None:
            self.setup(stage='validate')
        
        return create_dataloader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # No shuffling for validation
            pin_memory=self.pin_memory,
            drop_last=False,  # Don't drop last batch for validation
            collate_fn=collate_fn_controlnet
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader"""
        if self.test_dataset is None:
            self.setup(stage='test')
        
        return create_dataloader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # No shuffling for testing
            pin_memory=self.pin_memory,
            drop_last=False,  # Don't drop last batch for testing
            collate_fn=collate_fn_controlnet
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction DataLoader"""
        # For prediction, we might want to use the full dataset
        dataset = ControlNetDataset(
            img_dir=self.data_dir,
            img_size=self.img_size,
            caption_type=self.caption_type,
            control_type=self.control_type,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            center_crop=self.center_crop,
            normalize=self.normalize,
            **self.kwargs
        )
        
        return create_dataloader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # No shuffling for prediction
            pin_memory=self.pin_memory,
            drop_last=False,  # Don't drop last batch for prediction
            collate_fn=collate_fn_controlnet
        )


def create_datamodule_from_config(config: Dict[str, Any], **kwargs) -> pl.LightningDataModule:
    """
    Create DataModule from configuration
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Configured DataModule
    """
    data_config = config.get('data_config', {})
    
    # Check if this is a ControlNet configuration
    if 'controlnet_config' in config:
        return ControlNetDataModule(
            data_dir=data_config.get('img_dir', 'data/images/'),
            img_size=data_config.get('img_size', 512),
            batch_size=config.get('train_batch_size', 1),
            num_workers=data_config.get('num_workers', 4),
            train_split=0.9,
            caption_type='json',
            control_type=config['controlnet_config'].get('control_type', 'canny'),
            canny_low=config['controlnet_config'].get('canny', {}).get('low_threshold', 100),
            canny_high=config['controlnet_config'].get('canny', {}).get('high_threshold', 200),
            center_crop=True,
            random_ratio=data_config.get('random_ratio', False),
            **kwargs
        )
    else:
        return FlowDataModule(
            data_dir=data_config.get('img_dir', 'data/images/'),
            img_size=data_config.get('img_size', 512),
            batch_size=config.get('train_batch_size', 1),
            num_workers=data_config.get('num_workers', 4),
            train_split=0.9,
            caption_type='json',
            random_ratio=data_config.get('random_ratio', False),
            center_crop=False,
            **kwargs
        ) 