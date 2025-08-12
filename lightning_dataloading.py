import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from pathlib import Path
from torchvision import transforms
import pytorch_lightning as pl
from typing import Optional


class SimpleDataset(Dataset):
    """Simple dataset that loads images and prompts from a config file"""
    
    def __init__(self, config_path, size=512, center_crop=False):
        """
        Args:
            config_path: Path to config.yaml file
            size: Image size for resizing
            center_crop: Whether to center crop images
        """
        self.size = size
        self.center_crop = center_crop
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get dataset configuration
        dataset_config = self.config.get('dataset', {})
        self.data_dir = Path(dataset_config.get('data_dir', '.'))
        
        # Load captions from CSV file
        captions_file = self.data_dir / "captions.csv"
        if captions_file.exists():
            self.df = pd.read_csv(captions_file)
            self.image_files = []
            self.prompts = []
            
            # Get images directory
            images_dir = self.data_dir / "images"
            
            # Match captions with image files
            for _, row in self.df.iterrows():
                image_filename = row['filename']
                caption = row['caption']
                image_path = images_dir / image_filename
                
                if image_path.exists():
                    self.image_files.append(image_path)
                    self.prompts.append(caption)
        else:
            # Fallback to old method if no CSV file
            self.prompts = dataset_config.get('prompts', [])
            
            # Get image files
            self.image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.image_files.extend(list(self.data_dir.glob(ext)))
            
            # Ensure we have the same number of prompts as images
            if len(self.prompts) != len(self.image_files):
                # Repeat prompts if needed
                self.prompts = self.prompts * (len(self.image_files) // len(self.prompts) + 1)
                self.prompts = self.prompts[:len(self.image_files)]
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Get prompt
        prompt = self.prompts[idx]
        
        return {
            "instance_images": image,
            "instance_prompt": prompt
        }


def collate_fn(examples, with_prior_preservation=False):
    """
    Collate function compatible with Qwen - adds num_frames dimension
    """
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    # Qwen expects a `num_frames` dimension too.
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(2)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch


class QwenImageDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Qwen Image training
    Optimized for distributed training and memory efficiency
    """
    
    def __init__(
        self,
        config_path: str,
        batch_size: int = 2,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        with_prior_preservation: bool = False,
        size: int = 512,
        center_crop: bool = False,
        train_split: float = 0.9,
        **kwargs
    ):
        super().__init__()
        self.config_path = config_path
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.with_prior_preservation = with_prior_preservation
        self.size = size
        self.center_crop = center_crop
        self.train_split = train_split
        self.kwargs = kwargs
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.full_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation"""
        if stage == "fit" or stage is None:
            # Create full dataset
            self.full_dataset = SimpleDataset(
                self.config_path,
                size=self.size,
                center_crop=self.center_crop,
                **self.kwargs
            )
            
            # Split into train and validation
            dataset_size = len(self.full_dataset)
            train_size = int(self.train_split * dataset_size)
            val_size = dataset_size - train_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
    
    def train_dataloader(self):
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, self.with_prior_preservation),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=True,  # Important for distributed training
        )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=lambda examples: collate_fn(examples, self.with_prior_preservation),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
        )
    
    def predict_dataloader(self):
        """Create prediction dataloader"""
        return self.val_dataloader()
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up datasets"""
        self.train_dataset = None
        self.val_dataset = None
        self.full_dataset = None 