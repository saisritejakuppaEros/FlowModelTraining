import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from pathlib import Path
from torchvision import transforms


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


def create_dataloader(config_path, batch_size=4, shuffle=True, num_workers=0, 
                     with_prior_preservation=False, **kwargs):
    """
    Create a dataloader from config file
    
    Args:
        config_path: Path to config.yaml file
        batch_size: Batch size for dataloader
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading
        with_prior_preservation: Whether to use prior preservation
        **kwargs: Additional arguments for dataset
    
    Returns:
        DataLoader instance
    """
    dataset = SimpleDataset(config_path, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda examples: collate_fn(examples, with_prior_preservation),
        num_workers=num_workers
    )
    
    return dataloader



# if __name__ == "__main__":
#     dataloader = create_dataloader("config.yaml")
#     for batch in dataloader:
#         print(batch)
#         break