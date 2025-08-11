
from logzero import logger
from PIL import Image  
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler



class DatasetAugmentation:
    def __init__(self, dataset_configs):
        self.dataset_configs = dataset_configs
        
    def crop_to_square(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        min_dim = min(width, height)
        left, top, right, bottom = (
            (width - min_dim) // 2,
            (height - min_dim) // 2,
            (width + min_dim) // 2,
            (height + min_dim) // 2,
        )
        return image.crop((left, top, right, bottom))
    
    def get_all_augmentations(self):

        # Get image size from config (should be [width, height])
        image_size = self.dataset_configs.get("image_size", [512, 512])
        width, height = image_size

        # Always start with Resize
        self.image_transforms = [transforms.Resize((height, width))]

        # Add other augmentations in the order specified in config, skipping Resize (already added)
        for aug in self.dataset_configs.get("augmentation", []):
            if aug == "Resize":
                continue  # already handled
            elif aug == "RandomHorizontalFlip":
                self.image_transforms.append(transforms.RandomHorizontalFlip())
            elif aug == "Normalize":
                # Using ImageNet mean/std as default; adjust as needed
                self.image_transforms.append(
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                )
            # Add more augmentations here as needed

        # Always add ToTensor transform before Normalize if present, otherwise at the end
        normalize_idx = None
        for i, transform in enumerate(self.image_transforms):
            if isinstance(transform, transforms.Normalize):
                normalize_idx = i
                break
        
        if normalize_idx is not None:
            self.image_transforms.insert(normalize_idx, transforms.ToTensor())
        else:
            self.image_transforms.append(transforms.ToTensor())

        # Compose all transforms
        self.transform = transforms.Compose(self.image_transforms)
        return self.transform




from torch.utils.data import DataLoader
from einops import rearrange
import torch


def imagenet_collate_fn(batch: list[dict]) -> dict:
    return {
        "raw_images": rearrange(torch.stack([item["raw_images"] for item in batch]), "b c h w -> b c 1 h w"),
        "raw_texts": [item["raw_texts"] for item in batch],
    }




class ImageCaptionModule:
    def __init__(self, dataset_configs):
        self.dataset_configs = dataset_configs
        self.train_images_path = dataset_configs["train_images_path"]
        self.val_images_path = dataset_configs["val_images_path"]
        self.train_captions_path = dataset_configs["train_captions_path"]
        self.val_captions_path = dataset_configs["val_captions_path"]
        self.test_captions_path = dataset_configs["test_captions_path"]
        self.test_images_path = dataset_configs["test_images_path"]
        
        
    def get_dataloader(self):
        """ need to get the train, val and test dataloaders"""
        train_dataset = ImageCaptionDatabase(self.train_captions_path, self.train_images_path, self.dataset_configs)
        val_dataset = ImageCaptionDatabase(self.val_captions_path, self.val_images_path, self.dataset_configs)
        test_dataset = ImageCaptionDatabase(self.test_captions_path, self.test_images_path, self.dataset_configs)

        # Initialize datasets
        train_dataset.load_dataset()
        val_dataset.load_dataset()
        test_dataset.load_dataset()

        train_bs= self.dataset_configs["train_bs"]
        val_bs= self.dataset_configs["val_bs"]
        test_bs= self.dataset_configs["test_bs"]

        if dist.is_initialized():
            logger.info(f"Using distributed training with {dist.get_world_size()} GPUs")
            
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False   
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False
            )

            train_dataloader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, collate_fn=imagenet_collate_fn)
            val_dataloader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, collate_fn=imagenet_collate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=test_bs, sampler=test_sampler, collate_fn=imagenet_collate_fn)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, collate_fn=imagenet_collate_fn)
            val_dataloader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, collate_fn=imagenet_collate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False, collate_fn=imagenet_collate_fn)


    
        return train_dataloader, val_dataloader, test_dataloader
    
    
    
    

import os

class ImageCaptionDatabase:
    def __init__(self, caption_path, images_path, dataset_configs):
        self.caption_path = caption_path
        self.images_path = images_path

        # get all the caption and images paths
        self.caption_files = os.listdir(self.caption_path)
        self.image_files = os.listdir(self.images_path)
        self.image_size = dataset_configs["image_size"]
        self.image_transforms = DatasetAugmentation(dataset_configs).get_all_augmentations()

    def CaptionImageSanity(self):
        # Create base names (without extensions) for comparison
        caption_base_names = set()
        image_base_names = set()
        
        # Extract base names from caption files (.txt files)
        for caption_file in self.caption_files:
            if caption_file.endswith('.txt'):
                base_name = os.path.splitext(caption_file)[0]
                caption_base_names.add(base_name)
        
        # Extract base names from image files  
        for image_file in self.image_files:
            if any(image_file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']):
                base_name = os.path.splitext(image_file)[0]
                image_base_names.add(base_name)
        
        # Find common base names
        common_base_names = caption_base_names.intersection(image_base_names)
        
        # Filter files to keep only those with common base names
        self.caption_files = [f for f in self.caption_files 
                             if f.endswith('.txt') and os.path.splitext(f)[0] in common_base_names]
        self.image_files = [f for f in self.image_files 
                           if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'])
                           and os.path.splitext(f)[0] in common_base_names]
        
        # Sort both lists to ensure matching order
        self.caption_files.sort()
        self.image_files.sort(key=lambda x: os.path.splitext(x)[0])
        
        logger.info(f"Found {len(common_base_names)} matching image-caption pairs")

    def load_dataset(self):
        self.CaptionImageSanity()
        return self.caption_files, self.image_files


    def __getitem__(self, idx):
        caption_file = self.caption_files[idx]
        image_file = self.image_files[idx]
        
        caption_path = os.path.join(self.caption_path, caption_file)
        image_path = os.path.join(self.images_path, image_file)
        
        # Read caption
        with open(caption_path, "r", encoding='utf-8') as f:
            caption = f.read().strip()
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms (ToTensor is now included in the transform pipeline)
        image = self.image_transforms(image)
        
        # Ensure image is in [-1, 1] if Normalize is used
        if hasattr(image, "clip"):
            image = image.clip(-1.0, 1.0)
        
        return {
            "raw_images": image,
            "raw_texts": caption,
        }



    def __len__(self):
        return len(self.caption_files)
    
    