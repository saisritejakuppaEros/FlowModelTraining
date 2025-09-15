import os
import json
import yaml
from typing import Dict, List, Optional, Iterator, Tuple, Any
from multiprocessing import Pool
from glob import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import (
    CLIPTokenizer, 
    CLIPTextModel, 
    T5TokenizerFast, 
    T5EncoderModel
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft
from autoencoder import AutoEncoder
from einops import rearrange, repeat
import pandas as pd
import gc

from streaming import MDSWriter, StreamingDataset
from streaming.base.util import merge_index

# ----------------------- 
# Utilities (stateless)
# -----------------------
def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""

def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    return img.convert("RGB")

class CenterCropResize:
    """Center-crop to 1024x1024 then resize to target size"""
    def __init__(self, target_size: int = 256):
        self.crop_size = 1024
        self.target_size = target_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        # First, center crop to 1024x1024
        w, h = img.size
        
        # Find the minimum dimension to ensure we can crop to 1024x1024
        min_dim = min(w, h)
        if min_dim < self.crop_size:
            # If image is smaller than 1024, resize maintaining aspect ratio first
            if w < h:
                new_w = self.crop_size
                new_h = int(h * (self.crop_size / w))
            else:
                new_h = self.crop_size
                new_w = int(w * (self.crop_size / h))
            img = img.resize((new_w, new_h), resample=Image.BICUBIC)
            w, h = img.size
        
        # Center crop to 1024x1024
        left = (w - self.crop_size) // 2
        top = (h - self.crop_size) // 2
        img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        
        # Resize to target size
        if self.crop_size != self.target_size:
            img = img.resize((self.target_size, self.target_size), resample=Image.BICUBIC)
        
        return img

# FLUX HF Embedder
class HFEmbedder(torch.nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer = T5TokenizerFast.from_pretrained(version, max_length=max_length)
            self.hf_module = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: List[str]) -> torch.Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]

# Model configurations
from dataclasses import dataclass

@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: List[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float

@dataclass
class ModelSpec:
    ae_params: AutoEncoderParams
    repo_id: str
    repo_id_ae: str
    repo_ae: str

configs = {
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-schnell",
        repo_ae="ae.safetensors",
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    )
}

def load_flux_ae(name: str = "flux-schnell", device: str = "cuda", hf_download: bool = True):
    """Load FLUX AutoEncoder"""
    ckpt_path = os.getenv("AE")
    if (
        ckpt_path is None
        and configs[name].repo_id_ae is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id_ae, configs[name].repo_ae)

    print("Init FLUX AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        if missing or unexpected:
            print(f"AE load - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    return ae

def prepare_flux_batch(t5_embeds: torch.Tensor, clip_embeds: torch.Tensor, img_latents: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Prepare batch in FLUX format with proper spatial arrangements"""
    bs, c, h, w = img_latents.shape
    
    # Rearrange image latents from (b c h w) to (b h*w c*ph*pw) for FLUX
    img = rearrange(img_latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    
    # Create image position IDs
    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    
    # Create text position IDs
    txt_ids = torch.zeros(bs, t5_embeds.shape[1], 3)
    
    return {
        "img": img,
        "img_ids": img_ids.to(img_latents.device),
        "txt": t5_embeds,
        "txt_ids": txt_ids.to(t5_embeds.device),
        "vec": clip_embeds,
    }

# ----------------------- 
# FLUX Processing Class
# -----------------------
class FLUXProcessor:
    """
    FLUX processor for MosaicML Streaming:
    - Handles CLIP and T5 tokenization
    - Processes images for FLUX (center crop 1024 -> resize 256)
    - Encodes with FLUX models
    - Supports multi-GPU with manual GPU assignment
    """
    
    def __init__(
        self,
        resolution: int = 256,
        ae_name: str = "flux-schnell",
        clip_model: str = "openai/clip-vit-large-patch14",
        t5_model: str = "xlabs-ai/xflux_text_encoders",
        max_length_clip: int = 77,
        max_length_t5: int = 512,
        dtype: str = "bfloat16",
        device: Optional[str] = None,
        min_aesthetic: Optional[float] = None,
        drop_invalid: bool = True,
        gpu_id: Optional[int] = None,
    ):
        self.resolution = resolution
        self.max_length_clip = max_length_clip
        self.max_length_t5 = max_length_t5
        self.min_aesthetic = min_aesthetic
        self.drop_invalid = drop_invalid
        
        # Set dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
            self.numpy_dtype = np.float16
        else:
            self.torch_dtype = torch.float16
            self.numpy_dtype = np.float16
        
        # Set device - use gpu_id if provided, otherwise device parameter
        if gpu_id is not None:
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id}: {self.device}")
        elif device:
            self.device = torch.device(device)
            print(f"Using device: {self.device}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using default device: {self.device}")
        
        # Load models with proper device placement
        self._load_models(ae_name, clip_model, t5_model)
        
        # Initialize tokenizers for preprocessing
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        
        # Image transforms - center crop to 1024x1024 then resize to target
        self.image_transform = CenterCropResize(target_size=resolution)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Lambda(lambda x: x)  # Keep in [0,1] range
        
        # FLUX scaling factors
        self.ae_params = configs[ae_name].ae_params
    
    def _load_models(self, ae_name: str, clip_model: str, t5_model: str):
        """Load models with proper memory management"""
        print(f"Loading models on device: {self.device}")
        
        # Load FLUX AutoEncoder
        self.ae = load_flux_ae(ae_name, device=self.device)
        self.ae = self.ae.to(self.torch_dtype).eval()
        
        # Load text encoders using HFEmbedder
        self.clip_encoder = HFEmbedder(
            clip_model, 
            max_length=self.max_length_clip, 
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        self.t5_encoder = HFEmbedder(
            t5_model,
            max_length=self.max_length_t5,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def cleanup(self):
        """Clean up GPU memory"""
        del self.ae
        del self.clip_encoder  
        del self.t5_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def process_sample(self, img_path: str, caption: str, aesthetic_score: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Process one image-caption pair"""
        
        # Debug first sample
        debug_mode = False  # Set to True for debugging
        
        # Filter on aesthetic score
        if self.min_aesthetic is not None and aesthetic_score is not None:
            if float(aesthetic_score) < float(self.min_aesthetic):
                if debug_mode:
                    print(f"Filtered by aesthetic score: {aesthetic_score} < {self.min_aesthetic}")
                return None
        
        # Caption guard
        if not caption:
            if self.drop_invalid:
                if debug_mode:
                    print(f"Dropped for empty caption")
                return None
            caption = ""
        
        # Load and validate image
        if not img_path or not os.path.exists(img_path):
            if self.drop_invalid:
                if debug_mode:
                    print(f"Dropped for missing image: {img_path}")
                return None
            img = Image.new("RGB", (1024, 1024), (0, 0, 0))  # Use 1024 for consistency with crop
        else:
            try:
                img = Image.open(img_path)
                # Check minimum size for center crop
                w, h = img.size
                min_dim = min(w, h)
                if min_dim < 256:  # Too small even after processing
                    if self.drop_invalid:
                        if debug_mode:
                            print(f"Dropped for small image: {w}x{h}")
                        return None
            except Exception as e:
                if self.drop_invalid:
                    if debug_mode:
                        print(f"Dropped for image load error: {e}")
                    return None
                img = Image.new("RGB", (1024, 1024), (0, 0, 0))
        
        # Process image - center crop to 1024x1024 then resize to target
        img = _ensure_rgb(img)
        img = self.image_transform(img)
        tensor = self.to_tensor(img)
        tensor = self.normalize(tensor)
        
        # Convert to tensor for encoding
        img_tensor = tensor.unsqueeze(0).to(self.device, dtype=self.torch_dtype)
        
        with torch.no_grad():
            # Encode image with FLUX VAE
            img_latents = self.ae.encode(img_tensor)
            # Apply FLUX scaling and shifting
            img_latents = (img_latents - self.ae_params.shift_factor) * self.ae_params.scale_factor
            
            # Text embeddings
            clip_embeds = self.clip_encoder([caption])
            t5_embeds = self.t5_encoder([caption])
            
            # Prepare FLUX format
            flux_batch = prepare_flux_batch(t5_embeds, clip_embeds, img_latents)
        
        # Convert to numpy with proper dtype handling
        def to_numpy(tensor):
            if self.torch_dtype == torch.bfloat16:
                return tensor.detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
            else:
                return tensor.detach().cpu().numpy().astype(self.numpy_dtype)
        
        result = {
            "img_latents": to_numpy(flux_batch["img"].squeeze(0)),
            "img_ids": to_numpy(flux_batch["img_ids"].squeeze(0)),
            "txt_embeds": to_numpy(flux_batch["txt"].squeeze(0)),
            "txt_ids": to_numpy(flux_batch["txt_ids"].squeeze(0)),
            "vec_embeds": to_numpy(flux_batch["vec"].squeeze(0)),
            "clip_embeddings": to_numpy(clip_embeds.squeeze(0)),
            "t5_embeddings": to_numpy(t5_embeds.squeeze(0)),
            "raw_img_latents": to_numpy(img_latents.squeeze(0)),
            "aesthetic_score": float(aesthetic_score) if aesthetic_score is not None else np.nan,
            "caption_text": caption,
            "processed_image": tensor.numpy().astype(np.float32),  # Keep processed image
        }
        
        return result

def convert_dataset_partition_multigpu(samples: List[Dict], config: Dict, output_dir: str, partition_id: int, gpu_id: int) -> None:
    """Convert a partition of the dataset to MDS format using specific GPU"""
    
    sub_out_root = os.path.join(output_dir, str(partition_id))
    
    print(f"Processing partition {partition_id} with {len(samples)} samples on GPU {gpu_id}")
    
    # Initialize FLUX processor with specific GPU
    processor = FLUXProcessor(
        resolution=config.get("resolution", 256),
        ae_name=config.get("ae_name", "flux-schnell"),
        clip_model=config.get("clip_model", "openai/clip-vit-large-patch14"),
        t5_model=config.get("t5_model", "xlabs-ai/xflux_text_encoders"),
        max_length_clip=config.get("max_length_clip", 77),
        max_length_t5=config.get("max_length_t5", 512),
        dtype=config.get("dtype", "bfloat16"),
        min_aesthetic=config.get("min_aesthetic", None),
        drop_invalid=config.get("drop_invalid", True),
        gpu_id=gpu_id,
    )
    
    # Define MDS columns - using efficient numpy array encodings
    resolution = config.get("resolution", 256)
    max_length_clip = config.get("max_length_clip", 77)
    max_length_t5 = config.get("max_length_t5", 512)
    
    # Calculate latent dimensions for FLUX
    latent_h = latent_w = resolution // 8  # FLUX uses 8x downsampling
    flux_img_tokens = (latent_h // 2) * (latent_w // 2)  # After patchification
    flux_img_dim = 16 * 4  # c * ph * pw where ph=pw=2
    
    columns = {
        'img_latents': f'ndarray:float16:{flux_img_tokens},{flux_img_dim}',
        'img_ids': f'ndarray:float16:{flux_img_tokens},3',
        'txt_embeds': f'ndarray:float16:{max_length_t5},4096',  # T5 embedding dim
        'txt_ids': f'ndarray:float16:{max_length_t5},3',
        'vec_embeds': f'ndarray:float16:768',  # CLIP embedding dim
        'clip_embeddings': f'ndarray:float16:768',
        't5_embeddings': f'ndarray:float16:{max_length_t5},4096',
        'raw_img_latents': f'ndarray:float16:16,{latent_h},{latent_w}',
        'aesthetic_score': 'float32',
        'caption_text': 'str',
        'processed_image': f'ndarray:float32:3,{resolution},{resolution}',
    }
    
    # MDS Writer configuration
    mds_kwargs = {
        'out': sub_out_root,
        'columns': columns,
        'compression': config.get("compression", "zstd:3"),
        'hashes': config.get("hashes", ["sha256"]),
        'size_limit': config.get("size_limit", "100mb"),
    }
    
    successful_samples = 0
    failed_samples = 0
    
    try:
        with MDSWriter(**mds_kwargs) as out:
            for i, sample in enumerate(samples):
                try:
                    img_path = sample.get('image_path') or sample.get('img_path') or sample.get('image')
                    caption = sample.get('caption_text') or sample.get('caption') or sample.get('text') or sample.get('captions')
                    aesthetic_score = sample.get('aesthetic_score')
                    
                    # Check if caption is a file path
                    if caption and ('/' in str(caption) or '.txt' in str(caption)):
                        caption = _read_text_file(caption)
                    
                    result = processor.process_sample(img_path, caption, aesthetic_score)
                    
                    if result is not None:
                        out.write(result)
                        successful_samples += 1
                    else:
                        failed_samples += 1
                        # Debug first few failures
                        if failed_samples <= 5:
                            print(f"Sample {i} failed - img_path: {img_path}, caption: {caption[:50] if caption else 'None'}...")
                        
                    if (i + 1) % 100 == 0:
                        print(f"Partition {partition_id}: Processed {i+1}/{len(samples)} samples "
                              f"(Success: {successful_samples}, Failed: {failed_samples})")
                        # Clear cache periodically
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing sample {i} in partition {partition_id}: {e}")
                    print(f"Sample data: img_path={img_path}, caption={caption[:50] if caption else 'None'}...")
                    failed_samples += 1
                    continue
    finally:
        # Clean up processor to free GPU memory
        processor.cleanup()
    
    print(f"Partition {partition_id} completed: {successful_samples} successful, {failed_samples} failed")

def create_dataset_partitions(data_path: str, num_partitions: int, max_samples: int = None) -> List[List[Dict]]:
    """Create partitions from the dataset"""
    # Load dataset
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        samples = df.to_dict('records')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            samples = json.load(f)
    else:
        raise ValueError("Unsupported file format. Use .csv or .json")
    
    # Limit number of samples if specified
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]
        print(f"Limited dataset to first {max_samples} samples (from {len(samples)} total)")
    
    # Create partitions
    partition_size = len(samples) // num_partitions
    partitions = []
    
    for i in range(num_partitions):
        start_idx = i * partition_size
        if i == num_partitions - 1:  # Last partition gets remaining samples
            end_idx = len(samples)
        else:
            end_idx = (i + 1) * partition_size
        
        partition = samples[start_idx:end_idx]
        partitions.append(partition)
    
    return partitions

def main():
    # Load configuration
    try:
        with open("config_flux_mosaicml.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Default config
        config = {
            "data_path": "image_captions.csv",
            "output_dir": "./flux_mds_dataset",
            "resolution": 256,  # Final resolution after center crop 1024->256
            "num_partitions": 3,  # One per GPU (5, 6, 7)
            "gpu_ids": [5, 6, 7],  # Specific GPUs to use
            "ae_name": "flux-schnell",
            "clip_model": "openai/clip-vit-large-patch14",
            "t5_model": "xlabs-ai/xflux_text_encoders",
            "max_length_clip": 77,
            "max_length_t5": 512,
            "dtype": "bfloat16",
            "min_aesthetic": None,
            "drop_invalid": True,
            "compression": "zstd:3",
            "hashes": ["sha256"],
            "size_limit": "100mb",
            "max_samples": 50000,
        }
        print("Using default configuration. Create config_flux_mosaicml.yaml to customize.")
    
    data_path = config["data_path"]
    output_dir = config["output_dir"]
    num_partitions = config["num_partitions"]
    gpu_ids = config.get("gpu_ids", [5, 6, 7])
    
    print(f"Converting dataset from {data_path} to MDS format at {output_dir}")
    print(f"Using center crop 1024x1024 -> resize {config['resolution']}x{config['resolution']}")
    print(f"Using GPUs: {gpu_ids}")
    
    # Ensure we have enough GPUs
    if len(gpu_ids) < num_partitions:
        print(f"Warning: Only {len(gpu_ids)} GPUs available for {num_partitions} partitions. Some GPUs will process multiple partitions.")
    
    # Create partitions with limit
    max_samples = config.get("max_samples", 50000)
    print(f"Creating dataset partitions (max {max_samples} samples)...")
    partitions = create_dataset_partitions(data_path, num_partitions, max_samples)
    
    # Clean up output directory
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    
    # Prepare arguments for multiprocessing with GPU assignment
    args_list = []
    for i, partition in enumerate(partitions):
        gpu_id = gpu_ids[i % len(gpu_ids)]  # Cycle through available GPUs
        args_list.append((partition, config, output_dir, i, gpu_id))
    
    print(f"Processing {num_partitions} partitions with multiprocessing on GPUs {gpu_ids}...")
    
    # Use multiprocessing with manual GPU assignment
    with Pool(processes=min(num_partitions, len(gpu_ids))) as pool:
        pool.starmap(convert_dataset_partition_multigpu, args_list)
    
    # Merge index files
    print("Merging partition indices...")
    merge_index(output_dir, keep_local=True)
    
    print("Dataset conversion completed!")
    
    # Test loading
    print("Testing dataset loading...")
    try:
        dataset = StreamingDataset(local=output_dir, shuffle=False, batch_size=1)
        sample = next(iter(dataset))
        
        print("Sample keys:", list(sample.keys()))
        for key in ["img_latents", "txt_embeds", "vec_embeds"]:
            if key in sample:
                print(f"{key} shape:", sample[key].shape)
                
        print(f"Dataset contains {len(dataset)} samples")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")

if __name__ == "__main__":
    main()