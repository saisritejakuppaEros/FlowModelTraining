import os

# Set NCCL environment variables before importing PyTorch
os.environ['NCCL_NVLS_ENABLE'] = '0'
os.environ['NCCL_TREE_THRESHOLD'] = '0'
os.environ['NCCL_NET_GDR_LEVEL'] = '0'
os.environ['NCCL_P2P_LEVEL'] = 'SYS'
os.environ['NCCL_SHM_DISABLE'] = '0'
os.environ['NCCL_ALGO'] = 'Ring'
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['NCCL_DEBUG'] = 'WARN'

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft
from einops import rearrange, repeat
import gc
from accelerate import Accelerator
from streaming import MDSWriter, StreamingDataset
from streaming.base.util import merge_index
from tqdm import tqdm
from dataclasses import dataclass

# Import AutoEncoder from your existing code
from autoencoder import AutoEncoder


# ----------------------- 
# Model Configuration
# -----------------------
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


# ----------------------- 
# FLUX HF Embedder
# -----------------------
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


# ----------------------- 
# Model Loading
# -----------------------
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
    
    # Create image position IDs for the patchified image
    patch_h, patch_w = h // 2, w // 2
    img_ids = torch.zeros(patch_h, patch_w, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(patch_h)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(patch_w)[None, :]
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
# Image Preprocessing
# -----------------------
class CenterCropResize:
    """Center-crop to 1024x1024 then resize to target size"""
    def __init__(self, target_size: int = 256):
        self.crop_size = 1024
        self.target_size = target_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        
        # Find the minimum dimension to ensure we can crop to 1024x1024
        min_dim = min(w, h)
        if min_dim < self.crop_size:
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


# ----------------------- 
# FLUX Latent Generator
# -----------------------
class FLUXLatentGenerator:
    """
    Generate FLUX latents from MDS dataset with images and captions
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
    ):
        self.resolution = resolution
        self.max_length_clip = max_length_clip
        self.max_length_t5 = max_length_t5
        
        # Set dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
            self.numpy_dtype = np.float32
        else:
            self.torch_dtype = torch.float16
            self.numpy_dtype = np.float32
        
        # Set device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires GPU.")
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0")
        
        print(f"Using device: {self.device}")
        
        # Load models
        self._load_models(ae_name, clip_model, t5_model)
        
        # Image transforms
        self.image_transform = CenterCropResize(target_size=resolution)
        self.to_tensor = transforms.ToTensor()
        
        # FLUX scaling factors
        self.ae_params = configs[ae_name].ae_params
    
    def _load_models(self, ae_name: str, clip_model: str, t5_model: str):
        """Load models with proper memory management"""
        print(f"Loading models on device: {self.device}")
        
        # Load FLUX AutoEncoder
        self.ae = load_flux_ae(ae_name, device=self.device)
        self.ae = self.ae.to(self.torch_dtype).eval()
        
        # Load text encoders
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
        
        # Clear cache
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
    
    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process one sample from MDS dataset"""
        try:
            # Extract image and caption from MDS sample
            img = sample['jpg']  # PIL Image from MDS
            caption = sample['caption']  # String caption from MDS
            width = sample.get('width', img.size[0])
            height = sample.get('height', img.size[1])
            
            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Process image
            img = self.image_transform(img)
            tensor = self.to_tensor(img)
            
            # Convert to tensor for encoding
            img_tensor = tensor.unsqueeze(0).to(self.device, dtype=self.torch_dtype)
            
            with torch.no_grad():
                # Encode image with FLUX VAE
                img_latents = self.ae.encode(img_tensor)
                
                # Text embeddings
                clip_embeds = self.clip_encoder([caption])
                t5_embeds = self.t5_encoder([caption])
                
                # Prepare FLUX format
                flux_batch = prepare_flux_batch(t5_embeds, clip_embeds, img_latents)
            
            # Convert to numpy
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
                "caption_text": caption,
                "processed_image": tensor.numpy().astype(np.float32),
                "original_width": width,
                "original_height": height,
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None


def generate_latents_partition(
    input_mds_path: str,
    output_dir: str,
    config: Dict,
    accelerator: Accelerator
) -> None:
    """Generate latents for a partition of the dataset"""
    
    device = accelerator.device
    process_index = accelerator.process_index
    num_processes = accelerator.num_processes
    
    print(f"Process {process_index}: Loading dataset from {input_mds_path}")
    
    # Load MDS dataset
    dataset = StreamingDataset(
        local=input_mds_path,
        shuffle=False,
        batch_size=1
    )
    
    total_samples = len(dataset)
    print(f"Total samples in dataset: {total_samples}")
    
    # Split dataset across processes
    samples_per_process = total_samples // num_processes
    start_idx = process_index * samples_per_process
    if process_index == num_processes - 1:
        end_idx = total_samples
    else:
        end_idx = (process_index + 1) * samples_per_process
    
    print(f"Process {process_index}: Processing samples {start_idx}-{end_idx-1} on device {device}")
    
    # Initialize FLUX latent generator
    generator = FLUXLatentGenerator(
        resolution=config.get("resolution", 256),
        ae_name=config.get("ae_name", "flux-schnell"),
        clip_model=config.get("clip_model", "openai/clip-vit-large-patch14"),
        t5_model=config.get("t5_model", "xlabs-ai/xflux_text_encoders"),
        max_length_clip=config.get("max_length_clip", 77),
        max_length_t5=config.get("max_length_t5", 512),
        dtype=config.get("dtype", "bfloat16"),
        device=str(device),
    )
    
    # Calculate dimensions
    resolution = config.get("resolution", 256)
    max_length_clip = config.get("max_length_clip", 77)
    max_length_t5 = config.get("max_length_t5", 512)
    
    latent_h = latent_w = resolution // 8
    flux_img_tokens = (latent_h // 2) * (latent_w // 2)
    flux_img_dim = 16 * 4
    
    # Define MDS columns
    columns = {
        'img_latents': f'ndarray:float32:{flux_img_tokens},{flux_img_dim}',
        'img_ids': f'ndarray:float32:{flux_img_tokens},3',
        'txt_embeds': f'ndarray:float32:{max_length_t5},4096',
        'txt_ids': f'ndarray:float32:{max_length_t5},3',
        'vec_embeds': f'ndarray:float32:768',
        'clip_embeddings': f'ndarray:float32:768',
        't5_embeddings': f'ndarray:float32:{max_length_t5},4096',
        'raw_img_latents': f'ndarray:float32:16,{latent_h},{latent_w}',
        'caption_text': 'str',
        'processed_image': f'ndarray:float32:3,{resolution},{resolution}',
        'original_width': 'int32',
        'original_height': 'int32',
    }
    
    # Output directory for this process
    sub_out_root = os.path.join(output_dir, str(process_index))
    
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
            # Process assigned samples
            for idx in tqdm(range(start_idx, end_idx), desc=f"Process {process_index}"):
                try:
                    sample = dataset[idx]
                    result = generator.process_sample(sample)
                    
                    if result is not None:
                        out.write(result)
                        successful_samples += 1
                    else:
                        failed_samples += 1
                    
                    # Clear cache periodically
                    if (idx - start_idx + 1) % 100 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing sample {idx} in process {process_index}: {e}")
                    failed_samples += 1
                    continue
    finally:
        generator.cleanup()
    
    print(f"Process {process_index} completed: {successful_samples} successful, {failed_samples} failed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate FLUX latents from MDS dataset")
    parser.add_argument("--input_mds", type=str, required=True, help="Path to input MDS dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for latents MDS")
    parser.add_argument("--resolution", type=int, default=256, help="Target resolution")
    parser.add_argument("--ae_name", type=str, default="flux-schnell", help="AutoEncoder name")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--t5_model", type=str, default="xlabs-ai/xflux_text_encoders")
    parser.add_argument("--max_length_clip", type=int, default=77)
    parser.add_argument("--max_length_t5", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--compression", type=str, default="zstd:3")
    parser.add_argument("--size_limit", type=str, default="100mb")
    
    args = parser.parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    print(f"Using device: {accelerator.device}")
    print(f"Number of processes: {accelerator.num_processes}")
    print(f"Process index: {accelerator.process_index}")
    
    # Configuration
    config = {
        "resolution": args.resolution,
        "ae_name": args.ae_name,
        "clip_model": args.clip_model,
        "t5_model": args.t5_model,
        "max_length_clip": args.max_length_clip,
        "max_length_t5": args.max_length_t5,
        "dtype": args.dtype,
        "compression": args.compression,
        "hashes": ["sha256"],
        "size_limit": args.size_limit,
    }
    
    # Generate latents
    print(f"Generating FLUX latents from {args.input_mds} to {args.output_dir}")
    generate_latents_partition(args.input_mds, args.output_dir, config, accelerator)
    
    # Wait for all processes
    accelerator.wait_for_everyone()
    
    # Merge shards (only main process)
    if accelerator.is_main_process:
        print("Merging shards from all processes...")
        import time
        time.sleep(5)
        
        shards_metadata = [
            os.path.join(args.output_dir, str(i), 'index.json')
            for i in range(accelerator.num_processes)
        ]
        
        existing_shards = [shard for shard in shards_metadata if os.path.exists(shard)]
        print(f"Found {len(existing_shards)} shards to merge")
        
        if existing_shards:
            merge_index(existing_shards, out=args.output_dir, keep_local=True)
            print("Latent generation and merging completed!")
            
            # Test loading
            print("Testing dataset loading...")
            try:
                dataset = StreamingDataset(local=args.output_dir, shuffle=False, batch_size=1)
                sample = next(iter(dataset))
                
                print("Sample keys:", list(sample.keys()))
                for key in ["img_latents", "txt_embeds", "vec_embeds"]:
                    if key in sample:
                        print(f"{key} shape:", sample[key].shape)
                        
                print(f"Dataset contains {len(dataset)} samples")
                
            except Exception as e:
                print(f"Error testing dataset: {e}")
        else:
            print("No shards found to merge")


if __name__ == "__main__":
    main()