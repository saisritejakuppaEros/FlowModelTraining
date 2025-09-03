
import os
import io
from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image
import ray
import ray.data as rd
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

# ----------------------- 
# Utilities (stateless)
# -----------------------
def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""

def _read_image_bytes(path: str) -> str:
    """Return the image path instead of reading bytes to preserve original format"""
    try:
        return path
    except Exception:
        return ""

def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    return img.convert("RGB")

class LargestCenterSquare:
    """Center-crop the largest square then (optionally) resize to target."""
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        if s != self.size:
            img = img.resize((self.size, self.size), resample=Image.BICUBIC)
        return img

class FlexibleResize:
    """Resize to target resolution maintaining aspect ratio or exact size"""
    def __init__(self, target_height: int, target_width: Optional[int] = None, maintain_aspect: bool = True):
        self.target_height = target_height
        self.target_width = target_width if target_width else target_height
        self.maintain_aspect = maintain_aspect
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if self.maintain_aspect:
            # Maintain aspect ratio, resize shortest side to target
            w, h = img.size
            if w < h:
                new_w = self.target_width
                new_h = int(h * (self.target_width / w))
            else:
                new_h = self.target_height
                new_w = int(w * (self.target_height / h))
            img = img.resize((new_w, new_h), resample=Image.BICUBIC)
        else:
            # Exact resize
            img = img.resize((self.target_width, self.target_height), resample=Image.BICUBIC)
        return img

# FLUX HF Embedder (from your second document)
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

# Model configurations (from your second document)
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

def validate_image_paths_batch(
    batch: Dict[str, np.ndarray],
    image_col: str = "image_path",
    min_side: int = 8,
) -> Dict[str, np.ndarray]:
    """Validate images by path without loading full bytes into memory."""
    
    paths = batch.get(image_col)
    if paths is None:
        n = len(next(iter(batch.values()))) if batch else 0
        result = dict(batch)
        result.update({
            "is_image_valid": np.zeros((n,), dtype=bool),
            "invalid_reason": np.array(["missing_path_column"] * n, dtype=object),
            "image_width": np.zeros((n,), dtype=np.int32),
            "image_height": np.zeros((n,), dtype=np.int32),
        })
        return result
    
    is_valid = []
    reason = []
    widths = []
    heights = []
    
    for path in paths.tolist():
        if not path or not os.path.exists(path):
            is_valid.append(False)
            reason.append("path_not_found")
            widths.append(0)
            heights.append(0)
            continue
            
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w < min_side or h < min_side:
                    is_valid.append(False)
                    reason.append("too_small")
                else:
                    is_valid.append(True)
                    reason.append("")
                widths.append(w)
                heights.append(h)
                
        except Exception as e:
            is_valid.append(False)
            reason.append(type(e).__name__)
            widths.append(0)
            heights.append(0)
    
    result = dict(batch)
    result.update({
        "is_image_valid": np.array(is_valid, dtype=bool),
        "invalid_reason": np.array(reason, dtype=object),
        "image_width": np.array(widths, dtype=np.int32),
        "image_height": np.array(heights, dtype=np.int32),
    })
    return result

# --------------------------------------- 
# Stage 1: file IO (stateless map_batches)
# ---------------------------------------
def load_files_from_paths(batch: Dict[str, np.ndarray], 
                         caption_col: str = "caption_path", 
                         image_col: str = "image_path") -> Dict[str, Any]:
    """Given a batch of paths, load raw text and image paths."""
    available_columns = list(batch.keys())
    
    # Find caption column
    caption_col_found = None
    for col in available_columns:
        if col.lower() in ['caption_path', 'caption', 'text', 'caption_text']:
            caption_col_found = col
            break
    
    if caption_col_found is None:
        raise KeyError(f"Caption column not found. Available columns: {available_columns}")
    
    # Find image column
    image_col_found = None
    for col in available_columns:
        if col.lower() in ['image_path', 'image', 'img_path', 'img']:
            image_col_found = col
            break
    
    if image_col_found is None:
        raise KeyError(f"Image column not found. Available columns: {available_columns}")
    
    cap_paths = batch[caption_col_found].tolist()
    img_paths = batch[image_col_found].tolist()
    
    # Check if caption_path contains actual text or file paths
    if cap_paths and ('/' in str(cap_paths[0]) or '.txt' in str(cap_paths[0])):
        captions = [_read_text_file(p) for p in cap_paths]
    else:
        captions = cap_paths
    
    images = [_read_image_bytes(p) for p in img_paths]
    
    out = dict(batch)
    out["caption_text"] = np.array(captions, dtype=object)
    out["image_paths"] = np.array(images, dtype=object)
    return out

# ------------------------------------------------------ 
# Stage 2: stateful transform actor (tokenization + image processing)
# ------------------------------------------------------
class FLUXTransform:
    """
    Stateful Ray Data transformer for FLUX:
    - Handles CLIP and T5 tokenization
    - Processes images for FLUX
    - Filters by aesthetic score
    """
    
    def __init__(
        self,
        resolution: int = 512,
        clip_model: str = "openai/clip-vit-large-patch14",
        t5_model: str = "xlabs-ai/xflux_text_encoders",
        max_length_clip: int = 77,
        max_length_t5: int = 512,
        min_aesthetic: Optional[float] = None,
        drop_invalid: bool = True,
        maintain_aspect_ratio: bool = True,
    ):
        self.resolution = resolution
        self.max_length_clip = max_length_clip
        self.max_length_t5 = max_length_t5
        self.min_aesthetic = min_aesthetic
        self.drop_invalid = drop_invalid
        self.maintain_aspect_ratio = maintain_aspect_ratio
        
        # Initialize tokenizers (CPU only for this stage)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        
        # Image transforms
        if maintain_aspect_ratio:
            self.image_transform = FlexibleResize(resolution, resolution, maintain_aspect=True)
        else:
            self.image_transform = LargestCenterSquare(resolution)
        
        self.to_tensor = transforms.ToTensor()
        # FLUX uses [0, 1] normalization typically
        self.normalize = transforms.Lambda(lambda x: x)  # Keep in [0,1] range
    
    def _tokenize_text(self, caption: str):
        """Tokenize text with CLIP and T5"""
        
        # CLIP tokenization
        clip_tokens = self.clip_tokenizer(
            caption,
            max_length=self.max_length_clip,
            padding="max_length", 
            truncation=True,
            return_attention_mask=True,
            return_tensors=None,
        )
        
        # T5 tokenization
        t5_tokens = self.t5_tokenizer(
            caption,
            max_length=self.max_length_t5,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors=None,
        )
        
        return {
            "clip_ids": np.array(clip_tokens["input_ids"], dtype=np.int32),
            "clip_mask": np.array(clip_tokens["attention_mask"], dtype=np.int32),
            "t5_ids": np.array(t5_tokens["input_ids"], dtype=np.int32),
            "t5_mask": np.array(t5_tokens["attention_mask"], dtype=np.int32),
        }
    
    def _process_one(self, img_path: str, caption: str, aesthetic_score: Optional[float] = None):
        """Process one image-caption pair"""
        
        # Filter on aesthetic score
        if self.min_aesthetic is not None and aesthetic_score is not None:
            if float(aesthetic_score) < float(self.min_aesthetic):
                return None
        
        # Caption guard
        if not caption:
            if self.drop_invalid:
                return None
            caption = ""
        
        # Load image from path
        if not img_path or not os.path.exists(img_path):
            if self.drop_invalid:
                return None
            img = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        else:
            try:
                img = Image.open(img_path)
            except Exception:
                if self.drop_invalid:
                    return None
                img = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        
        # Process image
        img = _ensure_rgb(img)
        img = self.image_transform(img)
        tensor = self.to_tensor(img)
        tensor = self.normalize(tensor)
        img_arr = tensor.numpy().astype(np.float32)
        
        # Tokenize text
        tokens = self._tokenize_text(caption)
        
        return img_arr, tokens
    
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        images_out: List[np.ndarray] = []
        clip_ids_out: List[np.ndarray] = []
        clip_mask_out: List[np.ndarray] = []
        t5_ids_out: List[np.ndarray] = []
        t5_mask_out: List[np.ndarray] = []
        kept_aes: List[float] = []
        kept_captions: List[str] = []
        
        img_paths_batch = batch.get("image_paths")
        captions_batch = batch.get("caption_text") 
        aes_batch = batch.get("aesthetic_score", None)
        
        n = len(captions_batch)
        for i in range(n):
            aes = float(aes_batch[i]) if aes_batch is not None else None
            result = self._process_one(img_paths_batch[i], captions_batch[i], aes)
            
            if result is None:
                continue
                
            img_arr, tokens = result
            images_out.append(img_arr)
            clip_ids_out.append(tokens["clip_ids"])
            clip_mask_out.append(tokens["clip_mask"])
            t5_ids_out.append(tokens["t5_ids"])
            t5_mask_out.append(tokens["t5_mask"])
            kept_aes.append(aes if aes is not None else np.nan)
            kept_captions.append(captions_batch[i])
        
        # Handle empty batch
        if len(images_out) == 0:
            return {
                f"image_{self.resolution}": np.empty((0, 3, self.resolution, self.resolution), dtype=np.float32),
                "clip_ids": np.empty((0, self.max_length_clip), dtype=np.int32),
                "clip_mask": np.empty((0, self.max_length_clip), dtype=np.int32),
                "t5_ids": np.empty((0, self.max_length_t5), dtype=np.int32),
                "t5_mask": np.empty((0, self.max_length_t5), dtype=np.int32),
                "aesthetic_score": np.empty((0,), dtype=np.float32),
                "caption_text": np.empty((0,), dtype=object),
            }
        
        return {
            f"image_{self.resolution}": np.stack(images_out, axis=0),
            "clip_ids": np.stack(clip_ids_out, axis=0),
            "clip_mask": np.stack(clip_mask_out, axis=0),
            "t5_ids": np.stack(t5_ids_out, axis=0),
            "t5_mask": np.stack(t5_mask_out, axis=0),
            "aesthetic_score": np.array(kept_aes, dtype=np.float32),
            "caption_text": np.array(kept_captions, dtype=object),
        }

# Helper function to load FLUX AutoEncoder
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
# Stage 3: stateful FLUX encoder (GPU)
# -----------------------
class FLUXEncoder:
    """
    Loads FLUX encoders once per actor (GPU):
    - CLIP text encoder
    - T5 text encoder
    - FLUX AutoEncoder
    """
    
    def __init__(
        self,
        resolution: int = 512,
        ae_name: str = "flux-schnell",
        clip_model: str = "openai/clip-vit-large-patch14",
        t5_model: str = "xlabs-ai/xflux_text_encoders",
        max_length_clip: int = 77,
        max_length_t5: int = 512,
        dtype: str = "bfloat16",
        keep_intermediate_tensors: bool = False,
    ):
        self.resolution = resolution
        self.max_length_clip = max_length_clip
        self.max_length_t5 = max_length_t5
        self.keep_intermediate = keep_intermediate_tensors
        
        # Set dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
            self.numpy_dtype = np.float16
        else:
            self.torch_dtype = torch.float16
            self.numpy_dtype = np.float16
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load FLUX AutoEncoder
        self.ae = load_flux_ae(ae_name, device=self.device)
        self.ae = self.ae.to(self.torch_dtype).eval()
        
        # Load text encoders using HFEmbedder
        self.clip_encoder = HFEmbedder(
            clip_model, 
            max_length=max_length_clip, 
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        self.t5_encoder = HFEmbedder(
            t5_model,
            max_length=max_length_t5,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        # FLUX scaling factors
        self.ae_params = configs[ae_name].ae_params
    
    @torch.no_grad()
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        
        # Get batch data
        img_key = f"image_{self.resolution}"
        imgs_np = batch[img_key]  # (B,3,H,W), float32 in [0,1]
        captions = batch["caption_text"].tolist()
        
        # Convert images to tensor and encode
        imgs = torch.from_numpy(imgs_np).to(self.device, dtype=self.torch_dtype)
        
        # Encode images with FLUX VAE
        img_latents = self.ae.encode(imgs)
        
        # Apply FLUX scaling and shifting
        img_latents = (img_latents - self.ae_params.shift_factor) * self.ae_params.scale_factor
        
        # Text embeddings
        clip_embeds = self.clip_encoder(captions)  # (B, D_clip)
        t5_embeds = self.t5_encoder(captions)      # (B, S_t5, D_t5)
        
        # Prepare FLUX format
        flux_batch = prepare_flux_batch(t5_embeds, clip_embeds, img_latents)
        
        # Convert to numpy with proper dtype handling
        if self.torch_dtype == torch.bfloat16:
            out["img_latents"] = flux_batch["img"].detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
            out["img_ids"] = flux_batch["img_ids"].detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
            out["txt_embeds"] = flux_batch["txt"].detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
            out["txt_ids"] = flux_batch["txt_ids"].detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
            out["vec_embeds"] = flux_batch["vec"].detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
        else:
            out["img_latents"] = flux_batch["img"].detach().cpu().numpy().astype(self.numpy_dtype)
            out["img_ids"] = flux_batch["img_ids"].detach().cpu().numpy().astype(self.numpy_dtype)
            out["txt_embeds"] = flux_batch["txt"].detach().cpu().numpy().astype(self.numpy_dtype)
            out["txt_ids"] = flux_batch["txt_ids"].detach().cpu().numpy().astype(self.numpy_dtype)
            out["vec_embeds"] = flux_batch["vec"].detach().cpu().numpy().astype(self.numpy_dtype)
        
        # Store individual embeddings for inspection
        if self.torch_dtype == torch.bfloat16:
            out["clip_embeddings"] = clip_embeds.detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
            out["t5_embeddings"] = t5_embeds.detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
            out["raw_img_latents"] = img_latents.detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
        else:
            out["clip_embeddings"] = clip_embeds.detach().cpu().numpy().astype(self.numpy_dtype)
            out["t5_embeddings"] = t5_embeds.detach().cpu().numpy().astype(self.numpy_dtype)
            out["raw_img_latents"] = img_latents.detach().cpu().numpy().astype(self.numpy_dtype)
        
        # Keep intermediates if requested
        if self.keep_intermediate:
            out[img_key] = imgs_np
            out["clip_ids"] = batch["clip_ids"]
            out["clip_mask"] = batch["clip_mask"]
            out["t5_ids"] = batch["t5_ids"]
            out["t5_mask"] = batch["t5_mask"]
        
        # Copy over other fields
        for key in ["aesthetic_score", "caption_text"]:
            if key in batch:
                out[key] = batch[key]
        
        return out

# ----------------------- 
# Example usage
# -----------------------
if __name__ == "__main__":
    import yaml
    
    # Load config
    try:
        with open("config_flux.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Default config if no file
        config = {
            "csv": "../working_flux/dataset_preparation/eros_data/captions_dataset.csv",
            "resolution": 512,
            "batch_size": 32,
            "parallelism": 8,
            "min_aesthetic": None,
            "encoder": {
                "clip_model": "openai/clip-vit-large-patch14",
                "t5_model": "xlabs-ai/xflux_text_encoders",
                "dtype": "bfloat16",
                "num_actors": 2,
                "batch_size": 16,
                "keep_intermediate_tensors": False,
                "max_length_clip": 77,
                "max_length_t5": 512,
            }
        }
    
    csv_path = config.get("csv", "./data/dataset.csv")
    resolution = int(config.get("resolution", 512))
    batch_size = int(config.get("batch_size", 32))
    parallelism = int(config.get("parallelism", 8))
    min_aesthetic = config.get("min_aesthetic", None)
    
    enc_cfg = config.get("encoder", {})
    enc_clip_model = enc_cfg.get("clip_model", "openai/clip-vit-large-patch14")
    enc_t5_model = enc_cfg.get("t5_model", "xlabs-ai/xflux_text_encoders")
    enc_dtype = enc_cfg.get("dtype", "bfloat16")
    enc_num_actors = int(enc_cfg.get("num_actors", 2))
    enc_batch_size = int(enc_cfg.get("batch_size", batch_size // 2))
    enc_keep_intermediate = bool(enc_cfg.get("keep_intermediate_tensors", False))
    enc_max_length_clip = int(enc_cfg.get("max_length_clip", 77))
    enc_max_length_t5 = int(enc_cfg.get("max_length_t5", 512))
    
    ray.init(ignore_reinit_error=True)
    
    ds = rd.read_csv(csv_path, parallelism=parallelism)
    
    print(f"Initial dataset size: {ds.count()}")
    
    # Pipeline stages:
    
    # Step 1: Validate paths first (lightweight)
    print("Step 1: Validating image paths...")
    ds = ds.map_batches(validate_image_paths_batch, batch_size=batch_size, concurrency=parallelism)
    ds = ds.filter(lambda r: r["is_image_valid"])
    
    print(f"After validation: {ds.count()}")

    # Step 2: Load files (only for valid images)
    print("Step 2: Loading files from paths...")
    ds = ds.map_batches(load_files_from_paths, batch_size=batch_size, concurrency=parallelism)

    # Clean up validation columns
    ds = ds.drop_columns([
        "is_image_valid", "invalid_reason", "image_width", "image_height"
    ])

    # Step 3: FLUX Transform (CPU tokenization + image processing)
    print("Step 3: FLUX tokenization and image processing...")
    ds = ds.map_batches(
        FLUXTransform,
        fn_constructor_kwargs=dict(
            resolution=resolution,
            clip_model=enc_clip_model,
            t5_model=enc_t5_model,
            max_length_clip=enc_max_length_clip,
            max_length_t5=enc_max_length_t5,
            min_aesthetic=min_aesthetic,
            drop_invalid=True,
            maintain_aspect_ratio=False,
        ),
        batch_size=batch_size,
        num_cpus=2,
        concurrency=parallelism,
    )
    
    # Step 4: FLUX Encoder (GPU embedding + VAE encoding)
    print("Step 4: FLUX encoding (GPU)...")
    ds = ds.map_batches(
        FLUXEncoder,
        fn_constructor_kwargs=dict(
            resolution=resolution,
            ae_name="flux-schnell",
            clip_model=enc_clip_model,
            t5_model=enc_t5_model,
            max_length_clip=enc_max_length_clip,
            max_length_t5=enc_max_length_t5,
            dtype=enc_dtype,
            keep_intermediate_tensors=enc_keep_intermediate,
        ),
        batch_size=enc_batch_size,
        num_gpus=1,
        concurrency=enc_num_actors,
    )
    
    print(f"Final dataset: {ds}")
    
    # Test sample
    print("Testing sample...")
    sample = ds.take(1)
    if sample:
        keys = sample[0].keys()
        print("Sample keys:", keys)
        
        # Print shapes of key tensors
        for key in ["img_latents", "txt_embeds", "vec_embeds", "clip_embeddings", "t5_embeddings"]:
            if key in keys:
                print(f"{key} shape:", sample[0][key].shape)
    
    # Optional: Save processed dataset
    output_path = config.get("output_path", "./processed_flux_dataset")
    if output_path:
        print(f"Saving processed dataset to: {output_path}")
        ds.write_parquet(output_path)
        print("Dataset saved!")
