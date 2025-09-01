# transforms_sd3.py
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
from diffusers import AutoencoderKL

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
        # Just return the path to keep original format
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















from typing import Dict
import numpy as np
import io
from PIL import Image, UnidentifiedImageError, ImageFile

# Fail on truncated files (treat them as invalid)
ImageFile.LOAD_TRUNCATED_IMAGES = False

def validate_image_paths_batch(
    batch: Dict[str, np.ndarray],
    image_col: str = "image_path",
    min_side: int = 8,
) -> Dict[str, np.ndarray]:
    """Validate images by path without loading full bytes into memory."""
    
    paths = batch.get(image_col)
    if paths is None:
        n = len(next(iter(batch.values()))) if batch else 0
        result = dict(batch)  # Keep all original columns
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
            # Just get image info without loading full image
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
    
    # Preserve original columns and add validation info
    result = dict(batch)  # Keep all original columns
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
    """Given a batch of paths, load raw text and image bytes."""
    # Try to find the correct column names (case-insensitive)
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
    # If the first item looks like a file path (contains '/' or '.txt'), treat as file paths
    # Otherwise, treat as direct caption text
    if cap_paths and ('/' in str(cap_paths[0]) or '.txt' in str(cap_paths[0])):
        # Treat as file paths
        captions = [_read_text_file(p) for p in cap_paths]
    else:
        # Treat as direct caption text
        captions = cap_paths
    
    images = [_read_image_bytes(p) for p in img_paths]
    
    out = dict(batch)
    out["caption_text"] = np.array(captions, dtype=object)
    out["image_paths"] = np.array(images, dtype=object)  # Store paths instead of bytes
    return out

# ------------------------------------------------------ 
# Stage 2: stateful transform actor (multi-tokenize + image)
# ------------------------------------------------------
class SD3Transform:
    """
    Stateful Ray Data transformer for SD3:
    - Handles CLIP-L, CLIP-G, and T5 tokenization
    - Processes images for SD3 VAE
    - Filters by aesthetic score
    """
    
    def __init__(
        self,
        resolution: int = 512,
        clip_l_model: str = "openai/clip-vit-large-patch14",
        clip_g_model: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", 
        t5_model: str = "google/t5-v1_1-xxl",
        max_length_clip: int = 77,
        max_length_t5: int = 256,
        min_aesthetic: Optional[float] = None,
        drop_invalid: bool = True,
    ):
        self.resolution = resolution
        self.max_length_clip = max_length_clip
        self.max_length_t5 = max_length_t5
        self.min_aesthetic = min_aesthetic
        self.drop_invalid = drop_invalid
        
        # Initialize tokenizers
        self.clip_l_tokenizer = CLIPTokenizer.from_pretrained(clip_l_model)
        
        # For CLIP-G, we'll use a placeholder - adjust based on your actual model
        try:
            self.clip_g_tokenizer = CLIPTokenizer.from_pretrained(clip_g_model)
        except:
            # Fallback if model not available
            self.clip_g_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # T5 tokenizer
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        
        # Image transforms
        self.cropper = LargestCenterSquare(resolution)
        self.to_tensor = transforms.ToTensor()
        # SD3 typically uses [-1, 1] normalization
        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    def _tokenize_text(self, caption: str):
        """Tokenize text with all three encoders"""
        
        # CLIP-L tokenization
        clip_l_tokens = self.clip_l_tokenizer(
            caption,
            max_length=self.max_length_clip,
            padding="max_length", 
            truncation=True,
            return_attention_mask=True,
            return_tensors=None,
        )
        
        # CLIP-G tokenization  
        clip_g_tokens = self.clip_g_tokenizer(
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
            "clip_l_ids": np.array(clip_l_tokens["input_ids"], dtype=np.int32),
            "clip_l_mask": np.array(clip_l_tokens["attention_mask"], dtype=np.int32),
            "clip_g_ids": np.array(clip_g_tokens["input_ids"], dtype=np.int32), 
            "clip_g_mask": np.array(clip_g_tokens["attention_mask"], dtype=np.int32),
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
            img = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0)).resize((self.resolution, self.resolution))
            
        else:
            try:
                img = Image.open(img_path)
            except Exception:
                if self.drop_invalid:
                    return None
                img = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0)).resize((self.resolution, self.resolution))
        
        # Process image
        img = _ensure_rgb(img)
        img = self.cropper(img)
        tensor = self.to_tensor(img)
        tensor = self.normalize(tensor)
        img_arr = tensor.numpy().astype(np.float32)
        
        # Tokenize text
        tokens = self._tokenize_text(caption)
        
        return img_arr, tokens
    
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        images_out: List[np.ndarray] = []
        clip_l_ids_out: List[np.ndarray] = []
        clip_l_mask_out: List[np.ndarray] = []
        clip_g_ids_out: List[np.ndarray] = []
        clip_g_mask_out: List[np.ndarray] = []
        t5_ids_out: List[np.ndarray] = []
        t5_mask_out: List[np.ndarray] = []
        kept_aes: List[float] = []
        
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
            clip_l_ids_out.append(tokens["clip_l_ids"])
            clip_l_mask_out.append(tokens["clip_l_mask"])
            clip_g_ids_out.append(tokens["clip_g_ids"])
            clip_g_mask_out.append(tokens["clip_g_mask"])
            t5_ids_out.append(tokens["t5_ids"])
            t5_mask_out.append(tokens["t5_mask"])
            kept_aes.append(aes if aes is not None else np.nan)
        
        # Handle empty batch
        if len(images_out) == 0:
            return {
                f"image_{self.resolution}": np.empty((0, 3, self.resolution, self.resolution), dtype=np.float32),
                "clip_l_ids": np.empty((0, self.max_length_clip), dtype=np.int32),
                "clip_l_mask": np.empty((0, self.max_length_clip), dtype=np.int32),
                "clip_g_ids": np.empty((0, self.max_length_clip), dtype=np.int32),
                "clip_g_mask": np.empty((0, self.max_length_clip), dtype=np.int32),
                "t5_ids": np.empty((0, self.max_length_t5), dtype=np.int32),
                "t5_mask": np.empty((0, self.max_length_t5), dtype=np.int32),
                "aesthetic_score": np.empty((0,), dtype=np.float32),
            }
        
        return {
            f"image_{self.resolution}": np.stack(images_out, axis=0),
            "clip_l_ids": np.stack(clip_l_ids_out, axis=0),
            "clip_l_mask": np.stack(clip_l_mask_out, axis=0), 
            "clip_g_ids": np.stack(clip_g_ids_out, axis=0),
            "clip_g_mask": np.stack(clip_g_mask_out, axis=0),
            "t5_ids": np.stack(t5_ids_out, axis=0),
            "t5_mask": np.stack(t5_mask_out, axis=0),
            "aesthetic_score": np.array(kept_aes, dtype=np.float32),
        }




from model.vae import VAE
def load_vae(model, device: str = "cpu"):

    # checkpoints could be installed automatically from encoders/get_checkpoints.py

    DEBUG = False
    import os
    current_dir = '/data0/teja_codes/ImmersoAiResearch/FlowModelTraining/miniDiffusion'
    path = os.path.join(current_dir, os.path.join("encoders", "hub", "checkpoints", "vae.pth"))
    print(f"Loading VAE from {path}")
    checkpoint = torch.load(path, map_location = device)
    missing, unexpected = model.load_state_dict(checkpoint, strict = not DEBUG)

    # for debuggging
    if DEBUG:
        print(f"Missing keys ({len(missing)}):", missing)
        print(f"\nUnexpected keys ({len(unexpected)}):", unexpected)

    model.eval()

    return model








# ----------------------- 
# Stage 3: stateful SD3 encoder (GPU)
# -----------------------
class SD3Encoder:
    """
    Loads SD3 encoders once per actor (GPU):
    - CLIP-L text encoder
    - CLIP-G text encoder  
    - T5 text encoder
    - SD3 VAE
    """
    
    def __init__(
        self,
        resolution: int = 512,
        vae_model: str = "stabilityai/stable-diffusion-3-medium",
        clip_l_model: str = "openai/clip-vit-large-patch14",
        clip_g_model: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        t5_model: str = "google/t5-v1_1-xxl",
        dtype: str = "float16",
        keep_intermediate_tensors: bool = False,
    ):
        self.resolution = resolution
        self.keep_intermediate = keep_intermediate_tensors
        
        # Set dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
            self.numpy_dtype = np.float16  # numpy doesn't have bfloat16, use float16
        else:
            self.torch_dtype = torch.float16
            self.numpy_dtype = np.float16
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load VAE
        # try:
        self.vae = AutoencoderKL.from_pretrained(
            vae_model,
            subfolder="vae",
            torch_dtype=self.torch_dtype
        ).to(self.device)
        # except:
        #     # Fallback to SD2 VAE if SD3 not available
        #     self.vae = AutoencoderKL.from_pretrained(
        #         "stabilityai/stable-diffusion-2-base",
        #         subfolder="vae", 
        #         torch_dtype=self.torch_dtype
        #     ).to(self.device)
        
        
        # self.vae = VAE()
        # self.vae = load_vae(model = self.vae)
        # self.vae = self.vae.half().to(self.device)  # convert model to FP16
        # self.vae.eval()
        
        # convert to fp16
        self.vae.eval()
        
        # Load text encoders
        self.clip_l_encoder = CLIPTextModel.from_pretrained(
            clip_l_model,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.clip_l_encoder.eval()
        
        try:
            self.clip_g_encoder = CLIPTextModel.from_pretrained(
                clip_g_model,
                torch_dtype=self.torch_dtype
            ).to(self.device)
        except:
            # Fallback
            self.clip_g_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=self.torch_dtype
            ).to(self.device)
        self.clip_g_encoder.eval()
        
        self.t5_encoder = T5EncoderModel.from_pretrained(
            t5_model,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.t5_encoder.eval()
        
        # SD3 scaling factor (may be different from SD2)
        self.latent_scale = 1.5305  # SD3 scaling factor
    
    @torch.no_grad()
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        
        # ---- Images -> latents
        img_key = f"image_{self.resolution}"
        imgs_np = batch[img_key]  # (B,3,H,W), float32 in [-1,1]
        imgs = torch.from_numpy(imgs_np).to(self.device, dtype=self.torch_dtype)
        
        latent_dist = self.vae.encode(imgs).latent_dist
        latents = latent_dist.sample() * self.latent_scale
        # Convert to float32 first if using bfloat16, then to numpy
        if self.torch_dtype == torch.bfloat16:
            latents_np = latents.detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
        else:
            latents_np = latents.detach().cpu().numpy().astype(self.numpy_dtype)
        out[f"image_latents_{self.resolution}"] = latents_np
        

        # ---- CLIP-L text embeddings
        clip_l_ids  = torch.from_numpy(batch["clip_l_ids"]).to(self.device)
        clip_l_mask = torch.from_numpy(batch["clip_l_mask"]).to(self.device)
        clip_l_out  = self.clip_l_encoder(input_ids=clip_l_ids, attention_mask=clip_l_mask)
        clip_l_embeds = clip_l_out.last_hidden_state            # (B, S_clipL, D_l)
        clip_l_pooled = getattr(clip_l_out, "pooler_output", None)

        # ---- CLIP-G text embeddings  
        clip_g_ids  = torch.from_numpy(batch["clip_g_ids"]).to(self.device)
        clip_g_mask = torch.from_numpy(batch["clip_g_mask"]).to(self.device)
        clip_g_out  = self.clip_g_encoder(input_ids=clip_g_ids, attention_mask=clip_g_mask)
        clip_g_embeds = clip_g_out.last_hidden_state            # (B, S_clipG, D_g)
        clip_g_pooled = getattr(clip_g_out, "pooler_output", None)

        # ---- T5 text embeddings
        t5_ids   = torch.from_numpy(batch["t5_ids"]).to(self.device)
        t5_mask  = torch.from_numpy(batch["t5_mask"]).to(self.device)
        t5_out   = self.t5_encoder(input_ids=t5_ids, attention_mask=t5_mask)
        t5_embeds = t5_out.last_hidden_state                    # (B, S_t5, D_t5)  (D_t5 ~ 4096 for T5-XXL)

        # -------- Align CLIP sequences (must match to concat on features)
        # Use same tokenization for CLIP-L and CLIP-G; if seq lens differ, trim to min.
        if clip_l_embeds.size(1) != clip_g_embeds.size(1):
            L = min(clip_l_embeds.size(1), clip_g_embeds.size(1))
            clip_l_embeds = clip_l_embeds[:, :L, :]
            clip_g_embeds = clip_g_embeds[:, :L, :]

        # -------- Combine CLIP features along the *feature* axis
        # Example dims: D_l=768 (CLIP-L), D_g=1280 (OpenCLIP bigG) -> 768+1280=2048
        clip_embeddings = torch.cat([clip_l_embeds, clip_g_embeds], dim=-1)  # (B, S_clip, D_clip= D_l + D_g)

        # -------- Pooled features: concat CLIP pooled along feature axis
        def safe_pool(pooled, token_embeds):
            # If model doesn't return pooler_output, mean-pool tokens
            return pooled if pooled is not None else token_embeds.mean(dim=1)

        pooled_l = safe_pool(clip_l_pooled, clip_l_embeds)  # (B, D_l)
        pooled_g = safe_pool(clip_g_pooled, clip_g_embeds)  # (B, D_g)
        pooled_combined = torch.cat([pooled_l, pooled_g], dim=-1)  # (B, D_l + D_g) -> typically 2048

        # -------- Match CLIP feature dim to T5 feature dim (pad or project)
        D_clip = clip_embeddings.size(-1)         # usually 2048
        D_t5   = t5_embeds.size(-1)               # usually 4096

        if D_clip < D_t5:
            # zero-pad CLIP features to D_t5 (SD3-style padding)
            clip_embeddings = F.pad(clip_embeddings, (0, D_t5 - D_clip))  # pad last dim
        elif D_clip > D_t5:
            # (rare) project down to T5 dim (define once in __init__)
            clip_embeddings = self.clip_to_t5(clip_embeddings)  # nn.Linear(D_clip, D_t5)

        # -------- Final concatenation along the *sequence* axis
        # Shapes now: clip_embeddings (B, S_clip, D_t5), t5_embeds (B, S_t5, D_t5)
        text_embeddings = torch.cat([clip_embeddings, t5_embeds], dim=1)     # (B, S_clip+S_t5, D_t5)

        
        # Store outputs - handle bfloat16 conversion
        if self.torch_dtype == torch.bfloat16:
            out["text_embeddings"] = text_embeddings.detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
            out["pooled_embeddings"] = pooled_combined.detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
            
            # Individual embeddings for inspection/ablation
            out["clip_l_embeddings"] = clip_l_embeds.detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
            out["clip_g_embeddings"] = clip_g_embeds.detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype) 
            out["t5_embeddings"] = t5_embeds.detach().cpu().to(torch.float32).numpy().astype(self.numpy_dtype)
        else:
            out["text_embeddings"] = text_embeddings.detach().cpu().numpy().astype(self.numpy_dtype)
            out["pooled_embeddings"] = pooled_combined.detach().cpu().numpy().astype(self.numpy_dtype)
            
            # Individual embeddings for inspection/ablation
            out["clip_l_embeddings"] = clip_l_embeds.detach().cpu().numpy().astype(self.numpy_dtype)
            out["clip_g_embeddings"] = clip_g_embeds.detach().cpu().numpy().astype(self.numpy_dtype) 
            out["t5_embeddings"] = t5_embeds.detach().cpu().numpy().astype(self.numpy_dtype)
        
        # Keep intermediates if requested
        if self.keep_intermediate:
            out[img_key] = imgs_np
            out["clip_l_ids"] = batch["clip_l_ids"]
            out["clip_l_mask"] = batch["clip_l_mask"]
            out["clip_g_ids"] = batch["clip_g_ids"] 
            out["clip_g_mask"] = batch["clip_g_mask"]
            out["t5_ids"] = batch["t5_ids"]
            out["t5_mask"] = batch["t5_mask"]
        
        return out




import os, numpy as np
def check_paths(batch, col="image_path"):
    paths = batch[col].tolist()
    exists = [os.path.exists(p) for p in paths]
    return {
        "path_exists": np.array(exists, dtype=bool),
        "path_basename": np.array([os.path.basename(p) for p in paths], dtype=object),
    }


# ----------------------- 
# Example usage
# -----------------------
if __name__ == "__main__":
    import yaml
    
    with open("config_sd3.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    csv_path = config.get("csv", "./data/dataset.csv")
    resolution = int(config.get("resolution", 512))
    batch_size = int(config.get("batch_size", 32))
    parallelism = int(config.get("parallelism", 8))
    min_aesthetic = config.get("min_aesthetic", None)
    
    enc_cfg = config.get("encoder", {})
    enc_vae_model = enc_cfg.get("vae_model", "stabilityai/stable-diffusion-3-medium-diffusers")
    enc_clip_l_model = enc_cfg.get("clip_l_model", "openai/clip-vit-large-patch14")
    enc_clip_g_model = enc_cfg.get("clip_g_model", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    enc_t5_model = enc_cfg.get("t5_model", "google/t5-v1_1-xxl")
    enc_dtype = enc_cfg.get("dtype", "float16")
    enc_num_actors = int(enc_cfg.get("num_actors", 1))
    enc_batch_size = int(enc_cfg.get("batch_size", batch_size // 2))
    enc_keep_intermediate = bool(enc_cfg.get("keep_intermediate_tensors", False))
    
    ray.init(ignore_reinit_error=True)
    
    ds = rd.read_csv(csv_path, parallelism=parallelism)
    
    # Usage in pipeline:
    # Step 1: Validate paths first (lightweight)
    ds = ds.map_batches(validate_image_paths_batch, batch_size=batch_size, concurrency=parallelism)
    ds = ds.filter(lambda r: r["is_image_valid"])  # Filter early

    # Step 2: Load files (only for valid images)
    ds = ds.map_batches(load_files_from_paths, batch_size=batch_size, concurrency=parallelism)


    ds = ds.drop_columns([
        "is_image_valid","invalid_reason","image_width","image_height"
    ])

    # Stage 2: SD3 Transform (CPU actors)
    ds = ds.map_batches(
        SD3Transform,
        fn_constructor_kwargs=dict(
            resolution=resolution,
            clip_l_model=enc_clip_l_model,
            clip_g_model=enc_clip_g_model,
            t5_model=enc_t5_model,
            max_length_clip=77,
            max_length_t5=256,
            min_aesthetic=min_aesthetic,
            drop_invalid=True,
        ),
        batch_size=batch_size,
        num_cpus=30,
        concurrency=8,
    )
    
    # Stage 3: SD3 Encoder (GPU actors)
    ds = ds.map_batches(
        SD3Encoder,
        fn_constructor_kwargs=dict(
            resolution=resolution,
            vae_model=enc_vae_model,
            clip_l_model=enc_clip_l_model,
            clip_g_model=enc_clip_g_model, 
            t5_model=enc_t5_model,
            dtype=enc_dtype,
            keep_intermediate_tensors=enc_keep_intermediate,
        ),
        batch_size=enc_batch_size,
        num_gpus=1,
        concurrency=8,
    )
    
    print(ds)
    
    # Test sample
    sample = ds.take(1)
    if sample:
        keys = sample[0].keys()
        print("Sample keys:", keys)
        
        for key in ["image_latents_512", "text_embeddings", "pooled_embeddings"]:
            if key in keys:
                print(f"{key} shape:", sample[0][key].shape)