from huggingface_hub import hf_hub_download
import torch
import os
from dataclasses import dataclass
from autoencoder import AutoEncoder
from safetensors.torch import load_file as load_sft

from torch import Tensor, nn
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
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


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool



@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float

@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    repo_id_ae: str | None

configs = {
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
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



def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("xlabs-ai/xflux_text_encoders", max_length=max_length, torch_dtype=torch.bfloat16).to(device)

def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)

def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    ckpt_path = os.getenv("AE")
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id_ae, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae



# if __name__ == "__main__":

def test_ae_embeddings():
    load_ae("flux-schnell")

    # img =  torch.randn(1, 3, 512, 512)
    img = Image.open("eros_data/lena_256.png")
    img = img.resize((int(256), int(256)))
    img = np.array(img, dtype=np.float32)  # Convert to float32
    img = img / 255.0  # Normalize to [0, 1]
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 256, 256)
    img = torch.from_numpy(img)
    img = img.to("cuda")
    ae = load_ae("flux-schnell")
    z = ae.encode(img)
    print(z.shape)
    img_reconstructed = ae.decode(z)
    print(img_reconstructed.shape)

    # save the decoded image
    img_reconstructed = img_reconstructed.detach().cpu().numpy()
    img_reconstructed = img_reconstructed.transpose(0, 2, 3, 1)
    img_reconstructed = (img_reconstructed * 255).astype(np.uint8)
    img_reconstructed = Image.fromarray(img_reconstructed[0])
    img_reconstructed.save("img_reconstructed.png")

    prompt = "a beautiful sunset over a calm ocean"
    text_embed = load_t5("cuda").forward([prompt])
    print(text_embed.shape)

    clip_embed = load_clip("cuda").forward([prompt])
    print(clip_embed.shape)

import pandas as pd
from PIL import Image
import numpy as np


from einops import rearrange, repeat



def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

if __name__ == "__main__":

    df = pd.read_csv("/data0/teja_codes/ImmersoAiResearch/flow_matching_v2/FlowModelTraining/dataset_preparation/eros_data/captions_dataset.csv")


    # test_ae_embeddings()
    # exit()

    img_path = df["image_path"].iloc[0]
    caption = df["caption"].iloc[0]


    img_path = '/data0/teja_codes/ImmersoAiResearch/flow_matching_v2/FlowModelTraining/dataset_preparation/eros_data/lena_256.png'
    caption = 'a beautiful woman with long hair smiling from side view'

    # generate the latents for the image and the embeddings for the captions
    img = Image.open(img_path)
    img = img.resize((int(256*1.5), 256))
    img = np.array(img, dtype=np.float32)  # Convert to float32
    img = img / 255.0  # Normalize to [0, 1]
    img = img.transpose(2, 0, 1)
    # convert to torch
    img = img.reshape(1, 3, 256, int(256*1.5))
    img = torch.from_numpy(img)
    img = img.to("cuda")
    img_latent = load_ae("flux-schnell").encode(img)


    text_embed = load_t5("cuda").forward([caption])
    text_embed = text_embed.to("cuda")
    clip_embed = load_clip("cuda").forward([caption])
    clip_embed = clip_embed.to("cuda")

    print(img_latent.shape)
    print(text_embed.shape)
    print(clip_embed.shape)

    import os
    os.makedirs("data_cache", exist_ok=True)

    inp= prepare(load_t5("cuda"), load_clip("cuda"), img_latent, caption)


    for key, val in inp.items():
        print(key, val.shape)

    # save them as .pt file
    torch.save({
        "img_latent": img_latent,
        "text_embed": text_embed,
        "clip_embed": clip_embed,
        "inp": inp
    }, "data_cache/data.pt")
    # save the latents and the embeddings to a csv file


    # reconstrcutre back the latent
    img_reconstructed = load_ae("flux-schnell").decode(img_latent)
    img_reconstructed = img_reconstructed.detach().cpu().numpy()
    img_reconstructed = img_reconstructed.transpose(0, 2, 3, 1)
    img_reconstructed = (img_reconstructed * 255).astype(np.uint8)
    img_reconstructed = Image.fromarray(img_reconstructed[0])
    img_reconstructed.save("img_reconstructed.png")

    print("saved the reconstructed image")