import torch
from dataclasses import dataclass
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft

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
            depth=3,
            depth_single_blocks=8,
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

from flux import Flux

def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))



def load_flow_model2(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    # ckpt_path = configs[name].ckpt_path
    # if (
    #     ckpt_path is None
    #     and configs[name].repo_id is not None
    #     and configs[name].repo_flow is not None
    #     and hf_download
    # ):
    #     ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow.replace("sft", "safetensors"))

    # with torch.device("meta" if ckpt_path is not None else device):
    model = Flux(configs[name].params)

    # if ckpt_path is not None:
    #     print("Loading checkpoint")
    #     # load_sft doesn't support torch.device
    #     sd = load_sft(ckpt_path, device=str(device))
    #     missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    #     print_load_warning(missing, unexpected)
    return model


if __name__ == "__main__":
    dit = load_flow_model2("flux-schnell")
    dit.train()
    print(dit)


    # sample inputs
    # img torch.Size([1, 65536, 12])
    # img_ids torch.Size([1, 65536, 3])
    # txt torch.Size([1, 512, 4096])
    # txt_ids torch.Size([1, 512, 3])
    # vec torch.Size([1, 768])


    img = torch.randn(1, 384, 64)
    img_ids = torch.randn(1, 384, 3)
    txt = torch.randn(1, 512, 4096)
    txt_ids = torch.randn(1, 512, 3)
    vec = torch.randn(1, 768)

    timesteps = torch.tensor([0.5])  # Proper timestep values between 0 and 1
    guidance = torch.randn(1, 16, 64, 64)   

    print(f"timesteps shape: {timesteps.shape}")
    print(f"timesteps value: {timesteps}")
    print(f"vec shape: {vec.shape}")


    # Move all tensors to CUDA and bfloat16
    dit = dit.to("cuda").to(torch.bfloat16)
    img = img.to("cuda").to(torch.bfloat16)
    img_ids = img_ids.to("cuda").to(torch.bfloat16)
    txt = txt.to("cuda").to(torch.bfloat16)
    txt_ids = txt_ids.to("cuda").to(torch.bfloat16)
    timesteps = timesteps.to("cuda").to(torch.bfloat16)
    vec = vec.to("cuda").to(torch.bfloat16)
    guidance = guidance.to("cuda").to(torch.bfloat16)

    

    out = dit(img, img_ids, txt, txt_ids, timesteps, vec, guidance=guidance)
    print(out.shape)