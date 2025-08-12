from dataclasses import dataclass
import os
from pathlib import Path
from typing import cast

from einops import rearrange
import torch
from torch import Tensor, nn

from utils.config import BaseParams, ConfigurableModule


@dataclass
class AutoEncoderParams(BaseParams):
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159
    chunk_size: int = 64


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return cast(Tensor, rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))  # type: ignore[no-any-return]


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])  # type: ignore[index,operator]
                if len(self.down[i_level].attn) > 0:  # type: ignore[arg-type]
                    h = self.down[i_level].attn[i_block](h)  # type: ignore[index,operator]
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))  # type: ignore[operator]

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)  # type: ignore[operator]
        h = self.mid.attn_1(h)  # type: ignore[operator]
        h = self.mid.block_2(h)  # type: ignore[operator]
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h  # type: ignore[no-any-return]


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # compute in_ch_mult, block_in at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)  # type: ignore[operator]
        h = self.mid.attn_1(h)  # type: ignore[operator]
        h = self.mid.block_2(h)  # type: ignore[operator]

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)  # type: ignore[index,operator]
                if len(self.up[i_level].attn) > 0:  # type: ignore[arg-type]
                    h = self.up[i_level].attn[i_block](h)  # type: ignore[index,operator]
            if i_level != 0:
                h = self.up[i_level].upsample(h)  # type: ignore[operator]

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h  # type: ignore[no-any-return]


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class AutoEncoder(nn.Module, ConfigurableModule[AutoEncoderParams]):
    def __init__(
        self,
        params: AutoEncoderParams,
        from_pretrained: bool = True,
    ):
        nn.Module.__init__(self)
        self.params = params

        self.encoder = Encoder(
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=list(params.ch_mult),
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=list(params.ch_mult),
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

        self.chunk_size = params.chunk_size
        self.runtime_dtype: torch.dtype | None = None

        if from_pretrained:
            from safetensors.torch import load_file as load_sft

            cache_dir = os.environ.get("MINFM_CACHE_DIR", None)
            assert cache_dir is not None, "MINFM_CACHE_DIR is not set"
            assert Path(cache_dir).exists(), f"Cache directory {cache_dir} does not exist"
            state_dict = load_sft(
                Path(cache_dir) / "black-forest-labs/FLUX.1-dev/ae.safetensors"
            )
            self.load_state_dict(state_dict)

    @classmethod
    def get_default_params(cls) -> AutoEncoderParams:
        """Return the default parameters for AutoEncoder."""
        return AutoEncoderParams()

    def encode(self, x: Tensor) -> Tensor:
        if self.runtime_dtype is None:
            self.runtime_dtype = next(self.encoder.parameters()).dtype

        input_shape, input_dtype = x.shape, x.dtype
        if len(input_shape) == 5:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        x = x.type(self.runtime_dtype)
        z_chunks = []
        for i in range(0, x.shape[0], self.chunk_size):
            z_chunks.append(self.reg(self.encoder(x[i : i + self.chunk_size])))
        z = torch.cat(z_chunks, dim=0)
        z = self.scale_factor * (z - self.shift_factor)

        if len(input_shape) == 5:
            z = rearrange(z, "(b f) c h w -> b c f h w", b=input_shape[0])

        return z.type(input_dtype)  # type: ignore[no-any-return]

    def decode(self, z: Tensor) -> Tensor:
        if self.runtime_dtype is None:
            self.runtime_dtype = next(self.decoder.parameters()).dtype

        input_shape, input_dtype = z.shape, z.dtype
        if len(input_shape) == 5:
            z = rearrange(z, "b c f h w -> (b f) c h w")

        z = z.type(self.runtime_dtype) / self.scale_factor + self.shift_factor
        x_chunks = []
        for i in range(0, z.shape[0], self.chunk_size):
            x_chunks.append(self.decoder(z[i : i + self.chunk_size]))
        x = torch.cat(x_chunks, dim=0)

        if len(input_shape) == 5:
            x = rearrange(x, "(b f) c h w -> b c f h w", b=input_shape[0])

        return x.type(input_dtype)  # type: ignore[no-any-return]

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))