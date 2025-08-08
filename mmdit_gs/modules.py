import torch
import torch.nn as nn
from typing import Optional
import math
from logzero import logger

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding"""
    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            flatten: bool = True,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
            dtype=None,
            device=None,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None
            
        logger.debug("the no of patches is %s", self.num_patches)
        logger.debug("the grid size is %s", self.grid_size)
        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding.to(dtype=t.dtype)
        return embedding

    def forward(self, t, dtype, **kwargs):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class VectorEmbedder(nn.Module):
    """Embeds a flat vector of dimension input_dim"""

    def __init__(self, input_dim: int, hidden_size: int, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)




class Testing:
    
    def __init__(self):
        pass
    def testing_patch_embed():
        patch_embed = PatchEmbed(img_size=64, patch_size=4, in_chans=4, embed_dim=784, strict_img_size=False)
        x = torch.randn(1, 4, 64, 64)
        out = patch_embed(x)
        logger.debug("the output shape of the patch embedder is %s", out.shape)
        
    def testing_timestep_embedder():
        hidden_size = 784
        frequency_embedding_size = 256
        dtype = torch.float32
        device = "cpu"
        timestep_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size, dtype, device)
        t = torch.tensor([0.5])
        out = timestep_embedder(t, dtype)
        logger.debug("the output shape of the timestep embedder is %s", out.shape)

    def testing_vector_embedder():
        input_dim = 4096
        hidden_size = 784
        dtype = torch.float32
        device = "cpu"
        vector_embedder = VectorEmbedder(input_dim, hidden_size, dtype, device)
        x = torch.randn(1, input_dim)
        out = vector_embedder(x)
        logger.debug("the output shape of the vector embedder is %s", out.shape)






if __name__ == "__main__":
    Testing.testing_patch_embed()
    Testing.testing_timestep_embedder()
    Testing.testing_vector_embedder()