from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from utils.config import BaseParams, ConfigurableModule


@dataclass
class VanillaEmbedderParams(BaseParams):
    vocab_size: int = 1000
    embedding_dim: int = 768
    return_datum_lens: bool = False


class VanillaEmbedder(nn.Module, ConfigurableModule[VanillaEmbedderParams]):
    def __init__(self, params: VanillaEmbedderParams) -> None:
        nn.Module.__init__(self)
        self.params = params
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        self.return_datum_lens = params.return_datum_lens
        
        self.init_weights()
    
    def init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    @classmethod
    def get_default_params(cls) -> VanillaEmbedderParams:
        """Return the default parameters for VanillaEmbedder."""
        return VanillaEmbedderParams()
    
    def forward(self, input_ids: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the embedder.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with token ids
            
        Returns:
            If return_datum_lens is False: embeddings of shape (batch_size, seq_len, embedding_dim)
            If return_datum_lens is True: tuple of (embeddings, datum_lens)
        """
        embeddings = self.embedding(input_ids)
        
        if self.return_datum_lens:
            # Calculate the length of each sequence (assuming padding token is 0)
            datum_lens = (input_ids != 0).sum(dim=1)
            return embeddings, datum_lens
        
        return embeddings 