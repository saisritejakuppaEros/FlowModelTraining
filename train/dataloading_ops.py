import torch
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader
import numpy as np
from streaming import Stream, StreamingDataset


class FluxStreamingDataset(StreamingDataset):
    """Dataset class for loading FLUX precomputed data from mds format."""

    def __init__(
        self,
        streams: Optional[List[Stream]] = None,
        shuffle: bool = False,
        batch_size: int = 1,
        **kwargs
    ) -> None:
        # Pass batch_size to StreamingDataset as required
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        out = {}

        # Convert numpy arrays to tensors with correct shapes
        # Expected shapes from dataset preparation:
        # - img_latents: [256, 64] (256 tokens, 64 features each)
        # - txt_embeds: [512, 4096] (512 tokens, 4096 features each)
        # - vec_embeds: [768] (CLIP embedding)
        
        if 'img_latents' in sample:
            # Reshape from flattened to [256, 64] - use float32 to preserve precision
            img_data = sample['img_latents'].astype(np.float32)
            out['img_latents'] = torch.from_numpy(img_data.reshape(256, 64))
        
        if 'img_ids' in sample:
            # Reshape from flattened to [256, 3]
            img_ids_data = sample['img_ids'].astype(np.float32)
            out['img_ids'] = torch.from_numpy(img_ids_data.reshape(256, 3))
            
        if 'txt_embeds' in sample:
            # Reshape from flattened to [512, 4096] - use float32 to preserve precision
            txt_data = sample['txt_embeds'].astype(np.float32)
            out['txt_embeds'] = torch.from_numpy(txt_data.reshape(512, 4096))
            
        if 'txt_ids' in sample:
            # Reshape from flattened to [512, 3]
            txt_ids_data = sample['txt_ids'].astype(np.float32)
            out['txt_ids'] = torch.from_numpy(txt_ids_data.reshape(512, 3))
            
        if 'vec_embeds' in sample:
            # Keep as [768] - no reshaping needed, use float32 to preserve precision
            vec_data = sample['vec_embeds'].astype(np.float32)
            out['vec_embeds'] = torch.from_numpy(vec_data)
            
        # Include raw_img_latents for debugging/inference if available
        if 'raw_img_latents' in sample:
            # Reshape raw latents to [16, 32, 32] - use float32 to preserve precision
            raw_latents_data = sample['raw_img_latents'].astype(np.float32)
            out['raw_img_latents'] = torch.from_numpy(raw_latents_data)
            
        # Include caption text if available
        if 'caption_text' in sample:
            out['caption_text'] = sample['caption_text']

        return out


def build_flux_streaming_dataloader(
    datadir: Union[str, List[str]],
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    **dataloader_kwargs
) -> DataLoader:
    """Creates a DataLoader for FLUX streaming dataset - exact copy of working pattern."""
    
    if isinstance(datadir, str):
        datadir = [datadir]

    streams = [Stream(remote=None, local=d) for d in datadir]

    dataset = FluxStreamingDataset(
        streams=streams,
        shuffle=shuffle,
        batch_size=batch_size,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,  # Important: let dataset handle everything
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader

