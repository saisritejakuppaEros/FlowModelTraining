# MMDiT (Multi-Modal Diffusion Transformer) Testing Suite

This directory contains comprehensive testing and example scripts for the MMDiT model implementation.

## Files Overview

- `test_mmdit.py` - Comprehensive test suite for MMDiT model components
- `example_mmdit.py` - Usage examples demonstrating different MMDiT configurations
- `mmdit.py` - Core MMDiT model implementation
- `other_impls.py` - Supporting implementations (attention, MLP, etc.)

## Quick Start

### Running Tests

To run the complete test suite:

```bash
python test_mmdit.py
```

### Running Examples

To see usage examples:

```bash
python example_mmdit.py
```

## Test Coverage

The test suite covers the following components:

### Core Components
- **PatchEmbed** - Image to patch embedding
- **TimestepEmbedder** - Timestep embedding with sinusoidal encoding
- **VectorEmbedder** - Vector input embedding
- **SelfAttention** - Multi-head self-attention mechanism
- **RMSNorm** - Root Mean Square normalization
- **SwiGLUFeedForward** - SwiGLU activation feedforward network

### Model Blocks
- **DismantledBlock** - Individual transformer block with AdaLN conditioning
- **JointBlock** - Joint attention block for context and main sequence
- **FinalLayer** - Final output layer with AdaLN modulation

### Full Model
- **MMDiT** - Complete multi-modal diffusion transformer model

## Test Categories

### 1. Basic Functionality Tests
- Shape validation for all components
- Forward pass verification
- Parameter count validation

### 2. Configuration Variants
- Different patch sizes (2x2, 4x4, 16x16)
- Various attention modes (torch, xformers)
- RMSNorm vs LayerNorm
- SwiGLU vs GELU activation
- Learn sigma vs standard output

### 3. Edge Cases
- Minimal configurations
- Scale-only modulation
- Dynamic input sizing
- Different input/output channel counts

### 4. Training Tests
- Gradient flow verification
- Memory efficiency with gradient checkpointing
- Loss computation and backpropagation

## Model Configurations Tested

### Standard Configuration
```python
model = MMDiT(
    input_size=32,
    patch_size=2,
    in_channels=4,
    depth=4,
    mlp_ratio=4.0,
    learn_sigma=False,
    attn_mode="torch",
    rmsnorm=False,
    swiglu=False,
    pos_embed_max_size=16,  # Required for position embedding
    num_patches=256,  # 16*16=256 patches for 32x32 input with patch_size=2
    dtype=torch.float32
)
```

### With Conditioning
```python
model = MMDiT(
    input_size=32,
    patch_size=2,
    in_channels=4,
    depth=4,
    adm_in_channels=512,  # Class labels
    context_embedder_config={
        "target": "torch.nn.Linear",
        "params": {"in_features": 768, "out_features": 768}
    },
    dtype=torch.float32
)
```

### Memory Efficient
```python
model = MMDiT(
    input_size=64,
    patch_size=2,
    in_channels=4,
    depth=8,
    pos_embed_max_size=32,  # Required for position embedding
    num_patches=1024,  # 32*32=1024 patches for 64x64 input with patch_size=2
    dtype=torch.float32
)
model.gradient_checkpointing_enable()
```

## Expected Output Shapes

### Input/Output Shapes
- **Input**: `(batch_size, in_channels, height, width)`
- **Output**: `(batch_size, out_channels, height, width)`
- **With learn_sigma=True**: `out_channels = 2 * in_channels`

### Position Embedding Requirements
The MMDiT model requires both `pos_embed_max_size` and `num_patches` parameters to be set:

- **pos_embed_max_size**: Determines the maximum size of the position embedding grid
  - Should be at least as large as the maximum expected input size divided by the patch size
  - For example, if your input is 32x32 with patch_size=2, use `pos_embed_max_size=16`
  - For 64x64 input with patch_size=2, use `pos_embed_max_size=32`

- **num_patches**: The total number of patches that will be created from the input
  - Calculated as `(input_size // patch_size) ** 2`
  - For 32x32 input with patch_size=2: `num_patches = (32 // 2) ** 2 = 16 ** 2 = 256`
  - For 64x64 input with patch_size=2: `num_patches = (64 // 2) ** 2 = 32 ** 2 = 1024`

### Patch Embedding
- **Input**: `(batch_size, channels, height, width)`
- **Output**: `(batch_size, num_patches, embed_dim)`
- **num_patches**: `(height // patch_size) * (width // patch_size)`

### Timestep Embedding
- **Input**: `(batch_size,)` timestep values
- **Output**: `(batch_size, hidden_size)` embeddings

## Dependencies

Required packages:
```bash
pip install torch numpy einops
```

Optional packages for advanced features:
```bash
pip install xformers  # For xformers attention mode
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure `mmdit.py` and `other_impls.py` are in the same directory
2. **CUDA Errors**: Tests run on CPU by default; modify device parameter for GPU testing
3. **Memory Issues**: Reduce batch size or model depth for memory-constrained environments

### Performance Tips

1. **Use xformers attention** for better memory efficiency:
   ```python
   model = MMDiT(attn_mode="xformers", ...)
   ```

2. **Enable gradient checkpointing** for large models:
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Use mixed precision** for faster training:
   ```python
   model = MMDiT(dtype=torch.float16, ...)
   ```

## Example Usage

### Basic Inference
```python
from mmdit import MMDiT

# Create model
model = MMDiT(
    input_size=32,
    patch_size=2,
    in_channels=4,
    depth=4,
    pos_embed_max_size=16,  # Required for position embedding
    num_patches=256,  # 16*16=256 patches for 32x32 input with patch_size=2
    dtype=torch.float32
)
```
```

# Prepare inputs
x = torch.randn(1, 4, 32, 32)  # Input image
t = torch.tensor([0.5])        # Timestep

# Run inference
with torch.no_grad():
    output = model(x, t)
```

### Training Loop
```python
import torch.nn as nn
from torch.optim import AdamW

# Setup
model = MMDiT(...)
optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training step
model.train()
optimizer.zero_grad()

x = torch.randn(batch_size, 4, 32, 32)
t = torch.rand(batch_size)
target = torch.randn(batch_size, 4, 32, 32)

output = model(x, t)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

## Contributing

When adding new features to MMDiT:

1. Add corresponding tests in `test_mmdit.py`
2. Update examples in `example_mmdit.py`
3. Ensure all tests pass before submitting changes
4. Add documentation for new parameters or configurations

## License

This testing suite is provided as-is for educational and research purposes. 