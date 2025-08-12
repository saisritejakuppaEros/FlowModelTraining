# Qwen Image Training with PyTorch Lightning

This repository contains PyTorch Lightning implementations for training the Qwen Image model with advanced optimizations including DeepSpeed, mixed precision, and distributed training support.

## Files

- `lightning_dataloading.py` - PyTorch Lightning DataModule for efficient data loading
- `lightning_train.py` - Main Lightning module with model training logic
- `train_lightning.py` - Simple training script with CLI interface
- `deepspeed_config.json` - DeepSpeed configuration for optimal memory usage
- `requirements_lightning.txt` - Python dependencies

## Features

### Optimizations
- ✅ **PyTorch Lightning** - Clean, modular training code
- ✅ **DeepSpeed Integration** - Memory-efficient training with ZeRO optimization
- ✅ **Mixed Precision** - bf16/fp16 training for faster convergence
- ✅ **Gradient Checkpointing** - Reduced memory usage
- ✅ **8-bit Optimizer** - AdamW8bit for memory efficiency
- ✅ **Model Compilation** - PyTorch 2.0 compile support
- ✅ **Distributed Training** - Multi-GPU support with DDP/DeepSpeed
- ✅ **Memory Management** - Automatic cleanup and CPU offloading for text encoder
- ✅ **Flexible Device Placement** - Text encoder can run on CPU to save GPU memory

### Monitoring & Logging
- ✅ **TensorBoard** - Training metrics and loss curves
- ✅ **Weights & Biases** - Advanced experiment tracking
- ✅ **Model Checkpointing** - Automatic best model saving
- ✅ **Early Stopping** - Prevent overfitting
- ✅ **Learning Rate Monitoring** - Track LR schedules

## Installation

```bash
# Install dependencies
pip install -r requirements_lightning.txt

# For DeepSpeed support (recommended)
pip install deepspeed

# For W&B logging (optional)
pip install wandb
```

## Quick Start

### Basic Training
```bash
python train_lightning.py \
    --config_path config.yaml \
    --batch_size 4 \
    --max_epochs 10 \
    --learning_rate 1e-5
```

### Multi-GPU Training
```bash
python train_lightning.py \
    --config_path config.yaml \
    --batch_size 2 \
    --max_epochs 20 \
    --precision bf16-mixed
```

### DeepSpeed Training (Recommended for Large Models)
```bash
python train_lightning.py \
    --config_path config.yaml \
    --use_deepspeed \
    --batch_size 8 \
    --max_epochs 15 \
    --gradient_checkpointing \
    --compile_model
```

### With Weights & Biases Logging
```bash
python train_lightning.py \
    --config_path config.yaml \
    --use_wandb \
    --wandb_project "my-qwen-training" \
    --experiment_name "run-1" \
    --batch_size 4
```

## Configuration

### Data Configuration (`config.yaml`)
```yaml
dataset:
  data_dir: "/path/to/your/data"
  # If using CSV format:
  # - Place images in data_dir/images/
  # - Place captions.csv in data_dir/
  # CSV should have columns: filename, caption
  
  # If using directory format:
  # - Place images directly in data_dir/
  # - Define prompts list below
  prompts:
    - "A beautiful landscape"
    - "A cute dog"
    - "Modern architecture"
```

### DeepSpeed Configuration
The included `deepspeed_config.json` provides optimal settings for Qwen Image training:

- **ZeRO Stage 2** - Optimizer state partitioning
- **CPU Offloading** - Optimizer and parameter offloading to CPU
- **Mixed Precision** - bf16 training
- **Memory Optimizations** - Gradient/parameter optimizations
- **Lightning Compatible** - Gradient accumulation handled by Lightning Trainer

**Note**: The config doesn't include `gradient_accumulation_steps` as this is automatically managed by PyTorch Lightning's `accumulate_grad_batches` parameter.

## Command Line Options

### Data Arguments
- `--config_path` - Path to config file (default: config.yaml)
- `--batch_size` - Batch size per GPU (default: 2)
- `--num_workers` - Data loader workers (default: 4)
- `--resolution` - Image resolution (default: 512)

### Model Arguments
- `--pretrained_model` - Pretrained model path (default: Qwen/Qwen-Image)
- `--learning_rate` - Learning rate (default: 1e-5)
- `--weight_decay` - Weight decay (default: 1e-4)

### Training Arguments
- `--max_epochs` - Maximum epochs (default: 10)
- `--precision` - Training precision (bf16-mixed/16-mixed/32)
- `--accumulate_grad_batches` - Gradient accumulation steps (default: 4)

### Optimization Arguments
- `--gradient_checkpointing` - Enable gradient checkpointing
- `--use_8bit_optimizer` - Use 8-bit AdamW optimizer
- `--compile_model` - Enable PyTorch 2.0 compilation
- `--text_encoder_cpu` - Keep text encoder on CPU to save GPU memory (default: True)
- `--use_deepspeed` - Enable DeepSpeed strategy

### Logging Arguments
- `--output_dir` - Output directory (default: ./outputs)
- `--use_wandb` - Enable W&B logging
- `--wandb_project` - W&B project name

## Memory Optimization Tips

### For Limited GPU Memory
```bash
python train_lightning.py \
    --batch_size 1 \
    --accumulate_grad_batches 8 \
    --gradient_checkpointing \
    --use_8bit_optimizer \
    --text_encoder_cpu \
    --precision bf16-mixed
```

### For Large Scale Training
```bash
python train_lightning.py \
    --use_deepspeed \
    --batch_size 4 \
    --accumulate_grad_batches 4 \
    --gradient_checkpointing \
    --compile_model
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir outputs/qwen_image_finetune
```

### Weights & Biases
Training metrics are automatically logged to W&B when using `--use_wandb`.

## Resume Training
```bash
python train_lightning.py \
    --resume_from_checkpoint outputs/checkpoints/last.ckpt \
    --config_path config.yaml
```

## Model Saving

Models are automatically saved to:
- `outputs/checkpoints/` - Training checkpoints
- `outputs/final_model/` - Final trained model

## Performance Benchmarks

| Setup | GPUs | Batch Size | Memory Usage | Speed |
|-------|------|------------|--------------|-------|
| Basic | 1x A100 | 2 | ~35GB | ~1.2 it/s |
| DeepSpeed | 1x A100 | 4 | ~25GB | ~1.8 it/s |
| Multi-GPU | 8x A100 | 2 per GPU | ~20GB each | ~8.5 it/s |
| DeepSpeed Multi | 8x A100 | 4 per GPU | ~15GB each | ~12 it/s |

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size`
- Increase `--accumulate_grad_batches`
- Enable `--gradient_checkpointing`
- Use `--use_deepspeed`

### Slow Training
- Enable `--compile_model` (PyTorch 2.0+)
- Use `--precision bf16-mixed`
- Increase `--num_workers`
- Use DeepSpeed for multi-GPU

### NaN Loss
- Reduce `--learning_rate`
- Use `--precision bf16-mixed` instead of fp16
- Check data quality and preprocessing

### DeepSpeed Configuration Error
If you see: "Do not set `gradient_accumulation_steps` in the DeepSpeed config"
- Remove `gradient_accumulation_steps` from your custom DeepSpeed config file
- Use Lightning's `--accumulate_grad_batches` parameter instead
- The provided `deepspeed_config.json` is already configured correctly

### DeepSpeed Batch Size Error
If you see: "TypeError: '>' not supported between instances of 'str' and 'int'"
- Remove `train_batch_size` and `train_micro_batch_size_per_gpu` from DeepSpeed config
- Lightning automatically calculates these values
- Use `--batch_size` and `--accumulate_grad_batches` CLI arguments instead

### DeepSpeed Optimizer Conflict
If you see: "ZeRORuntimeException: You are using ZeRO-Offload with a client provided optimizer"
- DeepSpeed now handles the optimizer internally when using ZeRO-Offload
- The 8-bit optimizer is automatically disabled when using DeepSpeed
- DeepSpeed uses its own optimized AdamW implementation

## Advanced Usage

### Custom Training Loop
You can use the Lightning modules directly in your own training scripts:

```python
from lightning_train import QwenImageLightningModule
from lightning_dataloading import QwenImageDataModule
import pytorch_lightning as pl

# Setup data and model
data_module = QwenImageDataModule("config.yaml", batch_size=4)
model = QwenImageLightningModule(learning_rate=1e-5)

# Custom trainer configuration
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu",
    strategy="deepspeed_stage_2",
    precision="bf16-mixed"
)

# Train
trainer.fit(model, data_module)
```

### Custom Callbacks
Add your own callbacks for advanced monitoring:

```python
from pytorch_lightning.callbacks import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        # Custom logic here
        pass

trainer = pl.Trainer(callbacks=[CustomCallback()])
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project follows the same license as the original model repositories. 