# VLM Pretraining with Text-only Datasets

A novel approach to training Vision-Language Models using only text datasets by dynamically converting text segments into synthetic images.

## Key Innovation

Instead of requiring paired image-text datasets, this method:

1. **Dynamically splits** each text sample into three parts: prefix, infix, postfix
2. **Renders the infix** as a text-on-image using PIL with automatic font sizing
3. **Trains the model** to predict the postfix given `[tokenized_prefix, image_patches_from_infix]`

This enables VLM pretraining on abundant text-only corpora like FineWeb.

## Architecture

- **LLM**: Qwen 3-1.7B (configurable)
- **Vision Encoder**: ViT-PE-Core-Large (TIMM)
- **Projector**: Multi-modal projector with pooling
- **Training**: Mixed precision with Accelerate, distributed-safe

## Quick Start

```bash
# Install dependencies
uv venv
uv .venv/bin/activate
uv sync

# Run training
accelerate launch -m unsupervised.train

# View example of rendered infix
python -m unsupervised.image_utils
```

## Key Features

- **Distributed training** with proper gradient accumulation
- **Dynamic image generation** with optimal font sizing
- **Checkpoint management** with rotation
- **Mixed sampling** of text-only and image-conditioned generation
- **Robust tokenization** handling variable sequence lengths

## Configuration

Key parameters in `TrainingConfig`:
- `batch_size`: 2 (adjust for GPU memory)
- `max_length`: 1024 tokens
- `llm_lr`: 5e-5, `vit_lr`: 6e-6, `image_proj_lr`: 5e-4
- `dataset_name`: "HuggingFaceFW/fineweb" (any text dataset)

## File Structure

- `train.py` - Main training loop with Accelerate
- `tokenizing.py` - Text-to-image tokenization logic  
- `image_utils.py` - Dynamic image rendering utilities

## Why This Works

By treating rendered text as "images", the model learns to:
- Associate visual text representations with semantic content
- Bridge vision and language modalities without paired data
- Generalize to real images through shared visual-semantic patterns

This approach democratizes VLM training by removing the need for expensive image-text datasets.
