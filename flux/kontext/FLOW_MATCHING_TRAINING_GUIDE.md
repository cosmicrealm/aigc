# FLUX Kontext Flow Matching Training Guide

This document provides details about the Flow Matching-based training approach implemented in the Flux Kontext model for image enhancement and restoration tasks.

## New Training Script Overview

The latest training script introduces two major improvements:
1. Implementation of proper Flow Matching for velocity field learning
2. Using the preferred Kontext resolutions for optimal model training

## Flow Matching Implementation

### Theoretical Foundation

Flow Matching is a generative modeling approach where we model the vector field that guides the transformation from a source distribution to a target distribution. In our context:

- **Source Distribution**: Degraded/low-quality images (control images)
- **Target Distribution**: High-quality images (target images)

The model learns to predict the optimal velocity field at any point along the interpolation path between these distributions.

### Training Process

1. **Latent Space Interpolation**:
   ```python
   interpolated_latent = (1.0 - sigmas) * model_input_control + sigmas * model_input_target
   ```

2. **Velocity Field Learning**:
   ```python
   # True velocity field
   flow_target = model_input_target - model_input_control
   
   # Model predicts this velocity
   model_pred = transformer(...)
   ```

3. **Loss Calculation**:
   ```python
   l2_loss = weighted_mse(model_pred, flow_target)
   ```

The model is trained to predict how to move from any point along the path toward the target image, rather than just predicting the final result directly.

## Preferred Kontext Resolutions

We've updated the training script to use the officially preferred Kontext resolutions:

```
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672)
]
```

These resolutions are carefully selected to:
- Maintain efficient memory usage
- Support various aspect ratios
- Optimize for the model's internal architecture
- Enable high-quality generation for common image formats

## Training Parameters

The new script includes several parameter optimizations:

- **Degradation Parameters**: Fine-tuned noise level (0.02), blur radius (1.5), and JPEG quality (85)
- **Learning Schedule**: Cosine scheduler with warmup for more stable training
- **Validation**: Regular validation with custom prompts every 5 epochs
- **Guidance Scale**: 1.5 (increased from 1.0) for better controlled generation
- **Checkpointing**: Regular checkpoints every 500 steps

## Usage Example

```bash
# Basic usage with default parameters
bash run_scripts.sh

# Or modify specific parameters
export HR_ROOT="/path/to/your/high-res-images"
export OUTPUT_DIR="your-custom-output-dir"

# Then run the last training script in run_scripts.sh
```

## Best Practices

1. **Image Dataset Preparation**:
   - Ensure a diverse set of high-quality images
   - Consider pre-filtering for optimal aspect ratios
   - Clean your dataset for best results

2. **Training Duration**:
   - The default 3000 steps is suitable for most datasets of ~1000 images
   - For larger datasets, consider increasing steps proportionally
   - Monitor validation results to determine early stopping

3. **Hardware Recommendations**:
   - 16GB+ VRAM recommended (24GB+ ideal)
   - Use the fastest storage available for data loading
   - Consider distributed training for very large models

## Troubleshooting

- **OOM Errors**: Reduce batch size or use smaller resolution buckets
- **Poor Results**: Try adjusting degradation parameters to better match your use case
- **Slow Training**: Check data loading bottlenecks or consider caching latents

## References

- Flow Matching for Generative Modeling (Lipman et al.)
- Flux Kontext: Context-Aware Image Processing with Diffusion Models
- PEFT and LoRA for Efficient Fine-tuning of Large Models
