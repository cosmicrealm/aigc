#!/bin/bash
# train_flow_matching.sh - Script for training Flux Kontext with Flow Matching

# Configuration 
export MODEL_NAME="/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-Kontext-dev"
export HR_ROOT="/path/to/high-resolution-images"  # Change this to your image path
export OUTPUT_DIR="trained-flux-kontext-deg-lora-flow-matching"

# Prompt configuration
export DEFAULT_PROMPT="Restore this degraded photograph to high fidelity — remove noise, fix scratches, repair faded colors, and reconstruct missing details — while preserving the original structure, facial expressions, and composition. Enhance sharpness, clarity, and texture naturally, using photorealistic lighting and tonal consistency. The output should look like a restored, high-resolution version of the exact same scene."

# Preferred Kontext resolution buckets
export KONTEXT_BUCKETS="672:1568,688:1504,720:1456,752:1392,800:1328,832:1248,880:1184,944:1104,1024:1024,1104:944,1184:880,1248:832,1328:800,1392:752,1456:720,1504:688,1568:672"

# Run the training
accelerate launch train_deg_lora_flux_kontext.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --hr_root=$HR_ROOT \
  --output_dir=$OUTPUT_DIR \
  --rank=16 \
  --use_bucket_sampler \
  --aspect_ratio_buckets "$KONTEXT_BUCKETS" \
  --degradation_types "downsample,noise,blur,jpeg" \
  --noise_level=0.02 \
  --blur_radius=1.5 \
  --jpeg_quality=85 \
  --mixed_precision="bf16" \
  --instance_prompt="$DEFAULT_PROMPT" \
  --train_batch_size=1 \
  --guidance_scale=1.5 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --max_train_steps=3000 \
  --checkpointing_steps=500 \
  --validation_prompt="Enhance this low quality photo to high definition" \
  --validation_epochs=5 \
  --seed="42" \
  --weighting_scheme="uniform" \
  --perceptual_loss

echo "Flow Matching training completed!"
