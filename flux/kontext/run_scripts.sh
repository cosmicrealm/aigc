export MODEL_NAME="/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-Kontext-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-flux-kontext-lora"

accelerate launch train_dreambooth_lora_flux_kontext.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --seed="0" 




export MODEL_NAME="/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-Kontext-dev"
export INSTANCE_DIR="ffhq"
export OUTPUT_DIR="trained-flux-kontext-deg-lora"
export DEFAULT_PROMPT="Restore this degraded photograph to high fidelity — remove noise, fix scratches, repair faded colors, and reconstruct missing details — while preserving the original structure, facial expressions, and composition. Enhance sharpness, clarity, and texture naturally, using photorealistic lighting and tonal consistency. The output should look like a restored, high-resolution version of the exact same scene."

accelerate launch train_deg_lora_flux_kontext.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --rank=16 \
  --use_bucket_sampler \
  --aspect_ratio_buckets "512:512,768:512,512:768,1024:1024" \
  --degradation_types "downsample,noise,blur,jpeg" \
  --mixed_precision="bf16" \
  --instance_prompt="$DEFAULT_PROMPT" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --seed="0" 

accelerate launch train_deg_lora_flux_kontext.py \
  --pretrained_model_name_or_path="path/to/flux-kontext-model" \
  --hr_root="path/to/high-quality-images" \
  --output_dir="output/flux-kontext-deg-lora" \
  --instance_prompt="Restore this degraded photograph to high fidelity" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --degradation_types="downsample,noise,blur" \
  --noise_level=0.03 \
  --blur_radius=1.2 \
  --perceptual_loss


# 使用Flow Matching优化版本和Kontext首选分辨率
export MODEL_NAME="/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-Kontext-dev"
export HR_ROOT="/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/datasets/datasets/sr/DF2K/HR"
export OUTPUT_DIR="trained-flux-kontext-deg-lora-flow-matching"
export DEFAULT_PROMPT="Restore this degraded photograph to high fidelity — remove noise, fix scratches, repair faded colors, and reconstruct missing details — while preserving the original structure, facial expressions, and composition. Enhance sharpness, clarity, and texture naturally, using photorealistic lighting and tonal consistency. The output should look like a restored, high-resolution version of the exact same scene."

# 将首选Kontext分辨率格式化为参数形式
# /.cache/huggingface/accelerate/default_config.yaml
export KONTEXT_BUCKETS="672:1568,688:1504,720:1456,752:1392,800:1328,832:1248,880:1184,944:1104,1024:1024,1104:944,1184:880,1248:832,1328:800,1392:752,1456:720,1504:688,1568:672"

accelerate launch train_deg_lora_flux_kontext.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --hr_root=$HR_ROOT \
  --dataset_name="df2k" \
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
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --max_train_steps=300000 \
  --checkpointing_steps=500 \
  --validation_prompt="$DEFAULT_PROMPT" \
  --validation_epochs=5 \
  --seed="42" \
  --weighting_scheme="none"