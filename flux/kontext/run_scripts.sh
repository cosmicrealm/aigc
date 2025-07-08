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
  --cache_latents \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --seed="0" 


# Run tests
pytest test_dreambooth_lora_flux_kontext.py



export MODEL_NAME="/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-Kontext-dev"
export INSTANCE_DIR="ffhq"
export OUTPUT_DIR="trained-flux-kontext-deg-lora"
export DEFAULT_PROMPT="Restore this degraded photograph to high fidelity — remove noise, fix scratches, repair faded colors, and reconstruct missing details — while preserving the original structure, facial expressions, and composition. Enhance sharpness, clarity, and texture naturally, using photorealistic lighting and tonal consistency. The output should look like a restored, high-resolution version of the exact same scene."

accelerate launch train_deg_lora_flux_kontext.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="$DEFAULT_PROMPT" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --optimizer="adamw" \
  --use_8bit_adam \
  --cache_latents \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --seed="0" 
