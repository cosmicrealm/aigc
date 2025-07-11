import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, AutoencoderKL
from transformers import CLIPTokenizer, T5TokenizerFast
from PIL import Image
from torchvision import transforms
from diffusers.utils import load_image
import os
# set gpu id
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pretrained_model_name_or_path = "/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-Kontext-dev"
lora_weights_path = "./trained-flux-kontext-deg-lora/pytorch_lora_weights.safetensors"
prompt = "Restore this degraded photograph to high fidelity — remove noise, fix scratches, repair faded colors, and reconstruct missing details — while preserving the original structure, facial expressions, and composition. Enhance sharpness, clarity, and texture naturally, using photorealistic lighting and tonal consistency. The output should look like a restored, high-resolution version of the exact same scene. make background slightly blurred to reduce noise, focus on the subject in sharp focus, foreground details extremely clear, all in photorealistic style, fine textures and ultra-clear lighting."

# prompt = "enhance the image quality, remove noise, and restore details while preserving the original structure and composition. The output should look like a high-resolution version of the same scene."

# 参考图像路径
control_image_path = "./asserts/lr_image.png"  # 你需要准备一张图片
use_lora = False  # 是否使用 LoRA 权重

data_type = torch.bfloat16  # 使用 bfloat16 数据类型
# 1. 加载基础模型
transformer = FluxTransformer2DModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="transformer", torch_dtype=data_type
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae", torch_dtype=data_type
)
# vae to bfloat16
# vae.to(torch.bfloat16)
tokenizer_one = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=data_type
)
# tokenizer_one.to(torch.bfloat16)  # CLIPTokenizer does not support bfloat16, so we keep it in float32
tokenizer_two = T5TokenizerFast.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer_2", torch_dtype=data_type
)
# tokenizer_two.to(torch.bfloat16)  # T5TokenizerFast does not support bfloat16, so we keep it in float32
# 2. 构建pipeline
pipe = FluxKontextPipeline.from_pretrained(
    pretrained_model_name_or_path,
    transformer=transformer,
    vae=vae,
    torch_dtype=data_type,
    
)
if use_lora:
    pipe.load_lora_weights(lora_weights_path)

pipe.to("cuda")

control_image = load_image(control_image_path)
if isinstance(control_image, Image.Image):
    control_image = transforms.ToTensor()(control_image).unsqueeze(0)  # (1, 3, H, W)
control_image = control_image.to(dtype=data_type, device=pipe.device)

with torch.no_grad():
    images = pipe(prompt=prompt, image=control_image, 
                  num_inference_steps=30,
                  height=512,
                  width=512,
                  ).images
base_name = os.path.basename(control_image_path).split('.')[0]
if use_lora:
    output_image_name = f"output/generated_image_withlora_{base_name}.png"
else:
    # If not using LoRA, save with a different name
    output_image_name = f"output/generated_image_{base_name}.png"
os.makedirs(os.path.dirname(output_image_name), exist_ok=True)
for idx, img in enumerate(images):
    img.save(f"{output_image_name.replace('.png', f'_{idx}.png')}")
print("生成完成，图片已保存。")