import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, AutoencoderKL
from transformers import CLIPTokenizer, T5TokenizerFast
from PIL import Image
from torchvision import transforms
from diffusers.utils import load_image
import os

# 设置GPU ID
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 配置参数
pretrained_model_name_or_path = "/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-Kontext-dev"
lora_weights_path = "./trained-flux-kontext-deg-lora-flow-matching/checkpoint-5500/pytorch_lora_weights.safetensors"
prompt = "restore this degraded photograph to high fidelity — remove noise, fix scratches, repair faded colors, and reconstruct missing details — while preserving the original structure, facial expressions, and composition. Enhance sharpness, clarity, and texture naturally, using photorealistic lighting and tonal consistency. The output should look like a restored, high-resolution version of the exact same scene."

# 输出尺寸配置 - 可以尝试不同的尺寸
output_sizes = [
    (512, 512),    # 512x512cd co   co  
    (768, 768),    # 768x768
    (1024, 1024),  # 1024x1024
]
# 选择要使用的输出尺寸索引
selected_size_idx = 0  # 0=512x512, 1=768x768, 2=1024x1024

# 参考图像路径与LoRA设置
control_image_path = "./asserts/lr_image.png"  # 你需要准备一张图片
use_lora = True  # 是否使用 LoRA 权重

# 数据类型设置
data_type = torch.bfloat16  # 使用 bfloat16 数据类型

# 1. 加载基础模型
def load_models():
    # 2. 构建pipeline
    pipe = FluxKontextPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=data_type,
    )
    
    # 加载LoRA权重（如果启用）
    if use_lora:
        pipe.load_lora_weights(lora_weights_path)
    
    pipe.to("cuda")
    return pipe

# 主处理函数
def process_image(pipe, image_path, target_size, prompt):
    # 加载控制图像
    control_image = load_image(image_path)
    
    # 获取原始图像尺寸
    original_width, original_height = control_image.size
    print(f"原始图像尺寸: {original_width}x{original_height}")
    
    # 获取目标尺寸
    target_width, target_height = target_size
    print(f"目标输出尺寸: {target_width}x{target_height}")
    
    # 转换为张量格式
    control_image_tensor = transforms.ToTensor()(control_image).unsqueeze(0)
    control_image_tensor = control_image_tensor.to(dtype=data_type, device=pipe.device)
    
    # 生成图像
    with torch.no_grad():
        images = pipe(
            prompt=prompt, 
            image=control_image_tensor,
            num_inference_steps=30,
            height=target_height,
            width=target_width,
            guidance_scale=1,
            return_dict=True,
            # _auto_resize=False,
        ).images
    
    return images, target_width, target_height

# 保存生成的图像
def save_images(images, image_path, size_info, use_lora=False):
    base_name = os.path.basename(image_path).split('.')[0]
    
    if use_lora:
        output_prefix = f"output/0718/generated_image_withlora_{base_name}_{size_info}"
    else:
        output_prefix = f"output/0718/generated_image_{base_name}_{size_info}"

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    for idx, img in enumerate(images):
        output_path = f"{output_prefix}_{idx}.png"
        img.save(output_path)
        print(f"图像保存到: {output_path}")
    
    return output_prefix

# 主程序执行
def main():
    # 加载模型
    pipe = load_models()
    
    # 选择目标尺寸
    target_size = output_sizes[selected_size_idx]
    
    # 处理图像
    images, width, height = process_image(pipe, control_image_path, target_size, prompt)
    
    # 保存结果
    size_info = f"{width}x{height}"
    save_images(images, control_image_path, size_info, use_lora)
    
    print(f"生成完成，图片已保存。尺寸为: {size_info}")

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=7 python test_deg_lora_flux_kontext.py
    main()