import os
import torch
from diffusers import FluxTransformer2DModel, AutoencoderKL
from transformers import CLIPTokenizer, T5TokenizerFast
from PIL import Image
from torchvision import transforms
from diffusers.utils import load_image
from diffusers import FluxKontextDegPipeline
# from diffusers import FluxKontextPipeline


pretrained_model_name_or_path = "/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-Kontext-dev"
output_image_path = "deg_test.png"
prompt = "make the pic clear and sharp, add a red glasses and a purple hat"

# 参考图像路径
deg_image_path = "./asserts/00000000.png"  # 你需要准备一张图片

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
pipe = FluxKontextDegPipeline.from_pretrained(
    pretrained_model_name_or_path,
    transformer=transformer,
    vae=vae,
    torch_dtype=data_type,
)

pipe.to("cuda")

deg_image = load_image(deg_image_path)
if isinstance(deg_image, Image.Image):
    deg_image = transforms.ToTensor()(deg_image).unsqueeze(0)  # (1, 3, H, W)
deg_image = deg_image.to(dtype=data_type, device=pipe.device)

with torch.no_grad():
    images = pipe(prompt=prompt, image=deg_image, deg_image=deg_image,num_inference_steps=30,height=512,width=512).images

for idx, img in enumerate(images):
    img.save(f"{output_image_path.replace('.png', f'_{idx}.png')}")

print("生成完成，图片已保存。")