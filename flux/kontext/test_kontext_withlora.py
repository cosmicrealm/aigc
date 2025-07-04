import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, AutoencoderKL
from transformers import CLIPTokenizer, T5TokenizerFast
from PIL import Image
from torchvision import transforms
from diffusers.utils import load_image


pretrained_model_name_or_path = "/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-Kontext-dev"
lora_weights_path = "./trained-flux-kontext-lora/pytorch_lora_weights.safetensors"
output_image_path = "generated_image_withlora.png"
prompt = "add a glasses and a purple hat to the sks dog"

# 参考图像路径
reference_image_path = "./dog/alvan-nee-9M0tSjb-cpA-unsplash.jpeg"  # 你需要准备一张图片

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

pipe.load_lora_weights(lora_weights_path)

pipe.to("cuda")

ref_image = load_image(reference_image_path)
if isinstance(ref_image, Image.Image):
    ref_image = transforms.ToTensor()(ref_image).unsqueeze(0)  # (1, 3, H, W)
ref_image = ref_image.to(dtype=data_type, device=pipe.device)

with torch.no_grad():
    images = pipe(prompt=prompt, image=ref_image, num_inference_steps=30).images

for idx, img in enumerate(images):
    img.save(f"{output_image_path.replace('.png', f'_{idx}.png')}")

print("生成完成，图片已保存。")