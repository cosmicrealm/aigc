import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

input_image = load_image("/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/datasets/datasets/sr/face/celeba_512_validation/celeba_512_validation_lq/00000005.png")
prompt = "Enhance image clarity and sharpness: ultra high-resolution, ultra-detailed textures, crisp edges.the subject in sharp focus, foreground details extremely clear,  with soft and subtly blurred background to reduce noise,  all in photorealistic style, fine textures and ultra-clear lighting"
prompt = ""
image = pipe(
  image=input_image,
  prompt=prompt,
  guidance_scale=3.5
).images[0]
image.save("enhanced_celeba5-all.png")
