import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/models/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"
out = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=768,
    width=1360,
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("image.png")