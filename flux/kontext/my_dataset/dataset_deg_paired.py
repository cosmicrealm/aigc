import glob
import os
import random
import numpy as np
from PIL import Image, ImageFilter
from PIL.ImageOps import exif_transpose
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from io import BytesIO

import torchvision

class DegDataset(Dataset):
    def __init__(self, hr_data_root, instance_prompt, lr_scale=8, repeats=1, custom_instance_prompts=False, 
                 degradation_types=None, noise_level=0.05, blur_radius=1.0, jpeg_quality=75):
        super().__init__()
        self.hr_paths = glob.glob(os.path.join(hr_data_root, "**", "*.png"), recursive=True)
        if not self.hr_paths:
            # Try to find jpg files if no png files are found
            self.hr_paths = glob.glob(os.path.join(hr_data_root, "**", "*.jpg"), recursive=True) + \
                          glob.glob(os.path.join(hr_data_root, "**", "*.jpeg"), recursive=True)
        
        if not self.hr_paths:
            raise ValueError(f"No image files found in {hr_data_root}")
            
        self.instance_prompt = instance_prompt
        self.lr_scale = lr_scale
        # 每张图片重复 times 次
        self.paths = self.hr_paths * repeats
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.custom_instance_prompts = custom_instance_prompts
        
        # 退化类型参数
        self.degradation_types = degradation_types or ["downsample"]  # 默认只使用下采样
        self.noise_level = noise_level
        self.blur_radius = blur_radius
        self.jpeg_quality = jpeg_quality
        
        print(f"Dataset loaded with {len(self.paths)} images, degradation types: {self.degradation_types}")

    def __len__(self):
        return len(self.paths)
    
    def apply_degradation(self, img):
        """应用多种退化效果"""
        # 选择的退化类型应用在图像上
        if "downsample" in self.degradation_types:
            # 下采样然后上采样回原分辨率，造成细节丢失
            img = img.resize((img.width // self.lr_scale, img.height // self.lr_scale), Image.BICUBIC)
            img = img.resize((img.width * self.lr_scale, img.height * self.lr_scale), Image.BICUBIC)
        
        if "noise" in self.degradation_types and self.noise_level > 0:
            # 添加噪声
            img_np = np.array(img).astype(np.float32)
            noise = np.random.normal(0, self.noise_level * 255, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
        if "blur" in self.degradation_types and self.blur_radius > 0:
            # 添加模糊
            img = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
            
        if "jpeg" in self.degradation_types:
            # JPEG压缩退化
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=self.jpeg_quality)
            buffer.seek(0)
            img = Image.open(buffer)
            
        return img

    def __getitem__(self, idx):
        hr = Image.open(self.paths[idx])
        hr = exif_transpose(hr).convert("RGB")

        # 应用退化效果
        lr = self.apply_degradation(hr.copy())

        return {
            "control": self.to_tensor(lr),
            "target": self.to_tensor(hr),
            "prompts": self.instance_prompt,
        }

if __name__ == "__main__":
    dataset = DegDataset(
        hr_data_root="/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/datasets/datasets/sr/ffhq-dataset/images512x512",
        instance_prompt="Restore this degraded photograph to high fidelity — remove noise, fix scratches, repair faded colors, and reconstruct missing details.",
        lr_scale=8,
        repeats=1,
        custom_instance_prompts=False,
        degradation_types=["downsample", "noise", "blur"],  # 使用多种退化效果
        noise_level=0.03,
        blur_radius=1.2
    )
    output_dir = "output/dataset_deg_paired"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(dataset)):
        if i > 5:  # 只展示几个样本
            break
            
        sample = dataset[i]
        print(f"Sample {i} - Control shape: {sample['control'].shape}, min: {sample['control'].min()}, max: {sample['control'].max()}")

        # 将[-1,1]范围转回[0,1]范围用于保存
        control_img = (sample['control'] + 1) / 2
        target_img = (sample['target'] + 1) / 2
        
        # 水平拼接对比图
        combine_img = torch.cat([control_img, target_img], dim=2)
        
        # 保存图像
        torchvision.utils.save_image(control_img, os.path.join(output_dir, f"control_{i}.png"))
        torchvision.utils.save_image(target_img, os.path.join(output_dir, f"target_{i}.png"))
        torchvision.utils.save_image(combine_img, os.path.join(output_dir, f"compare_{i}.png"))

    print("Sample images saved in", output_dir)