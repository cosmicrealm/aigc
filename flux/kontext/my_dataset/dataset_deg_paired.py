import glob
import os
import random
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import torchvision

class DegDataset(Dataset):
    def __init__(self, hr_data_root, instance_prompt, lr_scale=8, repeats=1, custom_instance_prompts=False):
        super().__init__()
        self.hr_paths = glob.glob(os.path.join(hr_data_root, "**", "*.png"), recursive=False)
        self.instance_prompt = instance_prompt
        self.lr_scale = lr_scale
        # 每张图片重复 times 次
        self.paths = self.hr_paths * repeats
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.custom_instance_prompts = custom_instance_prompts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        hr = Image.open(self.paths[idx])
        hr = exif_transpose(hr).convert("RGB")

        # LR 下采样 with bicubic interpolation
        lr = hr.resize((hr.width // self.lr_scale, hr.height // self.lr_scale),
                       Image.BICUBIC)
        # resize lr to hr size
        lr = lr.resize((hr.width, hr.height), Image.BICUBIC)

        return {
            "control": self.to_tensor(lr),
            "target": self.to_tensor(hr),
            "prompts": self.instance_prompt,
        }

if __name__ == "__main__":
    dataset = DegDataset(
        hr_data_root="/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/datasets/datasets/sr/ffhq-dataset/images512x512",
        instance_prompt="A high-resolution image of a beautiful landscape.",
        lr_scale=8,
        repeats=1,
        custom_instance_prompts=False
    )
    output_dir = "output/dataset_deg_paired"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Control shape: {sample['control'].shape}, min: {sample['control'].min()}, max: {sample['control'].max()}")

        control_img = sample['control']
        target_img = sample['target']
        combine_img = torch.cat([control_img, target_img], dim=2)
        torchvision.utils.save_image(control_img, os.path.join(output_dir, f"control_{i}.png"), value_range=(0, 1))
        torchvision.utils.save_image(target_img, os.path.join(output_dir, f"target_{i}.png"), value_range=(0, 1))
        torchvision.utils.save_image(combine_img, os.path.join(output_dir, f"combine_{i}.png"), value_range=(0, 1))
        if i > 2:
            break

    print("Sample images saved in", output_dir)