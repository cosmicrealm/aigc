import os
import random
import itertools
import numpy as np
from PIL import Image, ImageFilter
from PIL.ImageOps import exif_transpose
import torch
from torchvision import transforms
from torch.utils.data import Dataset, BatchSampler
from torch.utils.data.sampler import BatchSampler
from io import BytesIO
from pathlib import Path
import glob

class DegBucketDataset(Dataset):
    """
    用于图像退化/恢复训练的数据集，支持多种退化效果和基于 bucket 的批处理策略。
    该数据集将高清图像分到不同的尺寸 bucket 中，并在每次获取时应用退化效果。
    """

    def __init__(
        self,
        hr_data_root,
        instance_prompt,
        buckets=None,
        repeats=1,
        lr_scale=8,
        center_crop=False,
        degradation_types=None,
        noise_level=0.05,
        blur_radius=1.0,
        jpeg_quality=75,
    ):
        self.hr_data_root = Path(hr_data_root)
        if not self.hr_data_root.exists():
            raise ValueError(f"高清图像根目录不存在：{hr_data_root}")
            
        # 收集所有图像文件
        self.hr_paths = glob.glob(os.path.join(hr_data_root, "*.png"), recursive=True)
        if not self.hr_paths:
            # 如果没有找到png文件，尝试查找jpg文件
            self.hr_paths = glob.glob(os.path.join(hr_data_root, "*.jpg"), recursive=True) + \
                          glob.glob(os.path.join(hr_data_root, "*.jpeg"), recursive=True)

        if not self.hr_paths:
            raise ValueError(f"在 {hr_data_root} 中没有找到图像文件")
        
        # 基本参数设置
        self.instance_prompt = instance_prompt
        self.lr_scale = lr_scale
        self.center_crop = center_crop
        self.custom_instance_prompts = None  # 恒定为None，使用统一prompt
        
        # 退化参数设置
        self.degradation_types = degradation_types or ["downsample"]
        self.noise_level = noise_level
        self.blur_radius = blur_radius
        self.jpeg_quality = jpeg_quality
        
        # 设置buckets，如果没有提供，默认使用1024x1024
        self.buckets = buckets or [(1024, 1024)]
        
        # 预处理高清图像
        self.hr_images = []
        for path in self.hr_paths:
            self.hr_images.extend(itertools.repeat(path, repeats))
        
        # 为每个图像找到合适的bucket并保存信息
        self.images_with_bucket = []
        for img_path in self.hr_images:
            try:
                with Image.open(img_path) as img:
                    img = exif_transpose(img)
                    if not img.mode == "RGB":
                        img = img.convert("RGB")
                    width, height = img.size
                    
                    # 找到最接近的bucket
                    bucket_idx = self.find_nearest_bucket(height, width)
                    target_height, target_width = self.buckets[bucket_idx]
                    
                    self.images_with_bucket.append((img_path, bucket_idx))
            except Exception as e:
                print(f"处理图像时出错 {img_path}: {e}")
                continue
        
        self._length = len(self.images_with_bucket)
        print(f"数据集加载完成，共 {self._length} 张图像，退化类型: {self.degradation_types}，buckets: {self.buckets}")
    
    def find_nearest_bucket(self, height, width):
        """找到最接近当前图像尺寸的bucket索引"""
        aspect_ratio = height / width
        
        min_diff = float('inf')
        nearest_idx = 0
        
        for idx, (bucket_h, bucket_w) in enumerate(self.buckets):
            bucket_ratio = bucket_h / bucket_w
            diff = abs(aspect_ratio - bucket_ratio)
            
            if diff < min_diff:
                min_diff = diff
                nearest_idx = idx
        
        return nearest_idx
    
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

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        img_path, bucket_idx = self.images_with_bucket[index]
        target_height, target_width = self.buckets[bucket_idx]
        
        # 加载原始高清图像
        hr_img = Image.open(img_path)
        hr_img = exif_transpose(hr_img)
        if not hr_img.mode == "RGB":
            hr_img = hr_img.convert("RGB")
        
        # 应用变换
        train_resize = transforms.Resize((target_height, target_width), interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop((target_height, target_width)) if self.center_crop else transforms.RandomCrop((target_height, target_width))
        
        # 调整尺寸
        hr_img = train_resize(hr_img)
        
        # 裁剪处理
        if self.center_crop:
            hr_img = train_crop(hr_img)
        else:
            # 获取随机裁剪参数
            i, j, h, w = train_crop.get_params(hr_img, (target_height, target_width))
            hr_img = transforms.functional.crop(hr_img, i, j, h, w)
        
        # 复制一份用于退化处理
        lr_img = hr_img.copy()
        
        # 应用退化效果
        lr_img = self.apply_degradation(lr_img)
        
        # 转换为tensor
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        hr_tensor = to_tensor(hr_img)
        lr_tensor = to_tensor(lr_img)
        
        return {
            "control": lr_tensor,  # 退化图像作为控制输入
            "target": hr_tensor,   # 高清图像作为目标
            "bucket_idx": bucket_idx,
            "prompts": self.instance_prompt,
        }


class BucketBatchSampler(BatchSampler):
    """
    按照bucket分组批量采样器，确保每个batch中的图像尺寸相似
    """
    def __init__(self, dataset: DegBucketDataset, batch_size: int, drop_last: bool = False):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size应为正整数，但得到 batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last应为布尔值，但得到 drop_last={drop_last}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 按bucket对索引进行分组
        self.bucket_indices = [[] for _ in range(len(self.dataset.buckets))]
        for idx, (_, bucket_idx) in enumerate(self.dataset.images_with_bucket):
            self.bucket_indices[bucket_idx].append(idx)

        self.sampler_len = 0
        self.batches = []

        # 为每个bucket预生成批次
        for indices_in_bucket in self.bucket_indices:
            # 打乱同一bucket内的索引
            random.shuffle(indices_in_bucket)
            # 创建批次
            for i in range(0, len(indices_in_bucket), self.batch_size):
                batch = indices_in_bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue  # 如果drop_last为True，跳过不完整的batch
                self.batches.append(batch)
                self.sampler_len += 1  # 计算batch数量

    def __iter__(self):
        # 每个epoch打乱批次顺序
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return self.sampler_len


def collate_fn(examples):
    """
    将多个样本整合成一个batch
    """
    control_images = [example["control"] for example in examples]
    target_images = [example["target"] for example in examples]
    prompts = [example["prompts"] for example in examples]

    # 堆叠图像
    control_images = torch.stack(control_images)
    target_images = torch.stack(target_images)
    
    # 确保内存布局连续
    control_images = control_images.to(memory_format=torch.contiguous_format).float()
    target_images = target_images.to(memory_format=torch.contiguous_format).float()

    # 创建batch字典
    batch = {
        "control": control_images, 
        "target": target_images, 
        "prompts": prompts
    }
    return batch


def parse_buckets_string(buckets_string):
    """
    解析表示尺寸buckets的字符串
    格式: "height1:width1,height2:width2,..."
    """
    buckets = []
    for bucket_str in buckets_string.split(','):
        h, w = map(int, bucket_str.split(':'))
        buckets.append((h, w))
    return buckets


# 测试代码
if __name__ == "__main__":
    import os
    import torchvision
    
    # 测试buckets解析
    buckets = parse_buckets_string("512:512,768:512,512:768,1024:1024")
    print(f"解析的buckets: {buckets}")
    hr_root = "/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/datasets/datasets/sr/ffhq-dataset/images1024x1024"
    hr_root = "/vepfs-d-data/q-xbyd/cv/users/zhangjinyang/datasets/datasets/sr/RealSR_V3/Test/2"
    # 测试数据集
    dataset = DegBucketDataset(
        hr_data_root=hr_root,
        instance_prompt="Restore this degraded photograph to high fidelity.",
        buckets=buckets,
        repeats=1,
        degradation_types=["downsample", "noise", "blur"],
        noise_level=0.03,
        blur_radius=1.2,
        jpeg_quality=80
    )
    
    # 测试BucketBatchSampler
    batch_sampler = BucketBatchSampler(dataset, batch_size=4, drop_last=False)
    print(f"总共有 {len(batch_sampler)} 个批次")
    
    # 测试collate_fn和数据加载
    output_dir = "output/dataset_deg_bucket"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取几个样本进行可视化
    for i, batch_indices in enumerate(batch_sampler):
        if i > 20:  # 只测试几个batch
            break
            
        batch = collate_fn([dataset[idx] for idx in batch_indices])
        print(f"Batch {i} - Control: {batch['control'].shape}, Target: {batch['target'].shape}")
        
        # 保存样本图像
        for j in range(min(2, len(batch_indices))):  # 每个batch保存前2张图片
            # 将[-1,1]范围转回[0,1]范围用于保存
            control_img = (batch['control'][j] + 1) / 2
            target_img = (batch['target'][j] + 1) / 2
            
            # 水平拼接对比图
            compare_img = torch.cat([control_img, target_img], dim=2)
            print(f"info compare_img shape: {compare_img.shape}, min: {compare_img.min()}, max: {compare_img.max()}")
            
            # 保存图像
            torchvision.utils.save_image(compare_img, os.path.join(output_dir, f"batch{i}_sample{j}.png"))
    
    print("样本图像保存在", output_dir)
