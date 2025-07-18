import os
import random
import itertools
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

def find_nearest_bucket(height, width, buckets):
    """
    找到最接近当前图像尺寸的bucket索引
    
    Args:
        height (int): 图像高度
        width (int): 图像宽度
        buckets (list): bucket列表，每个元素为(height, width)元组
        
    Returns:
        int: 最近的bucket索引
    """
    aspect_ratio = height / width
    
    min_diff = float('inf')
    nearest_idx = 0
    
    for idx, (bucket_h, bucket_w) in enumerate(buckets):
        bucket_ratio = bucket_h / bucket_w
        diff = abs(aspect_ratio - bucket_ratio)
        
        if diff < min_diff:
            min_diff = diff
            nearest_idx = idx
    
    return nearest_idx


def parse_buckets_string(buckets_string):
    """
    解析表示尺寸buckets的字符串
    格式: "height1:width1,height2:width2,..."
    
    Args:
        buckets_string (str): 格式为"height1:width1,height2:width2,..."的字符串
        
    Returns:
        list: 解析后的buckets列表，每个元素为(height, width)元组
    """
    buckets = []
    for bucket_str in buckets_string.split(','):
        h, w = map(int, bucket_str.split(':'))
        buckets.append((h, w))
    return buckets


class DreamBoothDataset(Dataset):
    """
    用于DreamBooth微调的数据集类，支持实例图像和类别图像的处理。
    支持bucket-based采样，根据图像宽高比将图像分组。
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt=None,
        class_data_root=None,
        class_num=None,
        repeats=1,
        center_crop=False,
        random_flip=False,
        buckets=None,
        image_column=None,
        caption_column=None,
        dataset_name=None,
        dataset_config_name=None,
        cache_dir=None,
    ):
        """
        初始化DreamBoothDataset
        
        Args:
            instance_data_root (str): 实例图像根目录
            instance_prompt (str): 实例提示词
            class_prompt (str, optional): 类别提示词，用于先验保存
            class_data_root (str, optional): 类别图像根目录
            class_num (int, optional): 使用的类别图像数量
            repeats (int, optional): 图像重复次数
            center_crop (bool, optional): 是否使用中心裁剪
            random_flip (bool, optional): 是否随机水平翻转
            buckets (list, optional): bucket列表，每个元素为(height, width)元组
            image_column (str, optional): 数据集中图像列名
            caption_column (str, optional): 数据集中描述列名
            dataset_name (str, optional): 要加载的Hugging Face数据集名称
            dataset_config_name (str, optional): 数据集配置名称
            cache_dir (str, optional): 缓存目录
        """
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt

        self.buckets = buckets or [(1024, 1024)]  # 默认bucket

        # 如果提供了dataset_name，从HuggingFace加载数据集
        if dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "您正在使用datasets库加载数据。如果希望使用自定义描述词训练，请安装datasets库: "
                    "`pip install datasets`。如果只想加载包含图像的本地文件夹，请指定--instance_data_dir。"
                )
            # 加载数据集
            dataset = load_dataset(
                dataset_name,
                dataset_config_name,
                cache_dir=cache_dir,
            )
            # 处理数据集
            column_names = dataset["train"].column_names

            # 获取输入/目标的列名
            if image_column is None:
                image_column = column_names[0]
                print(f"图像列默认为 {image_column}")
            else:
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` 值 '{image_column}' 在数据集列中未找到。数据集列为: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]

            if caption_column is None:
                print(
                    "未提供描述列，为所有图像默认使用instance_prompt。如果数据集包含图像的描述/提示词，"
                    "确保使用--caption_column指定列。"
                )
                self.custom_instance_prompts = None
            else:
                if caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` 值 '{caption_column}' 在数据集列中未找到。数据集列为: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][caption_column]
                # 根据--repeats创建最终描述列表
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            # 从本地文件夹加载图像
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError(f"实例图像根目录不存在：{instance_data_root}")

            # 从目录中收集所有图像
            instance_images_paths = []
            for ext in ["jpg", "jpeg", "png", "bmp", "gif"]:
                instance_images_paths.extend(list(self.instance_data_root.glob(f"**/*.{ext}")))
                instance_images_paths.extend(list(self.instance_data_root.glob(f"**/*.{ext.upper()}")))
                
            if not instance_images_paths:
                raise ValueError(f"在 {instance_data_root} 中未找到图像文件")
                
            instance_images = [Image.open(path) for path in instance_images_paths]
            self.custom_instance_prompts = None

        # 按照repeats复制图像
        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        # 预处理图像
        self.pixel_values = []
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            width, height = image.size

            # 找到最接近的bucket
            bucket_idx = find_nearest_bucket(height, width, self.buckets)
            target_height, target_width = self.buckets[bucket_idx]
            self.size = (target_height, target_width)

            # 根据bucket分配定义变换
            train_resize = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)
            train_crop = transforms.CenterCrop(self.size) if center_crop else transforms.RandomCrop(self.size)
            train_flip = transforms.RandomHorizontalFlip(p=1.0)
            train_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            image = train_resize(image)
            if center_crop:
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, self.size)
                image = crop(image, y1, x1, h, w)
            if random_flip and random.random() < 0.5:
                image = train_flip(image)
            image = train_transforms(image)
            self.pixel_values.append((image, bucket_idx))

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        # 处理类别图像（如果提供）
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        # 图像变换
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if center_crop else transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image, bucket_idx = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["bucket_idx"] = bucket_idx

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt
        else:
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example


def collate_fn(examples, with_prior_preservation=False):
    """
    将多个样本整合成一个batch
    
    Args:
        examples (list): 样本列表
        with_prior_preservation (bool): 是否使用先验保存
        
    Returns:
        dict: batch字典
    """
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # 连接类别和实例样本用于先验保存
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch


# 测试代码
if __name__ == "__main__":
    import os
    import torchvision
    
    # 测试参数
    instance_data_root = "dog"  # 使用测试目录
    instance_prompt = "a photo of sks dog"
    
    # 测试parse_buckets_string函数
    buckets_str = "512:512,768:512,512:768,1024:1024"
    buckets = parse_buckets_string(buckets_str)
    print(f"解析的buckets: {buckets}")
    
    # 创建数据集
    dataset = DreamBoothDataset(
        instance_data_root=instance_data_root,
        instance_prompt=instance_prompt,
        buckets=buckets,
        repeats=1,
        center_crop=False,
        random_flip=True,
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"图像数量: {dataset.num_instance_images}")
    
    # 测试获取样本
    output_dir = "output/dataset_dreambooth_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化一些样本
    for i in range(min(4, len(dataset))):
        sample = dataset[i]
        print(f"样本 {i} - 提示词: {sample['instance_prompt']}, min: {sample['instance_images'].min()}, max: {sample['instance_images'].max()}")

        # 转换回[0,1]范围用于保存
        image = (sample["instance_images"] + 1) / 2
        
        # 保存图像
        torchvision.utils.save_image(image, os.path.join(output_dir, f"sample_{i}.png"))
    
    # 测试collate_fn
    batch_size = 2
    samples = [dataset[i] for i in range(batch_size)]
    batch = collate_fn(samples, with_prior_preservation=False)
    
    print(f"Batch - pixel_values shape: {batch['pixel_values'].shape}")
    print(f"Batch - prompts: {batch['prompts']}, min: {batch['pixel_values'].min()}, max: {batch['pixel_values'].max()}")
    
    # 保存批次中的图像
    for i in range(batch_size):
        image = (batch["pixel_values"][i] + 1) / 2
        torchvision.utils.save_image(image, os.path.join(output_dir, f"batch_sample_{i}.png"))
    
    print(f"测试图像保存在 {output_dir}")
