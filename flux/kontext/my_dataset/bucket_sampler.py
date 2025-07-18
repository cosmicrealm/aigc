import random
import torch
from torch.utils.data import BatchSampler
from .dataset_dreambooth import DreamBoothDataset

class BucketBatchSampler(BatchSampler):
    """
    按照bucket分组批量采样器，确保每个batch中的图像尺寸相似。
    用于DreamBooth或其他支持bucket的数据集。
    """
    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        """
        初始化BucketBatchSampler
        
        Args:
            dataset: 数据集对象，必须包含buckets属性和images_with_bucket或pixel_values属性
            batch_size (int): 批次大小
            drop_last (bool): 是否丢弃不完整的最后一批次
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size应为正整数，但得到 batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last应为布尔值，但得到 drop_last={drop_last}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 按bucket对索引进行分组
        self.bucket_indices = [[] for _ in range(len(self.dataset.buckets))]
        
        # 对于DreamBoothDataset，使用pixel_values属性
        if hasattr(self.dataset, 'pixel_values'):
            for idx, (_, bucket_idx) in enumerate(self.dataset.pixel_values):
                self.bucket_indices[bucket_idx].append(idx)
        # 对于DegBucketDataset，使用images_with_bucket属性
        elif hasattr(self.dataset, 'images_with_bucket'):
            for idx, (_, bucket_idx) in enumerate(self.dataset.images_with_bucket):
                self.bucket_indices[bucket_idx].append(idx)
        else:
            raise ValueError("数据集必须包含pixel_values或images_with_bucket属性")

        self.sampler_len = 0
        self.batches = []

        # 为每个bucket预生成批次
        for indices_in_bucket in self.bucket_indices:
            # 跳过空的bucket
            if not indices_in_bucket:
                continue
                
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


# 测试代码
if __name__ == "__main__":
    import os
    import torchvision
    from .dataset_dreambooth import DreamBoothDataset, parse_buckets_string, collate_fn
    
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
    )
    
    # 创建批次采样器
    batch_sampler = BucketBatchSampler(dataset, batch_size=2, drop_last=False)
    print(f"总共有 {len(batch_sampler)} 个批次")
    
    # 测试批次采样
    output_dir = "output/bucket_sampler_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取几个批次进行可视化
    for i, batch_indices in enumerate(batch_sampler):
        if i >= 2:  # 只测试前两个批次
            break
            
        print(f"批次 {i} - 索引: {batch_indices}")
        
        # 整合批次
        batch = collate_fn([dataset[idx] for idx in batch_indices])
        
        # 保存批次中的图像
        for j, img in enumerate(batch["pixel_values"]):
            # 转换回[0,1]范围用于保存
            img = (img + 1) / 2
            
            # 保存图像
            torchvision.utils.save_image(img, os.path.join(output_dir, f"batch{i}_sample{j}.png"))
            
        print(f"批次 {i} - 提示词: {batch['prompts']}")
    
    print(f"测试图像保存在 {output_dir}")
