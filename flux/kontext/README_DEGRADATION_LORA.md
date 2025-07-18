# FLUX Kontext 图像退化修复 LoRA 训练指南

## 项目概述

这个项目实现了基于FLUX Kontext模型的图像退化修复LoRA训练流程。通过低秩适应(LoRA)技术，我们可以高效地训练小型模型来修复和增强各种退化的图像，如低分辨率、噪声、模糊等问题。

## 退化修复原理

本项目采用有监督学习方法，使用高质量图像对作为训练数据：

1. **控制图像(Control Image)**: 退化的低质量图像，作为模型的输入
2. **目标图像(Target Image)**: 对应的高质量图像，作为学习目标

训练过程中，模型学习从退化图像到高质量图像的映射关系，通过LoRA技术在不修改原始FLUX Kontext模型权重的情况下，添加小型可训练参数来实现图像退化修复功能。

## 关键特性

- **多样化退化模拟**: 支持多种图像退化类型，包括:
  - 下采样(分辨率降低)
  - 噪声添加
  - 高斯模糊
  - JPEG压缩伪影

- **感知损失**: 除了像素级L2损失外，还可选择性地添加VGG感知损失，提高恢复图像的视觉质量

- **高效训练**: 
  - 使用LoRA技术，仅训练少量参数
  - 支持混合精度训练和梯度检查点，降低内存需求
  - 支持分布式训练加速

## 数据集设计

### `DegDataset` 类

我们实现了一个灵活的 `DegDataset` 类，可以动态生成多种类型的退化图像：

```python
class DegDataset(Dataset):
    def __init__(self, hr_data_root, instance_prompt, lr_scale=8, repeats=1, 
                 degradation_types=None, noise_level=0.05, blur_radius=1.0, jpeg_quality=75):
        # ...初始化代码...

    def apply_degradation(self, img):
        """应用多种退化效果"""
        # 下采样
        if "downsample" in self.degradation_types:
            img = img.resize((img.width // self.lr_scale, img.height // self.lr_scale), Image.BICUBIC)
            img = img.resize((img.width * self.lr_scale, img.height * self.lr_scale), Image.BICUBIC)
        
        # 添加噪声
        if "noise" in self.degradation_types:
            # ...噪声添加代码...
            
        # 添加模糊
        if "blur" in self.degradation_types:
            # ...模糊处理代码...
            
        # JPEG压缩
        if "jpeg" in self.degradation_types:
            # ...JPEG压缩处理代码...
            
        return img
```

### 数据配对

每个训练样本包含:
- `control`: 退化图像
- `target`: 对应的高质量图像
- `prompts`: 图像修复提示文本

## 损失函数设计

### 基础L2损失

像素级MSE损失是基础损失函数，确保生成图像与目标图像在像素值上接近：

```python
l2_loss = torch.mean(
    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
    1,
)
l2_loss = l2_loss.mean()
```

### VGG感知损失(可选)

感知损失使用预训练的VGG网络提取图像特征，比较深层特征的差异，有助于恢复更自然的纹理和细节：

```python
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        # ...初始化VGG特征提取器...

    def forward(self, input, target):
        # ...计算深层特征差异...
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss
```

## 使用方法

### 训练命令

```bash
accelerate launch train_deg_lora_flux_kontext.py \
  --pretrained_model_name_or_path="path/to/flux-kontext-model" \
  --hr_root="path/to/high-quality-images" \
  --output_dir="output/flux-kontext-deg-lora" \
  --instance_prompt="Restore this degraded photograph to high fidelity" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --degradation_types="downsample,noise,blur" \
  --noise_level=0.03 \
  --blur_radius=1.2 \
  --perceptual_loss
```

### 关键参数说明

- **退化控制参数**:
  - `--degradation_types`: 设置需要应用的退化类型，可选"downsample", "noise", "blur", "jpeg"
  - `--noise_level`: 噪声强度，值越大图像噪点越多
  - `--blur_radius`: 高斯模糊半径，值越大图像越模糊
  - `--jpeg_quality`: JPEG压缩质量，值越低压缩伪影越明显

- **训练参数**:
  - `--perceptual_loss`: 启用VGG感知损失
  - `--lr_scale`: 下采样比例，控制分辨率降低程度

### 推理使用

训练完成后，使用以下代码加载和应用训练好的LoRA权重：

```python
from diffusers import FluxKontextPipeline
import torch

# 加载基础模型
pipeline = FluxKontextPipeline.from_pretrained(
    "path/to/flux-kontext-model", 
    torch_dtype=torch.bfloat16
).to("cuda")

# 加载LoRA权重
pipeline.load_lora_weights("path/to/trained-lora-weights")

# 处理退化图像
input_image = load_image("degraded_image.png")
prompt = "Restore this degraded photograph to high fidelity"
enhanced_image = pipeline(
    image=input_image,
    prompt=prompt,
    guidance_scale=3.5
).images[0]

# 保存结果
enhanced_image.save("restored_image.png")
```

## 最佳实践与建议

1. **退化参数调整**:
   - 针对不同场景选择合适的退化类型组合
   - 建议先用少量图像测试退化效果，确保与真实场景退化类似

2. **提示词设计**:
   - 详细描述需要修复的问题能提高修复效果
   - 例如："Restore this degraded photograph to high fidelity — remove noise, fix scratches, repair faded colors"

3. **训练技巧**:
   - 使用较小的学习率(1e-4到5e-5)获得更稳定的训练
   - 如果主要关注细节恢复，推荐启用感知损失
   - 对于噪声较多的训练数据，可以适当增加训练步数

4. **验证流程**:
   - 定期使用验证图像检查修复效果
   - 关注边缘清晰度、细节恢复程度和整体自然度

## 控制图像和目标图像处理逻辑

在训练过程中，控制图像(退化图像)和目标图像(高清图像)的处理流程如下：

1. **数据加载阶段**:
   ```python
   # 从数据集获取一个批次
   batch = next(dataloader)
   pixel_values_control = batch["control"]  # 退化图像
   pixel_values_target = batch["target"]    # 高清图像
   ```

2. **VAE编码阶段**:
   ```python
   # 将像素空间的图像编码到潜在空间
   model_input_control = vae.encode(pixel_values_control).latent_dist.mode()
   model_input_target = vae.encode(pixel_values_target).latent_dist.mode()
   ```

3. **训练阶段**:
   ```python
   # 模型使用退化图像作为输入
   model_pred = transformer(model_input_control, ...)
   
   # 目标为噪声-目标图像潜在表示
   target = noise - model_input_target
   
   # 计算损失
   loss = torch.mean((model_pred - target) ** 2)
   ```

4. **注意事项**:
   - 控制图像(退化图像)作为模型的输入，用于生成预测结果
   - 目标图像(高清图像)用于构建训练目标，模型学习将退化图像映射到高清图像
   - 在流程中，两者均需通过VAE编码到潜在空间中进行处理

## 总结

通过本项目，我们可以高效地训练FLUX Kontext模型的LoRA适配器，专门用于图像退化修复任务。其核心在于构建多样化的退化-高清图像对，使用合适的损失函数引导模型学习复原能力，并通过LoRA技术实现参数高效的微调。
