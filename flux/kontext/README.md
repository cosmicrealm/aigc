# FLUX Kontext Project

这个项目是基于FLUX模型和Kontext技术的图像处理工具集，主要用于图像增强、图像修复以及特定图像生成任务。

## 项目概述

FLUX Kontext 是一个基于扩散模型的图像处理框架，它利用了最新的AI技术来提高图像质量、增强细节，以及根据文本提示进行图像修改。该项目包含了训练和测试脚本，以及一些实用工具和数据集处理组件。

## 主要功能

- **图像增强**：提高低质量图像的清晰度和细节
- **图像修复**：修复受损图像，包括去除噪点、修复划痕、恢复褪色颜色等
- **自定义图像生成**：基于文本提示和参考图像生成定制化图像
- **LoRA微调**：支持对模型进行LoRA (Low-Rank Adaptation) 微调，以适应特定任务或领域
- **退化图像修复**：专门用于修复多种类型退化图像的LoRA训练流程，包括低分辨率、噪声和模糊等

## 项目结构

```
/
├── asserts/                    # 测试和示例图片
├── dog/                        # 示例数据集（狗的图片）
├── my_dataset/                 # 自定义数据集加载器
│   └── dataset_deg_paired.py   # 降质图像对数据集实现
│   └── dataset_deg_bucket.py   # 基于Bucket的降质图像数据集实现
├── scripts/                    # 实用脚本集合
│   ├── demo_enhance_image.py   # 图像增强演示脚本
│   ├── download_dataset.py     # 数据集下载脚本
│   ├── img_downsample.py       # 图像下采样工具
│   ├── test_kontext.py         # Kontext模型测试
│   ├── test_kontext_deg.py     # 降质图像处理测试
│   └── test_kontext_withlora.py# 带LoRA的模型测试
├── utils/                      # 工具函数
│   ├── parse.py                # 参数解析工具
│   └── utils.py                # 通用工具函数
├── run_scripts.sh              # 运行脚本示例
├── train_dreambooth_lora_flux_kontext.py   # Dreambooth LoRA训练脚本
├── train_deg_lora_flux_kontext.py          # 降质修复LoRA训练脚本
└── test_deg_lora_flux_kontext.py           # 降质修复LoRA测试脚本
```

## 核心组件

### 训练脚本

- **train_dreambooth_lora_flux_kontext.py**: 使用Dreambooth技术结合LoRA进行模型微调，用于生成符合特定主题的图像
- **train_deg_lora_flux_kontext.py**: 训练用于图像降质修复的LoRA模型

### 测试脚本

- **test_deg_lora_flux_kontext.py**: 测试降质修复LoRA模型的效果
- **scripts/test_kontext.py**: 测试基础Kontext模型
- **scripts/test_kontext_deg.py**: 测试降质图像处理功能
- **scripts/test_kontext_withlora.py**: 测试使用LoRA微调后的模型

### 数据集

- **my_dataset/dataset_deg_paired.py**: 实现了一个高分辨率-低分辨率图像对的数据集加载器，用于训练图像增强模型

### 工具脚本

- **scripts/demo_enhance_image.py**: 演示如何使用训练好的模型增强图像
- **scripts/img_downsample.py**: 工具脚本，用于生成低分辨率图像用于测试和训练
- **scripts/download_dataset.py**: 用于下载训练和测试所需的数据集

## 技术栈

- **PyTorch**: 深度学习框架
- **Diffusers**: Hugging Face的扩散模型库
- **Transformers**: 用于处理文本提示的模型
- **PEFT/LoRA**: 参数高效微调技术
- **Accelerate**: 分布式训练支持

## 使用示例

### 训练LoRA模型

```bash
export MODEL_NAME="/path/to/base/model"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-flux-kontext-lora"

accelerate launch train_dreambooth_lora_flux_kontext.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --max_train_steps=500
```

### 图像增强示例

```python
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

# 加载模型
pipe = FluxKontextPipeline.from_pretrained("/path/to/model", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# 加载输入图像
input_image = load_image("path/to/input/image.png")

# 定义增强提示
prompt = "Enhance image clarity and sharpness: ultra high-resolution, ultra-detailed textures, crisp edges."

# 执行增强
enhanced_image = pipe(
  image=input_image,
  prompt=prompt,
  guidance_scale=3.5
).images[0]

# 保存结果
enhanced_image.save("enhanced_image.png")
```

## 退化图像修复LoRA训练

我们对`train_deg_lora_flux_kontext.py`脚本进行了改进，添加了更全面的图像退化模拟和训练功能。详细说明请查看 [README_DEGRADATION_LORA.md](README_DEGRADATION_LORA.md)。

```bash
accelerate launch train_deg_lora_flux_kontext.py \
  --pretrained_model_name_or_path="/path/to/flux-kontext-model" \
  --hr_root="/path/to/high-resolution-images" \
  --output_dir="trained-flux-kontext-deg-lora" \
  --mixed_precision="bf16" \
  --instance_prompt="Restore this degraded photograph to high fidelity" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --max_train_steps=3000 \
  --degradation_types="downsample,noise,blur" \
  --perceptual_loss
```

### 退化修复流程

退化图像修复LoRA训练的核心是构建控制图像(退化图像)和目标图像(高清图像)对，主要流程如下：

1. **退化模拟**：对高质量图像应用多种退化效果(下采样、噪声、模糊等)
2. **训练过程**：以退化图像为输入，训练模型生成接近高质量图像的输出
3. **损失计算**：结合L2损失和可选的感知损失，引导模型学习细节恢复

主要特点:
- 支持多种退化类型组合
- 可使用VGG感知损失提高视觉质量
- 完整的数据加载和训练流程

## Flow Matching 训练实现

最近，我们对训练流程进行了重要更新，引入了基于Flow Matching理论的训练方法，使模型学习更加高效和理论上更加合理。

### Flow Matching 基本概念

在Flow Matching训练框架中，我们使用以下关键概念：

- **控制图像(Control Image)**：作为流的起点(z₀)，通常是待处理的退化图像
- **目标图像(Target Image)**：作为流的终点(z₁)，通常是高质量参考图像
- **流路径(Flow Path)**：从z₀到z₁的路径，由时间参数t控制
- **速度场(Velocity Field)**：模型学习预测的方向，指向从当前点到目标的最优路径

### 训练流程改进

1. **潜在空间插值**：
   ```python
   # 在控制图像和目标图像之间进行插值
   interpolated_latent = (1.0 - sigmas) * model_input_control + sigmas * model_input_target
   ```

2. **速度场预测与学习**：
   ```python
   # 真实的速度场方向
   flow_target = model_input_target - model_input_control
   
   # 模型预测的速度场
   model_pred = transformer(...)
   
   # 损失计算 - 预测与真实速度场的差异
   l2_loss = weighted_mse(model_pred, flow_target)
   ```

3. **感知损失优化**：
   ```python
   # 解码预测方向和真实方向到像素空间
   pred_direction_pixels = vae.decode(model_pred)
   true_direction_pixels = vae.decode(flow_target)
   
   # 计算像素空间的感知相似度
   p_loss = perceptual_loss_model(pred_direction_pixels, true_direction_pixels)
   ```

### Flow Matching 理论基础

Flow Matching 损失函数理论表述：

```
L_FM = E_{t,x₀,x₁} || f_θ(x_t, t) - (x₁ - x₀) ||²
```

其中：
- f_θ(x_t, t) 是模型在时间点t的预测
- (x₁ - x₀) 是从起点到终点的真实方向向量
- x_t = (1-t) * x₀ + t * x₁ 是流路径上的中间点

这种方法让模型学习最优的速度场，使得从任意时间点出发都能正确地向目标移动。

## 环境要求

- CUDA兼容的GPU
- PyTorch 2.0+
- Python 3.8+
- 足够的GPU内存(推荐16GB+)

## 快速开始

1. **基本训练脚本**：
   ```bash
   # 使用基本退化图像修复训练
   bash run_scripts.sh
   
   # 使用Flow Matching优化版本训练（推荐）
   bash train_flow_matching.sh
   ```

2. **查看训练指南**：
   - 基本训练流程：参考本文档
   - Flow Matching优化版本：查看 [FLOW_MATCHING_TRAINING_GUIDE.md](FLOW_MATCHING_TRAINING_GUIDE.md)

## 注意事项

- 模型训练可能需要大量计算资源
- 处理高分辨率图像时需要注意内存使用
- 为获得最佳效果，可能需要针对特定任务调整模型参数
- Flow Matching实现更接近理论基础，通常能产生更好的结果

## 许可证

基于Apache License 2.0
