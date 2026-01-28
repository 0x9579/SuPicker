# SuPicker 设计文档

Cryo-EM 颗粒识别神经网络模型

## 概述

SuPicker 是一个用于冷冻电镜（cryo-EM）图像颗粒识别的深度学习模型，支持训练和推理两个模块。

### 核心技术选型

- **Backbone**: ConvNeXt-Tiny（可配置其他变体）
- **Neck**: FPN（特征金字塔网络）
- **Head**: CenterNet（基于中心点的检测）
- **框架**: PyTorch

## 项目结构

```
SuPicker/
├── supicker/
│   ├── __init__.py
│   ├── config/                    # 配置模块
│   │   ├── __init__.py
│   │   ├── base.py               # 基础配置类
│   │   ├── model.py              # 模型配置（backbone、neck、head）
│   │   ├── training.py           # 训练配置（优化器、学习率、损失函数）
│   │   ├── data.py               # 数据配置（增强、预处理）
│   │   └── inference.py          # 推理配置
│   │
│   ├── models/                    # 模型模块
│   │   ├── __init__.py
│   │   ├── backbone/             # ConvNeXt 实现
│   │   ├── neck/                 # FPN 实现
│   │   ├── head/                 # CenterNet Head 实现
│   │   └── detector.py           # 组装完整检测器
│   │
│   ├── data/                      # 数据处理模块
│   │   ├── __init__.py
│   │   ├── dataset.py            # Dataset 类
│   │   ├── star_parser.py        # STAR 文件解析
│   │   ├── transforms.py         # 数据增强
│   │   └── target_generator.py   # 生成训练目标（热图、尺寸图）
│   │
│   ├── losses/                    # 损失函数模块
│   │   ├── __init__.py
│   │   ├── focal_loss.py
│   │   └── regression_loss.py
│   │
│   ├── engine/                    # 训练和推理引擎
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── predictor.py
│   │
│   └── utils/                     # 工具函数
│       ├── __init__.py
│       ├── checkpoint.py
│       ├── logger.py
│       └── export.py             # 结果导出（STAR/JSON/CSV）
│
├── scripts/
│   ├── train.py                  # 训练入口
│   └── predict.py                # 推理入口
│
├── docs/
│   └── plans/
│
└── main.py
```

## 配置系统设计

使用 Python `dataclass` 实现类型安全的配置系统。

### 模型配置

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class ConvNeXtVariant(Enum):
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"

@dataclass
class BackboneConfig:
    variant: ConvNeXtVariant = ConvNeXtVariant.TINY
    pretrained: bool = True
    pretrained_path: Optional[str] = None  # 自定义权重路径

@dataclass
class FPNConfig:
    in_channels: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    out_channels: int = 256

@dataclass
class HeadConfig:
    num_classes: int = 1
    feat_channels: int = 256
    heatmap_loss: str = "focal"           # focal, gaussian_focal
    heatmap_loss_params: dict = field(default_factory=lambda: {"alpha": 2, "beta": 4})
    size_loss: str = "l1"                  # l1, smooth_l1
    size_loss_params: dict = field(default_factory=dict)
```

### 数据增强配置

```python
@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_90: bool = True
    random_rotation: bool = True
    rotation_range: tuple[float, float] = (-180, 180)
    brightness: bool = True
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast: bool = True
    contrast_range: tuple[float, float] = (0.8, 1.2)
    gaussian_noise: bool = True
    noise_std: float = 0.02
    ctf_simulation: bool = False          # CTF 模拟默认关闭
```

### 训练配置

```python
@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 100
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
```

### 损失函数配置

```python
@dataclass
class LossConfig:
    # Heatmap 损失
    heatmap_type: str = "focal"  # focal, gaussian_focal, mse
    heatmap_weight: float = 1.0
    focal_alpha: float = 2.0
    focal_beta: float = 4.0

    # 尺寸回归损失
    size_type: str = "l1"  # l1, smooth_l1, iou
    size_weight: float = 0.1
    smooth_l1_beta: float = 1.0

    # 偏移量损失
    offset_type: str = "l1"
    offset_weight: float = 1.0
```

### 推理配置

```python
@dataclass
class InferenceConfig:
    score_threshold: float = 0.3
    nms_enabled: bool = True
    nms_radius: float = 20.0  # 像素单位
    output_format: str = "star"  # star, json, csv
    batch_size: int = 1
```

## 模型架构

### 整体数据流

```
Input Image (H×W×1)
       ↓
   ConvNeXt-Tiny (Backbone)
       ↓
   [C1, C2, C3, C4]  ← 4个尺度特征图
       ↓
      FPN (Neck)
       ↓
   [P2, P3, P4, P5]  ← 多尺度融合特征
       ↓
   CenterNet Head
       ↓
   ┌─────────────┬─────────────┐
   ↓             ↓             ↓
 Heatmap      Size Map     Offset Map
 (H/4×W/4×C)  (H/4×W/4×2)  (H/4×W/4×2)
```

### ConvNeXt Backbone 输出

| Stage | 输出尺寸 | 通道数 (Tiny) |
|-------|---------|---------------|
| C1    | H/4×W/4 | 96            |
| C2    | H/8×W/8 | 192           |
| C3    | H/16×W/16 | 384         |
| C4    | H/32×W/32 | 768         |

### CenterNet Head 输出

- **Heatmap**: 每个类别一个通道，值表示该位置是颗粒中心的概率
- **Size Map**: 2 通道，预测颗粒的宽度和高度
- **Offset Map**: 2 通道，预测中心点的亚像素偏移（补偿下采样精度损失）

### 关键设计决策

1. **单尺度输出**: Head 仅在 P2（1/4 下采样）上预测，保持足够分辨率
2. **通道适配**: Backbone 输出为灰度图适配（首层卷积接受单通道输入）
3. **权重加载**: 预训练权重首层卷积通过通道平均适配单通道输入

## 数据处理流程

### 输入格式

- **图像**: TIFF 格式，支持任意尺寸（自适应）
- **标注**: STAR 文件格式（RELION/cryoSPARC 兼容）

### STAR 文件解析

```python
# 解析 STAR 文件，提取关键字段
{
    "micrograph": "image_001.tiff",
    "particles": [
        {"x": 512.3, "y": 234.1, "class_id": 0},
        {"x": 789.6, "y": 456.2, "class_id": 1},
        ...
    ]
}
```

支持 RELION 3.x/4.x 和 cryoSPARC 导出的 STAR 格式，自动检测字段名差异。

### 训练目标生成

将点标注转换为 CenterNet 所需的监督信号：

1. **Heatmap 生成**: 在每个颗粒中心位置渲染 2D 高斯核
   - 高斯核半径根据颗粒尺寸自适应计算
   - 多个颗粒重叠时取最大值

2. **Size Map 生成**: 在颗粒中心位置填入 (width, height)

3. **Offset Map 生成**: 记录量化误差 `offset = center - floor(center / stride) * stride`

### Dataset 类设计

```python
class CryoEMDataset(Dataset):
    def __init__(self, image_dir, star_file, config: DataConfig, transforms=None):
        ...

    def __getitem__(self, idx):
        image = load_tiff(self.images[idx])       # (H, W)
        particles = self.annotations[idx]          # List of particles

        if self.transforms:
            image, particles = self.transforms(image, particles)

        targets = self.generate_targets(image, particles)
        # targets: {"heatmap": ..., "size_map": ..., "offset_map": ...}

        return image, targets
```

数据增强同步作用于图像和标注坐标，确保一致性。

## 训练流程

### 训练循环核心逻辑

```python
class Trainer:
    def train_epoch(self):
        self.model.train()
        for images, targets in self.dataloader:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss, loss_dict = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            self.logger.log(loss_dict)

        self.scheduler.step()
```

### 训练功能

- **检查点保存**: 定期保存 + 最优模型保存（基于验证集指标）
- **断点续训**: 加载检查点恢复训练状态（模型、优化器、epoch）
- **TensorBoard**: 记录 loss 曲线、学习率、验证指标
- **早停机制**: 可选，验证指标无改善时提前终止

### 损失函数

```python
# 总损失计算
loss = (heatmap_weight * heatmap_loss +
        size_weight * size_loss +
        offset_weight * offset_loss)
```

## 推理与后处理

### 推理流程

```python
class Predictor:
    def __init__(self, model, config: InferenceConfig):
        self.model = model.eval()
        self.config = config

    def predict(self, image_path: str) -> list[Particle]:
        image = load_tiff(image_path)
        image_tensor = self.preprocess(image)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        particles = self.postprocess(outputs)
        return particles
```

### 后处理步骤

1. **热图解码**: 提取局部峰值点（3×3 max pooling 比较）
2. **置信度过滤**: 保留得分 > `score_threshold` 的点
3. **尺寸提取**: 在对应位置读取预测的 (width, height)
4. **偏移校正**: 加上亚像素偏移量，恢复精确坐标
5. **NMS（可选）**: 基于距离的非极大值抑制，去除重复检测

### 输出数据结构

```python
@dataclass
class Particle:
    x: float           # 中心 x 坐标
    y: float           # 中心 y 坐标
    width: float       # 预测宽度
    height: float      # 预测高度
    score: float       # 置信度
    class_id: int      # 类别 ID
    class_name: str    # 类别名称
```

### 输出格式

支持导出为 STAR（兼容 RELION）、JSON、CSV 三种格式。

## 使用示例

### 训练

```python
from supicker.config import TrainingConfig, ModelConfig, DataConfig
from supicker.engine import Trainer

config = TrainingConfig(
    model=ModelConfig(
        backbone=BackboneConfig(variant="tiny", pretrained=True),
        head=HeadConfig(num_classes=3)
    ),
    data=DataConfig(
        train_image_dir="./data/train/images",
        train_star_file="./data/train/particles.star",
        val_image_dir="./data/val/images",
        val_star_file="./data/val/particles.star",
        augmentation=AugmentationConfig(ctf_simulation=True)
    ),
    epochs=100,
    learning_rate=1e-4
)

trainer = Trainer(config)
trainer.fit()
```

### 推理

```python
from supicker.config import InferenceConfig
from supicker.engine import Predictor

config = InferenceConfig(
    checkpoint_path="./checkpoints/best.pth",
    score_threshold=0.3,
    output_format="star"
)

predictor = Predictor(config)

# 单张图像
particles = predictor.predict("micrograph_001.tiff")

# 批量处理目录
predictor.predict_dir(
    input_dir="./data/test/images",
    output_path="./results/predictions.star"
)
```

## 设计决策总结

| 模块 | 设计决策 |
|------|---------|
| **输入格式** | TIFF 图像 + STAR 标注文件 |
| **网络架构** | ConvNeXt-Tiny + FPN + CenterNet |
| **输出** | 中心点 + 尺寸 + 类别（多类别） |
| **预训练权重** | 可配置是否加载 ImageNet 权重 |
| **数据增强** | 丰富增强，各项可单独开关 |
| **配置管理** | Python dataclass，类型安全 |
| **损失函数** | 可配置类型和参数 |
| **训练监控** | TensorBoard + 控制台日志 + 检查点 |
| **推理输出** | STAR / JSON / CSV 多格式支持 |
