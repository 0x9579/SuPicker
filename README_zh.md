# SuPicker

基于深度学习的冷冻电镜（Cryo-EM）显微图像颗粒自动拾取框架，采用 CenterNet 风格的目标检测方法。

## 特性

- **ConvNeXt 骨干网络**：采用现代视觉 Transformer 启发的架构，支持 Tiny、Small、Base 三种规模
- **特征金字塔网络（FPN）**：多尺度特征提取，适用于检测不同大小的颗粒
- **CenterNet 检测头**：无锚框检测，输出热力图、尺寸和偏移预测
- **灵活的数据管道**：支持 TIFF、MRC 及标准图像格式，使用 STAR 文件标注
- **随机裁剪训练**：自动随机裁剪大尺寸显微图像，避免显存不足（OOM）
- **混合精度训练（AMP）**：自动混合精度，节省 30-50% 显存并加速训练
- **可配置训练**：模块化配置系统，覆盖所有训练超参数
- **分布式训练**：通过 PyTorch DistributedDataParallel 支持多 GPU 训练
- **预训练权重**：可选加载 ImageNet 预训练骨干网络权重
- **评估指标**：内置精确率、召回率、F1 和平均精度（AP）计算
- **训练日志评估**：训练时同步在控制台显示当前的精确率（P）、召回率（R）和 F1
- **中断自动保存**：优雅处理 Ctrl+C 退出，自动保存训练进度防止丢失

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/supicker.git
cd supicker

# 开发模式安装
pip install -e .

# 安装开发依赖
pip install -e ".[dev]"

# 安装 MRC 文件支持
pip install -e ".[mrc]"
```

## 快速开始

### 训练

```bash
# 基础训练
python scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --backbone tiny \
    --epochs 100 \
    --batch-size 8

# 指定 GPU 训练
python scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --backbone tiny \
    --epochs 100 \
    --batch-size 8 \
    --device cuda:7

# 带验证集训练
python scripts/train.py \
    --train-images ./data/train \
    --train-star ./data/train.star \
    --val-images ./data/val \
    --val-star ./data/val.star \
    --pretrained

# 多 GPU 训练（4 卡）
torchrun --nproc_per_node=4 scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --distributed

# 指定特定 GPU
CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --distributed
```

### 推理

```bash
# 在显微图像上运行预测
python scripts/predict.py \
    --checkpoint ./checkpoints/best.pth \
    --input ./data/test \
    --output ./results \
    --format star
```

## 配置说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backbone` | `tiny` | ConvNeXt 变体：tiny、small、base |
| `--batch-size` | `8` | 训练批次大小 |
| `--val-batch-size` | `2` | 验证批次大小（默认较小以节省显存） |
| `--epochs` | `100` | 训练轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--optimizer` | `adamw` | 优化器：adam、adamw、sgd |
| `--scheduler` | `cosine` | 学习率调度器：cosine、step、none |
| `--weight-decay` | `0.01` | 权重衰减（正则化） |
| `--warmup-epochs` | `5` | 线性预热轮数 |
| `--pretrained` | `False` | 使用 ImageNet 预训练权重 |
| `--device` | `cuda` | 设备选择（如 `cuda:0`、`cuda:7`、`cpu`） |
| `--distributed` | `False` | 启用多 GPU 分布式训练（配合 `torchrun` 使用） |
| `--resume` | `None` | 继续训练的断点文件路径（checkpoint） |
| `--no-amp` | `False` | 禁用自动混合精度 |
| `--no-augmentation` | `False` | 禁用数据增强 |

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--score-threshold` | `0.3` | 最小检测置信度阈值 |
| `--nms-radius` | `10` | NMS 抑制半径（像素） |
| `--format` | `star` | 输出格式：star、json、csv |

## Python API

### 训练

```python
from supicker.config import (
    ModelConfig, BackboneConfig, ConvNeXtVariant,
    TrainingConfig, AugmentationConfig
)
from supicker.models import Detector
from supicker.data import create_dataloader, build_transforms
from supicker.engine import Trainer

# 配置模型
model_config = ModelConfig(
    backbone=BackboneConfig(
        variant=ConvNeXtVariant.TINY,
        pretrained=True,
        in_channels=1,
    )
)

# 创建模型
model = Detector(model_config)

# 创建数据加载器
train_loader = create_dataloader(
    image_dir="./data/train",
    star_file="./data/train.star",
    batch_size=8,
    transforms=build_transforms(AugmentationConfig()),
)

# 配置并运行训练
training_config = TrainingConfig(
    epochs=100,
    learning_rate=1e-4,
)

trainer = Trainer(
    model=model,
    config=training_config,
    checkpoint_dir="./checkpoints",
    log_dir="./logs",
)

trainer.train(train_loader)
```

### 推理

```python
from supicker.config import ModelConfig, InferenceConfig
from supicker.models import Detector
from supicker.engine import Predictor
from supicker.utils import export_to_star
import tifffile
import torch

# 加载模型
model = Detector(ModelConfig())
predictor = Predictor.from_checkpoint(
    checkpoint_path="./checkpoints/best.pth",
    model=model,
    config=InferenceConfig(score_threshold=0.3),
)

# 加载并预处理图像
image = tifffile.imread("micrograph.tiff").astype("float32")
image = (image - image.min()) / (image.max() - image.min() + 1e-8)
image = torch.from_numpy(image).unsqueeze(0)  # 添加通道维度

# 运行预测
particles = predictor.predict(image)

# 导出结果
export_to_star(particles, "output.star", micrograph_name="micrograph.tiff")
```

### 评估

```python
from supicker.utils.metrics import compute_detection_metrics, MetricAggregator

# 单张图像评估
metrics = compute_detection_metrics(
    predictions=detected_particles,
    ground_truth=true_particles,
    distance_threshold=10.0,
)
print(f"精确率: {metrics.precision:.3f}")
print(f"召回率: {metrics.recall:.3f}")
print(f"F1 分数: {metrics.f1_score:.3f}")

# 跨数据集聚合指标
aggregator = MetricAggregator(distance_threshold=10.0)
for preds, gts in zip(all_predictions, all_ground_truths):
    aggregator.add_image(preds, gts)
aggregate_metrics = aggregator.compute_aggregate()
```

## 输出格式

### STAR 格式

标准 RELION 兼容的 STAR 文件：

```
data_particles

loop_
_rlnCoordinateX
_rlnCoordinateY
_rlnMicrographName
_rlnAutopickFigureOfMerit
100.5    200.3    micrograph_001.tiff    0.95
150.2    300.1    micrograph_001.tiff    0.88
...
```

### JSON 格式

```json
{
  "micrograph_001.tiff": [
    {"x": 100.5, "y": 200.3, "score": 0.95, "width": 64, "height": 64},
    {"x": 150.2, "y": 300.1, "score": 0.88, "width": 64, "height": 64}
  ]
}
```

### CSV 格式

```csv
micrograph,x,y,score,width,height
micrograph_001.tiff,100.5,200.3,0.95,64,64
micrograph_001.tiff,150.2,300.1,0.88,64,64
```

## 数据工具

SuPicker 内置了 STAR 文件检查和拆分工具：

```bash
# 查看 STAR 文件统计信息
python scripts/star_tool.py info particles.star

# 列出所有 micrograph
python scripts/star_tool.py info particles.star --list

# 提取前/后 N 张图像
python scripts/star_tool.py split particles.star -n 10 -o subset.star
python scripts/star_tool.py split particles.star -n 50 --from-end -o val.star

# 一键拆分训练/验证集（可随机打乱）
python scripts/star_tool.py split-trainval particles.star \
    --val-images 50 \
    --train-output train.star --val-output val.star \
    --shuffle --seed 42
```

## 项目结构

```
supicker/
├── config/          # 配置数据类
├── data/            # 数据集与数据加载
├── engine/          # 训练与推理引擎
├── losses/          # 损失函数
├── models/          # 模型架构
│   ├── backbone/    # ConvNeXt 骨干网络
│   ├── fpn/         # 特征金字塔网络
│   └── head/        # CenterNet 检测头
└── utils/           # 工具类（日志、导出、评估指标）
scripts/
├── train.py         # 训练脚本
├── predict.py       # 推理脚本
└── star_tool.py     # STAR 文件检查与拆分工具
```

## 训练建议

- **批次大小**：建议 8-20，越大梯度越稳定，但受显存限制
- **学习率**：随 batch size 等比缩放。`batch_size=8` 用 `1e-4`，`batch_size=16+` 用 `2e-4`
- **裁剪尺寸**：默认 1024×1024，训练时自动随机裁剪以适应显存，可通过 `AugmentationConfig.crop_size` 调整
- **混合精度**：默认开启，节省 30-50% 显存并加速约 1.5 倍。如遇数值问题用 `--no-amp` 关闭
- **GPU 选择**：使用 `--device cuda:N` 指定 GPU，或用环境变量 `CUDA_VISIBLE_DEVICES=N`
- **多卡训练**：始终使用 `torchrun` 配合 `--distributed` 标志，不要同时使用 `--device`
- **恢复训练**：使用 `--resume ./checkpoints/...` 继续之前中断的训练。如果在训练时按下 `Ctrl+C`，框架会自动保存当前的进度到 checkpoint。
- **验证集指标**：如果提供了验证集，训练日志中的 `P`、`R`、`F1` 分别代表 精确率 (Precision)、召回率 (Recall) 和 F1 分数。

## 许可证

MIT 许可证
