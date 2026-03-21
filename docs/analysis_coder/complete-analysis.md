# SuPicker 项目完全解析

> 冷冻电镜颗粒拾取深度学习框架全面指南

**版本**: 0.1.0  
**最后更新**: 2026-03-21  
**项目地址**: https://github.com/your-org/supicker

---

## 📋 目录

- [一、项目概述](#一项目概述)
- [二、项目架构](#二项目架构)
- [三、核心模块详解](#三核心模块详解)
- [四、使用方法详解](#四使用方法详解)
- [五、注意事项与最佳实践](#五注意事项与最佳实践)
- [六、输出格式说明](#六输出格式说明)
- [七、扩展开发指南](#七扩展开发指南)
- [八、总结](#八总结)

---

## 一、项目概述

### 1.1 什么是 SuPicker？

**SuPicker** 是一个用于冷冻电镜 (Cryo-EM) 显微图像颗粒拾取的深度学习框架，采用 CenterNet 风格的无锚框目标检测架构。

### 1.2 核心特性

- ✅ **ConvNeXt 骨干网络**：现代化视觉 Transformer 风格架构，支持 Tiny/Small/Base 变体
- ✅ **特征金字塔网络 (FPN)**：多尺度特征融合，检测不同尺寸的颗粒
- ✅ **CenterNet 检测头**：无锚框设计，输出热图、尺寸和偏移预测
- ✅ **灵活数据管道**：支持 TIFF、MRC 等格式，STAR 文件标注
- ✅ **RandomCrop 训练**：自动随机裁剪，无需担心显存溢出
- ✅ **自动混合精度 (AMP)**：节省 30-50% 显存，加速训练约 1.5 倍
- ✅ **分布式训练**：多 GPU 支持（PyTorch DDP）
- ✅ **预训练权重**：可选 ImageNet 预训练 backbone
- ✅ **完整评估指标**：Precision、Recall、F1、Average Precision
- ✅ **中断保护**：Ctrl+C 自动保存检查点

### 1.3 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch ≥ 2.0.0 |
| 图像处理 | TiffFile, NumPy, PIL |
| 日志监控 | TensorBoard |
| 科学计算 | SciPy |
| 文件格式 | RELION STAR, JSON, CSV |

---

## 二、项目架构

### 2.1 整体架构图

```
输入图像 (B, 1, H, W)
    ↓
┌─────────────────────┐
│  ConvNeXt Backbone  │ → 输出 [C1, C2, C3, C4]
│  (特征提取)          │    strides: [4, 8, 16, 32]
└─────────────────────┘
    ↓
┌─────────────────────┐
│      FPN Neck       │ → 输出 [P2, P3, P4, P5]
│  (多尺度特征融合)     │   通道数：256
└─────────────────────┘
    ↓
┌─────────────────────┐
│   CenterNet Head    │ → heatmap, size, offset
│  (检测头)            │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   后处理 + NMS      │ → 颗粒坐标 (x, y, score)
└─────────────────────┘
```

### 2.2 项目结构

```
supicker/
├── config/              # 配置数据类
│   ├── __init__.py
│   ├── base.py         # 配置基类
│   ├── model.py        # 模型配置
│   ├── data.py         # 数据配置
│   ├── training.py     # 训练配置
│   └── inference.py    # 推理配置
│
├── models/              # 模型架构
│   ├── __init__.py
│   ├── detector.py     # 完整检测器
│   ├── backbone/
│   │   ├── __init__.py
│   │   └── convnext.py # ConvNeXt 骨干
│   ├── neck/
│   │   ├── __init__.py
│   │   └── fpn.py      # 特征金字塔
│   └── head/
│       ├── __init__.py
│       └── centernet.py # CenterNet 头
│
├── data/                # 数据处理
│   ├── __init__.py
│   ├── dataset.py      # 数据集类
│   ├── star_parser.py  # STAR 文件解析
│   ├── target_generator.py # 目标生成
│   └── transforms.py   # 数据增强
│
├── engine/              # 训练/推理引擎
│   ├── __init__.py
│   ├── trainer.py      # 训练器
│   └── predictor.py    # 预测器
│
├── losses/              # 损失函数
│   ├── __init__.py
│   ├── combined.py     # 组合损失
│   ├── focal_loss.py   # Focal 损失
│   └── regression_loss.py # 回归损失
│
├── utils/               # 工具函数
│   ├── __init__.py
│   ├── checkpoint.py   # 检查点管理
│   ├── logger.py       # 日志记录
│   ├── metrics.py      # 评估指标
│   ├── export.py       # 结果导出
│   └── coordinate_validation.py # 坐标验证
│
└── scripts/             # 命令行脚本
    ├── train.py        # 训练脚本
    ├── predict.py      # 预测脚本
    └── star_tool.py    # STAR 文件工具
```

---

## 三、核心模块详解

### 3.1 模型架构 (`models/`)

#### 3.1.1 Detector - 完整检测器

**文件**: `supicker/models/detector.py`

```python
class Detector(nn.Module):
    """完整的颗粒检测器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.backbone = ConvNeXt(config.backbone)  # 特征提取
        self.neck = FPN(config.fpn)                # 特征融合
        self.head = CenterNetHead(config.head)     # 预测输出
    
    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)           # [C1, C2, C3, C4]
        fpn_features = self.neck(features)    # [P2, P3, P4, P5]
        outputs = self.head(fpn_features[0])  # 使用 P2(最高分辨率)
        return {'heatmap': ..., 'size': ..., 'offset': ...}
```

**关键设计**：
- 输入：单通道灰度图 `(B, 1, H, W)`
- 输出 stride=4（相对原图下采样 4 倍）
- 使用 P2 层进行最终预测（保留最多空间信息）

---

#### 3.1.2 ConvNeXt 骨干网络

**文件**: `supicker/models/backbone/convnext.py`

**架构细节**：

```python
# 配置表
CONVNEXT_CONFIGS = {
    ConvNeXtVariant.TINY:  {"depths": [3, 3, 9, 3],  "dims": [96, 192, 384, 768]},
    ConvNeXtVariant.SMALL: {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
    ConvNeXtVariant.BASE:  {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
}
```

**网络结构**：

```
输入 (B, 1, H, W)
    ↓
Stem: Conv2d(1→96, 4×4, stride=4) + LayerNorm
    ↓ (1/4 分辨率)
Stage1: [ConvNeXtBlock × 3] → C1
    ↓
Downsample: Conv2d(96→192, 2×2, stride=2)
    ↓ (1/8 分辨率)
Stage2: [ConvNeXtBlock × 3] → C2
    ↓
Downsample: Conv2d(192→384, 2×2, stride=2)
    ↓ (1/16 分辨率)
Stage3: [ConvNeXtBlock × 9/27] → C3
    ↓
Downsample: Conv2d(384→768/1024, 2×2, stride=2)
    ↓ (1/32 分辨率)
Stage4: [ConvNeXtBlock × 3] → C4
```

**ConvNeXt Block 结构**：

```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)  # 深度卷积
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4*dim)  # 扩展
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4*dim, dim)  # 压缩
        self.gamma = nn.Parameter(1e-6 * ones(dim))  # LayerScale
        self.drop_path = DropPath(drop_path)
    
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)  # NCHW → NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0,3,1,2)  # NHWC → NCHW
        return shortcut + self.drop_path(x)
```

**预训练权重适配**：

```python
def _adapt_input_channels(self, state_dict):
    """将 RGB 预训练权重适配到灰度输入"""
    stem_weight = state_dict['stem.0.weight']  # (out_ch, 3, H, W)
    gray_weight = stem_weight.mean(dim=1, keepdim=True)  # 平均 RGB 通道
    if self.config.in_channels > 1:
        gray_weight = gray_weight.repeat(1, self.config.in_channels, 1, 1)
    state_dict['stem.0.weight'] = gray_weight
```

---

#### 3.1.3 FPN 颈部网络

**文件**: `supicker/models/neck/fpn.py`

**结构图**：

```
C4 (768ch) → lateral_conv(1×1) → P4 (256ch)
                   ↑                    ↑
            upsample(×2)         output_conv(3×3)
                   ↑
C3 (384ch) → lateral_conv(1×1) → P3 (256ch)
                   ↑                    ↑
            upsample(×2)         output_conv(3×3)
                   ↑
C2 (192ch) → lateral_conv(1×1) → P2 (256ch)
                                      ↓
                                 output_conv(3×3)
```

**代码实现**：

```python
class FPN(nn.Module):
    def __init__(self, config):
        self.lateral_convs = ModuleList([
            Conv2d(in_ch, 256, 1) for in_ch in [96, 192, 384, 768]
        ])
        self.output_convs = ModuleList([
            Conv2d(256, 256, 3, padding=1) for _ in range(4)
        ])
    
    def forward(self, features):
        # 侧向连接
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # 自顶向下融合
        for i in range(3, 0, -1):
            upsampled = interpolate(laterals[i], size=laterals[i-1].shape[2:])
            laterals[i-1] = laterals[i-1] + upsampled
        
        # 输出卷积
        return [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
```

---

#### 3.1.4 CenterNet 检测头

**文件**: `supicker/models/head/centernet.py`

**三个预测分支**：

```python
class CenterNetHead(nn.Module):
    def __init__(self, config, in_channels=256):
        feat_channels = 256
        
        # 共享卷积层
        self.shared_conv = Sequential(
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True)
        )
        
        # 热图分支 (预测中心点概率)
        self.heatmap_head = Sequential(
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, num_classes, 1),
            Sigmoid()  # 输出范围 [0, 1]
        )
        
        # 尺寸分支 (预测宽高)
        self.size_head = Sequential(
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 2, 1),
            ReLU()  # 确保尺寸为正
        )
        
        # 偏移分支 (亚像素校正)
        self.offset_head = Sequential(
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, 2, 1)  # 无激活，可正可负
        )
        
        # 偏置初始化技巧
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)  # sigmoid(-2.19) ≈ 0.1
```

**前向传播**：

```python
def forward(self, feature):
    x = self.shared_conv(feature)
    heatmap = self.heatmap_head(x)  # (B, num_classes, H, W)
    size = self.size_head(x)        # (B, 2, H, W) → width, height
    offset = self.offset_head(x)    # (B, 2, H, W) → offset_x, offset_y
    return {'heatmap': heatmap, 'size': size, 'offset': offset}
```

---

### 3.2 数据管道 (`data/`)

#### 3.2.1 STAR 文件解析

**文件**: `supicker/data/star_parser.py`

**解析流程**：

```python
def parse_star_file(star_path: Path) -> dict[str, list[dict]]:
    """
    返回：{micrograph_name: [{x, y, score, class_id, ...}, ...]}
    """
    particles_by_mic = {}
    
    with open(star_path) as f:
        lines = f.readlines()
    
    # 解析 header，找到列索引
    column_indices = {}
    for i, line in enumerate(lines):
        if line.startswith('_rln'):
            col_name = line.split()[0]
            column_indices[col_name] = len(column_indices)
        elif line.strip() and not line.startswith('_'):
            # 数据行
            parts = line.split()
            mic_name = parts[column_indices['_rlnMicrographName']]
            particle = {
                'x': float(parts[column_indices['_rlnCoordinateX']]),
                'y': float(parts[column_indices['_rlnCoordinateY']]),
                'score': float(parts.get('_rlnAutopickFigureOfMerit', 1.0)),
            }
            particles_by_mic.setdefault(mic_name, []).append(particle)
    
    return particles_by_mic
```

---

#### 3.2.2 ParticleDataset

**文件**: `supicker/data/dataset.py`

**关键方法**：

```python
class ParticleDataset(Dataset):
    def __getitem__(self, idx):
        # 1. 加载图像
        image = self._load_image(image_path)  # (H, W) → float32 [0,1]
        
        # 2. 获取颗粒标注
        particles = self.particles_by_micrograph[image_name]
        
        # 3. 应用数据增强（同步变换图像和坐标）
        if self.transforms:
            image, particles = self.transforms(image, particles)
        
        # 4. 生成训练目标
        targets = self.target_generator(particles, image_size=image.shape[1:])
        
        return {
            'image': image,              # (1, H, W)
            'heatmap': targets['heatmap'],  # (num_classes, H/4, W/4)
            'size': targets['size'],        # (2, H/4, W/4)
            'offset': targets['offset'],    # (2, H/4, W/4)
            'mask': targets['mask'],        # (H/4, W/4)
            'particles': particles,         # 原始颗粒列表
        }
```

**自定义 Collate 函数**：

```python
def particle_collate_fn(batch):
    """处理变长颗粒列表"""
    collated = {}
    
    # 张量字段直接堆叠
    for key in ['image', 'heatmap', 'size', 'offset', 'mask']:
        collated[key] = torch.stack([sample[key] for sample in batch])
    
    # 颗粒列表保持为 list of lists
    collated['particles'] = [sample['particles'] for sample in batch]
    
    return collated
```

---

#### 3.2.3 数据增强

**文件**: `supicker/data/transforms.py`

**增强类型**：

```python
@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True      # 水平翻转
    vertical_flip: bool = True        # 垂直翻转
    rotation_90: bool = True          # 90 度旋转
    random_rotation: bool = True      # 任意角度旋转
    rotation_range: tuple = (-180.0, 180.0)
    brightness: bool = True           # 亮度调整
    brightness_range: tuple = (0.8, 1.2)
    contrast: bool = True             # 对比度调整
    contrast_range: tuple = (0.8, 1.2)
    gaussian_noise: bool = True       # 高斯噪声
    noise_std: float = 0.02
    crop_size: int = 1024             # 随机裁剪尺寸
```

**随机裁剪实现**：

```python
class RandomCrop:
    def __init__(self, crop_size=1024):
        self.crop_size = crop_size
    
    def apply(self, image, particles):
        h, w = image.shape[-2:]
        
        # 随机选择裁剪起点
        top = random.randint(0, max(0, h - self.crop_size))
        left = random.randint(0, max(0, w - self.crop_size))
        
        # 裁剪图像
        image = image[..., top:top+self.crop_size, left:left+self.crop_size]
        
        # 调整颗粒坐标
        for p in particles:
            p['x'] -= left
            p['y'] -= top
        
        # 过滤掉在裁剪区域外的颗粒
        particles = [p for p in particles 
                     if 0 <= p['x'] < self.crop_size and 0 <= p['y'] < self.crop_size]
        
        return image, particles
```

---

#### 3.2.4 目标生成器

**文件**: `supicker/data/target_generator.py`

**生成热图**：

```python
class TargetGenerator:
    def __init__(self, num_classes=1, output_stride=4, gaussian_sigma=2.0):
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.gaussian_sigma = gaussian_sigma
    
    def __call__(self, particles, image_size):
        h, w = image_size
        out_h, out_w = h // self.output_stride, w // self.output_stride
        
        # 初始化热图（全 0）
        heatmap = torch.zeros(self.num_classes, out_h, out_w)
        size = torch.zeros(2, out_h, out_w)
        offset = torch.zeros(2, out_h, out_w)
        mask = torch.zeros(out_h, out_w)
        
        for p in particles:
            # 映射到输出分辨率
            x_out = p['x'] / self.output_stride
            y_out = p['y'] / self.output_stride
            
            x_int, y_int = int(x_out), int(y_out)
            
            # 绘制 2D 高斯分布
            self._draw_gaussian(heatmap[0], (x_int, y_int), sigma=self.gaussian_sigma)
            
            # 存储尺寸
            size[0, y_int, x_int] = p.get('width', 64)
            size[1, y_int, x_int] = p.get('height', 64)
            
            # 存储偏移残差
            offset[0, y_int, x_int] = x_out - x_int
            offset[1, y_int, x_int] = y_out - y_int
            
            # 标记有效区域
            mask[y_int, x_int] = 1
        
        return {'heatmap': heatmap, 'size': size, 'offset': offset, 'mask': mask}
```

---

### 3.3 训练引擎 (`engine/trainer.py`)

#### 3.3.1 Trainer 初始化

```python
class Trainer:
    def __init__(self, model, config, checkpoint_dir, log_dir, device='cuda'):
        self.config = config
        self.device = device
        
        # 分布式训练设置
        self.is_distributed = config.distributed
        if self.is_distributed:
            self._setup_distributed()
            model = DDP(model, device_ids=[self.local_rank])
        
        self.model = model.to(device)
        self.criterion = CombinedLoss(config.loss)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # AMP 混合精度
        self.use_amp = config.use_amp and 'cuda' in device
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        
        # 检查点和日志（仅主进程）
        self.checkpoint_manager = CheckpointManager(checkpoint_dir) if self.is_main_process else None
        self.logger = Logger(log_dir) if self.is_main_process else None
```

---

#### 3.3.2 训练循环

**单步训练**：

```python
def train_step(self, batch):
    self.model.train()
    
    # 准备数据
    image = batch['image'].to(self.device)
    targets = {
        'heatmap': batch['heatmap'].to(self.device),
        'size': batch['size'].to(self.device),
        'offset': batch['offset'].to(self.device),
        'mask': batch['mask'].to(self.device),
    }
    
    # 前向传播（AMP）
    self.optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
        outputs = self.model(image)
        loss, loss_dict = self.criterion(outputs, targets)
    
    # 反向传播（GradScaler）
    self.scaler.scale(loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
    
    return loss, loss_dict
```

**完整训练流程**：

```python
def train(self, train_loader, val_loader=None, epochs=None):
    epochs = epochs or self.config.epochs
    
    try:
        for epoch in range(self.current_epoch, epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            if val_loader and (epoch + 1) % self.config.val_interval == 0:
                val_loss, val_metrics = self.validate(val_loader, compute_metrics=True)
            
            # 更新学习率
            if self.scheduler and epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # 日志记录
            if self.is_main_process and self.logger:
                self.logger.log_epoch(
                    epoch=epoch+1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    lr=self.optimizer.param_groups[0]['lr']
                )
            
            # 保存检查点
            if self.is_main_process and (epoch + 1) % self.config.save_interval == 0:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch+1,
                    loss=val_loss or train_loss
                )
    
    except KeyboardInterrupt:
        # Ctrl+C 中断保护
        if self.is_main_process:
            print(f"\n训练在 epoch {self.current_epoch} 被中断，保存检查点...")
            self.checkpoint_manager.save(...)
            print("检查点已保存，可使用 --resume 恢复训练")
```

---

#### 3.3.3 验证与评估

```python
def validate(self, val_loader, compute_metrics=True):
    self.model.eval()
    total_loss = 0.0
    metric_aggregator = MetricAggregator(self.config.val_distance_threshold)
    
    with torch.no_grad():
        for batch in val_loader:
            image = batch['image'].to(self.device)
            targets = {...}
            
            # 前向传播
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                outputs = self.model(image)
                loss, _ = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # 计算检测指标
            if compute_metrics:
                predictions = self._extract_predictions(
                    outputs,
                    score_threshold=self.config.val_score_threshold,
                    nms_radius=self.config.val_nms_radius
                )
                ground_truth = batch['particles']
                metric_aggregator.add_image(predictions, ground_truth)
    
    avg_loss = total_loss / len(val_loader)
    metrics = metric_aggregator.compute_aggregate()
    
    return avg_loss, {
        'precision': metrics.precision,
        'recall': metrics.recall,
        'f1_score': metrics.f1_score,
        'ap': metrics.ap
    }
```

---

### 3.4 推理引擎 (`engine/predictor.py`)

#### 3.4.1 Predictor 解码流程

```python
class Predictor:
    def predict(self, image: torch.Tensor) -> list[dict]:
        # 确保有 batch 维度
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(image.to(self.device))
        
        # 解码输出
        return self.decode_outputs(
            outputs,
            score_threshold=self.config.score_threshold,
            nms_enabled=self.config.nms_enabled,
            nms_radius=self.config.nms_radius,
            output_stride=4
        )
```

---

#### 3.4.2 热图峰值提取

```python
@staticmethod
def extract_peaks_from_heatmap(heatmap, score_threshold, min_distance=1):
    batch_size, num_classes, h, w = heatmap.shape
    particles = []
    
    # 最大池化找局部最大值
    kernel_size = 2 * min_distance + 1
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=min_distance)
    
    # 保留局部最大值且超过阈值
    keep = (heatmap == hmax) & (heatmap >= score_threshold)
    
    for b in range(batch_size):
        for c in range(num_classes):
            y_coords, x_coords = torch.where(keep[b, c])
            
            for y, x in zip(y_coords.tolist(), x_coords.tolist()):
                score = float(heatmap[b, c, y, x])
                particles.append({
                    'x': x, 'y': y, 'score': score, 'class_id': c, 'batch_idx': b
                })
    
    # 按分数降序排序
    particles.sort(key=lambda p: p['score'], reverse=True)
    return particles
```

---

#### 3.4.3 NMS（非极大值抑制）

```python
@staticmethod
def apply_nms_to_particles(particles, radius):
    if not particles:
        return []
    
    radius_sq = radius ** 2
    particles = sorted(particles, key=lambda p: p['score'], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i, p in enumerate(particles):
        if i in suppressed:
            continue
        
        keep.append(p)
        
        # 抑制附近的低分粒子
        for j in range(i + 1, len(particles)):
            if j in suppressed:
                continue
            
            other = particles[j]
            if p['batch_idx'] != other['batch_idx']:
                continue
            if p['class_id'] != other['class_id']:
                continue
            
            dist_sq = (p['x'] - other['x'])**2 + **(p['y'] - other['y'])2
            if dist_sq < radius_sq:
                suppressed.add(j)
    
    return keep
```

---

### 3.5 损失函数 (`losses/`)

#### 3.5.1 组合损失

```python
class CombinedLoss(nn.Module):
    def __init__(self, config):
        # 热图损失（默认 Focal Loss）
        self.heatmap_loss = FocalLoss(alpha=2.0, beta=4.0)
        
        # 尺寸损失（默认 L1）
        self.size_loss = RegL1Loss()
        
        # 偏移损失（默认 L1）
        self.offset_loss = RegL1Loss()
        
        # 损失权重
        self.heatmap_weight = 1.0
        self.size_weight = 0.1
        self.offset_weight = 1.0
    
    def forward(self, outputs, targets):
        heatmap_loss = self.heatmap_loss(outputs['heatmap'], targets['heatmap'])
        size_loss = self.size_loss(outputs['size'], targets['size'], targets['mask'])
        offset_loss = self.offset_loss(outputs['offset'], targets['offset'], targets['mask'])
        
        total_loss = (
            self.heatmap_weight * heatmap_loss +
            self.size_weight * size_loss +
            self.offset_weight * offset_loss
        )
        
        return total_loss, {
            'heatmap_loss': heatmap_loss.item(),
            'size_loss': size_loss.item(),
            'offset_loss': offset_loss.item(),
            'total_loss': total_loss.item()
        }
```

---

#### 3.5.2 Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        self.alpha = alpha  # 平衡因子
        self.beta = beta    # 调制因子
    
    def forward(self, pred, target):
        # pred: sigmoid 后的热图，target: 0-1 热图
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        
        pos_loss = -self.alpha * (1 - pred)**self.beta * torch.log(pred + 1e-8) * pos_mask
        neg_loss = -(1 - self.alpha) * pred**self.beta * torch.log(1 - pred + 1e-8) * neg_mask
        
        return (pos_loss.sum() + neg_loss.sum()) / max(pos_mask.sum(), 1)
```

---

### 3.6 评估指标 (`utils/metrics.py`)

#### 3.6.1 匈牙利算法匹配

```python
def match_particles_by_distance(predictions, ground_truth, threshold):
    if len(predictions) == 0 or len(ground_truth) == 0:
        return [], list(range(len(predictions))), list(range(len(ground_truth)))
    
    # 构建距离矩阵
    cost_matrix = np.zeros((len(predictions), len(ground_truth)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            dx = pred['x'] - gt['x']
            dy = pred['y'] - gt['y']
            cost_matrix[i, j] = np.sqrt(dx*dx + dy*dy)
    
    # 匈牙利算法
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # 过滤超过阈值的匹配
    matched_pairs = []
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] <= threshold:
            matched_pairs.append((pred_idx, gt_idx))
    
    # 找出未匹配的
    matched_pred_set = {p[0] for p in matched_pairs}
    matched_gt_set = {p[1] for p in matched_pairs}
    
    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_pred_set]
    unmatched_gts = [i for i in range(len(ground_truth)) if i not in matched_gt_set]
    
    return matched_pairs, unmatched_preds, unmatched_gts
```

---

#### 3.6.2 检测指标计算

```python
def compute_detection_metrics(predictions, ground_truth, distance_threshold=10.0):
    matched, unmatched_preds, unmatched_gts = match_particles_by_distance(
        predictions, ground_truth, distance_threshold
    )
    
    tp = len(matched)
    fp = len(unmatched_preds)
    fn = len(unmatched_gts)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 平均匹配距离
    if matched:
        total_dist = sum(
            np.sqrt((predictions[i]['x']-ground_truth[j]['x'])**2 + 
                    **(predictions[i]['y']-ground_truth[j]['y'])2)
            for i, j in matched
        )
        avg_distance = total_dist / len(matched)
    else:
        avg_distance = 0.0
    
    return DetectionMetrics(
        precision=precision, recall=recall, f1_score=f1_score,
        true_positives=tp, false_positives=fp, false_negatives=fn,
        avg_distance=avg_distance
    )
```

---

## 四、使用方法详解

### 4.1 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/supicker.git
cd supicker

# 基础安装
pip install -e .

# 开发模式（带测试工具）
pip install -e ".[dev]"

# 支持 MRC 格式
pip install -e ".[mrc]"
```

**依赖要求**：
- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- CUDA 11.7+（推荐）

---

### 4.2 训练流程

#### 4.2.1 准备数据

```bash
# 查看 STAR 文件信息
python scripts/star_tool.py info particles.star

# 列出所有微镜图像
python scripts/star_tool.py info particles.star --list

# 划分训练集/验证集（随机打散）
python scripts/star_tool.py split-trainval particles.star \
    --val-images 50 \
    --train-output train.star \
    --val-output val.star \
    --shuffle --seed 42
```

---

#### 4.2.2 基础训练

```bash
# 单 GPU 训练
python scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --backbone tiny \
    --epochs 100 \
    --batch-size 8 \
    --pretrained \
    --device cuda:0
```

**输出示例**：
```
Creating model with tiny backbone...
Loading training data from ./data/micrographs...
  Found 500 training images
Starting training for 100 epochs...
  Device: cuda:0
  Batch size: 8
  Learning rate: 0.0001

Epoch 1/100: train_loss=0.523, val_loss=0.412, P=0.78, R=0.85, F1=0.81
Epoch 2/100: train_loss=0.398, val_loss=0.389, P=0.82, R=0.87, F1=0.84
...
```

---

#### 4.2.3 带验证的训练

```bash
python scripts/train.py \
    --train-images ./data/train \
    --train-star ./data/train.star \
    --val-images ./data/val \
    --val-star ./data/val.star \
    --backbone small \
    --epochs 150 \
    --batch-size 16 \
    --lr 2e-4 \
    --pretrained \
    --save-interval 10 \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs
```

---

#### 4.2.4 多 GPU 分布式训练

```bash
# 4 GPU 训练
torchrun --nproc_per_node=4 scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --distributed

# 指定特定 GPU（例如只用 GPU 0 和 2）
CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --distributed
```

**注意**：
- 必须使用 `torchrun` 启动
- 不要同时使用 `--device` 和 `--distributed`
- 自动使用 SyncBatchNorm 跨 GPU 同步统计量

---

#### 4.2.5 恢复训练

```bash
# 从第 50 轮的检查点继续
python scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --resume ./checkpoints/checkpoint_50.pth
```

**中断保护**：训练时按 `Ctrl+C` 会自动保存当前状态

---

### 4.3 推理流程

#### 4.3.1 单张图像预测

```bash
python scripts/predict.py \
    --checkpoint ./checkpoints/best.pth \
    --input ./data/test/micrograph_001.tiff \
    --output ./results \
    --format star \
    --threshold 0.3 \
    --nms-radius 20
```

---

#### 4.3.2 批量预测

```bash
python scripts/predict.py \
    --checkpoint ./checkpoints/best.pth \
    --input ./data/test_micrographs \
    --output ./results \
    --format star \
    --merge-output ./results/all_particles.star
```

**输出**：
- 每张图像一个文件：`micrograph_001.star`, `micrograph_002.star`, ...
- 合并文件：`all_particles.star`（包含所有图像的颗粒）

---

#### 4.3.3 Python API 推理

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
    device="cuda:0"
)

# 加载并预处理图像
image = tifffile.imread("micrograph.tiff").astype("float32")
image = (image - image.min()) / (image.max() - image.min() + 1e-8)
image = torch.from_numpy(image).unsqueeze(0)  # (1, 1, H, W)

# 运行预测
particles = predictor.predict(image)
print(f"检测到 {len(particles)} 个颗粒")

# 导出结果
export_to_star(particles, "output.star", micrograph_name="micrograph.tiff")
```

---

### 4.4 评估指标

#### 4.4.1 单图像评估

```python
from supicker.utils.metrics import compute_detection_metrics

# 加载预测和真实标注
predictions = [...]  # [{x, y, score, ...}, ...]
ground_truth = [...] # [{x, y, ...}, ...]

metrics = compute_detection_metrics(
    predictions=predictions,
    ground_truth=ground_truth,
    distance_threshold=10.0  # 匹配距离阈值（像素）
)

print(f"Precision: {metrics.precision:.3f}")
print(f"Recall: {metrics.recall:.3f}")
print(f"F1 Score: {metrics.f1_score:.3f}")
print(f"AP: {metrics.ap:.3f}")
print(f"平均匹配距离：{metrics.avg_distance:.2f}px")
```

---

#### 4.4.2 数据集级别评估

```python
from supicker.utils.metrics import MetricAggregator

aggregator = MetricAggregator(distance_threshold=10.0)

for preds, gts in zip(all_predictions, all_ground_truths):
    aggregator.add_image(preds, gts)

aggregate_metrics = aggregator.compute_aggregate(compute_ap=True)

print(f"数据集级别指标:")
print(f"  Precision: {aggregate_metrics.precision:.3f}")
print(f"  Recall: {aggregate_metrics.recall:.3f}")
print(f"  F1 Score: {aggregate_metrics.f1_score:.3f}")
print(f"  AP: {aggregate_metrics.ap:.3f}")
```

---

## 五、注意事项与最佳实践

### 5.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | 8GB 显存 | 16-24GB 显存 (V100/A100/RTX 3090+) |
| **批次大小** | 4 | 8-20 |
| **训练时间** | ~12 小时 | 6-8 小时 (V100/A100) |
| **CPU** | 4 核 | 8-16 核 |
| **内存** | 16GB | 32-64GB |
| **存储** | SSD 推荐 | NVMe SSD |

---

### 5.2 超参数调优指南

#### 5.2.1 批次大小与学习率

**经验法则**：

```python
# 线性缩放规则
batch_size=8   → lr=1e-4
batch_size=16  → lr=2e-4
batch_size=32  → lr=4e-4
```

**原理**：大批次需要更高学习率以保持梯度方差稳定

**建议**：
- 从 `batch_size=8, lr=1e-4` 开始
- 如果显存允许，逐步增大 batch_size 并相应调整 lr
- 观察训练损失曲线，如果震荡则降低 lr

---

#### 5.2.2 数据增强配置

**强增强（推荐）**：

```python
AugmentationConfig(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_90=True,
    random_rotation=True,
    rotation_range=(-180.0, 180.0),
    brightness=True,
    brightness_range=(0.8, 1.2),
    contrast=True,
    contrast_range=(0.8, 1.2),
    gaussian_noise=True,
    noise_std=0.02,
    crop_size=1024
)
```

**弱增强（快速实验）**：

```python
AugmentationConfig(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_90=False,
    random_rotation=False,
    brightness=False,
    contrast=False,
    gaussian_noise=False,
    crop_size=1024
)
```

**禁用增强**：

```bash
python scripts/train.py ... --no-augmentation
```

---

#### 5.2.3 损失权重调整

**默认配置**：

```python
LossConfig(
    heatmap_weight=1.0,
    size_weight=0.1,
    offset_weight=1.0
)
```

**如果热图训练不稳定**：

```python
LossConfig(
    heatmap_weight=2.0,   # 增加热图权重
    size_weight=0.05,     # 减小尺寸权重
    offset_weight=1.0
)
```

**如果定位不准**：

```python
LossConfig(
    heatmap_weight=1.0,
    size_weight=0.2,      # 增加尺寸权重
    offset_weight=2.0     # 增加偏移权重
)
```

---

### 5.3 常见问题排查

#### 5.3.1 OOM（显存不足）

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate ...
```

**解决方案**：

1. **减小批次大小**
   ```bash
   python scripts/train.py ... --batch-size 4
   ```

2. **启用 AMP（默认已开启）**
   ```bash
   # 不要加 --no-amp
   ```

3. **减小裁剪尺寸**
   ```python
   # 修改配置文件
   AugmentationConfig(crop_size=512)  # 默认 1024
   ```

4. **使用更小的 backbone**
   ```bash
   python scripts/train.py ... --backbone tiny  # vs small/base
   ```

5. **梯度累积（需自行实现）**
   ```python
   # 多次前向后一次反向
   for i, batch in enumerate(loader):
       loss = train_step(batch)
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

---

#### 5.3.2 训练损失震荡

**可能原因**：
- 学习率过高
- 批次太小
- 数据增强过强
- 热身轮数不足

**解决方法**：

```python
TrainingConfig(
    learning_rate=5e-5,   # 降低学习率
    batch_size=16,        # 增大批次
    warmup_epochs=10,     # 延长热身（默认 5）
    scheduler='cosine'    # 使用余弦退火
)
```

---

#### 5.3.3 检测效果差

**诊断步骤**：

1. **检查数据质量**
   ```bash
   python scripts/star_tool.py info particles.star --list
   ```
   
   - 确认微镜图像文件名正确
   - 检查颗粒数量是否合理

2. **可视化验证**
   ```python
   # 在训练集中抽样，可视化热图和原图叠加
   import matplotlib.pyplot as plt
   
   sample = dataset[0]
   plt.imshow(sample['image'][0].numpy(), cmap='gray')
   plt.imshow(sample['heatmap'][0].numpy(), alpha=0.5, cmap='hot')
   plt.show()
   ```

3. **调整检测阈值**
   ```bash
   python scripts/predict.py ... --threshold 0.2  # 降低阈值（默认 0.3）
   ```

4. **增加训练轮数**
   ```bash
   python scripts/train.py ... --epochs 200
   ```

5. **检查验证集指标**
   - 如果验证集 F1 远低于训练集 → 过拟合
   - 如果两者都低 → 欠拟合或数据质量问题

---

### 5.4 数据格式规范

#### 5.4.1 图像格式

**支持的格式**：

| 格式 | 扩展名 | 推荐度 | 备注 |
|------|--------|--------|------|
| TIFF | `.tiff`, `.tif` | ⭐⭐⭐⭐⭐ | 无损，支持大文件 |
| MRC | `.mrc` | ⭐⭐⭐⭐ | Cryo-EM 标准格式 |
| PNG | `.png` | ⭐⭐ | 无损但文件大 |
| JPEG | `.jpg` | ⭐ | 有损压缩，不推荐 |

**预处理要求**：
- 单通道灰度图
- 浮点类型（float32）
- 值域归一化到 [0, 1]

---

#### 5.4.2 STAR 文件格式

**必需列**：
```star
_rlnCoordinateX        # X 坐标（像素）
_rlnCoordinateY        # Y 坐标（像素）
_rlnMicrographName     # 微镜图像文件名
```

**可选列**：
```star
_rlnAutopickFigureOfMerit  # 置信度分数（0-1）
_rlnClassNumber            # 颗粒类别 ID
_rlnAnglePsi               # 旋转角度
```

**坐标系统**：
- 原点：左上角
- X 轴：向右增加
- Y 轴：向下增加
- 单位：像素
- 类型：浮点数（支持亚像素精度）

**示例**：
```star
data_particles

loop_
_rlnCoordinateX
_rlnCoordinateY
_rlnMicrographName
_rlnAutopickFigureOfMerit
100.5    200.3    micrograph_001.tiff    0.95
150.2    300.1    micrograph_001.tiff    0.88
250.7    400.9    micrograph_002.tiff    0.92
```

---

### 5.5 性能优化建议

#### 5.5.1 训练加速

1. **使用预训练权重**
   ```bash
   python scripts/train.py ... --pretrained
   ```
   - 收敛更快
   - 小数据集效果更好

2. **混合精度训练（AMP）**
   ```bash
   # 默认开启，无需额外参数
   # 可节省 30-50% 显存，加速约 1.5 倍
   ```

3. **增加 DataLoader workers**
   ```bash
   python scripts/train.py ... --num-workers 8
   ```

4. **使用更快的存储**
   - NVMe SSD > SATA SSD > HDD
   - 考虑将数据加载到 RAM disk

---

#### 5.5.2 推理加速

1. **批量推理**
   ```bash
   python scripts/predict.py ... --batch-size 4
   ```

2. **禁用 NMS（如果需要速度）**
   ```bash
   python scripts/predict.py ... --no-nms
   ```

3. **降低输入分辨率（需重新训练）**
   - 训练时使用较小的 `crop_size`
   - 推理时速度更快

4. **模型导出（未来支持）**
   - ONNX Runtime
   - TensorRT

---

## 六、输出格式说明

### 6.1 STAR 格式（推荐）

**文件扩展名**：`.star`

**格式**：
```star
data_particles

loop_
_rlnCoordinateX
_rlnCoordinateY
_rlnMicrographName
_rlnAutopickFigureOfMerit
100.5    200.3    micrograph_001.tiff    0.95
150.2    300.1    micrograph_001.tiff    0.88
```

**特点**：
- ✅ RELION 原生支持
- ✅ 可直接导入后续处理流程
- ✅ 支持元数据
- ❌ 人类可读性较差

**使用场景**：生产环境、RELION 集成

---

### 6.2 JSON 格式

**文件扩展名**：`.json`

**格式**：
```json
{
  "micrograph": "micrograph_001.tiff",
  "num_particles": 2,
  "particles": [
    {
      "x": 100.5,
      "y": 200.3,
      "score": 0.95,
      "width": 64,
      "height": 64,
      "class_id": 0
    },
    {
      "x": 150.2,
      "y": 300.1,
      "score": 0.88,
      "width": 64,
      "height": 64,
      "class_id": 0
    }
  ]
}
```

**特点**：
- ✅ 人类可读
- ✅ 易于解析（所有语言支持）
- ✅ 支持嵌套结构
- ❌ 文件体积较大

**使用场景**：Web 应用、Python 脚本集成、调试

---

### 6.3 CSV 格式

**文件扩展名**：`.csv`

**格式**：
```csv
micrograph,x,y,score,class_id,width,height
micrograph_001.tiff,100.5,200.3,0.95,0,64,64
micrograph_001.tiff,150.2,300.1,0.88,0,64,64
```

**特点**：
- ✅ 人类可读
- ✅ Excel 可直接打开
- ✅ 文件体积小
- ❌ 不支持嵌套结构
- ❌ 需要记住列顺序

**使用场景**：Excel 分析、其他软件导入

---

## 七、扩展开发指南

### 7.1 添加新 Backbone

**步骤 1：实现新骨干网络**

```python
# supicker/models/backbone/resnet.py
import torch.nn as nn
from torchvision.models import resnet50

class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 加载预训练 ResNet
        backbone = resnet50(pretrained=config.pretrained)
        
        # 修改第一层以支持单通道输入
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        # 保留 4 个阶段
        self.layer1 = backbone.layer1  # C1
        self.layer2 = backbone.layer2  # C2
        self.layer3 = backbone.layer3  # C3
        self.layer4 = backbone.layer4  # C4
    
    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)  # C1
        
        x = self.layer2(x)
        features.append(x)  # C2
        
        x = self.layer3(x)
        features.append(x)  # C3
        
        x = self.layer4(x)
        features.append(x)  # C4
        
        return features
```

---

**步骤 2：注册配置**

```python
# supicker/config/model.py
class BackboneConfig:
    variant: str = "resnet50"  # 新增变体
    pretrained: bool = True
    in_channels: int = 1
```

---

**步骤 3：更新 Detector**

```python
# supicker/models/detector.py
from .backbone.resnet import ResNet

class Detector(nn.Module):
    def __init__(self, config):
        # 根据配置选择 backbone
        if config.backbone.variant == "resnet50":
            self.backbone = ResNet(config.backbone)
        elif config.backbone.variant == "convnext_tiny":
            self.backbone = ConvNeXt(config.backbone)
        # ...
```

---

### 7.2 自定义数据增强

**步骤 1：实现新变换**

```python
# supicker/data/transforms.py
import random
import numpy as np

class CTFSimulation:
    """模拟 CTF（衬度传递函数）效应"""
    
    def __init__(self, defocus_range=(1.0, 3.0), voltage=300):
        self.defocus_range = defocus_range
        self.voltage = voltage
    
    def apply(self, image, particles):
        # 随机选择离焦值
        defocus = random.uniform(*self.defocus_range)
        
        # 计算 CTF（简化版）
        h, w = image.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        r = torch.sqrt((x - w/2)**2 + **(y - h/2)2)
        
        # 应用 CTF 调制
        ctf = torch.sin(2 * np.pi * defocus * r**2 / (2 * self.voltage))
        image = image * ctf
        
        return image, particles
```

---

**步骤 2：添加到配置**

```python
# supicker/config/data.py
@dataclass
class AugmentationConfig:
    # ... 现有字段 ...
    ctf_simulation: bool = False
    defocus_range: tuple = (1.0, 3.0)
```

---

**步骤 3：集成到 transforms 构建器**

```python
# supicker/data/transforms.py
def build_transforms(config: AugmentationConfig):
    transforms = []
    
    if config.crop_size > 0:
        transforms.append(RandomCrop(config.crop_size))
    
    if config.horizontal_flip:
        transforms.append(RandomHorizontalFlip())
    
    # ... 其他变换 ...
    
    if config.ctf_simulation:
        transforms.append(CTFSimulation(config.defocus_range))
    
    return Compose(transforms)
```

---

### 7.3 新的损失函数

**步骤 1：实现损失**

```python
# supicker/losses/dice_loss.py
import torch.nn as nn

class DiceLoss(nn.Module):
    """Dice Loss 用于热图预测"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred: sigmoid 后的热图 (B, C, H, W)
        # target: 0-1 热图 (B, C, H, W)
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return (1.0 - dice.mean())
```

---

**步骤 2：集成到 CombinedLoss**

```python
# supicker/losses/combined.py
from .dice_loss import DiceLoss

class CombinedLoss(nn.Module):
    def __init__(self, config):
        # ... 现有代码 ...
        
        if config.heatmap_type == "dice":
            self.heatmap_loss = DiceLoss()
        elif config.heatmap_type == "focal":
            self.heatmap_loss = FocalLoss(...)
        # ...
```

---

**步骤 3：更新配置**

```python
# supicker/config/training.py
@dataclass
class LossConfig:
    heatmap_type: str = "dice"  # 改为 dice
    heatmap_weight: float = 1.0
    # ...
```

---

### 7.4 自定义评估指标

**步骤 1：实现新指标**

```python
# supicker/utils/metrics.py
def compute_localization_accuracy(predictions, ground_truth, threshold=5.0):
    """
    计算定位准确率（严格匹配）
    
    Args:
        predictions: 预测颗粒列表
        ground_truth: 真实颗粒列表
        threshold: 严格匹配阈值（像素）
    
    Returns:
        localization_accuracy: 定位准确率
    """
    matched, _, _ = match_particles_by_distance(
        predictions, ground_truth, threshold
    )
    
    # 严格匹配下的召回率
    accuracy = len(matched) / max(len(ground_truth), 1)
    
    return accuracy
```

---

**步骤 2：集成到 MetricAggregator**

```python
class MetricAggregator:
    def add_image(self, predictions, ground_truth):
        # ... 现有代码 ...
        
        # 计算新指标
        loc_acc = compute_localization_accuracy(
            predictions, ground_truth, threshold=5.0
        )
        self.loc_accuracies.append(loc_acc)
    
    def compute_aggregate(self, compute_ap=True):
        # ... 现有代码 ...
        
        metrics_dict['localization_accuracy'] = np.mean(self.loc_accuracies)
        
        return metrics_dict
```

---

## 八、总结

### 8.1 SuPicker 核心优势

| 特性 | 描述 | 优势 |
|------|------|------|
| **现代化架构** | ConvNeXt + FPN + CenterNet | 高性能、无锚框设计 |
| **灵活配置** | 完整的数据类配置系统 | 易于调整超参数 |
| **高效训练** | AMP、分布式、多 GPU | 节省显存、加速训练 |
| **生产就绪** | 检查点管理、中断恢复、日志 | 稳定可靠 |
| **生态兼容** | RELION STAR 格式 | 无缝集成现有流程 |
| **完善测试** | 单元测试覆盖各模块 | 代码质量有保障 |

---

### 8.2 适用场景

🎯 **主要场景**：
- Cryo-EM 颗粒自动拾取
- 电子显微镜图像分析

🎯 **扩展场景**：
- 圆形/椭圆形物体检测
- 小目标检测研究
- 医学图像分析

---

### 8.3 快速上手路线图

**第 1 步：环境搭建**（30 分钟）
```bash
git clone https://github.com/your-org/supicker.git
cd supicker
pip install -e .
```

**第 2 步：数据准备**（1 小时）
- 收集微镜图像（TIFF/MRC 格式）
- 准备 STAR 标注文件
- 使用 `star_tool.py` 划分训练/验证集

**第 3 步：快速实验**（2 小时）
```bash
python scripts/train.py \
    --train-images ./data/train \
    --train-star ./data/train.star \
    --backbone tiny \
    --epochs 50 \
    --pretrained
```

**第 4 步：评估调优**（2-3 天）
- 在验证集上评估
- 调整阈值、学习率、数据增强
- 观察训练曲线

**第 5 步：大规模训练**（1-2 天）
- 使用更多数据
- 增加训练轮数
- 考虑分布式训练

**第 6 步：部署应用**
- 批量预测测试集
- 导出结果到 STAR 文件
- 导入 RELION 进行后续处理

---

### 8.4 下一步建议

**对于初学者**：
1. 从 Tiny backbone 开始
2. 使用预训练权重
3. 遵循默认超参数
4. 先跑通整个流程再调优

**对于研究者**：
1. 尝试不同 backbone（ResNet、EfficientNet）
2. 修改 neck 结构（BiFPN、PANet）
3. 实验不同的损失函数
4. 添加自定义数据增强

**对于工程师**：
1. 优化推理速度（ONNX、TensorRT）
2. 集成到现有 pipeline
3. 开发 GUI 界面
4. 部署到生产环境

---

### 8.5 资源链接

- **官方文档**：https://github.com/your-org/supicker
- **PyTorch 文档**：https://pytorch.org/docs/
- **CenterNet 论文**：https://arxiv.org/abs/1904.07850
- **ConvNeXt 论文**：https://arxiv.org/abs/2201.03545
- **RELION 教程**：https://www3.mrc-lmb.cam.ac.uk/relion/

---

## 附录

### A. 常用命令速查

```bash
# 训练
python scripts/train.py --train-images IMG --train-star STAR --backbone tiny --epochs 100

# 验证集训练
python scripts/train.py --train-images IMG --train-star STAR --val-images VIMG --val-star VSTAR

# 多 GPU
torchrun --nproc_per_node=4 scripts/train.py --distributed

# 恢复训练
python scripts/train.py --resume checkpoint.pth

# 预测
python scripts/predict.py --checkpoint best.pth --input IMG --output OUT

# STAR 文件工具
python scripts/star_tool.py info file.star
python scripts/star_tool.py split file.star -n 10 -o subset.star
python scripts/star_tool.py split-trainval file.star --val-images 50 --train-output train.star --val-output val.star
```

---

### B. 配置参数速查表

#### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backbone` | `tiny` | backbone 变体：tiny/small/base |
| `--batch-size` | `8` | 训练批次大小 |
| `--epochs` | `100` | 训练轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--optimizer` | `adamw` | 优化器：adam/adamw/sgd |
| `--scheduler` | `cosine` | 学习率调度器 |
| `--weight-decay` | `0.01` | 权重衰减 |
| `--warmup-epochs` | `5` | 热身轮数 |
| `--pretrained` | `False` | 使用预训练权重 |
| `--device` | `cuda` | 设备选择 |
| `--distributed` | `False` | 启用分布式训练 |
| `--resume` | `None` | 检查点路径 |
| `--no-amp` | `False` | 禁用混合精度 |

#### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--threshold` | `0.3` | 置信度阈值 |
| `--nms-radius` | `20.0` | NMS 半径（像素） |
| `--format` | `star` | 输出格式：star/json/csv |
| `--merge-output` | `None` | 合并输出文件路径 |

---

### C. 故障排除清单

✅ **训练无法收敛**
- [ ] 检查学习率是否过高
- [ ] 确认数据标注是否正确
- [ ] 验证热图生成是否准确
- [ ] 尝试使用预训练权重

✅ **显存不足**
- [ ] 减小批次大小
- [ ] 启用 AMP
- [ ] 减小裁剪尺寸
- [ ] 使用更小的 backbone

✅ **检测效果差**
- [ ] 调整检测阈值
- [ ] 检查数据质量
- [ ] 增加训练轮数
- [ ] 验证标注坐标准确性

✅ **分布式训练失败**
- [ ] 确认使用 `torchrun` 启动
- [ ] 检查 NCCL 后端是否正常
- [ ] 验证 GPU 可见性
- [ ] 确保 SyncBatchNorm 正确配置

---

**文档结束**

---

*如果您发现任何错误或需要改进的地方，欢迎提交 Issue 或 Pull Request！*
