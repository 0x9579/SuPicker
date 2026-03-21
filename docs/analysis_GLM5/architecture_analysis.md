# SuPicker 架构分析报告

> 分析日期: 2026-03-22
> 分析模型: GLM-5

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 冷冻电镜图像与颗粒特性](#2-冷冻电镜图像与颗粒特性)
- [3. 当前网络架构分析](#3-当前网络架构分析)
- [4. 架构适用性评估](#4-架构适用性评估)
- [5. 优化空间分析](#5-优化空间分析)
- [6. 改进建议优先级](#6-改进建议优先级)

---

## 1. 项目概述

SuPicker 是一个基于深度学习的冷冻电镜颗粒自动挑选框架，采用 **ConvNeXt + FPN + CenterNet** 的架构组合。

### 1.1 核心模块

| 模块 | 文件路径 | 功能描述 |
|------|----------|----------|
| Backbone | `supicker/models/backbone/convnext.py` | ConvNeXt 特征提取骨干网络 |
| Neck | `supicker/models/neck/fpn.py` | Feature Pyramid Network 多尺度特征融合 |
| Head | `supicker/models/head/centernet.py` | CenterNet 检测头，输出 heatmap/size/offset |
| Detector | `supicker/models/detector.py` | 完整检测器组装 |

### 1.2 架构配置

```python
# 默认配置 (supicker/config/model.py)
BackboneConfig:
  variant: TINY/SMALL/BASE
  pretrained: True
  in_channels: 1  # 灰度图像

FPNConfig:
  in_channels: [96, 192, 384, 768]  # ConvNeXt 输出通道
  out_channels: 256

HeadConfig:
  num_classes: 1
  feat_channels: 256
```

---

## 2. 冷冻电镜图像与颗粒特性

### 2.1 图像特性

| 特性 | 描述 | 对检测的影响 |
|------|------|--------------|
| **极低信噪比** | SNR 通常 < 0.1，噪声主导图像 | 目标信号被淹没，需强降噪能力 |
| **低对比度** | 生物样本与冰层对比度极低 | 边缘模糊，定位困难 |
| **大尺寸** | 4K×4K 或更大 | 显存限制，需裁剪训练 |
| **灰度单通道** | 无颜色信息 | 纹理特征更关键 |
| **CTF 效应** | 对比度传递函数导致频率畸变 | 特定频率信息丢失/反转 |

### 2.2 颗粒特性

| 特性 | 描述 | 检测挑战 |
|------|------|----------|
| **尺寸变化大** | 几十 Å 到数百 Å | 需多尺度检测能力 |
| **随机取向** | 无固定方向 | 需旋转不变性 |
| **尺寸相对稳定** | 同一蛋白尺寸集中 | Size 预测价值有限 |
| **密度不均匀** | 不同区域颗粒密度差异大 | 需处理类别不平衡 |

### 2.3 干扰因素

- 碳膜边缘
- 污染物/杂质
- 冰晶缺陷
- 样品支撑膜纹理

---

## 3. 当前网络架构分析

### 3.1 Backbone: ConvNeXt

```
Input (B, 1, H, W)
    ↓
Stem: Conv2d(1, 96, kernel=4, stride=4)  # 下采样4x
LayerNorm2d
    ↓
Stage 1: 3x ConvNeXtBlock(dim=96)  #输出 C1, stride=4
    ↓
Downsample: LayerNorm + Conv2d(2x2)
    ↓
Stage 2: 3x ConvNeXtBlock(dim=192)  # 输出 C2, stride=8
    ↓
Downsample
    ↓
Stage 3: 9x ConvNeXtBlock(dim=384)  # 输出 C3, stride=16
    ↓
Downsample
    ↓
Stage 4: 3x ConvNeXtBlock(dim=768)  # 输出 C4, stride=32
```

**关键特性:**
- 7×7 大核深度卷积，扩展感受野
- LayerNorm 替代 BatchNorm，更稳定
- Inverted bottleneck 结构
- 支持 ImageNet 预训练权重（RGB→灰度适配）

### 3.2 Neck: FPN

```
C4 (stride=32) ──→ Lateral Conv(1x1) ──→ Upsample ──┐
                                                    │ +→ P4
C3 (stride=16) ──→ Lateral Conv(1x1) ←──────────────┘
                         ↓
                      Upsample ───────────────────────┐
                                                    │ +→ P3
C2 (stride=8)  ──→ Lateral Conv(1x1) ←──────────────┘
                         ↓
                      Upsample ───────────────────────┐
                                                    │ +→ P2
C1 (stride=4)  ──→ Lateral Conv(1x1) ←──────────────┘
```

**实际使用:**仅取 P2 层（最高分辨率）进行检测。

### 3.3 Head: CenterNet

```python
# 输入: P2 特征 (B, 256, H/4, W/4)

shared_conv: Conv(256→256, 3x3) + BN + ReLU

分支 1 - heatmap:Conv(256→256, 3x3) + BN + ReLU → Conv(256→1, 1x1)→ Sigmoid
分支 2 - size:Conv(256→256, 3x3) + BN + ReLU → Conv(256→2, 1x1) → ReLU
分支 3 - offset: Conv(256→256, 3x3) + BN + ReLU → Conv(256→2, 1x1)
```

### 3.4 损失函数

```python
Total Loss = λ_hm * HeatmapLoss + λ_size * SizeLoss + λ_off * OffsetLoss

# Default weights
λ_hm = 1.0, λ_size = 0.1, λ_off = 1.0

# Heatmap: Focal Loss (α=2, β=4)
# Size/Offset: L1 Loss
```

### 3.5 推理流程

```
Image → Backbone → FPN → P2 → Head
                                    ↓
    Heatmap → NMS → Peak Detection
                                    ↓
    Size + Offset → Coordinate Refinement
                                    ↓
    Output: (x, y, score, width, height)
```

---

## 4. 架构适用性评估

### 4.1 优点分析

| 方面 | 评估 | 说明 |
|------|------|------|
| **Backbone 选择** | ✓ 良好 | ConvNeXt 大核(7×7)适合捕捉低SNR图像的全局上下文 |
| **FPN 结构** | ✓ 合理 | 多尺度特征融合理论上支持不同尺寸颗粒 |
| **Anchor-free** | ✓ 合适 | CenterNet 无需 anchor 调参，适合稀疏目标检测 |
| **输出分辨率** | ✓ 可接受 | stride=4 提供稳定的亚像素定位精度 |
| **单通道输入** | ✓ 正确 | 符合冷冻电镜图像特性 |
| **预训练适配** | ✓ 有价值 | RGB→灰度权重平均，利用 ImageNet 知识迁移 |

### 4.2 不足分析

| 问题 | 严重程度 | 影响范围 |
|------|----------|----------|
| 低SNR处理不足 | 高 | 核心性能瓶颈 |
| FPN利用不充分 | 中 | 大尺寸颗粒召回 |
| Heatmap半径固定 | 中 | 定位精度 |
| 数据增强保守 | 中 | 泛化能力 |
| Size分支冗余 | 低 | 计算效率 |
| 缺乏CTF感知 | 低-中 | 特定场景 |

---

## 5. 优化空间分析

### 5.1 低信噪比处理不足（优先级：高）

**现状:**
```python
# supicker/data/transforms.py
class Normalize(BaseTransform):
    def apply(self, image, particles):
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
        return image, particles
```

仅进行简单的均值方差归一化，未针对低SNR特性优化。

**问题:**
- 冷冻电镜噪声谱复杂，包含高斯噪声、结构噪声
- 简单归一化无法有效增强信号

**建议方案:**

1. **带通滤波预处理**
   ```python
   def bandpass_filter(image, low_freq=0.02, high_freq=0.5):
       # FFT频域滤波
       fft = torch.fft.fft2(image)
       # 移除低频漂移和高频噪声
       # ... 实现 ...
       return filtered_image
   ```

2. **可学习降噪模块**
   ```
   Input → Denoiser (DnCNN-style) → ConvNeXt → ...
   ```

3. **增强噪声注入**
   ```python
   # 当前: noise_std = 0.02
   # 建议: noise_std = 0.05~0.1
   GaussianNoise(std=0.05, p=0.5)
   ```

### 5.2 FPN利用不充分（优先级：中）

**现状:**
```python
# supicker/models/detector.py:42
def forward(self, x):
    features = self.backbone(x)
    fpn_features = self.neck(features)
    outputs = self.head(fpn_features[0])  # 仅使用 P2
    return outputs
```

**问题:**
- 仅使用P2层，丢弃了P3/P4信息
- 大尺寸颗粒可能在低分辨率层有更好响应

**建议方案:**

1. **多尺度检测头**
   ```python
   # 方案 A: 多层级检测融合
   detections_p2 = self.head_p2(fpn_features[0])
   detections_p3 = self.head_p3(fpn_features[1])
   # 尺度自适应融合
   ```

2. **尺度感知检测**
   ```python
   # 方案 B: 添加尺度预测分支
   scale_pred = self.scale_head(fpn_features[0])
   # 根据预测尺度选择检测层级
   ```

### 5.3 Heatmap 半径固定（优先级：中）

**现状:**
```python
# supicker/data/target_generator.py:52
radius = max(int(self.gaussian_sigma * 3), 1)
```

**问题:**
-固定半径未考虑颗粒实际尺寸
- 小颗粒热图过于弥散，大颗粒热图过于尖锐

**建议方案:**
```python
def generate_heatmap(self, particles, image_size):
    for p in particles:
        width = p.get("width", 64) / self.output_stride
        height = p.get("height", 64) / self.output_stride
        # 动态半径：基于颗粒尺寸
        radius = max(int(min(width, height) / 3), 1)
        self._draw_gaussian(heatmap[class_id], cx, cy, radius)
```

### 5.4 Size 预测分支（优先级：低）

**现状:**
CenterNet头同时预测 heatmap、size、offset 三个输出。

**分析:**
- 冷冻电镜颗粒尺寸相对稳定（同一蛋白）
- Size 预测增加模型复杂度
- Size 分支损失权重仅 0.1，贡献有限

**建议:**
- 考虑移除 Size 分支，简化模型
- 或改为"质量/可信度"预测

### 5.5 数据增强策略

**现状:**
```python
# supicker/config/data.py
AugmentationConfig:
  gaussian_noise: True
  noise_std: 0.02  # 较保守
  ctf_simulation: False  #未启用
```

**建议:**
```python
AugmentationConfig:
  gaussian_noise: True
  noise_std: 0.05  # 增强
  ctf_simulation: True  # 启用CTF模拟
  bandpass_augmentation: True  # 新增
  local_contrast_normalization: True  # 新增
```

### 5.6 缺乏CTF感知

**背景:**
CTF (Contrast Transfer Function) 导致不同频率信号的对比度变化，是冷冻电镜图像的核心畸变源。

**建议:**
```python
# 方案 1: CTF 参数作为辅助输入
class DetectorWithCTF(nn.Module):
    def forward(self, x, ctf_params):
        ctf_embedding = self.ctf_encoder(ctf_params)
        features = self.backbone(x, ctf_embedding)
        # ...

# 方案 2: CTF 感注意力模块
class CTFAttention(nn.Module):
    def __init__(self, channels):
        self.ctf_conv = nn.Conv2d(channels, channels, 1)
    def forward(self, x, ctf_params):
        attention = self.ctf_conv(ctf_params)
        return x * attention
```

---

## 6. 改进建议优先级

### 6.1 优先级矩阵

| 优先级 | 改进点 | 预期收益 | 实现难度 | ROI |
|--------|--------|----------|----------|-----|
| **P0** | 带通滤波预处理 | 高 | 低 | 高 |
| **P0** | 动态 Heatmap 半径 | 中 | 低 | 高 |
| **P1** | 增强噪声注入 | 中 | 低 | 高 |
| **P1** | 多尺度检测头 | 中 | 中 | 中 |
| **P2** | CTF 数据增强 | 中 | 中 | 中 |
| **P2** | 移除 Size 分支 | 低 | 低 | 中 |
| **P3** | 频域 Attention | 中 | 高 | 低 |
| **P3** | CTF 辅助输入 | 中 | 高 | 低 |

### 6.2 具体实施建议

#### 阶段一：基础设施优化（1-2周）

```python
# 1. 添加带通滤波预处理
class BandpassFilter(BaseTransform):
    def __init__(self, low_cutoff=0.02, high_cutoff=0.5):
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
    
    def apply(self, image, particles):
        # FFT-based bandpass filtering
        ...

# 2. 动态 heatmap 半径
class TargetGenerator:
    def generate_heatmap(self, particles, image_size):
        for p in particles:
            dynamic_radius = self._compute_radius(p)
            ...

# 3. 增强噪声注入
AugmentationConfig:
    noise_std: 0.05  # 从 0.02 提升
```

#### 阶段二：架构优化（2-4周）

```python
# 1. 多尺度检测头
class MultiScaleCenterNetHead(nn.Module):
    def __init__(self, ...):
        self.head_p2 = CenterNetHead(...)
        self.head_p3 = CenterNetHead(...)
        self.scale_aware_fusion = ...
    
    def forward(self, fpn_features):
        det_p2 = self.head_p2(fpn_features[0])
        det_p3 = self.head_p3(fpn_features[1])
        return self.scale_aware_fusion(det_p2, det_p3)

# 2. CTF 模拟增强
class CTFSimulation(BaseTransform):
    def apply(self, image, particles):
        # 模拟随机 CTF 参数
        ctf_params = self._random_ctf()
        image = apply_ctf(image, ctf_params)
        ...
```

#### 阶段三：实验验证

```bash
# 消融实验设计
1. Baseline: 当前架构
2. + Bandpass: 添加带通滤波
3. + DynamicRadius: 动态 heatmap 半径
4. + MultiScale: 多尺度检测头
5. Full: 所有优化组合

# 评估指标
- Precision@threshold
- Recall@threshold
- F1-score
- Average Precision (AP)
- 推理速度 (FPS)
```

---

## 7. 总结

### 7.1 当前架构评价

SuPicker 采用的 **ConvNeXt + FPN + CenterNet** 架构是一个**标准且合理**的选择，具备以下特点：

- **现代性**: ConvNeXt 是 2022 年的先进架构，大核设计适合低SNR图像
- **完整性**: FPN 多尺度融合 + CenterNet anchor-free 检测是成熟的组合
- **工程性**: 代码结构清晰，支持分布式训练、AMP等

### 7.2 核心改进方向

针对冷冻电镜领域特性，最重要的改进方向是：

1. **低SNR处理** - 这是冷冻电镜的核心挑战
2. **尺度自适应** - 颗粒尺寸变化大的特性要求
3. **领域增强** - CTF、噪声等冷冻电镜特有数据增强

### 7.3 风险提示

- 过度工程化可能引入新问题
-需在真实数据集上验证每个改进的有效性
- 保持模型简单性vs 性能的平衡

---

*本分析基于代码审查和领域知识，具体改进效果需要通过实验验证。*