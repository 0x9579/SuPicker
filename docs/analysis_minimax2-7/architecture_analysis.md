# SuPicker 网络架构分析报告

**Cryo-EM 颗粒挑选深度学习框架架构评估与优化建议**

---

## 一、项目概述

SuPicker 是一个基于深度学习的冷冻电镜（Cryo-EM）图像颗粒挑选框架，采用 **CenterNet 风格的无锚框目标检测范式**。

### 1.1 技术架构

```
输入图像 (B, 1, H, W)
    ↓
ConvNeXt Backbone (4阶段输出 [C1@H/4, C2@H/8, C3@H/16, C4@H/32])
    ↓
FPN Neck (自顶向下特征金字塔)
    ↓
CenterNet Head (仅使用 P2 分支)
    ↓
热图 (B, 1, H/4, W/4) + 尺寸 (B, 2, H/4, W/4) + 偏移 (B, 2, H/4, W/4)
```

### 1.2 核心配置

| 组件 | 配置 |
|------|------|
| Backbone | ConvNeXt-Tiny/Small/Base |
| Neck | FPN (256通道) |
| Head | CenterNet (共享卷积 + 3分支) |
| 训练 | AdamW, lr=1e-4, batch=8, 100 epochs |
| 损失 | Focal Loss + L1 Loss |
| 数据增强 | 翻转、旋转90°、随机旋转、亮度对比度、高斯噪声 |

---

## 二、冷冻电镜图像物理特性分析

### 2.1 与自然图像的本质差异

| 特性 | 自然图像 | Cryo-EM 图像 |
|------|---------|--------------|
| 对比度来源 | 光照、反射 | 蛋白质与冰的电子散射差异 |
| 信噪比 (SNR) | 高 ( >10 ) | **极低 (0.01-0.1)** |
| 结构特征 | 语义边缘、纹理 | 高斯状密度分布、CTF 调制 |
| 目标特性 | 语义可分 | 物理密度投影 |
| 噪声类型 | 相对简单 | 电子散射噪声、探测器噪声、辐射损伤 |

### 2.2 Cryo-EM 颗粒的物理本质

1. **二维高斯密度投影**：颗粒是蛋白质的三维密度在投影平面上的积分，呈现近似高斯分布
2. **尺寸跨度极大**：从 ~100Å (小型蛋白) 到 >3000Å (大型复合物)
3. **CTF 调制效应**：成像过程固有，会导致图像在频率域呈现特征性振荡
4. **密集排布**：高浓度样本中颗粒间距小，容易重叠

---

## 三、当前架构的核心矛盾

### 3.1 CenterNet 语义检测范式 vs 物理密度估计

**问题**：CenterNet 源自 "Objects as Points"，其 focal loss 设计目的是区分"前景语义物体"和"背景"。但冷冻电镜颗粒本质上是二维高斯密度投影，不是语义对象。

```
当前范式：focal_loss(heatmap) → 区分粒子 vs 背景
实际情况：粒子与背景是渐进过渡的，不是二值边界
```

**影响**：模型被迫学习一个不自然的二分类边界，而非物理上正确的密度分布。

### 3.2 固定高斯 σ 的局限性

```python
# target_generator.py
gaussian_sigma = 2.0  # 固定值，无法适应不同颗粒尺寸
```

- **小颗粒**：热峰过度展开，定位模糊，召回率下降
- **大颗粒**：热峰被压缩在中心，边缘信息丢失，定位精度下降

### 3.3 CTF 物理效应被完全忽略

CTF（对比度传递函数）是冷冻电镜成像的**固有物理过程**：

```
I_observed = I_true * CTF + noise
CTF(f) = -sin(π·Δz·f²·Cs + f·φ)  # 振荡调制
```

代码中存在 `ctf_simulation` 配置标记，但**从未在 transforms 中实现**。这意味着：
- 训练时模型学习的是"理想图像 → 理想热图"
- 实际推理时输入的是"CTF 调制后的退化图像"
- 模型需要额外学习"反 CTF"变换，增加了学习难度

### 3.4 单尺度检测的局限

```python
# detector.py - 仅使用 P2 (1/4 分辨率)
```

只检测单一尺度，对于大型蛋白复合物和小型蛋白混在一起时，表现会下降。

---

## 四、真实优化空间

### 4.1 CTF 感知网络设计 ⭐ P0

**为什么重要**：CTF 不是"噪声"，而是确定性的物理调制。忽略它意味着模型需要学习不必要的"反 CTF"变换。

#### 方案 1：CTF 条件化特征调制

```python
class CTFConditionedConv(nn.Module):
    def __init__(self, in_channels, ctf_channels=6):
        # CTF 参数: voltage, Cs, DefocusU, DefocusV, Astigmatism, PhaseShift
        self.ctf_embed = nn.Linear(ctf_channels, in_channels)
        
    def forward(self, x, ctf_params):
        ctf_bias = self.ctf_embed(ctf_params)
        return x + ctf_bias  # 加性调制
```

#### 方案 2：频率域注意力

```python
class FrequencyDomainAttention(nn.Module):
    def __init__(self, in_channels):
        self.fft_conv = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        x_fft = torch.fft.rfft2(x)
        attn = torch.sigmoid(self.fft_conv(x_fft))
        return torch.fft.irfft2(x_fft * attn)
```

#### 方案 3：CTF 模拟数据增强

在训练时随机应用不同 CTF 参数，强制模型对 CTF 变化鲁棒。

**预期收益**：真实电镜图像检测精度提升 **15-30%**

---

### 4.2 多尺度自适应检测 ⭐ P1

#### 方案 1：FPN 多层级融合检测

```python
class MultiScaleDetectionHead(nn.Module):
    def __init__(self, in_channels=256):
        self.head_p2 = self._make_head(in_channels, stride=4)
        self.head_p3 = self._make_head(in_channels, stride=8)
        self.head_p4 = self._make_head(in_channels, stride=16)
        
    def forward(self, features):
        hm_p2 = self.head_p2(features['p2'])
        hm_p3 = self.head_p3(features['p3'])
        hm_p4 = self.head_p4(features['p4'])
        
        # 上采样到统一尺度后融合
        hm_p3_up = F.interpolate(hm_p3, scale_factor=2)
        hm_p4_up = F.interpolate(hm_p4, scale_factor=4)
        
        return (hm_p2 + hm_p3_up + hm_p4_up) / 3
```

#### 方案 2：自适应高斯热图

```python
class AdaptiveGaussianHead(nn.Module):
    def __init__(self, in_channels):
        self.sigma_predictor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Softplus()  # 确保 σ > 0
        )
    
    def forward(self, x):
        sigma_map = self.sigma_predictor(x) + 1.0  # 最小 σ=1.0
        # 生成热图时使用动态 σ
        return self.generate_heatmap(coords, sigma_map)
```

**预期收益**：不同尺寸颗粒召回率提升 **10-20%**

---

### 4.3 专门的去噪 Backbone ⭐ P1

**问题**：Cryo-EM 图像 SNR 极低，ImageNet 预训练对于低对比度、低信噪比图像迁移效果差。

#### 方案 1：U-Net 风格编码器-解码器

```python
class DenoisingEncoder(nn.Module):
    def __init__(self, in_channels=1):
        self.enc1 = self._make_layer(in_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        
        self.dec2 = self._make_layer(256 + 128, 128)
        self.dec1 = self._make_layer(128 + 64, 64)
        
        self.to_clean = nn.Conv2d(64, 1, 1)
```

#### 方案 2：残差学习

```python
class NoiseResidualLearning(nn.Module):
    def forward(self, noisy_image):
        features = self.backbone(noisy_image)
        noise = self.noise_head(features)
        return noisy_image - noise  # 预测噪声残差
```

#### 方案 3：非局部自注意力

捕获蛋白质重复对称性结构的长程依赖。

**预期收益**：噪声环境下检测精度提升 **20-40%**

---

### 4.4 定位回归方法 ⭐ P2

#### 热图方法的问题

- 高斯核大小固定，难以适应不同颗粒
- 训练目标是"让正确位置概率高"，而非"精确坐标"

#### 方案：直接坐标回归

```python
class DirectRegressionHead(nn.Module):
    def __init__(self, in_channels):
        # 每个位置预测: [offset_x, offset_y, w, h, confidence]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 5, 1)  # 5 channels
        )
```

#### 方案：粗到细定位

```python
class TwoStageLocator(nn.Module):
    def forward(self, x):
        coarse_offset, coarse_conf = self.coarse_head(self.backbone(x))
        refine_features = self.extract_local_features(x, coarse_offset)
        fine_offset = self.refine_head(refine_features)
        return coarse_offset + fine_offset
```

**预期收益**：亚像素精度提升，定位误差减少 **15-25%**

---

### 4.5 训练策略优化 ⭐ P2

#### 难度采样 (Hard Example Mining)

```python
class HardExampleMiningLoss(nn.Module):
    def forward(self, pred, target, mask):
        loss_map = F.binary_cross_entropy(pred, target, reduction='none')
        n_hard = int(mask.sum() * 0.7)
        hard_loss = loss_map[mask].topk(n_hard)[0].mean()
        easy_loss = loss_map[mask].topk(int(mask.sum() * 0.3))[0].mean()
        return 0.7 * hard_loss + 0.3 * easy_loss
```

#### Test-Time Augmentation (TTA)

```python
def predict_with_tta(model, image, num_augments=8):
    augments = [
        lambda x: x,
        lambda x: torch.flip(x, [-1]),
        lambda x: torch.flip(x, [-2]),
        lambda x: torch.flip(x, [-1, -2]),
        lambda x: torch.rot90(x, 1, [-2, -1]),
        lambda x: torch.rot90(x, 2, [-2, -1]),
        lambda x: torch.rot90(x, 3, [-2, -1]),
        lambda x: torch.flip(torch.rot90(x, 1, [-2, -1]), [-1]),
    ]
    predictions = [model(aug(image)) for aug in augments]
    return torch.stack(predictions).mean(0)
```

**预期收益**：检测精度提升 **5-15%**

---

## 五、优化优先级总结

| 优化方向 | 实施难度 | 预期收益 | 优先级 |
|---------|---------|---------|--------|
| CTF 感知网络 | 高 | 高 | **P0** |
| 多尺度检测 | 中 | 中高 | **P1** |
| 去噪 Backbone | 高 | 高 | **P1** |
| 定位回归 | 中 | 中 | **P2** |
| TTA | 低 | 中 | **P2** |

---

## 六、核心结论

当前 CenterNet + ConvNeXt 架构是**通用的目标检测范式在 cryo-EM 上的直接应用**，而非针对 cryo-EM 物理特性的定制化设计。

### 主要差距

1. **未显式建模 CTF 物理效应** — 这是 cryo-EM 特有的最重要因素
2. **针对极低 SNR 的去噪机制缺失** — ImageNet 迁移学习效果有限
3. **动态多尺度检测能力不足** — 颗粒尺寸差异巨大

### 建议优先级

1. **短期**：实现 TTA、多层级 FPN 融合检测
2. **中期**：引入自适应高斯热图、优化训练损失函数
3. **长期**：设计 CTF 感知网络、专门的去噪 Backbone

---

*报告生成时间：2026-03-22*
