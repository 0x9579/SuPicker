# SuPicker 网络结构深度解析

> 从张量维度到架构设计的全面剖析

**版本**: 1.0  
**最后更新**: 2026-03-21

---

## 📋 目录

- [一、整体架构概览](#一整体架构概览)
- [二、输入输出规格](#二输入输出规格)
- [三、ConvNeXt 骨干网络详解](#三 convnext 骨干网络详解)
- [四、FPN 颈部网络详解](#四 fpn 颈部网络详解)
- [五、CenterNet 检测头详解](#五 centernet 检测头详解)
- [六、完整前向传播流程](#六完整前向传播流程)
- [七、设计原理分析](#七设计原理分析)
- [八、参数量与计算量分析](#八参数量与计算量分析)
- [九、常见修改方案](#九常见修改方案)

---

## 一、整体架构概览

### 1.1 架构图（详细版）

```
输入图像 (B, 1, H, W)
    │
    │ Conv2d(1→96, 4×4, s=4)
    │ LayerNorm
    ▼
Stem: (B, 96, H/4, W/4)
    │
    ├─ Stage1: [ConvNeXtBlock × 3]
    │           ↓
    │         C1: (B, 96, H/4, W/4) ────────┐
    │                                        │
    │  Conv2d(96→192, 2×2, s=2)             │
    ▼                                        │
Stage2: [ConvNeXtBlock × 3]                  │
          ↓                                   │
        C2: (B, 192, H/8, W/8) ───────────────┤
                                               │
    Conv2d(192→384, 2×2, s=2)                 │
    ▼                                          │
Stage3: [ConvNeXtBlock × 9/27]                │
          ↓                                    │
        C3: (B, 384, H/16, W/16) ──────────────┤
                                                │
    Conv2d(384→768/1024, 2×2, s=2)            │
    ▼                                           │
Stage4: [ConvNeXtBlock × 3]                    │
          ↓                                     │
        C4: (B, 768, H/32, W/32) ←──────────────┘
                                                 
    FPN Feature Pyramid Network
    ├─ Lateral: 1×1 Conv (统一通道数到 256)
    ├─ Top-down: Upsample + Add
    └─ Output: 3×3 Conv
    
    ▼
    P2: (B, 256, H/4, W/4) ← 用于检测
    P3: (B, 256, H/8, W/8)
    P4: (B, 256, H/16, W/16)
    P5: (B, 256, H/32, W/32)
    
    CenterNet Head
    ├─ Shared Conv: 3×3 Conv (256→256)
    ├─ Heatmap Head: 1×1 Conv (256→num_classes) + Sigmoid
    ├─ Size Head: 1×1 Conv (256→2) + ReLU
    └─ Offset Head: 1×1 Conv (256→2)
    
    ▼
    Heatmap: (B, num_classes, H/4, W/4)
    Size: (B, 2, H/4, W/4)
    Offset: (B, 2, H/4, W/4)
    
    Post-processing
    ├─ Extract peaks from heatmap
    ├─ Apply size & offset
    └─ NMS filtering
    
    ▼
    Particles: [{x, y, score, width, height}, ...]
```

---

### 1.2 关键设计决策

| 组件 | 设计选择 | 原因 |
|------|---------|------|
| **Backbone** | ConvNeXt | 结合 CNN 和 Transformer 优势，性能优于 ResNet |
| **Neck** | FPN | 多尺度特征融合，增强小目标检测 |
| **Head** | CenterNet | 无锚框设计，简化流程，适合圆形颗粒 |
| **Output stride** | 4 | 保留更多空间信息，提高定位精度 |
| **Feature for detection** | P2 | 最高分辨率 (1/4)，适合小颗粒检测 |

---

## 二、输入输出规格

### 2.1 输入规格

#### 训练时输入

```python
# 图像输入
image: torch.Tensor
shape: (B, 1, H, W)
dtype: torch.float32
range: [0, 1]

# B: batch size (通常 8-16)
# 1: 单通道灰度图
# H, W: 图像高宽 (通常裁剪到 1024×1024)
```

#### 推理时输入

```python
# 单张图像
image: torch.Tensor
shape: (1, 1, H, W)  # 或 (C, H, W)，自动添加 batch 维

# 支持任意尺寸，但建议保持长边 ≤ 2048
```

---

### 2.2 输出规格

#### 模型原始输出

```python
outputs = {
    'heatmap': torch.Tensor(B, num_classes, H/4, W/4),
    'size': torch.Tensor(B, 2, H/4, W/4),
    'offset': torch.Tensor(B, 2, H/4, W/4)
}

# heatmap: 每个位置是颗粒中心的概率 (0-1)
# size: [width, height] 预测值
# offset: [offset_x, offset_y] 亚像素校正
```

#### 后处理后输出

```python
particles: List[Dict]
[
    {
        'x': float,          # 像素坐标（原图尺度）
        'y': float,
        'score': float,      # 置信度 (0-1)
        'width': float,      # 颗粒宽度（像素）
        'height': float,     # 颗粒高度（像素）
        'class_id': int,     # 类别 ID
        'batch_idx': int     # batch 索引（批量推理时）
    },
    ...
]
```

---

### 2.3 维度变化总览表

| 阶段 | 输出张量 | 空间分辨率 | 通道数 | stride |
|------|---------|-----------|--------|--------|
| **Input** | `(B, 1, H, W)` | `H×W` | 1 | 1× |
| **Stem** | `(B, 96, H/4, W/4)` | `H/4 × W/4` | 96 | 4× |
| **C1** | `(B, 96, H/4, W/4)` | `H/4 × W/4` | 96 | 4× |
| **C2** | `(B, 192, H/8, W/8)` | `H/8 × W/8` | 192 | 8× |
| **C3** | `(B, 384, H/16, W/16)` | `H/16 × W/16` | 384 | 16× |
| **C4** | `(B, 768, H/32, W/32)` | `H/32 × W/32` | 768 | 32× |
| **P2** | `(B, 256, H/4, W/4)` | `H/4 × W/4` | 256 | 4× |
| **Heatmap** | `(B, 1, H/4, W/4)` | `H/4 × W/4` | 1 | 4× |

---

## 三、ConvNeXt 骨干网络详解

### 3.1 Stem 层（茎干层）

**作用**：将输入图像转换为特征图，并进行 4 倍下采样

```python
class Stem(nn.Sequential):
    def __init__(self, in_channels=1, out_channels=96):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=4, stride=4),  # 4×4 卷积，stride=4
            LayerNorm2d(out_channels, eps=1e-6)   # LayerNorm
        )

# 输入：(B, 1, H, W)
# 输出：(B, 96, H/4, W/4)
# 参数量：1×4×4×96 + 96(bias) + 96×2(LayerNorm) ≈ 1.6K + 192 ≈ 1.8K
# FLOPs: (H/4 × W/4) × (1×4×4×96) ≈ 96HW
```

**设计要点**：
- **4×4 大卷积核**：一次性下采样 4 倍，减少后续计算量
- **LayerNorm2d**：自定义的 Channel-first LayerNorm，适配 ConvNeXt

---

### 3.2 LayerNorm2d 实现

```python
class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (B, C, H, W)"""
    
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均值和方差（在 H, W 维度）
        u = x.mean(1, keepdim=True)           # (B, 1, H, W)
        s = (x - u).pow(2).mean(1, keepdim=True)  # (B, 1, H, W)
        
        # 归一化
        x = (x - u) / torch.sqrt(s + self.eps)
        
        # 仿射变换（按通道缩放和平移）
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        
        return x

# 输入：(B, C, H, W)
# 输出：(B, C, H, W)
# 参数量：C(weight) + C(bias) = 2C
```

**为什么不用 BatchNorm？**
- LayerNorm 不依赖 batch 统计量，更适合小 batch 训练
- 与 Transformer 风格一致，便于迁移预训练权重

---

### 3.3 ConvNeXt Block 详细结构

**单个 Block 的数据流**：

```
输入 x: (B, C, H, W)
    │
    ├─ shortcut 分支（恒等映射）
    │
    └─ main 分支:
        │
        ├─ Depthwise Conv (7×7, groups=C)
        │   output: (B, C, H, W)
        │   参数量：C × 7×7 = 49C
        │
        ├─ Permute: NCHW → NHWC
        │   output: (B, H, W, C)
        │
        ├─ LayerNorm
        │   output: (B, H, W, C)
        │   参数量：2C
        │
        ├─ Linear (C → 4C)  [pwconv1]
        │   output: (B, H, W, 4C)
        │   参数量：C × 4C + 4C = 4C² + 4C
        │
        ├─ GELU 激活
        │   output: (B, H, W, 4C)
        │
        ├─ Linear (4C → C)  [pwconv2]
        │   output: (B, H, W, C)
        │   参数量：4C × C + C = 4C² + C
        │
        ├─ LayerScale (gamma * x)
        │   output: (B, H, W, C)
        │   参数量：C (gamma)
        │
        ├─ Permute: NHWC → NCHW
        │   output: (B, C, H, W)
        │
        └─ DropPath (随机丢弃)
            output: (B, C, H, W)
            │
            + shortcut
            │
            ▼
        输出：(B, C, H, W)
```

---

### 3.4 ConvNeXt Block 代码实现

```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        
        # 深度可分离卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm (NHWC 格式)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # 逐点卷积 1: 升维
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        
        # 激活函数
        self.act = nn.GELU()
        
        # 逐点卷积 2: 降维
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # LayerScale (可选)
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim)
        ) if layer_scale_init > 0 else None
        
        # Stochastic Depth (可选)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        
        # 主分支
        x = self.dwconv(x)                          # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)                   # (B, H, W, C)
        x = self.norm(x)                            # LayerNorm
        x = self.pwconv1(x)                         # (B, H, W, 4C)
        x = self.act(x)                             # GELU
        x = self.pwconv2(x)                         # (B, H, W, C)
        
        if self.gamma is not None:
            x = self.gamma * x                      # LayerScale
        
        x = x.permute(0, 3, 1, 2)                   # (B, C, H, W)
        x = self.drop_path(x)                       # DropPath
        
        # 残差连接
        return shortcut + x
```

---

### 3.5 Inverted Bottleneck 设计

**核心思想**：先升维再降维，类似 MobileNetV2

```python
# 传统 Bottleneck (ResNet)
x → Conv(C→C/4) → Conv(C/4→C/4) → Conv(C/4→C)

# Inverted Bottleneck (ConvNeXt)
x → Conv(C→4C) → Conv(4C→4C) → Conv(4C→C)
       ↑              ↑             ↑
     升维          深度卷积        降维
```

**优势**：
- 在更高维空间进行特征变换
- 深度卷积在 4C 通道上进行，表达能力更强
- 参数量增加不多（主要在线性层）

---

### 3.6 LayerScale 机制

```python
# 初始化
self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

# 使用
x = self.gamma * x  # 逐通道缩放
```

**作用**：
- 训练初期 gamma 接近 0，block 近似为恒等映射
- 随着训练进行，gamma 逐渐增大，学习到的特征才起作用
- 帮助深层网络稳定训练

---

### 3.7 DropPath（随机深度）

```python
class DropPath(nn.Module):
    """Drop paths (stochastic depth) for regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, 1)
        
        # 生成随机 mask
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化 (0 或 1)
        
        # 缩放以保持期望不变
        return x.div(keep_prob) * random_tensor

# 示例：drop_prob=0.1
# 训练时：10% 的概率将整个样本的特征置为 0
# 测试时：恒等映射
```

**与普通 Dropout 的区别**：
- Dropout: 随机丢弃单个神经元
- DropPath: 随机丢弃整个路径（整个 block 的输出）
- 更适合残差网络结构

---

### 3.8 不同变体的配置

```python
CONVNEXT_CONFIGS = {
    ConvNeXtVariant.TINY: {
        "depths": [3, 3, 9, 3],      # 各 stage 的 block 数量
        "dims": [96, 192, 384, 768], # 各 stage 的通道数
        "total_params": "~28M"
    },
    ConvNeXtVariant.SMALL: {
        "depths": [3, 3, 27, 3],
        "dims": [96, 192, 384, 768],
        "total_params": "~50M"
    },
    ConvNeXtVariant.BASE: {
        "depths": [3, 3, 27, 3],
        "dims": [128, 256, 512, 1024],
        "total_params": "~89M"
    }
}
```

**参数量对比**：
- Tiny: ~28M (推荐用于快速实验)
- Small: ~50M (平衡性能和速度)
- Base: ~89M (追求最佳性能)

---

### 3.9 预训练权重加载

```python
def _load_pretrained(self, pretrained_path: Optional[str] = None):
    # 1. 加载权重
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
    else:
        url = CONVNEXT_PRETRAINED_URLS[self.config.variant]
        state_dict = torch.hub.load_state_dict_from_url(url)
    
    # 2. 映射键名（torchvision → SuPicker）
    mapped_state_dict = self._map_pretrained_keys(state_dict)
    
    # 3. 适配输入通道（RGB → 灰度）
    if self.config.in_channels != 3:
        mapped_state_dict = self._adapt_input_channels(mapped_state_dict)
    
    # 4. 加载（允许部分键不匹配）
    missing, unexpected = self.load_state_dict(mapped_state_dict, strict=False)
```

**通道适配策略**：

```python
def _adapt_input_channels(self, state_dict):
    stem_weight = state_dict['stem.0.weight']  # (96, 3, 4, 4)
    
    # RGB 权重平均得到灰度权重
    gray_weight = stem_weight.mean(dim=1, keepdim=True)  # (96, 1, 4, 4)
    
    # 如果是多通道输入（如 4 通道），则重复
    if self.config.in_channels > 1:
        gray_weight = gray_weight.repeat(1, self.config.in_channels, 1, 1)
    
    state_dict['stem.0.weight'] = gray_weight
    return state_dict
```

---

## 四、FPN 颈部网络详解

### 4.1 FPN 的作用

**问题**：Backbone 输出的 C1-C4 具有不同分辨率和通道数，如何有效融合？

**解决方案**：Feature Pyramid Network (FPN)

```
C4 (高分辨率，少通道) ─┬→ 语义信息弱，空间信息强
C3                    │
C2                    │
C1 (低分辨率，多通道) ─┴→ 语义信息强，空间信息弱

FPN 融合后:
P4 (高分辨率，中等通道) ← 兼具语义和空间信息
P3
P2
P1 (低分辨率，中等通道)
```

---

### 4.2 FPN 详细结构

```python
class FPN(nn.Module):
    def __init__(self, config: FPNConfig):
        super().__init__()
        in_channels = config.in_channels  # [96, 192, 384, 768]
        out_channels = config.out_channels  # 256
        
        # 侧向连接：1×1 卷积统一通道数
        self.lateral_convs = ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])
        
        # 输出卷积：3×3 卷积平滑特征
        self.output_convs = ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])
```

---

### 4.3 FPN 前向传播（自顶向下）

```python
def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
    # features = [C1, C2, C3, C4]
    
    # Step 1: 应用侧向卷积（统一通道数到 256）
    laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
    # laterals = [L1, L2, L3, L4], 每个都是 (B, 256, H_i, W_i)
    
    # Step 2: 自顶向下融合
    for i in range(len(laterals) - 1, 0, -1):
        # 上采样高分辨率特征
        upsampled = F.interpolate(
            laterals[i], 
            size=laterals[i - 1].shape[2:],  # 对齐到前一层的大小
            mode='nearest'
        )
        
        # 逐元素相加
        laterals[i - 1] = laterals[i - 1] + upsampled
    
    # Step 3: 应用输出卷积（消除混叠效应）
    outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
    
    return outputs  # [P1, P2, P3, P4]
```

---

### 4.4 FPN 数据流可视化

假设输入图像为 `1024×1024`：

```
Backbone 输出:
C1: (B, 96, 256, 256)   [H/4, W/4]
C2: (B, 192, 128, 128)  [H/8, W/8]
C3: (B, 384, 64, 64)    [H/16, W/16]
C4: (B, 768, 32, 32)    [H/32, W/32]

经过 lateral_conv (1×1):
L1: (B, 256, 256, 256)
L2: (B, 256, 128, 128)
L3: (B, 256, 64, 64)
L4: (B, 256, 32, 32)

自顶向下融合:
Step 1:
    upsample(L4): (B, 256, 64, 64)  [32→64, ×2]
    L3 = L3 + upsample(L4): (B, 256, 64, 64)

Step 2:
    upsample(L3): (B, 256, 128, 128)  [64→128, ×2]
    L2 = L2 + upsample(L3): (B, 256, 128, 128)

Step 3:
    upsample(L2): (B, 256, 256, 256)  [128→256, ×2]
    L1 = L1 + upsample(L2): (B, 256, 256, 256)

经过 output_conv (3×3):
P1: (B, 256, 256, 256)
P2: (B, 256, 128, 128)
P3: (B, 256, 64, 64)
P4: (B, 256, 32, 32)
```

---

### 4.5 为什么使用 P2 进行检测？

**分辨率对比**：
- P2: `H/4 × W/4` (256×256 for 1024 input)
- P3: `H/8 × W/8` (128×128)
- P4: `H/16 × W/16` (64×64)

**原因**：
1. **更高的空间分辨率**：保留更多位置信息，适合小颗粒
2. **更精确的定位**：4 倍下采样 vs 8 倍/16 倍，量化误差更小
3. **Cryo-EM 颗粒特点**：通常较小（直径 64-256 像素），需要高分辨率特征

**代价**：
- 计算量更大（256×256 vs 128×128）
- 显存占用更多

---

### 4.6 FPN 参数量计算

```python
# 假设 in_channels = [96, 192, 384, 768], out_channels = 256

# Lateral convolutions (1×1)
lateral_params = sum(in_ch * 256 for in_ch in [96, 192, 384, 768])
# = 96×256 + 192×256 + 384×256 + 768×256
# = 24,576 + 49,152 + 98,304 + 196,608 = 368,640

# Output convolutions (3×3)
output_params = 4 * (256 * 256 * 9)  # 4 个 3×3 卷积
# = 4 * 589,824 = 2,359,296

# Total FPN params
total = 368,640 + 2,359,296 ≈ 2.7M
```

---

## 五、CenterNet 检测头详解

### 5.1 检测头的三个分支

```python
class CenterNetHead(nn.Module):
    def __init__(self, config: HeadConfig, in_channels: int = 256):
        super().__init__()
        feat_channels = config.feat_channels  # 256
        
        # 共享卷积层
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )
        
        # Heatmap 分支
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, config.num_classes, 1),
            nn.Sigmoid()
        )
        
        # Size 分支
        self.size_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 2, 1),
            nn.ReLU()
        )
        
        # Offset 分支
        self.offset_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 2, 1)
        )
```

---

### 5.2 Heatmap 分支

**作用**：预测每个位置是颗粒中心的概率

```python
# 输入：P2 (B, 256, H/4, W/4)
x = self.shared_conv(feature)  # (B, 256, H/4, W/4)
heatmap = self.heatmap_head(x)  # (B, num_classes, H/4, W/4)

# Sigmoid 确保输出范围 [0, 1]
# 值越接近 1，表示该位置越可能是颗粒中心
```

**偏置初始化技巧**：

```python
# 初始化 heatmap 偏置为 -2.19
nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

# 效果：sigmoid(-2.19) ≈ 0.1
# 目的：训练初期让网络输出较低置信度，避免过度自信
```

**为什么这样初始化？**
- 图像中大部分位置不是颗粒中心（负样本占主导）
- 初始低置信度可以让 Focal Loss 更好地工作
- 避免训练初期的梯度爆炸

---

### 5.3 Size 分支

**作用**：预测颗粒的宽度和高度

```python
size = self.size_head(x)  # (B, 2, H/4, W/4)

# size[:, 0, :, :] → width 预测
# size[:, 1, :, :] → height 预测

# ReLU 确保尺寸为正值
```

**为什么需要预测尺寸？**
- 不同颗粒可能有不同大小
- 尺寸信息可用于后续的 NMS 和可视化
- 对于圆形颗粒，width ≈ height

---

### 5.4 Offset 分支

**作用**：预测亚像素级偏移，修正量化误差

```python
offset = self.offset_head(x)  # (B, 2, H/4, W/4)

# offset[:, 0, :, :] → offset_x ∈ (-∞, +∞)
# offset[:, 1, :, :] → offset_y ∈ (-∞, +∞)

# 无激活函数，允许正负值
```

**量化误差问题**：

```
原图坐标：(100.7, 200.3)
↓ 除以 stride=4
热图坐标：(25.175, 50.075)
↓ 取整
离散坐标：(25, 50)
↓ 乘以 stride=4
还原坐标：(100, 200)  # 丢失了 0.7 和 0.3

Offset 的作用：
offset_x = 0.175, offset_y = 0.075
最终坐标 = (25 + 0.175) * 4 = 100.7 ✓
```

---

### 5.5 检测头参数量

```python
# 假设 feat_channels=256, num_classes=1

# Shared conv: 3×3
shared_params = 256 * 256 * 9 + 256 = 590,080

# Heatmap head:
#   3×3 conv: 256*256*9 + 256 = 590,080
#   1×1 conv: 256*1*1 + 1 = 257
heatmap_params = 590,337

# Size head:
#   3×3 conv: 256*256*9 + 256 = 590,080
#   1×1 conv: 256*2*1 + 2 = 514
size_params = 590,594

# Offset head:
#   3×3 conv: 256*256*9 + 256 = 590,080
#   1×1 conv: 256*2*1 + 2 = 514
offset_params = 590,594

# Total head params
total = 590,080 + 590,337 + 590,594 + 590,594 ≈ 2.36M
```

---

## 六、完整前向传播流程

### 6.1 逐步维度追踪

假设输入 `image: (B, 1, 1024, 1024)`

```python
# ========== Backbone ==========
# Stem
x = self.stem(image)  # (B, 96, 256, 256)  [÷4]

# Stage 1
for block in self.stages[0]:
    x = block(x)  # (B, 96, 256, 256)
C1 = x  # (B, 96, 256, 256)

# Downsample 1
x = self.downsamples[0](C1)  # (B, 192, 128, 128)  [÷2]

# Stage 2
for block in self.stages[1]:
    x = block(x)
C2 = x  # (B, 192, 128, 128)

# Downsample 2
x = self.downsamples[1](C2)  # (B, 384, 64, 64)  [÷2]

# Stage 3
for block in self.stages[2]:
    x = block(x)
C3 = x  # (B, 384, 64, 64)

# Downsample 3
x = self.downsamples[2](C3)  # (B, 768, 32, 32)  [÷2]

# Stage 4
for block in self.stages[3]:
    x = block(x)
C4 = x  # (B, 768, 32, 32)

features = [C1, C2, C3, C4]

# ========== FPN ==========
# Lateral convolutions
L1 = self.lateral_convs[0](C1)  # (B, 256, 256, 256)
L2 = self.lateral_convs[1](C2)  # (B, 256, 128, 128)
L3 = self.lateral_convs[2](C3)  # (B, 256, 64, 64)
L4 = self.lateral_convs[3](C4)  # (B, 256, 32, 32)

# Top-down fusion
L3 = L3 + interpolate(L4, size=(64, 64))  # (B, 256, 64, 64)
L2 = L2 + interpolate(L3, size=(128, 128))  # (B, 256, 128, 128)
L1 = L1 + interpolate(L2, size=(256, 256))  # (B, 256, 256, 256)

# Output convolutions
P1 = self.output_convs[0](L1)  # (B, 256, 256, 256)
P2 = self.output_convs[1](L2)  # (B, 256, 128, 128)
P3 = self.output_convs[2](L3)  # (B, 256, 64, 64)
P4 = self.output_convs[3](L4)  # (B, 256, 32, 32)

# ========== Head ==========
# 使用 P2 进行检测
feature = P2  # (B, 256, 128, 128)

# Shared conv
x = self.shared_conv(feature)  # (B, 256, 128, 128)

# Heatmap
heatmap = self.heatmap_head(x)  # (B, 1, 128, 128)

# Size
size = self.size_head(x)  # (B, 2, 128, 128)

# Offset
offset = self.offset_head(x)  # (B, 2, 128, 128)

outputs = {'heatmap': heatmap, 'size': size, 'offset': offset}
```

---

### 6.2 解码为颗粒坐标

```python
# ========== Decode ==========
heatmap = outputs['heatmap']  # (B, 1, 128, 128)
size = outputs['size']        # (B, 2, 128, 128)
offset = outputs['offset']    # (B, 2, 128, 128)

# Step 1: 提取热图峰值
particles = extract_peaks(heatmap, score_threshold=0.3)
# particles = [{x, y, score, class_id, batch_idx}, ...]
# x, y 是热图上的整数坐标 (0-127)

# Step 2: 应用尺寸和偏移
for p in particles:
    batch_idx = p['batch_idx']
    x_out, y_out = p['x'], p['y']
    
    # 读取尺寸
    p['width'] = size[batch_idx, 0, y_out, x_out]
    p['height'] = size[batch_idx, 1, y_out, x_out]
    
    # 读取偏移并修正坐标
    offset_x = offset[batch_idx, 0, y_out, x_out]
    offset_y = offset[batch_idx, 1, y_out, x_out]
    
    p['x'] = (x_out + offset_x) * 4  # 乘回 stride
    p['y'] = (y_out + offset_y) * 4

# Step 3: NMS
particles = apply_nms(particles, radius=20)
# 移除距离 < 20 像素的低分粒子

# 最终输出
# particles = [{x, y, score, width, height, class_id}, ...]
```

---

## 七、设计原理分析

### 7.1 为什么选择 ConvNeXt？

**与传统 CNN 对比**：

| 特性 | ResNet | ConvNeXt | 优势 |
|------|--------|----------|------|
| **Block 设计** | Bottleneck | Inverted Bottleneck | 更高维特征空间 |
| **激活函数** | ReLU | GELU | 更平滑的梯度 |
| **归一化** | BatchNorm | LayerNorm | 不依赖 batch 统计 |
| **下采样** | 逐层 2×2 | 直接 4×4 | 更快收敛 |
| **大卷积核** | 3×3 | 7×7 depthwise | 更大感受野 |
| **正则化** | Dropout | DropPath | 更适合残差结构 |

**与 Transformer 对比**：

| 特性 | ViT | ConvNeXt | 优势 |
|------|-----|----------|------|
| **归纳偏置** | 无 | 有（局部性、平移不变性） | 更适合视觉任务 |
| **计算效率** | O(n²) | O(n) | 更快推理 |
| **训练难度** | 需要大量数据 | 较容易 | 小数据集友好 |

---

### 7.2 为什么使用 FPN？

**单尺度检测的问题**：

```
只用 C4 (32×32):
- 小物体（< 16 像素）可能完全消失
- 定位粗糙（每格代表 32 像素）

只用 C1 (256×256):
- 缺乏语义信息（感受野太小）
- 难以区分相似物体
```

**FPN 的优势**：

```
融合 C1+C2+C3+C4 → P2:
- 保留高分辨率（256×256）
- 融入深层语义信息
- 适合多尺度物体检测
```

---

### 7.3 为什么选择 CenterNet？

**与 Anchor-based 方法对比**：

| 特性 | Faster R-CNN | CenterNet | SuPicker 适用性 |
|------|-------------|-----------|----------------|
| **Anchor** | 需要预设 | 无需 anchor | ✅ 颗粒形状多变 |
| **NMS** | 必需 | 可选 | ✅ 简化流程 |
| **速度** | 较慢 | 较快 | ✅ 实时性要求 |
| **精度** | 高 | 高 | ✅ 满足需求 |

**与 YOLO 系列对比**：

| 特性 | YOLOv5 | CenterNet | SuPicker 适用性 |
|------|--------|-----------|----------------|
| **网格划分** | 粗 (S×S) | 细 (H/4×W/4) | ✅ 精确定位 |
| **多标签** | 困难 | 天然支持 | ✅ 多类颗粒 |
| **小物体** | 较弱 | 较强 | ✅ 小颗粒检测 |

---

### 7.4 损失函数设计原理

**Combined Loss 的组成**：

```python
total_loss = w1*L_heatmap + w2*L_size + w3*L_offset

# 默认权重
w1 = 1.0  # 热图损失（Focal Loss）
w2 = 0.1  # 尺寸损失（L1）
w3 = 1.0  # 偏移损失（L1）
```

**为什么这样设置权重？**

1. **热图损失最重要** (w1=1.0)
   - 决定是否有颗粒（分类问题）
   - Focal Loss 处理正负样本不平衡

2. **尺寸损失权重较低** (w2=0.1)
   - 尺寸是回归问题，相对容易
   - 对最终检测效果影响较小

3. **偏移损失重要** (w3=1.0)
   - 直接影响定位精度
   - 亚像素级修正很关键

---

### 7.5 Focal Loss 的设计

```python
FL(p_t) = -α_t * (1-p_t)^β * log(p_t)

# α=2.0: 平衡因子
# β=4.0: 调制因子
```

**工作原理**：

```python
# 对于正样本 (target=1)
p_t = pred  # 预测概率
FL = -2.0 * (1-p_t)^4 * log(p_t)

# 示例：
# p_t = 0.9 → FL = -2*(0.1)^4*log(0.9) ≈ 0.0002  (简单样本，损失很小)
# p_t = 0.1 → FL = -2*(0.9)^4*log(0.1) ≈ 3.0    (困难样本，损失很大)

# 对于负样本 (target=0)
p_t = 1-pred
FL = -(1-2.0) * (1-(1-p_t))^4 * log(1-p_t)
   = 1.0 * p_t^4 * log(1-p_t)
```

**效果**：
- 降低简单负样本的权重（大部分背景区域）
- 聚焦于困难样本（颗粒中心附近）
- 解决正负样本极度不平衡问题（1:1000）

---

## 八、参数量与计算量分析

### 8.1 参数量 breakdown

以 ConvNeXt-Tiny 为例：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| **Stem** | 1.8K | <0.01% |
| **Stage1** (3 blocks) | 0.8M | 2.9% |
| **Stage2** (3 blocks) | 1.5M | 5.4% |
| **Stage3** (9 blocks) | 5.2M | 18.6% |
| **Stage4** (3 blocks) | 5.8M | 20.7% |
| **Downsamples** | 0.6M | 2.1% |
| **Backbone Total** | ~14M | 50% |
| **FPN** | 2.7M | 9.6% |
| **Head** | 2.4M | 8.6% |
| **Total** | ~28M | 100% |

---

### 8.2 计算量分析 (FLOPs)

对于 `1024×1024` 输入：

| 阶段 | 输出尺寸 | FLOPs | 占比 |
|------|---------|-------|------|
| **Stem** | 256×256 | 6.2G | 5% |
| **Stage1** | 256×256 | 18.7G | 15% |
| **Stage2** | 128×128 | 18.7G | 15% |
| **Stage3** | 64×64 | 18.7G | 15% |
| **Stage4** | 32×32 | 9.3G | 7.5% |
| **FPN** | 多尺度 | 25G | 20% |
| **Head** | 256×256 | 28G | 22.5% |
| **Total** | - | ~125G | 100% |

**注意**：FLOPs 随输入尺寸平方增长

---

### 8.3 显存占用分析

训练时显存占用（batch_size=8, 1024×1024）：

| 项目 | 显存 (GB) |
|------|----------|
| **模型参数** | 0.1 (28M × 4 bytes) |
| **梯度** | 0.1 |
| **优化器状态** | 0.2 (Adam: 2×params) |
| ** activations** | 8-10 (中间特征图) |
| **AMP 缓冲** | 0.5 |
| **Total** | ~9-11 GB |

**优化建议**：
- 减小 batch_size: 8 → 4 (节省 4-5GB)
- 启用 AMP: 节省 30-50%
- 减小 crop_size: 1024 → 512 (节省 75% activations)

---

## 九、常见修改方案

### 9.1 轻量化方案

**方案 1: 使用 MobileNetV3**

```python
from torchvision.models import mobilenet_v3_small

class MobileNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        backbone = mobilenet_v3_small(pretrained=pretrained)
        
        # 修改第一层支持单通道
        self.features = backbone.features
        self.features[0][0] = nn.Conv2d(1, 16, 3, 2, 1)
    
    def forward(self, x):
        features = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i in [2, 4, 6, 12]:  # 提取多尺度特征
                features.append(x)
        return features
```

**效果**：
- 参数量：~2.5M (vs 28M)
- 速度：快 3-5 倍
- 精度：下降 5-10%

---

**方案 2: 减小 ConvNeXt 通道数**

```python
# 自定义 Tiny+ 配置
CONVNEXT_CONFIGS['tiny_plus'] = {
    "depths": [3, 3, 9, 3],
    "dims": [64, 128, 256, 512],  # 原来的一半
}

# 效果:
# 参数量：~7M (vs 28M)
# 速度：快 2 倍
# 精度：下降 2-5%
```

---

### 9.2 高精度方案

**方案 1: 使用更大 backbone**

```python
# ConvNeXt-Large (需自行添加配置)
CONVNEXT_CONFIGS['large'] = {
    "depths": [3, 3, 27, 3],
    "dims": [192, 384, 768, 1536],
}

# 参数量：~197M (vs 89M Base)
# 精度：提升 1-3%
```

---

**方案 2: 添加 Deformable Convolution**

```python
from torchvision.ops import DeformConv2d

class DeformableBlock(nn.Module):
    def __init__(self, dim):
        # 学习偏移量场
        self.offset_conv = nn.Conv2d(dim, 18, 3, padding=1)
        self.dcn = DeformConv2d(dim, dim, 3, padding=1)
    
    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.dcn(x, offset)
        return x
```

**效果**：
- 更好适应形变物体
- 参数量增加 ~10%
- 计算量增加 ~20%

---

### 9.3 多尺度检测方案

**当前方案**：只用 P2

```python
# Detector.forward
outputs = self.head(fpn_features[0])  # 只用 P2
```

**改进方案**：融合 P2+P3

```python
class MultiScaleHead(nn.Module):
    def __init__(self, config, in_channels=256):
        # P2 分支 (高分辨率)
        self.p2_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.p2_head = CenterNetHead(config, 256)
        
        # P3 分支 (低分辨率，大感受野)
        self.p3_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.p3_head = CenterNetHead(config, 256)
    
    def forward(self, fpn_features):
        P2, P3 = fpn_features[0], fpn_features[1]
        
        # P2 预测
        x2 = self.p2_conv(P2)
        outputs2 = self.p2_head(x2)
        
        # P3 预测
        x3 = self.p3_conv(P3)
        outputs3 = self.p3_head(x3)
        
        # 融合（例如：加权平均）
        outputs = {
            'heatmap': 0.7*outputs2['heatmap'] + 0.3*outputs3['heatmap'],
            'size': outputs2['size'],
            'offset': outputs2['offset']
        }
        
        return outputs
```

**效果**：
- 更好检测多尺度颗粒
- 计算量增加 ~50%
- 显存增加 ~30%

---

### 9.4 注意力机制增强

**添加 SE Block**：

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weight = self.fc(x)
        return x * weight

# 集成到 ConvNeXt Block
class ConvNeXtBlockWithSE(nn.Module):
    def __init__(self, dim, ...):
        super().__init__()
        # ... 原有代码 ...
        self.se = SEBlock(dim)
    
    def forward(self, x):
        x = super().forward(x)
        x = self.se(x)  # 添加通道注意力
        return x
```

**效果**：
- 增强重要通道，抑制无关通道
- 参数量增加 ~5%
- 精度提升 1-2%

---

### 9.5 替换为 BiFPN

**BiFPN vs FPN**：

```
FPN (单向自顶向下):
C4 → P4
     ↑
C3 → P3
     ↑
C2 → P2
     ↑
C1 → P1

BiFPN (双向融合):
C4 ↔ P4
     ↕
C3 ↔ P3
     ↕
C2 ↔ P2
     ↕
C1 ↔ P1
```

**BiFPN 实现**：

```python
class BiFPN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 添加自底向上路径
        self.bottom_up_convs = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1)
            for _ in range(3)
        ])
    
    def forward(self, features):
        # 自顶向下（同 FPN）
        laterals = [...]
        for i in range(3, 0, -1):
            laterals[i-1] += interpolate(laterals[i])
        
        # 自底向上（新增）
        for i in range(0, 3):
            laterals[i+1] += F.interpolate(
                self.bottom_up_convs[i](laterals[i]),
                size=laterals[i+1].shape[2:]
            )
        
        return [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
```

**效果**：
- 更强的特征融合
- 参数量增加 ~20%
- 精度提升 2-4%

---

## 十、总结

### 10.1 架构特点总结

✅ **优势**：
- 现代化 ConvNeXt backbone，性能优异
- FPN 多尺度特征融合
- CenterNet 无锚框设计，简洁高效
- 4 倍输出 stride，精确定位
- 完整的训练/推理 pipeline

⚠️ **局限**：
- 只使用 P2 层，忽略其他尺度信息
- 固定 4 倍下采样，对极小/极大颗粒不够灵活
- 单阶段检测，精度略低于两阶段方法

---

### 10.2 适用场景

🎯 **最适合**：
- Cryo-EM 颗粒拾取（圆形/椭圆形）
- 小目标检测（直径 32-256 像素）
- 需要精确定位的场景

❌ **不太适合**：
- 极大物体（>512 像素）
- 极度形变的物体
- 实时性要求极高（>30 FPS）

---

### 10.3 未来改进方向

1. **架构层面**：
   - 尝试 EfficientNet、Swin Transformer 等 backbone
   - 引入可变形卷积
   - 添加注意力机制

2. **训练策略**：
   - 知识蒸馏（大模型→小模型）
   - 半监督学习（利用未标注数据）
   - 自监督预训练

3. **推理优化**：
   - 模型量化（FP16/INT8）
   - 剪枝和压缩
   - ONNX/TensorRT 部署

---

**文档结束**

---

*如需了解具体模块的实现细节，请参考源代码或相关论文。*
