# Changelog: 2026-03-08 ~ 2026-03-09

## 概述

本次更新聚焦于 **训练流水线的稳定性和大尺寸 micrograph 的显存优化**，共涉及两个分支：

- **`main`** 分支：基础 bug 修复和功能完善
- **`feat/random-crop-oom-fix`** 分支：GPU 显存优化（RandomCrop + AMP）

---

## `main` 分支更新

### 1. 分布式训练初始化修复

**提交**: `c45418f`

- 确保 `torch.distributed.init_process_group` 在 DataLoader 创建之前调用
- 添加 `MASTER_ADDR`（默认 `localhost`）和 `MASTER_PORT`（默认 `29500`）的环境变量回退
- 当 `torch.cuda.is_available()` 为 False 时，自动将分布式后端从 `nccl` 切换到 `gloo`

### 2. Target Generator 鲁棒性增强

**提交**: `89f3562`

**修复的错误**:
- `IndexError: index 33 is out of bounds for dimension 0 with size 1` — 生成 heatmap 时 class_id 越界
- `RuntimeError` — Gaussian 核范围无效时的崩溃

**改动内容**:
- `target_generator.py` — 添加边界检查，跳过超出输出图范围或 class_id 无效的颗粒
- `_draw_gaussian()` — 检查 Gaussian 核范围有效性，无效时提前返回
- `dataset.py` — 当 `num_classes=1` 时，强制所有 class_id 为 0；class_id 越界时回退到 0

### 3. STAR 文件解析改进

- 支持 `data_` 和 `data_particles` 两种 header 格式
- 修复 `in_data_particles` 未初始化导致的 `UnboundLocalError`
- micrograph 名称提取改为直接使用列值，避免路径解析错误

### 4. 自定义 Collate 函数

**提交**: `f1ec1e8`

- 添加 `particle_collate_fn`，将 tensor 字段（image/heatmap/size/offset/mask）正常堆叠
- `particles` 字段保持为 list-of-lists（每张图颗粒数量不同，无法堆叠为 tensor）
- 解决 `RuntimeError: each element in list of batch should be of equal size`

### 5. 设备选择修复

**提交**: `8ac6b0e`

- 修复 `--device cuda:7` 被分布式初始化覆盖为 `cuda:0` 的问题
- 仅当环境变量 `LOCAL_RANK` 存在时（即通过 `torchrun` 启动），才用 `cuda:{LOCAL_RANK}` 覆盖用户指定的 device

---

## `feat/random-crop-oom-fix` 分支更新

该分支在 `main` 之上新增以下功能（待合并）：

### 6. RandomCrop Transform — 解决大图 OOM

**提交**: `7e2c31d`

冷冻电镜 micrograph 通常 4096×4096 或更大，全分辨率送入网络即使 batch_size=1 也可能 OOM。

**实现**:
- 添加 `RandomCrop` transform，从大图中随机裁出固定大小区域（默认 1024×1024）
- 自动过滤裁剪区域外的颗粒并调整坐标
- 在 `AugmentationConfig` 中新增 `crop_size` 参数（默认 1024，设为 0 禁用）
- 裁剪作为 transform pipeline 的第一步执行
- 修复 `image_size` 在 transforms 之后获取，确保和裁剪后的尺寸匹配

### 7. AMP 混合精度训练

**提交**: `c3ce3f3`

**效果**: 节省约 30-50% 显存，并加速训练 1.5-2x。

**实现**:
- 在 `TrainingConfig` 中添加 `use_amp: bool = True`
- `Trainer.__init__` 中初始化 `torch.amp.GradScaler`
- `train_step` 使用 `torch.amp.autocast` 包裹前向传播
- `validate` 同样使用 `autocast`
- `scripts/train.py` 添加 `--no-amp` 参数用于禁用
- 在 CPU 上自动禁用 AMP

### 8. 修复 AMP 下 Loss 变 NaN

**提交**: `5d4fb02`

AMP 使用 FP16 计算，导致 Focal Loss 中的 `log()` 和 `pow()` 操作溢出产生 NaN。

**修复**:
- 所有 Loss 函数内部强制 `.float()`（FP32）计算
- `torch.clamp` 的 epsilon 从 `1e-6` 放大到 `1e-4`
- `num_pos.clamp(min=1)` 替代 `if num_pos == 0` 分支，避免空 patch 时的未归一化求和
- 修复范围：`FocalLoss`、`GaussianFocalLoss`、`RegL1Loss`、`SmoothL1Loss`

### 9. 验证集 OOM 修复

**提交**: `8b22614`

验证集之前使用 `transforms=None`，导致全分辨率 micrograph 直接送入网络。

**修复**: 验证集使用独立的 transform（仅 Crop + Normalize，关闭所有增强）。

---

## GPU 指定与多卡训练指南

### 单卡训练

```bash
# 默认使用 GPU 0
python scripts/train.py --train-images ... --train-star ...

# 指定 GPU
python scripts/train.py --train-images ... --train-star ... --device cuda:7

# 通过环境变量指定
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --train-images ... --train-star ...
```

### 多卡训练

```bash
# 使用所有 GPU
torchrun --nproc_per_node=4 scripts/train.py \
    --train-images ... --train-star ... --distributed

# 指定特定 GPU
CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 scripts/train.py \
    --train-images ... --train-star ... --distributed
```

> **注意**: 多卡训练时不要使用 `--device` 参数，`torchrun` 会自动通过 `LOCAL_RANK` 分配设备。

---

## 训练建议

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--batch-size` | 8-20 | 越大越稳定，受显存限制 |
| `--lr` | `1e-4` ~ `2e-4` | 大 batch 可适当提高 |
| `--backbone` | `tiny` | 资源受限时首选 |
| `crop_size` | 1024 | 默认值，可按需调整 |
| AMP | 默认开启 | 如遇精度问题用 `--no-amp` 关闭 |

### Loss 正常范围参考

- **Epoch 1**: 数万级（模型随机初始化，大量假阳性）
- **Epoch 2-3**: 快速下降到两位数
- **Epoch 10+**: 应降到个位数范围
- **最终收敛**: 1-10（Focal Loss 典型值）

### 异常排查

| 现象 | 可能原因 | 解决方案 |
|---|---|---|
| Loss = NaN | AMP 数值溢出 | 已修复；如仍出现加 `--no-amp` |
| Loss 不下降 | 学习率过小 | 提高 `--lr` |
| Loss 震荡 | 学习率过大/batch 太小 | 降低 `--lr`或增加 `--batch-size` |
| OOM | 图像太大/batch 太大 | 降低 `--batch-size` 或减小 `crop_size` |
