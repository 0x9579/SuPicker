# SuPicker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a cryo-EM particle detection neural network with ConvNeXt backbone, FPN neck, and CenterNet head.

**Architecture:** PyTorch-based detector with modular config system, STAR file parsing, rich data augmentation, and multi-format output support.

**Tech Stack:** Python 3.10+, PyTorch, tifffile, numpy, tensorboard

---

## Phase 1: Project Setup & Configuration

### Task 1: Initialize Project Structure

**Files:**
- Create: `supicker/__init__.py`
- Create: `supicker/config/__init__.py`
- Create: `supicker/models/__init__.py`
- Create: `supicker/data/__init__.py`
- Create: `supicker/losses/__init__.py`
- Create: `supicker/engine/__init__.py`
- Create: `supicker/utils/__init__.py`
- Create: `scripts/__init__.py`
- Create: `tests/__init__.py`
- Create: `requirements.txt`

**Step 1: Create directory structure**

```bash
mkdir -p supicker/config supicker/models/backbone supicker/models/neck supicker/models/head supicker/data supicker/losses supicker/engine supicker/utils scripts tests
```

**Step 2: Create __init__.py files**

`supicker/__init__.py`:
```python
__version__ = "0.1.0"
```

All other `__init__.py` files: empty.

**Step 3: Create requirements.txt**

```text
torch>=2.0.0
torchvision>=0.15.0
tifffile>=2023.1.1
numpy>=1.24.0
tensorboard>=2.12.0
```

**Step 4: Commit**

```bash
git init
git add .
git commit -m "chore: initialize project structure"
```

---

### Task 2: Base Configuration Classes

**Files:**
- Create: `supicker/config/base.py`
- Create: `tests/config/__init__.py`
- Create: `tests/config/test_base.py`

**Step 1: Write failing test**

`tests/config/test_base.py`:
```python
import pytest
from supicker.config.base import Config


def test_config_to_dict():
    config = Config()
    result = config.to_dict()
    assert isinstance(result, dict)


def test_config_from_dict():
    data = {"key": "value"}
    config = Config.from_dict(data)
    assert isinstance(config, Config)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/config/test_base.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/config/base.py`:
```python
from dataclasses import dataclass, asdict, fields
from typing import TypeVar, Type

T = TypeVar("T", bound="Config")


@dataclass
class Config:
    """Base configuration class with serialization support."""

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """Create config from dictionary."""
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/config/test_base.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/config/base.py tests/config/
git commit -m "feat(config): add base Config class with serialization"
```

---

### Task 3: Model Configuration Classes

**Files:**
- Create: `supicker/config/model.py`
- Create: `tests/config/test_model.py`

**Step 1: Write failing test**

`tests/config/test_model.py`:
```python
import pytest
from supicker.config.model import (
    ConvNeXtVariant,
    BackboneConfig,
    FPNConfig,
    HeadConfig,
    ModelConfig,
)


def test_convnext_variant_values():
    assert ConvNeXtVariant.TINY.value == "tiny"
    assert ConvNeXtVariant.SMALL.value == "small"
    assert ConvNeXtVariant.BASE.value == "base"


def test_backbone_config_defaults():
    config = BackboneConfig()
    assert config.variant == ConvNeXtVariant.TINY
    assert config.pretrained is True
    assert config.pretrained_path is None
    assert config.in_channels == 1


def test_fpn_config_defaults():
    config = FPNConfig()
    assert config.in_channels == [96, 192, 384, 768]
    assert config.out_channels == 256


def test_head_config_defaults():
    config = HeadConfig()
    assert config.num_classes == 1
    assert config.feat_channels == 256


def test_model_config_composition():
    config = ModelConfig()
    assert isinstance(config.backbone, BackboneConfig)
    assert isinstance(config.fpn, FPNConfig)
    assert isinstance(config.head, HeadConfig)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/config/test_model.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/config/model.py`:
```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .base import Config


class ConvNeXtVariant(Enum):
    """ConvNeXt model variants."""
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"


@dataclass
class BackboneConfig(Config):
    """ConvNeXt backbone configuration."""
    variant: ConvNeXtVariant = ConvNeXtVariant.TINY
    pretrained: bool = True
    pretrained_path: Optional[str] = None
    in_channels: int = 1  # Grayscale cryo-EM images


@dataclass
class FPNConfig(Config):
    """Feature Pyramid Network configuration."""
    in_channels: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    out_channels: int = 256


@dataclass
class HeadConfig(Config):
    """CenterNet head configuration."""
    num_classes: int = 1
    feat_channels: int = 256
    heatmap_loss: str = "focal"
    heatmap_loss_params: dict = field(default_factory=lambda: {"alpha": 2.0, "beta": 4.0})
    size_loss: str = "l1"
    size_loss_params: dict = field(default_factory=dict)
    offset_loss: str = "l1"
    offset_loss_params: dict = field(default_factory=dict)


@dataclass
class ModelConfig(Config):
    """Complete model configuration."""
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/config/test_model.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/config/model.py tests/config/test_model.py
git commit -m "feat(config): add model configuration classes"
```

---

### Task 4: Data Configuration Classes

**Files:**
- Create: `supicker/config/data.py`
- Create: `tests/config/test_data.py`

**Step 1: Write failing test**

`tests/config/test_data.py`:
```python
import pytest
from supicker.config.data import AugmentationConfig, DataConfig


def test_augmentation_config_defaults():
    config = AugmentationConfig()
    assert config.horizontal_flip is True
    assert config.vertical_flip is True
    assert config.rotation_90 is True
    assert config.random_rotation is True
    assert config.rotation_range == (-180.0, 180.0)
    assert config.gaussian_noise is True
    assert config.ctf_simulation is False


def test_data_config_defaults():
    config = DataConfig()
    assert config.train_image_dir == ""
    assert config.train_star_file == ""
    assert isinstance(config.augmentation, AugmentationConfig)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/config/test_data.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/config/data.py`:
```python
from dataclasses import dataclass, field
from typing import Optional

from .base import Config


@dataclass
class AugmentationConfig(Config):
    """Data augmentation configuration."""
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_90: bool = True
    random_rotation: bool = True
    rotation_range: tuple[float, float] = (-180.0, 180.0)
    brightness: bool = True
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast: bool = True
    contrast_range: tuple[float, float] = (0.8, 1.2)
    gaussian_noise: bool = True
    noise_std: float = 0.02
    ctf_simulation: bool = False


@dataclass
class DataConfig(Config):
    """Data loading and preprocessing configuration."""
    train_image_dir: str = ""
    train_star_file: str = ""
    val_image_dir: Optional[str] = None
    val_star_file: Optional[str] = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    num_workers: int = 4
    pin_memory: bool = True
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/config/test_data.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/config/data.py tests/config/test_data.py
git commit -m "feat(config): add data and augmentation configuration"
```

---

### Task 5: Training Configuration Classes

**Files:**
- Create: `supicker/config/training.py`
- Create: `tests/config/test_training.py`

**Step 1: Write failing test**

`tests/config/test_training.py`:
```python
import pytest
from supicker.config.training import LossConfig, TrainingConfig


def test_loss_config_defaults():
    config = LossConfig()
    assert config.heatmap_type == "focal"
    assert config.heatmap_weight == 1.0
    assert config.focal_alpha == 2.0
    assert config.focal_beta == 4.0
    assert config.size_type == "l1"
    assert config.size_weight == 0.1


def test_training_config_defaults():
    config = TrainingConfig()
    assert config.batch_size == 8
    assert config.epochs == 100
    assert config.optimizer == "adamw"
    assert config.learning_rate == 1e-4
    assert config.scheduler == "cosine"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/config/test_training.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/config/training.py`:
```python
from dataclasses import dataclass, field

from .base import Config


@dataclass
class LossConfig(Config):
    """Loss function configuration."""
    # Heatmap loss
    heatmap_type: str = "focal"  # focal, gaussian_focal, mse
    heatmap_weight: float = 1.0
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    # Size regression loss
    size_type: str = "l1"  # l1, smooth_l1
    size_weight: float = 0.1
    smooth_l1_beta: float = 1.0
    # Offset loss
    offset_type: str = "l1"
    offset_weight: float = 1.0


@dataclass
class TrainingConfig(Config):
    """Training configuration."""
    batch_size: int = 8
    epochs: int = 100
    optimizer: str = "adamw"  # adam, adamw, sgd
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"  # cosine, step, none
    warmup_epochs: int = 5
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_interval: int = 10
    val_interval: int = 1
    loss: LossConfig = field(default_factory=LossConfig)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/config/test_training.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/config/training.py tests/config/test_training.py
git commit -m "feat(config): add training and loss configuration"
```

---

### Task 6: Inference Configuration Classes

**Files:**
- Create: `supicker/config/inference.py`
- Create: `tests/config/test_inference.py`

**Step 1: Write failing test**

`tests/config/test_inference.py`:
```python
import pytest
from supicker.config.inference import InferenceConfig


def test_inference_config_defaults():
    config = InferenceConfig()
    assert config.checkpoint_path == ""
    assert config.score_threshold == 0.3
    assert config.nms_enabled is True
    assert config.nms_radius == 20.0
    assert config.output_format == "star"
    assert config.batch_size == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/config/test_inference.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/config/inference.py`:
```python
from dataclasses import dataclass

from .base import Config


@dataclass
class InferenceConfig(Config):
    """Inference configuration."""
    checkpoint_path: str = ""
    score_threshold: float = 0.3
    nms_enabled: bool = True
    nms_radius: float = 20.0  # pixels
    output_format: str = "star"  # star, json, csv
    batch_size: int = 1
    device: str = "cuda"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/config/test_inference.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/config/inference.py tests/config/test_inference.py
git commit -m "feat(config): add inference configuration"
```

---

### Task 7: Config Module Exports

**Files:**
- Modify: `supicker/config/__init__.py`

**Step 1: Update config __init__.py**

`supicker/config/__init__.py`:
```python
from .base import Config
from .model import (
    ConvNeXtVariant,
    BackboneConfig,
    FPNConfig,
    HeadConfig,
    ModelConfig,
)
from .data import AugmentationConfig, DataConfig
from .training import LossConfig, TrainingConfig
from .inference import InferenceConfig

__all__ = [
    "Config",
    "ConvNeXtVariant",
    "BackboneConfig",
    "FPNConfig",
    "HeadConfig",
    "ModelConfig",
    "AugmentationConfig",
    "DataConfig",
    "LossConfig",
    "TrainingConfig",
    "InferenceConfig",
]
```

**Step 2: Run all config tests**

```bash
pytest tests/config/ -v
```
Expected: All PASS

**Step 3: Commit**

```bash
git add supicker/config/__init__.py
git commit -m "feat(config): export all configuration classes"
```

---

## Phase 2: Model Architecture

### Task 8: ConvNeXt Block

**Files:**
- Create: `supicker/models/backbone/__init__.py`
- Create: `supicker/models/backbone/convnext.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/backbone/__init__.py`
- Create: `tests/models/backbone/test_convnext.py`

**Step 1: Write failing test**

`tests/models/backbone/test_convnext.py`:
```python
import pytest
import torch
from supicker.models.backbone.convnext import ConvNeXtBlock, LayerNorm2d


def test_layer_norm_2d_shape():
    norm = LayerNorm2d(96)
    x = torch.randn(2, 96, 32, 32)
    out = norm(x)
    assert out.shape == (2, 96, 32, 32)


def test_convnext_block_shape():
    block = ConvNeXtBlock(dim=96)
    x = torch.randn(2, 96, 32, 32)
    out = block(x)
    assert out.shape == (2, 96, 32, 32)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/models/backbone/test_convnext.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/models/backbone/convnext.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with depthwise conv and inverted bottleneck."""

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = shortcut + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Drop paths (stochastic depth) for regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/models/backbone/test_convnext.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/models/backbone/ tests/models/
git commit -m "feat(model): add ConvNeXt block and LayerNorm2d"
```

---

### Task 9: ConvNeXt Backbone

**Files:**
- Modify: `supicker/models/backbone/convnext.py`
- Modify: `tests/models/backbone/test_convnext.py`

**Step 1: Write failing test**

Add to `tests/models/backbone/test_convnext.py`:
```python
from supicker.config import BackboneConfig, ConvNeXtVariant


def test_convnext_tiny_output_shapes():
    from supicker.models.backbone.convnext import ConvNeXt

    config = BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False)
    model = ConvNeXt(config)
    x = torch.randn(2, 1, 256, 256)
    features = model(x)

    assert len(features) == 4
    assert features[0].shape == (2, 96, 64, 64)    # C1: H/4
    assert features[1].shape == (2, 192, 32, 32)   # C2: H/8
    assert features[2].shape == (2, 384, 16, 16)   # C3: H/16
    assert features[3].shape == (2, 768, 8, 8)     # C4: H/32


def test_convnext_single_channel_input():
    from supicker.models.backbone.convnext import ConvNeXt

    config = BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False, in_channels=1)
    model = ConvNeXt(config)
    x = torch.randn(2, 1, 128, 128)
    features = model(x)
    assert features[0].shape[1] == 96
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/models/backbone/test_convnext.py::test_convnext_tiny_output_shapes -v
```
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

Add to `supicker/models/backbone/convnext.py`:
```python
from supicker.config import BackboneConfig, ConvNeXtVariant

# Model configurations for each variant
CONVNEXT_CONFIGS = {
    ConvNeXtVariant.TINY: {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
    ConvNeXtVariant.SMALL: {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
    ConvNeXtVariant.BASE: {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
}


class ConvNeXt(nn.Module):
    """ConvNeXt backbone for feature extraction."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        variant_config = CONVNEXT_CONFIGS[config.variant]
        depths = variant_config["depths"]
        dims = variant_config["dims"]

        # Stem: patchify with 4x4 conv, stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        # Build stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        # Downsampling layers between stages
        self.downsamples = nn.ModuleList()
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsamples.append(downsample)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of feature maps [C1, C2, C3, C4] at strides [4, 8, 16, 32]
        """
        features = []
        x = self.stem(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if i < 3:
                x = self.downsamples[i](x)

        return features
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/models/backbone/test_convnext.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/models/backbone/convnext.py tests/models/backbone/test_convnext.py
git commit -m "feat(model): add ConvNeXt backbone with multi-scale output"
```

---

### Task 10: FPN Neck

**Files:**
- Create: `supicker/models/neck/__init__.py`
- Create: `supicker/models/neck/fpn.py`
- Create: `tests/models/neck/__init__.py`
- Create: `tests/models/neck/test_fpn.py`

**Step 1: Write failing test**

`tests/models/neck/test_fpn.py`:
```python
import pytest
import torch
from supicker.config import FPNConfig


def test_fpn_output_shapes():
    from supicker.models.neck.fpn import FPN

    config = FPNConfig(in_channels=[96, 192, 384, 768], out_channels=256)
    fpn = FPN(config)

    # Simulate backbone outputs at different scales
    features = [
        torch.randn(2, 96, 64, 64),   # C1
        torch.randn(2, 192, 32, 32),  # C2
        torch.randn(2, 384, 16, 16),  # C3
        torch.randn(2, 768, 8, 8),    # C4
    ]

    outputs = fpn(features)

    assert len(outputs) == 4
    assert outputs[0].shape == (2, 256, 64, 64)   # P2
    assert outputs[1].shape == (2, 256, 32, 32)   # P3
    assert outputs[2].shape == (2, 256, 16, 16)   # P4
    assert outputs[3].shape == (2, 256, 8, 8)     # P5
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/models/neck/test_fpn.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/models/neck/fpn.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from supicker.config import FPNConfig


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""

    def __init__(self, config: FPNConfig):
        super().__init__()
        self.config = config
        in_channels = config.in_channels
        out_channels = config.out_channels

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])

        # Output convolutions (3x3 conv to smooth features)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fuse multi-scale features with top-down pathway.

        Args:
            features: List of feature maps [C1, C2, C3, C4] from backbone

        Returns:
            List of fused feature maps [P2, P3, P4, P5]
        """
        # Apply lateral convolutions
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest"
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        return outputs
```

`supicker/models/neck/__init__.py`:
```python
from .fpn import FPN

__all__ = ["FPN"]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/models/neck/test_fpn.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/models/neck/ tests/models/neck/
git commit -m "feat(model): add FPN neck for multi-scale feature fusion"
```

---

### Task 11: CenterNet Head

**Files:**
- Create: `supicker/models/head/__init__.py`
- Create: `supicker/models/head/centernet.py`
- Create: `tests/models/head/__init__.py`
- Create: `tests/models/head/test_centernet.py`

**Step 1: Write failing test**

`tests/models/head/test_centernet.py`:
```python
import pytest
import torch
from supicker.config import HeadConfig


def test_centernet_head_output_shapes():
    from supicker.models.head.centernet import CenterNetHead

    config = HeadConfig(num_classes=3, feat_channels=256)
    head = CenterNetHead(config, in_channels=256)

    # Single scale feature map (P2 from FPN)
    feature = torch.randn(2, 256, 64, 64)
    outputs = head(feature)

    assert "heatmap" in outputs
    assert "size" in outputs
    assert "offset" in outputs
    assert outputs["heatmap"].shape == (2, 3, 64, 64)   # num_classes channels
    assert outputs["size"].shape == (2, 2, 64, 64)      # width, height
    assert outputs["offset"].shape == (2, 2, 64, 64)    # offset_x, offset_y


def test_centernet_head_heatmap_activation():
    from supicker.models.head.centernet import CenterNetHead

    config = HeadConfig(num_classes=1)
    head = CenterNetHead(config, in_channels=256)

    feature = torch.randn(2, 256, 32, 32)
    outputs = head(feature)

    # Heatmap should be in [0, 1] after sigmoid
    assert outputs["heatmap"].min() >= 0.0
    assert outputs["heatmap"].max() <= 1.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/models/head/test_centernet.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/models/head/centernet.py`:
```python
import torch
import torch.nn as nn

from supicker.config import HeadConfig


class CenterNetHead(nn.Module):
    """CenterNet detection head for heatmap, size, and offset prediction."""

    def __init__(self, config: HeadConfig, in_channels: int):
        super().__init__()
        self.config = config
        feat_channels = config.feat_channels

        # Shared feature layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        # Heatmap branch (class probability at each location)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, config.num_classes, kernel_size=1),
        )

        # Size branch (width, height prediction)
        self.size_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 2, kernel_size=1),
        )

        # Offset branch (sub-pixel offset)
        self.offset_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 2, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize heatmap bias for focal loss (low initial confidence)
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)  # sigmoid(-2.19) ≈ 0.1

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict heatmap, size, and offset from feature map.

        Args:
            feature: Feature map from FPN (B, C, H, W)

        Returns:
            Dictionary with 'heatmap', 'size', 'offset' tensors
        """
        x = self.shared_conv(feature)

        heatmap = self.heatmap_head(x)
        heatmap = torch.sigmoid(heatmap)

        size = self.size_head(x)
        size = torch.relu(size)  # Size must be positive

        offset = self.offset_head(x)

        return {
            "heatmap": heatmap,
            "size": size,
            "offset": offset,
        }
```

`supicker/models/head/__init__.py`:
```python
from .centernet import CenterNetHead

__all__ = ["CenterNetHead"]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/models/head/test_centernet.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/models/head/ tests/models/head/
git commit -m "feat(model): add CenterNet head for detection"
```

---

### Task 12: Complete Detector

**Files:**
- Create: `supicker/models/detector.py`
- Create: `tests/models/test_detector.py`

**Step 1: Write failing test**

`tests/models/test_detector.py`:
```python
import pytest
import torch
from supicker.config import ModelConfig, BackboneConfig, HeadConfig, ConvNeXtVariant


def test_detector_forward():
    from supicker.models.detector import Detector

    config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False),
        head=HeadConfig(num_classes=3),
    )
    model = Detector(config)

    x = torch.randn(2, 1, 256, 256)
    outputs = model(x)

    assert "heatmap" in outputs
    assert "size" in outputs
    assert "offset" in outputs
    # Output at 1/4 resolution
    assert outputs["heatmap"].shape == (2, 3, 64, 64)
    assert outputs["size"].shape == (2, 2, 64, 64)
    assert outputs["offset"].shape == (2, 2, 64, 64)


def test_detector_different_input_sizes():
    from supicker.models.detector import Detector

    config = ModelConfig(
        backbone=BackboneConfig(variant=ConvNeXtVariant.TINY, pretrained=False),
        head=HeadConfig(num_classes=1),
    )
    model = Detector(config)

    # Test various input sizes (must be divisible by 32)
    for size in [128, 256, 512]:
        x = torch.randn(1, 1, size, size)
        outputs = model(x)
        expected_out_size = size // 4
        assert outputs["heatmap"].shape == (1, 1, expected_out_size, expected_out_size)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/models/test_detector.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/models/detector.py`:
```python
import torch
import torch.nn as nn

from supicker.config import ModelConfig
from supicker.models.backbone.convnext import ConvNeXt
from supicker.models.neck.fpn import FPN
from supicker.models.head.centernet import CenterNetHead


class Detector(nn.Module):
    """Complete particle detector: ConvNeXt + FPN + CenterNet."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Build backbone
        self.backbone = ConvNeXt(config.backbone)

        # Build FPN neck
        self.neck = FPN(config.fpn)

        # Build detection head (uses P2 feature at 1/4 resolution)
        self.head = CenterNetHead(config.head, in_channels=config.fpn.out_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the detector.

        Args:
            x: Input image (B, 1, H, W)

        Returns:
            Dictionary with 'heatmap', 'size', 'offset' predictions
        """
        # Extract multi-scale features
        features = self.backbone(x)

        # Fuse features with FPN
        fpn_features = self.neck(features)

        # Detect on P2 (highest resolution FPN output)
        outputs = self.head(fpn_features[0])

        return outputs
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/models/test_detector.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/models/detector.py tests/models/test_detector.py
git commit -m "feat(model): add complete Detector class"
```

---

### Task 13: Model Module Exports

**Files:**
- Modify: `supicker/models/__init__.py`
- Modify: `supicker/models/backbone/__init__.py`

**Step 1: Update exports**

`supicker/models/backbone/__init__.py`:
```python
from .convnext import ConvNeXt, ConvNeXtBlock, LayerNorm2d

__all__ = ["ConvNeXt", "ConvNeXtBlock", "LayerNorm2d"]
```

`supicker/models/__init__.py`:
```python
from .detector import Detector
from .backbone import ConvNeXt
from .neck import FPN
from .head import CenterNetHead

__all__ = ["Detector", "ConvNeXt", "FPN", "CenterNetHead"]
```

**Step 2: Run all model tests**

```bash
pytest tests/models/ -v
```
Expected: All PASS

**Step 3: Commit**

```bash
git add supicker/models/
git commit -m "feat(model): export all model classes"
```

---

## Phase 3: Loss Functions

### Task 14: Focal Loss

**Files:**
- Create: `supicker/losses/focal_loss.py`
- Create: `tests/losses/__init__.py`
- Create: `tests/losses/test_focal_loss.py`

**Step 1: Write failing test**

`tests/losses/test_focal_loss.py`:
```python
import pytest
import torch


def test_focal_loss_shape():
    from supicker.losses.focal_loss import FocalLoss

    loss_fn = FocalLoss(alpha=2.0, beta=4.0)
    pred = torch.sigmoid(torch.randn(2, 3, 64, 64))
    target = torch.zeros(2, 3, 64, 64)
    # Add some positive samples
    target[0, 0, 32, 32] = 1.0
    target[1, 1, 16, 16] = 1.0

    loss = loss_fn(pred, target)
    assert loss.ndim == 0  # Scalar
    assert loss >= 0


def test_focal_loss_positive_sample():
    from supicker.losses.focal_loss import FocalLoss

    loss_fn = FocalLoss(alpha=2.0, beta=4.0)

    # High confidence at positive location should give low loss
    pred_good = torch.ones(1, 1, 4, 4) * 0.9
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 2, 2] = 1.0
    loss_good = loss_fn(pred_good, target)

    # Low confidence at positive location should give high loss
    pred_bad = torch.ones(1, 1, 4, 4) * 0.1
    loss_bad = loss_fn(pred_bad, target)

    assert loss_bad > loss_good
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/losses/test_focal_loss.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/losses/focal_loss.py`:
```python
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss for heatmap prediction in CenterNet.

    Based on CornerNet: https://arxiv.org/abs/1808.01244
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Predicted heatmap (B, C, H, W), values in [0, 1]
            target: Target heatmap (B, C, H, W), values in [0, 1]

        Returns:
            Scalar loss value
        """
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-6, max=1 - 1e-6)

        # Positive samples (target == 1)
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        # Positive loss
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred) * pos_mask

        # Negative loss with reduced weight near positive samples
        neg_weight = torch.pow(1 - target, self.beta)
        neg_loss = -neg_weight * torch.pow(pred, self.alpha) * torch.log(1 - pred) * neg_mask

        # Normalize by number of positive samples
        num_pos = pos_mask.sum()
        if num_pos == 0:
            loss = neg_loss.sum()
        else:
            loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss


class GaussianFocalLoss(nn.Module):
    """Gaussian focal loss variant with quality-aware weighting."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian focal loss."""
        pred = torch.clamp(pred, min=1e-6, max=1 - 1e-6)

        pos_mask = target.ge(0.99).float()
        neg_mask = target.lt(0.99).float()

        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred) * pos_mask
        neg_weight = torch.pow(1 - target, self.beta)
        neg_loss = -neg_weight * torch.pow(pred, self.alpha) * torch.log(1 - pred) * neg_mask

        num_pos = pos_mask.sum()
        if num_pos == 0:
            loss = neg_loss.sum()
        else:
            loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/losses/test_focal_loss.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/losses/ tests/losses/
git commit -m "feat(loss): add FocalLoss and GaussianFocalLoss"
```

---

### Task 15: Regression Losses

**Files:**
- Create: `supicker/losses/regression_loss.py`
- Create: `tests/losses/test_regression_loss.py`

**Step 1: Write failing test**

`tests/losses/test_regression_loss.py`:
```python
import pytest
import torch


def test_l1_loss_at_locations():
    from supicker.losses.regression_loss import RegL1Loss

    loss_fn = RegL1Loss()
    pred = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    mask = torch.zeros(2, 64, 64)
    mask[0, 32, 32] = 1
    mask[1, 16, 16] = 1

    loss = loss_fn(pred, target, mask)
    assert loss.ndim == 0
    assert loss >= 0


def test_smooth_l1_loss_at_locations():
    from supicker.losses.regression_loss import SmoothL1Loss

    loss_fn = SmoothL1Loss(beta=1.0)
    pred = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    mask = torch.zeros(2, 64, 64)
    mask[0, 32, 32] = 1

    loss = loss_fn(pred, target, mask)
    assert loss.ndim == 0
    assert loss >= 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/losses/test_regression_loss.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/losses/regression_loss.py`:
```python
import torch
import torch.nn as nn


class RegL1Loss(nn.Module):
    """L1 loss computed only at specified locations."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute L1 loss at masked locations.

        Args:
            pred: Predictions (B, C, H, W)
            target: Targets (B, C, H, W)
            mask: Binary mask (B, H, W) indicating valid locations

        Returns:
            Scalar loss value
        """
        # Expand mask to match prediction channels
        mask = mask.unsqueeze(1).expand_as(pred)

        loss = torch.abs(pred - target) * mask
        num_pos = mask.sum()

        if num_pos == 0:
            return loss.sum()
        return loss.sum() / num_pos


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss computed only at specified locations."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute Smooth L1 loss at masked locations.

        Args:
            pred: Predictions (B, C, H, W)
            target: Targets (B, C, H, W)
            mask: Binary mask (B, H, W) indicating valid locations

        Returns:
            Scalar loss value
        """
        mask = mask.unsqueeze(1).expand_as(pred)
        diff = torch.abs(pred - target)

        # Smooth L1 formula
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta,
        )
        loss = loss * mask

        num_pos = mask.sum()
        if num_pos == 0:
            return loss.sum()
        return loss.sum() / num_pos
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/losses/test_regression_loss.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add supicker/losses/regression_loss.py tests/losses/test_regression_loss.py
git commit -m "feat(loss): add RegL1Loss and SmoothL1Loss"
```

---

### Task 16: Combined Loss and Module Exports

**Files:**
- Create: `supicker/losses/combined.py`
- Modify: `supicker/losses/__init__.py`
- Create: `tests/losses/test_combined.py`

**Step 1: Write failing test**

`tests/losses/test_combined.py`:
```python
import pytest
import torch
from supicker.config import LossConfig


def test_combined_loss():
    from supicker.losses.combined import CombinedLoss

    config = LossConfig()
    loss_fn = CombinedLoss(config)

    outputs = {
        "heatmap": torch.sigmoid(torch.randn(2, 3, 64, 64)),
        "size": torch.abs(torch.randn(2, 2, 64, 64)),
        "offset": torch.randn(2, 2, 64, 64),
    }

    targets = {
        "heatmap": torch.zeros(2, 3, 64, 64),
        "size": torch.zeros(2, 2, 64, 64),
        "offset": torch.zeros(2, 2, 64, 64),
        "mask": torch.zeros(2, 64, 64),
    }
    # Add positive sample
    targets["heatmap"][0, 0, 32, 32] = 1.0
    targets["mask"][0, 32, 32] = 1.0

    loss, loss_dict = loss_fn(outputs, targets)

    assert loss.ndim == 0
    assert "heatmap_loss" in loss_dict
    assert "size_loss" in loss_dict
    assert "offset_loss" in loss_dict
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/losses/test_combined.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

`supicker/losses/combined.py`:
```python
import torch
import torch.nn as nn

from supicker.config import LossConfig
from .focal_loss import FocalLoss, GaussianFocalLoss
from .regression_loss import RegL1Loss, SmoothL1Loss


class CombinedLoss(nn.Module):
    """Combined loss for CenterNet training."""

    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

        # Build heatmap loss
        if config.heatmap_type == "focal":
            self.heatmap_loss = FocalLoss(
                alpha=config.focal_alpha, beta=config.focal_beta
            )
        elif config.heatmap_type == "gaussian_focal":
            self.heatmap_loss = GaussianFocalLoss(
                alpha=config.focal_alpha, beta=config.focal_beta
            )
        else:
            self.heatmap_loss = nn.MSELoss(reduction="mean")

        # Build size loss
        if config.size_type == "l1":
            self.size_loss = RegL1Loss()
        else:
            self.size_loss = SmoothL1Loss(beta=config.smooth_l1_beta)

        # Build offset loss
        if config.offset_type == "l1":
            self.offset_loss = RegL1Loss()
        else:
            self.offset_loss = SmoothL1Loss(beta=config.smooth_l1_beta)

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            outputs: Model outputs with 'heatmap', 'size', 'offset'
            targets: Ground truth with 'heatmap', 'size', 'offset', 'mask'

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        heatmap_loss = self.heatmap_loss(outputs["heatmap"], targets["heatmap"])
        size_loss = self.size_loss(outputs["size"], targets["size"], targets["mask"])
        offset_loss = self.offset_loss(outputs["offset"], targets["offset"], targets["mask"])

        total_loss = (
            self.config.heatmap_weight * heatmap_loss
            + self.config.size_weight * size_loss
            + self.config.offset_weight * offset_loss
        )

        loss_dict = {
            "heatmap_loss": heatmap_loss.item(),
            "size_loss": size_loss.item(),
            "offset_loss": offset_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict
```

`supicker/losses/__init__.py`:
```python
from .focal_loss import FocalLoss, GaussianFocalLoss
from .regression_loss import RegL1Loss, SmoothL1Loss
from .combined import CombinedLoss

__all__ = [
    "FocalLoss",
    "GaussianFocalLoss",
    "RegL1Loss",
    "SmoothL1Loss",
    "CombinedLoss",
]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/losses/ -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add supicker/losses/
git commit -m "feat(loss): add CombinedLoss and export all losses"
```

---

## Phase 4: Data Processing

### Task 17: STAR File Parser

**Files:**
- Create: `supicker/data/star_parser.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_star_parser.py`
- Create: `tests/data/fixtures/test.star`

**Step 1: Create test fixture**

`tests/data/fixtures/test.star`:
```text
data_

loop_
_rlnMicrographName
_rlnCoordinateX
_rlnCoordinateY
_rlnClassNumber
image_001.tiff 100.5 200.3 1
image_001.tiff 300.2 400.1 2
image_002.tiff 150.0 250.0 1
```

**Step 2: Write failing test**

`tests/data/test_star_parser.py`:
```python
import pytest
from pathlib import Path


def test_parse_star_file():
    from supicker.data.star_parser import parse_star_file

    star_path = Path(__file__).parent / "fixtures" / "test.star"
    result = parse_star_file(star_path)

    assert "image_001.tiff" in result
    assert "image_002.tiff" in result
    assert len(result["image_001.tiff"]) == 2
    assert len(result["image_002.tiff"]) == 1

    particle = result["image_001.tiff"][0]
    assert particle["x"] == 100.5
    assert particle["y"] == 200.3
    assert particle["class_id"] == 0  # 0-indexed


def test_parse_star_cryosparc_format():
    from supicker.data.star_parser import parse_star_file

    # cryoSPARC uses different column names
    star_content = """data_

loop_
_rlnMicrographName
_rlnCoordinateX
_rlnCoordinateY
mic_001.tiff 50.0 60.0
"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".star", delete=False) as f:
        f.write(star_content)
        f.flush()
        result = parse_star_file(f.name)

    assert "mic_001.tiff" in result
    assert result["mic_001.tiff"][0]["class_id"] == 0  # Default class
```

**Step 3: Run test to verify it fails**

```bash
mkdir -p tests/data/fixtures
pytest tests/data/test_star_parser.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 4: Write implementation**

`supicker/data/star_parser.py`:
```python
from pathlib import Path
from typing import Union


def parse_star_file(star_path: Union[str, Path]) -> dict[str, list[dict]]:
    """Parse STAR file and group particles by micrograph.

    Supports both RELION and cryoSPARC formats.

    Args:
        star_path: Path to STAR file

    Returns:
        Dictionary mapping micrograph names to list of particles.
        Each particle has keys: 'x', 'y', 'class_id'
    """
    star_path = Path(star_path)
    particles_by_micrograph: dict[str, list[dict]] = {}

    with open(star_path, "r") as f:
        lines = f.readlines()

    # Find column indices
    column_indices = {}
    in_loop = False
    data_start = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("loop_"):
            in_loop = True
            continue
        if in_loop and line.startswith("_"):
            # Parse column name
            parts = line.split()
            col_name = parts[0]
            col_idx = len(column_indices)
            column_indices[col_name] = col_idx
        elif in_loop and line and not line.startswith("_") and not line.startswith("#"):
            data_start = i
            break

    # Map common column name variants
    mic_col = None
    x_col = None
    y_col = None
    class_col = None

    for name, idx in column_indices.items():
        name_lower = name.lower()
        if "micrograph" in name_lower:
            mic_col = idx
        elif "coordinatex" in name_lower:
            x_col = idx
        elif "coordinatey" in name_lower:
            y_col = idx
        elif "class" in name_lower:
            class_col = idx

    if mic_col is None or x_col is None or y_col is None:
        raise ValueError(f"Missing required columns in STAR file: {star_path}")

    # Parse data lines
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("data_"):
            continue

        parts = line.split()
        if len(parts) < len(column_indices):
            continue

        micrograph = Path(parts[mic_col]).name
        x = float(parts[x_col])
        y = float(parts[y_col])
        class_id = int(parts[class_col]) - 1 if class_col is not None else 0

        if micrograph not in particles_by_micrograph:
            particles_by_micrograph[micrograph] = []

        particles_by_micrograph[micrograph].append({
            "x": x,
            "y": y,
            "class_id": class_id,
        })

    return particles_by_micrograph


def write_star_file(
    particles: list[dict],
    output_path: Union[str, Path],
    micrograph_name: str = "micrograph.tiff",
) -> None:
    """Write particles to STAR file format.

    Args:
        particles: List of particle dicts with 'x', 'y', 'class_id', 'score', 'width', 'height'
        output_path: Output file path
        micrograph_name: Name of the micrograph
    """
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write("data_\n\n")
        f.write("loop_\n")
        f.write("_rlnMicrographName\n")
        f.write("_rlnCoordinateX\n")
        f.write("_rlnCoordinateY\n")
        f.write("_rlnClassNumber\n")
        f.write("_rlnAutopickFigureOfMerit\n")
        f.write("_rlnParticleBoxSize\n")

        for p in particles:
            mic = p.get("micrograph", micrograph_name)
            x = p["x"]
            y = p["y"]
            cls = p.get("class_id", 0) + 1  # 1-indexed in STAR
            score = p.get("score", 1.0)
            box_size = max(p.get("width", 100), p.get("height", 100))
            f.write(f"{mic} {x:.2f} {y:.2f} {cls} {score:.4f} {box_size:.0f}\n")
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/data/test_star_parser.py -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add supicker/data/ tests/data/
git commit -m "feat(data): add STAR file parser and writer"
```

---

## Remaining Tasks Summary

Due to space constraints, the remaining tasks follow the same TDD pattern:

### Phase 4 (continued): Data Processing
- **Task 18**: Target Generator (heatmap, size map, offset map)
- **Task 19**: Data Transforms (augmentations)
- **Task 20**: Dataset Class
- **Task 21**: Data Module Exports

### Phase 5: Utilities
- **Task 22**: Checkpoint Manager
- **Task 23**: Logger (TensorBoard + Console)
- **Task 24**: Export Utils (JSON, CSV output)

### Phase 6: Engine
- **Task 25**: Trainer Class
- **Task 26**: Predictor Class (inference + post-processing)

### Phase 7: Scripts
- **Task 27**: Training Script (scripts/train.py)
- **Task 28**: Prediction Script (scripts/predict.py)

### Phase 8: Integration
- **Task 29**: End-to-End Test
- **Task 30**: Final Cleanup and Documentation

---

Plan complete and saved to `docs/plans/2026-01-28-supicker-implementation.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
