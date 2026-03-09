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
    val_score_threshold: float = 0.1
    val_distance_threshold: float = 20.0
    val_nms_radius: float = 20.0
    loss: LossConfig = field(default_factory=LossConfig)
    # Distributed training settings
    distributed: bool = False
    world_size: int = 1
    local_rank: int = 0
    dist_backend: str = "nccl"  # nccl (GPU) or gloo (CPU)
    sync_bn: bool = True  # Convert BatchNorm to SyncBatchNorm
    use_amp: bool = True  # Use automatic mixed precision (FP16) to save memory
