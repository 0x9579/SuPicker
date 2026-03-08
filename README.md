# SuPicker

A deep learning framework for particle picking in Cryo-EM micrographs using CenterNet-style detection.

## Features

- **ConvNeXt Backbone**: Modern vision transformer-inspired architecture with support for Tiny, Small, and Base variants
- **Feature Pyramid Network**: Multi-scale feature extraction for detecting particles of varying sizes
- **CenterNet Detection Head**: Anchor-free detection with heatmap, size, and offset predictions
- **Flexible Data Pipeline**: Support for TIFF, MRC, and standard image formats with STAR file annotations
- **RandomCrop Training**: Automatic random cropping for training on large micrographs without OOM
- **AMP (Mixed Precision)**: Automatic mixed precision training for 30-50% memory savings and faster training
- **Configurable Training**: Modular configuration system for all training hyperparameters
- **Distributed Training**: Multi-GPU support via PyTorch DistributedDataParallel
- **Pretrained Weights**: Optional ImageNet pretrained backbone weights
- **Evaluation Metrics**: Precision, Recall, F1, and Average Precision computation
- **Training Metrics Logging**: Real-time display of precision and recall in training logs
- **Auto-save on Interrupt**: Safely save training progress when gracefully exiting (Ctrl+C)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/supicker.git
cd supicker

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with MRC file support
pip install -e ".[mrc]"
```

## Quick Start

### Training

```bash
# Basic training
python scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --backbone tiny \
    --epochs 100 \
    --batch-size 8

# Training on a specific GPU
python scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --backbone tiny \
    --epochs 100 \
    --batch-size 8 \
    --device cuda:7

# Training with validation
python scripts/train.py \
    --train-images ./data/train \
    --train-star ./data/train.star \
    --val-images ./data/val \
    --val-star ./data/val.star \
    --pretrained

# Multi-GPU training (4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --distributed

# Specific GPUs only
CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 scripts/train.py \
    --train-images ./data/micrographs \
    --train-star ./data/particles.star \
    --distributed
```

### Inference

```bash
# Run prediction on micrographs
python scripts/predict.py \
    --checkpoint ./checkpoints/best.pth \
    --input ./data/test \
    --output ./results \
    --format star
```

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backbone` | `tiny` | ConvNeXt variant: tiny, small, base |
| `--batch-size` | `8` | Training batch size |
| `--val-batch-size` | `2` | Validation batch size (smaller to save memory) |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--optimizer` | `adamw` | Optimizer: adam, adamw, sgd |
| `--scheduler` | `cosine` | LR scheduler: cosine, step, none |
| `--weight-decay` | `0.01` | Weight decay for regularization |
| `--warmup-epochs` | `5` | Linear warmup epochs |
| `--pretrained` | `False` | Use ImageNet pretrained weights |
| `--device` | `cuda` | Device to use (e.g. `cuda:0`, `cuda:7`, `cpu`) |
| `--distributed` | `False` | Enable multi-GPU training (use with `torchrun`) |
| `--resume` | `None` | Path to checkpoint file to resume training from |
| `--no-amp` | `False` | Disable automatic mixed precision |
| `--no-augmentation` | `False` | Disable data augmentation |

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--score-threshold` | `0.3` | Minimum detection score |
| `--nms-radius` | `10` | NMS suppression radius (pixels) |
| `--format` | `star` | Output format: star, json, csv |

## Python API

### Training

```python
from supicker.config import (
    ModelConfig, BackboneConfig, ConvNeXtVariant,
    TrainingConfig, AugmentationConfig
)
from supicker.models import Detector
from supicker.data import create_dataloader, build_transforms
from supicker.engine import Trainer

# Configure model
model_config = ModelConfig(
    backbone=BackboneConfig(
        variant=ConvNeXtVariant.TINY,
        pretrained=True,
        in_channels=1,
    )
)

# Create model
model = Detector(model_config)

# Create data loaders
train_loader = create_dataloader(
    image_dir="./data/train",
    star_file="./data/train.star",
    batch_size=8,
    transforms=build_transforms(AugmentationConfig()),
)

# Configure and run training
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

### Inference

```python
from supicker.config import ModelConfig, InferenceConfig
from supicker.models import Detector
from supicker.engine import Predictor
from supicker.utils import export_to_star
import tifffile
import torch

# Load model
model = Detector(ModelConfig())
predictor = Predictor.from_checkpoint(
    checkpoint_path="./checkpoints/best.pth",
    model=model,
    config=InferenceConfig(score_threshold=0.3),
)

# Load and preprocess image
image = tifffile.imread("micrograph.tiff").astype("float32")
image = (image - image.min()) / (image.max() - image.min() + 1e-8)
image = torch.from_numpy(image).unsqueeze(0)  # Add channel dim

# Run prediction
particles = predictor.predict(image)

# Export results
export_to_star(particles, "output.star", micrograph_name="micrograph.tiff")
```

### Evaluation

```python
from supicker.utils.metrics import compute_detection_metrics, MetricAggregator

# Single image evaluation
metrics = compute_detection_metrics(
    predictions=detected_particles,
    ground_truth=true_particles,
    distance_threshold=10.0,
)
print(f"Precision: {metrics.precision:.3f}")
print(f"Recall: {metrics.recall:.3f}")
print(f"F1 Score: {metrics.f1_score:.3f}")

# Aggregate metrics across dataset
aggregator = MetricAggregator(distance_threshold=10.0)
for preds, gts in zip(all_predictions, all_ground_truths):
    aggregator.add_image(preds, gts)
aggregate_metrics = aggregator.compute_aggregate()
```

## Output Formats

### STAR Format

Standard RELION-compatible STAR file:

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

### JSON Format

```json
{
  "micrograph_001.tiff": [
    {"x": 100.5, "y": 200.3, "score": 0.95, "width": 64, "height": 64},
    {"x": 150.2, "y": 300.1, "score": 0.88, "width": 64, "height": 64}
  ]
}
```

### CSV Format

```csv
micrograph,x,y,score,width,height
micrograph_001.tiff,100.5,200.3,0.95,64,64
micrograph_001.tiff,150.2,300.1,0.88,64,64
```

## Data Tools

SuPicker includes a STAR file tool for data inspection and splitting:

```bash
# Show STAR file statistics
python scripts/star_tool.py info particles.star

# List all micrographs
python scripts/star_tool.py info particles.star --list

# Extract first/last N images
python scripts/star_tool.py split particles.star -n 10 -o subset.star
python scripts/star_tool.py split particles.star -n 50 --from-end -o val.star

# Split into train/val sets (with optional shuffle)
python scripts/star_tool.py split-trainval particles.star \
    --val-images 50 \
    --train-output train.star --val-output val.star \
    --shuffle --seed 42
```

## Project Structure

```
supicker/
├── config/          # Configuration dataclasses
├── data/            # Dataset and data loading
├── engine/          # Training and inference
├── losses/          # Loss functions
├── models/          # Model architectures
│   ├── backbone/    # ConvNeXt backbone
│   ├── fpn/         # Feature Pyramid Network
│   └── head/        # CenterNet detection head
└── utils/           # Utilities (logging, export, metrics)
scripts/
├── train.py         # Training script
├── predict.py       # Inference script
└── star_tool.py     # STAR file inspection & splitting
```

## Training Tips

- **Batch size**: Use 8-20 for stable training. Larger batches smooth gradients and reduce loss oscillation.
- **Learning rate**: Scale with batch size. `1e-4` for batch_size=8, `2e-4` for batch_size=16+.
- **Crop size**: Default 1024×1024. Images are randomly cropped during training to fit in GPU memory. Adjust via `AugmentationConfig.crop_size`.
- **AMP**: Enabled by default. Saves 30-50% GPU memory and speeds up training ~1.5x. Disable with `--no-amp` if you encounter numerical issues.
- **GPU selection**: Use `--device cuda:N` to select a specific GPU, or `CUDA_VISIBLE_DEVICES=N` environment variable.
- **Multi-GPU**: Always use `torchrun` with `--distributed` flag. Do not combine `--device` with `--distributed`.
- **Resuming Training**: Use `--resume ./checkpoints/...` to continue training from a checkpoint. If you press `Ctrl+C` during training, SuPicker will automatically save the latest state so you don't lose progress.
- **Validation Metrics**: When using a validation set, you will see `P` (Precision), `R` (Recall), and `F1` (F1 Score) in the console alongside the loss values.

## License

MIT License
