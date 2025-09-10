# YOLOv11 Segmentation Training Guide for im2fit

This guide provides step-by-step instructions for training a YOLOv11 segmentation model for nail detection and measurement in the im2fit pipeline.

## Prerequisites

### Hardware Requirements
- **Recommended**: NVIDIA RTX 3060 or better with 8GB+ VRAM
- **Minimum**: 16GB RAM, 50GB free disk space
- **CPU**: Multi-core processor for data loading

### Software Requirements
- Python 3.9-3.11
- CUDA 11.8+ (for GPU training)
- All dependencies from `app/requirements.txt`

### Dataset Requirements
- Roboflow dataset in YOLO segmentation format
- Images with nail objects
- Binary masks converted to polygon annotations
- Minimum 500 labeled images (recommended 1000+)

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r app/requirements.txt

# Verify CUDA availability (for GPU training)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Dataset Preparation

#### Option A: Use Existing Roboflow Dataset
```bash
# Download dataset from Roboflow in YOLO format
# Extract to data/roboflow/
# Verify data.yaml paths are correct
```

#### Option B: Convert Custom Dataset
```bash
# Convert binary masks to YOLO format
python tools/to_yolo_seg.py \
  --raw-dir path/to/your/raw/data \
  --out-dir data/roboflow \
  --class-name nail
```

### 3. Training Configuration

Edit `ultralytics.yaml` or use command-line parameters:

```yaml
# Key parameters for nail segmentation
model: yolo11m-seg.pt
data: data/roboflow/data.yaml
epochs: 40
batch: -1  # Auto-detect
imgsz: 640
device: 0  # GPU 0
patience: 10
```

### 4. Start Training

#### Using the Training Script
```bash
python scripts/train_yolo_seg.py \
  --data data/roboflow/data.yaml \
  --model yolo11m-seg.pt \
  --epochs 40 \
  --patience 10 \
  --device 0
```

#### Using Ultralytics CLI
```bash
yolo segment train \
  model=yolo11m-seg.pt \
  data=data/roboflow/data.yaml \
  epochs=40 \
  imgsz=640 \
  batch=-1 \
  device=0 \
  project=runs/segment \
  name=im2fit_nail_seg
```

#### Using Makefile (Windows)
```bash
make train
```

## Advanced Training Options

### Two-Phase Training (Recommended)

The training script implements a two-phase approach:

1. **Phase 1 - Generalization** (832px, heavy augmentation)
   - Build robust feature representations
   - Heavy data augmentation (mosaic, mixup)
   - 80 epochs with early stopping

2. **Phase 2 - Fine-tuning** (1024px, light augmentation)
   - Improve boundary precision
   - Reduced learning rate
   - Minimal augmentation for crisp edges
   - 30 epochs

### Model Escalation

Automatic escalation to larger models if performance targets aren't met:

```bash
# Disable escalation
python scripts/train_yolo_seg.py --auto_escalate False

# Adjust escalation threshold
python scripts/train_yolo_seg.py --escalate_margin 0.05
```

### Hyperparameter Tuning

Key parameters for nail segmentation:

```yaml
# Loss weights (increase seg for better boundaries)
seg: 12.0  # Segmentation loss weight
box: 7.5   # Bounding box loss weight
cls: 0.5   # Classification loss weight

# Augmentation (tuned for nail images)
mosaic: 0.7      # Good for small objects
mixup: 0.10      # Conservative mixing
fliplr: 0.5      # Horizontal flip OK
flipud: 0.0      # No vertical flip for nails
erasing: 0.20    # Random erasing
```

## Monitoring Training

### Real-time Monitoring

Training progress is logged to:
- `runs/segment/im2fit_final_seg/`
- TensorBoard logs (if available)
- Console output

### Key Metrics to Watch

1. **mAP50 (Segmentation)**: Primary metric for model selection
   - Target: >0.90 for production use
   - >0.85 for demo/development

2. **Loss Components**:
   - `train/seg_loss`: Segmentation loss (should decrease steadily)
   - `val/seg_loss`: Validation segmentation loss
   - `train/box_loss`: Bounding box regression loss

3. **Learning Rate**: Should follow cosine schedule

### Early Stopping

Training automatically stops if validation mAP50 doesn't improve for `patience` epochs (default: 10).

## Model Export

### Automatic ONNX Export

The training script automatically exports to ONNX:

```bash
# ONNX model saved to: model/best.onnx
# Validation report: model/onnx_validation_report.json
```

### Manual Export

```bash
# Export best model to ONNX
python scripts/export_best_onnx.py \
  --weights runs/segment/im2fit_final_seg/weights/best.pt \
  --output model/best.onnx
```

### Validation

```bash
# Validate ONNX model
python scripts/validate_onnx.py --model model/best.onnx
```

## Evaluation and Testing

### Validation on Test Set

```bash
# Evaluate on validation set
yolo segment val \
  model=runs/segment/im2fit_final_seg/weights/best.pt \
  data=data/roboflow/data.yaml \
  split=val

# Evaluate on test set  
yolo segment val \
  model=runs/segment/im2fit_final_seg/weights/best.pt \
  data=data/roboflow/data.yaml \
  split=test
```

### End-to-End Pipeline Testing

```bash
# Test complete pipeline with trained model
python demo_local.py sample_image.png
```

### Generate Client Deliverables

```bash
# Generate evaluation report for handsy_out/
python scripts/run_handsy_evaluation.py
python scripts/write_methods_note.py --run-name im2fit_final_seg
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```bash
# Reduce batch size
python scripts/train_yolo_seg.py --batch 8

# Use smaller image size
python scripts/train_yolo_seg.py --imgsz 512

# Disable caching
python scripts/train_yolo_seg.py --cache off
```

#### Poor Convergence
```bash
# Increase learning rate
python scripts/train_yolo_seg.py --lr0 0.005

# Reduce augmentation
python scripts/train_yolo_seg.py --mosaic 0.3 --mixup 0.0

# Increase training time
python scripts/train_yolo_seg.py --epochs 60 --patience 15
```

#### Data Loading Issues
```bash
# Reduce workers
python scripts/train_yolo_seg.py --workers 4

# Check dataset paths in data.yaml
# Verify image/label file correspondence
```

### Performance Optimization

#### For RTX 3060 (8GB VRAM)
```bash
python scripts/train_yolo_seg.py \
  --batch 16 \
  --workers 8 \
  --cache disk \
  --device 0
```

#### For CPU Training
```bash
python scripts/train_yolo_seg.py \
  --batch 4 \
  --workers 2 \
  --device cpu \
  --cache off
```

## Best Practices

### Dataset Quality
- Ensure consistent labeling quality
- Include diverse lighting conditions
- Balance nail shapes and sizes
- Verify mask-to-image alignment

### Training Strategy
- Start with pre-trained weights (yolo11m-seg.pt)
- Use two-phase training for best results
- Monitor both training and validation metrics
- Save checkpoints regularly

### Production Deployment
- Export to ONNX for inference speed
- Validate model on held-out test set
- Document model performance metrics
- Version control trained models

## File Structure

After successful training:

```
runs/segment/im2fit_final_seg/
├── weights/
│   ├── best.pt          # Best model weights
│   ├── last.pt          # Last epoch weights
│   └── best.onnx        # ONNX export
├── train_batch*.jpg     # Training visualizations
├── val_batch*.jpg       # Validation visualizations
├── results.png          # Training curves
├── confusion_matrix.png # Classification matrix
├── PR_curve.png         # Precision-Recall curve
├── F1_curve.png         # F1 score curve
└── args_used.json      # Training arguments
```

## References

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [YOLO Segmentation Tutorial](https://docs.ultralytics.com/tasks/segment/)
- [Roboflow Dataset Format](https://roboflow.com/formats/yolov8-pytorch-txt)