#!/usr/bin/env python3
"""
🎯 PRAGMATIC Nail Segmentation Fine-tuning - No Bullshit Edition
================================================================

Fast, efficient fine-tuning for nail segmentation that's already good at 68.8%.
We're fine-tuning, not solving rocket science!

Features:
- Pragmatic epoch counts (10-20 epochs max)
- Automatic batch sizing with batch=-1
- Fast training focused on improvement, not perfection
- No over-engineering

Author: AI Assistant (Pragmatic Mode)
Date: 2025-01-13
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any

import torch
import psutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def check_dependencies():
    """Check YOLO dependency."""
    try:
        import ultralytics
        print("✓ Ultralytics YOLO found")
        return True
    except ImportError:
        print("✗ Ultralytics YOLO not found. Install with: pip install ultralytics")
        return False

def get_system_resources() -> Dict[str, Any]:
    """Quick system check."""
    resources = {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'gpu_count': 0,
        'gpu_memory_gb': 0,
    }

    if torch.cuda.is_available():
        resources['gpu_count'] = torch.cuda.device_count()
        for i in range(resources['gpu_count']):
            props = torch.cuda.get_device_properties(i)
            resources['gpu_memory_gb'] = max(resources['gpu_memory_gb'], props.total_memory / (1024**3))

    return resources

def pragmatic_finetune(
    model_path: str,
    data_yaml: str,
    epochs: int = 15,  # DEFAULT: Pragmatic 15 epochs
    device: str = "0",
    output_dir: str = "runs/pragmatic"
) -> str:
    """
    PRAGMATIC fine-tuning: Fast, efficient, no bullshit.
    Model is already good - we just need small improvements.
    """
    from ultralytics import YOLO

    print("⚡ PRAGMATIC Nail Fine-tuning - Fast & Efficient")
    print("=" * 50)

    resources = get_system_resources()
    print(f"💻 GPU: {resources['gpu_memory_gb']:.1f}GB | CPU: {resources['cpu_count']} cores")

    # Load existing good model
    model = YOLO(model_path)
    print(f"🎯 Fine-tuning from baseline: 68.8% → Target: >75%")

    # PRAGMATIC configuration - focused on quick improvement
    training_args = {
        # Core settings
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': 640,
        'device': device,
        'batch': -1,          # Auto batch sizing
        'workers': 4,         # Conservative workers for stability

        # Output
        'project': output_dir,
        'name': 'nail_pragmatic',
        'exist_ok': True,
        'save_period': max(5, epochs // 3),  # Save checkpoints frequently

        # Performance (keep it simple)
        'amp': True,          # Mixed precision for speed
        'cache': False,       # Don't cache large dataset
        'multi_scale': True,  # Good for generalization

        # Optimizer (let YOLO decide)
        'optimizer': 'auto',
        'close_mosaic': max(5, epochs // 3),

        # Minimal augmentation (model is already good)
        'hsv_h': 0.01,        # Light hue variation
        'hsv_s': 0.4,         # Moderate saturation
        'hsv_v': 0.3,         # Light value changes
        'degrees': 5.0,       # Small rotations
        'translate': 0.05,    # Minimal translation
        'scale': 0.3,         # Conservative scaling
        'shear': 1.0,         # Light shear
        'perspective': 0.0001,# Minimal perspective
        'flipud': 0.0,        # No vertical flip (nails are oriented)
        'fliplr': 0.3,        # Light horizontal flip
        'mosaic': 0.7,        # Moderate mosaic
        'mixup': 0.05,        # Light mixup
        'copy_paste': 0.1,    # Light copy-paste
        'cutmix': 0.0,        # Skip cutmix for speed

        # Learning rate (conservative for fine-tuning)
        'lr0': 0.005,         # Lower initial LR for fine-tuning
        'lrf': 0.1,           # Conservative final LR
        'momentum': 0.9,      # Standard momentum
        'weight_decay': 0.0003, # Light regularization
        'warmup_epochs': 1.0, # Minimal warmup
        'cos_lr': True,       # Cosine scheduling

        # Early stopping (pragmatic)
        'patience': min(10, epochs // 2),  # Stop if no improvement
        'val': True,
        'plots': True,
        'save': True,
        'verbose': True,
    }

    print(f"\n📦 PRAGMATIC Configuration:")
    print(f"  Epochs: {epochs} (fine-tuning, not training from scratch!)")
    print(f"  Batch: -1 (auto-detected)")
    print(f"  Learning Rate: {training_args['lr0']} (conservative for fine-tuning)")
    print(f"  Patience: {training_args['patience']} (stop if no improvement)")
    print(f"  Augmentation: Minimal (model already knows nails)")

    print(f"\n🚀 Starting PRAGMATIC fine-tuning...")
    print(f"📁 Output: {output_dir}/nail_pragmatic")

    # Start training
    results = model.train(**training_args)

    print(f"✅ PRAGMATIC fine-tuning completed!")
    print(f"📊 Best model: {output_dir}/nail_pragmatic/weights/best.pt")

    # Return path to best model
    best_model_path = Path(output_dir) / "nail_pragmatic" / "weights" / "best.pt"
    return str(best_model_path)

def quick_test(
    model_path: str,
    data_yaml: str,
    epochs: int = 5,  # SUPER QUICK TEST
    device: str = "0",
    output_dir: str = "runs/quick_test"
) -> str:
    """
    SUPER QUICK test - just 5 epochs to see if improvement direction is right.
    """
    from ultralytics import YOLO

    print("🏃 QUICK TEST - 5 Epochs Sanity Check")
    print("=" * 40)

    model = YOLO(model_path)

    # Minimal configuration for quick test
    training_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': 640,
        'device': device,
        'batch': -1,          # Auto batch
        'workers': 2,         # Minimal workers

        'project': output_dir,
        'name': 'nail_quick_test',
        'exist_ok': True,
        'save_period': epochs,  # Save only at end

        'amp': True,
        'cache': False,

        # Minimal augmentation
        'hsv_h': 0.005,
        'hsv_s': 0.2,
        'hsv_v': 0.2,
        'degrees': 2.0,
        'translate': 0.02,
        'scale': 0.1,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'mosaic': 0.5,

        'lr0': 0.003,
        'lrf': 0.3,
        'warmup_epochs': 0.5,
        'patience': epochs,   # No early stopping for quick test

        'val': True,
        'plots': False,       # Skip plots for speed
        'save': True,
        'verbose': True,
    }

    print(f"⚡ Quick test: {epochs} epochs only")
    print(f"🎯 Goal: Verify improvement direction")

    results = model.train(**training_args)

    print(f"✅ Quick test completed!")

    best_model_path = Path(output_dir) / "nail_quick_test" / "weights" / "best.pt"
    return str(best_model_path)

def main():
    """Main function - PRAGMATIC argument parsing."""
    parser = argparse.ArgumentParser(
        description="🎯 PRAGMATIC Nail Fine-tuning - No Bullshit Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pragmatic fine-tuning (15 epochs - reasonable for improvement)
  python pragmatic_nail_finetune.py --mode pragmatic --epochs 15

  # Quick sanity check (5 epochs - just to see if we're going right direction)
  python pragmatic_nail_finetune.py --mode quick --epochs 5

  # Custom pragmatic (10-20 epochs max recommended)
  python pragmatic_nail_finetune.py --mode pragmatic --epochs 20
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['pragmatic', 'quick'],
        default='pragmatic',
        help='Training mode: pragmatic (15 epochs) or quick (5 epochs test)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (default: 15 pragmatic, 5 quick)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='GPU device'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='models/production/nailseg-best.pt',
        help='Base model path'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='datasets/nail_Segmentation.v3i.yolov11/data.yaml',
        help='Dataset YAML'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='runs/pragmatic',
        help='Output directory'
    )

    args = parser.parse_args()

    # Set default epochs based on mode
    if args.epochs is None:
        args.epochs = 15 if args.mode == 'pragmatic' else 5

    # Validate epoch counts
    if args.mode == 'pragmatic' and args.epochs > 25:
        print(f"⚠️  WARNING: {args.epochs} epochs is excessive for fine-tuning!")
        print("   Recommended: 10-20 epochs max for pragmatic fine-tuning")
        print("   Model is already good at 68.8% - we're improving, not starting over!")

    # Print header
    print("🎯 PRAGMATIC Nail Segmentation Fine-tuning")
    print("📊 Baseline: 68.8% mask mAP50-95 (already good!)")
    print("🎯 Target: >75% mask mAP50-95 (modest improvement)")
    print(f"📁 Dataset: {args.data}")
    print(f"🤖 Model: {args.model}")
    print(f"📂 Output: {args.output}")
    print(f"🖥️  Device: {args.device}")
    print(f"⏰ Mode: {args.mode.upper()} ({args.epochs} epochs)")
    print()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Verify files
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)

    if not Path(args.data).exists():
        print(f"❌ Dataset not found: {args.data}")
        sys.exit(1)

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Execute training
    try:
        if args.mode == 'pragmatic':
            result_path = pragmatic_finetune(
                args.model, args.data, args.epochs, args.device, args.output
            )
        elif args.mode == 'quick':
            result_path = quick_test(
                args.model, args.data, args.epochs, args.device, args.output
            )

        print(f"\n🏆 PRAGMATIC training completed!")
        print(f"📁 Best model: {result_path}")
        print(f"🔍 Results in: {args.output}")

    except KeyboardInterrupt:
        print("\n⚠️  Training stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()