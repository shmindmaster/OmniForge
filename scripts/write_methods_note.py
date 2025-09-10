from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import pandas as pd  # type: ignore
import yaml  # type: ignore

ROOT = Path(__file__).resolve().parents[1]


def safe_get(d, key, default=0.0):
    try:
        v = d.get(key, default)
        if v in (None, ""):
            return default
        return float(v)
    except Exception:
        return default


def load_final_metrics(run_dir: Path) -> dict:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {}
    df = pd.read_csv(results_csv)
    if df.empty:
        return {}
    row = df.iloc[-1].to_dict()
    return row


def load_args(run_dir: Path) -> dict:
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.exists():
        return {}
    try:
        return yaml.safe_load(args_yaml.read_text()) or {}
    except Exception:
        return {}


def onnx_hash(model_path: Path) -> str:
    if not model_path.exists():
        return "N/A"
    h = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return h[:16]


def render_markdown(run_name: str, metrics: dict, args: dict, onnx_h: str) -> str:
    mask_map50 = metrics.get('metrics/mAP50(M)') or metrics.get('metrics/mAP50(Mask)') or metrics.get('metrics/mAP50(M)')
    mask_map50_95 = metrics.get('metrics/mAP50-95(M)') or metrics.get('metrics/mAP50-95(Mask)')
    box_map50_95 = metrics.get('metrics/mAP50-95(B)') or metrics.get('metrics/mAP50-95(Box)')
    prec_m = metrics.get('metrics/precision(M)')
    rec_m = metrics.get('metrics/recall(M)')
    epochs_done = int(metrics.get('epoch', -1)) + 1 if 'epoch' in metrics else 'N/A'

    return f"""# Methods Note: Handsy Phase‑0

**Run Name:** `{run_name}`
**ONNX Hash (first 16 SHA256):** `{onnx_h}`

## 1. Objective
Segment fingernails and compute per‑nail physical metrics (length and sectional widths) with reproducible pixel→mm scaling.

## 2. Model & Training
Model: **YOLOv11 Medium Segmentation** (`yolov11m-seg.pt`) fine‑tuned on a Roboflow nail dataset.

Key settings:
- Image size: {args.get('imgsz')} px
- Batch: {args.get('batch')} (auto-fit if -1)
- Epochs completed: {epochs_done}
- Optimizer: {args.get('optimizer')}
- Cosine LR: {args.get('cos_lr')}
- close_mosaic: {args.get('close_mosaic')}
- Augmentations: mosaic, hsv, scale, flip, random erasing

### Final Validation Metrics
| Metric | Value |
|--------|-------|
| Mask mAP@50 | {mask_map50} |
| Mask mAP@50-95 | {mask_map50_95} |
| Box mAP@50-95 | {box_map50_95} |
| Mask Precision | {prec_m} |
| Mask Recall | {rec_m} |

## 3. Inference Pipeline
1. Load ONNX (CUDA → CPU fallback).
2. Generate raw segmentation mask.
3. Clean mask: keep top 4 external contours, morphological close (3×3).
4. Scaling hierarchy:
   - ArUco 4x4 (20 mm marker) → confidence 1.0
   - ID‑1 card fallback (credit/debit) → confidence 0.7
   - Default heuristic (0.25 mm/px) → confidence 0.2
5. Per‑contour PCA axis length + widths at 25%, 50%, 75% along axis.
6. Export per‑nail rows to `measurements.csv` (one row per nail).

## 4. Outputs
- `*_mask.png` cleaned binary masks
- `*_overlay.png` visual overlay with metrics
- `measurements.csv` aggregated measurements (length & sectional widths)
- `methods_note.md` (this file)

## 5. Reproducibility
Commands (GNU Make targets):
```
make train
make val
make export-onnx
make evaluate RUN_NAME={run_name}
```

## 6. Notes
- Mixed precision (AMP) enabled.
- Deterministic disabled for throughput; set `deterministic=True` if exact repeatability required.
- ONNX exported with dynamic shapes + simplification.
""".strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description="Generate enhanced methods note from a YOLO run")
    ap.add_argument('--run-name', help='YOLO run folder under runs/segment', default=None)
    ap.add_argument('--handsy-out', help='Output directory for note', default='handsy_out')
    args = ap.parse_args()

    segment_root = ROOT / 'runs' / 'segment'
    if not args.run_name:
        # pick most recent
        candidates = [p for p in segment_root.iterdir() if p.is_dir()]
        if not candidates:
            raise SystemExit('No runs found under runs/segment; provide --run-name')
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        run_name = candidates[0].name
    else:
        run_name = args.run_name

    run_dir = segment_root / run_name
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    metrics = load_final_metrics(run_dir)
    train_args = load_args(run_dir)
    onnx_h = onnx_hash(ROOT / 'model' / 'best.onnx')
    md = render_markdown(run_name, metrics, train_args, onnx_h)

    out_dir = ROOT / args.handsy_out
    out_dir.mkdir(parents=True, exist_ok=True)
    note_path = out_dir / 'methods_note.md'
    note_path.write_text(md, encoding='utf-8')
    print(f"✅ Methods note written: {note_path}")


if __name__ == '__main__':
    main()

