from __future__ import annotations
import argparse
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd  # type: ignore
import yaml  # type: ignore

ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except Exception:
        return {}


def load_results(run_dir: Path) -> Dict:
    csv_path = run_dir / 'results.csv'
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {}
        df.columns = [c.strip() for c in df.columns]
        return df.iloc[-1].to_dict()
    except Exception:
        return {}


def load_args(run_dir: Path) -> Dict:
    return read_yaml(run_dir / 'args.yaml')


def short_hash(onnx_path: Path) -> str:
    if not onnx_path.exists():
        return 'N/A'
    try:
        return hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:16]
    except Exception:
        return 'N/A'


def read_cached_hash_file() -> str:
    f = ROOT / 'model' / 'onnx_sha256.txt'
    if f.exists():
        line = f.read_text(encoding='utf-8').strip()
        return line[:16]
    return 'N/A'


def dataset_counts(data_yaml: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    data_cfg = read_yaml(Path(data_yaml))
    if not data_cfg:
        return counts
    for split in ('train', 'val', 'test', 'valid'):
        rel = data_cfg.get(split)
        if not rel:
            continue
        p = (ROOT / rel).resolve()
        imgs = list(p.glob('*.jpg')) + list(p.glob('*.png'))
        if imgs:
            if split == 'valid':
                split = 'val'
            counts[split] = len(imgs)
    return counts


def extract_runtime(metrics: Dict) -> str:
    preferred = [
        'speed/inference(ms)', 'metrics/latency(ms)', 'metrics/latency', 'speed/mean_ms',
        'inference_speed', 'speed'
    ]
    for k in preferred:
        if k in metrics:
            try:
                return f"{float(metrics[k]):.2f} ms/image"
            except Exception:
                pass
    # heuristic scan
    for k, v in metrics.items():
        lk = k.lower()
        if any(t in lk for t in ('speed', 'latency')):
            try:
                return f"{float(v):.2f} ms/image"
            except Exception:
                continue
    return 'N/A'


def version_block() -> str:
    try:
        import ultralytics, torch, onnxruntime, numpy
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
        return (
            f"ultralytics={ultralytics.__version__} torch={torch.__version__} "
            f"onnxruntime={onnxruntime.__version__} numpy={numpy.__version__} gpu={gpu}"
        )
    except Exception:
        return 'versions=N/A'


def build_markdown(run_name: str, metrics: Dict, args: Dict, onnx_h: str) -> str:
    # Core metrics
    m_mask50 = metrics.get('metrics/mAP50(M)', metrics.get('metrics/mAP50(Mask)', 'N/A'))
    m_mask5095 = metrics.get('metrics/mAP50-95(M)', metrics.get('metrics/mAP50-95(Mask)', 'N/A'))
    m_box5095 = metrics.get('metrics/mAP50-95(B)', metrics.get('metrics/mAP50-95(Box)', 'N/A'))
    m_prec = metrics.get('metrics/precision(M)', 'N/A')
    m_rec = metrics.get('metrics/recall(M)', 'N/A')
    epochs_done = (int(metrics.get('epoch', -1)) + 1) if 'epoch' in metrics else 'N/A'
    model_cfg = (args.get('model') or '').lower()
    if 'yolo11' in model_cfg:
        model_desc = 'YOLOv11 Medium Segmentation (yolo11m-seg.pt)'
    elif 'yolo12' in model_cfg:
        model_desc = 'YOLOv12 Medium Segmentation'
    else:
        model_desc = args.get('model', 'YOLO Segmentation')
    aug_line = []
    for k in ('mixup', 'erasing', 'close_mosaic'):
        if k in args:
            aug_line.append(f"{k}={args.get(k)}")
    aug_line = ', '.join(aug_line) if aug_line else 'standard (mosaic,hsv,scale,flip)'
    counts = dataset_counts(args.get('data', 'data/roboflow/data.yaml'))
    counts_line = ', '.join(f"{k}={v}" for k, v in counts.items()) if counts else 'N/A'
    runtime_line = extract_runtime(metrics)
    seed = args.get('seed', 'N/A')
    versions = version_block()

    md = f"""# Methods Note: Handsy Phase-0 Evaluation\n\n"""
    md += f"**Run Name:** `{run_name}`  \n"
    md += f"**ONNX Hash (first 16 SHA256):** `{onnx_h}`  \n"
    md += f"**Versions:** {versions}  \n"
    md += f"**Seed:** {seed}  \n"
    md += f"**Average Inference Runtime:** {runtime_line}\n\n"

    md += "## 1. Pipeline Overview\n"\
          "The pipeline performs training, dual-split evaluation (val & test), ONNX export, and artifact generation (masks, overlays, measurements, methods note).\n\n"

    md += "## 2. Model & Training\n"
    md += f"Model: **{model_desc}** fine-tuned on a curated single-class nail segmentation dataset.\n\n"
    md += "Key settings:\n"
    md += f"- Image size: {args.get('imgsz')} px\n"
    md += f"- Batch: {args.get('batch')} (auto-fit if -1)\n"
    md += f"- Epochs completed: {epochs_done} (early stopping patience {args.get('patience')})\n"
    md += f"- Optimizer: {args.get('optimizer')} (cos_lr={args.get('cos_lr')})\n"
    md += f"- close_mosaic: {args.get('close_mosaic')}\n"
    md += f"- Augmentations: {aug_line}\n"
    md += f"- cache: {args.get('cache')} workers: {args.get('workers')}\n\n"

    md += "## 3. Dataset\n"
    md += f"Source config: `{args.get('data', 'data/roboflow/data.yaml')}`  \n"
    md += f"Counts: {counts_line}\n\n"

    md += "### 3.1 Validation Metrics (val split, retina_masks=True)\n"
    md += "| Metric | Value |\n|--------|-------|\n"
    md += f"| Mask mAP@50 | {m_mask50} |\n"
    md += f"| Mask mAP@50-95 | {m_mask5095} |\n"
    md += f"| Box mAP@50-95 | {m_box5095} |\n"
    md += f"| Mask Precision | {m_prec} |\n"
    md += f"| Mask Recall | {m_rec} |\n\n"

    md += "### 3.2 Test Split Metrics\n"
    md += ("Run executed with `split=test` separately; metrics captured in console output. (Future enhancement: parse and persist test metrics.)\n\n")

    md += "## 4. Measurement & Scaling\n"
    md += ("A hierarchical scaling strategy converts pixel distances to millimeters with provenance per nail:\n\n")
    md += ("1. **ArUco Marker (Primary)** — Detect 4×4 DICT_4X4_50 20 mm marker; if found, derive precise mm/px (confidence=1.0).\n")
    md += ("2. **ID-1 Card Fallback** — If no marker, attempt OWL-ViT open-vocabulary detection of a credit/debit (ID‑1) card (85.60 mm long side). (confidence≈0.7).\n")
    md += ("3. **Heuristic Default** — Fallback constant (0.25 mm/px) flagged with low confidence (0.2).\n\n")
    md += ("For each retained nail contour: \n")
    md += ("- Largest up to four external contours kept after morphological close.\n")
    md += ("- PCA major axis → length_mm.\n")
    md += ("- Three orthogonal sectional widths sampled at 25%, 50%, 75% of axis length → width_prox_mm, width_mid_mm, width_dist_mm.\n")
    md += ("- Each row in `measurements.csv` includes: image, nail_index, length_mm, width_prox_mm, width_mid_mm, width_dist_mm, mask_area_px, mm_per_px, scale_confidence, scale_method.\n")
    md += ("- Provenance captured via `scale_method` (aruco | id1 | heuristic).\n\n")

    md += "## 5. Inference & Post-Processing\n"
    md += ("1. Load ONNX (CUDAExecutionProvider preferred with CPU fallback).\n"
            "2. Forward pass to obtain raw mask tensor.\n"
            "3. Threshold + contour extraction; keep top 4 nails.\n"
            "4. Morphological smoothing (3×3 close).\n"
            "5. Scaling hierarchy application.\n"
            "6. Metric computation & overlay rendering (retina masks only at eval time).\n\n")

    md += "## 6. Outputs\n"
    md += ("- `*_mask.png` cleaned binary masks\n"
            "- `*_overlay.png` overlays with length + widths (mm)\n"
            "- `measurements.csv` per-nail metrics + scaling provenance\n"
            "- `methods_note.md` (this document)\n\n")

    md += "## 7. Reproducibility\n"
    md += ("Training & evaluation key environment: TF32 enabled, mixed precision (AMP), seed captured. "
            "ONNX exported with dynamic axes + simplification (opset>=14).\n")
    md += ("Example commands:\n\n```\n"
            f"yolo segment train model={args.get('model')} data={args.get('data')} epochs={args.get('epochs')} imgsz={args.get('imgsz')}\n"
            f"yolo segment val model=runs/segment/{run_name}/weights/best.pt data={args.get('data')} retina_masks=True\n"
            f"yolo export model=runs/segment/{run_name}/weights/best.pt format=onnx dynamic=True simplify=True opset=14\n"
            "```\n\n")

    md += "## 8. Notes\n"
    md += ("- Retina masks used only for evaluation overlays.\n"
            "- Deterministic=False for throughput; enable if exact reproducibility required.\n"
            "- Further test metrics parsing can be added if persisted separately.\n")
    return md.strip() + '\n'


def resolve_run_name(passed: str | None) -> str:
    if passed:
        return passed
    seg_root = ROOT / 'runs' / 'segment'
    if not seg_root.exists():
        raise SystemExit('No runs/segment directory found')
    candidates = [p for p in seg_root.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit('No run directories under runs/segment')
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name


def main():
    ap = argparse.ArgumentParser(description='Generate detailed methods note with scaling & measurement section')
    ap.add_argument('--run-name', required=False, help='YOLO run folder name under runs/segment')
    ap.add_argument('--handsy-out', default='handsy_out', help='Output directory')
    args = ap.parse_args()

    run_name = resolve_run_name(args.run_name)
    run_dir = ROOT / 'runs' / 'segment' / run_name
    if not run_dir.exists():
        raise SystemExit(f'Run directory not found: {run_dir}')

    metrics = load_results(run_dir)
    train_args = load_args(run_dir)
    onnx_h = read_cached_hash_file()
    if onnx_h == 'N/A':  # fallback direct hash if cache absent
        onnx_h = short_hash(ROOT / 'model' / 'best.onnx')
    md = build_markdown(run_name, metrics, train_args, onnx_h)

    out_dir = ROOT / args.handsy_out
    out_dir.mkdir(parents=True, exist_ok=True)
    note_path = out_dir / 'methods_note.md'
    note_path.write_text(md, encoding='utf-8')
    print(f'✅ Methods note written: {note_path}')


if __name__ == '__main__':
    main()

