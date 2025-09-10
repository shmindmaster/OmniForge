import os
import csv
import cv2
import json
import glob
import zipfile
import numpy as np
from datetime import datetime

try:
    from app.onnx_infer import infer_mask
    from app.scale_aruco import mm_per_pixel_from_aruco
    from app.measure import min_area_rect_metrics
except Exception:
    import sys
    sys.path.append(os.path.abspath("app"))
    from onnx_infer import infer_mask  # type: ignore
    from scale_aruco import mm_per_pixel_from_aruco  # type: ignore
    from measure import min_area_rect_metrics  # type: ignore

HANDSY_DIR = "data/handsytest"
OUT_ROOT = "handsy_submission"
MASK_DIR = os.path.join(OUT_ROOT, "masks")
OVERLAY_DIR = os.path.join(OUT_ROOT, "overlays")
MEASURE_CSV = os.path.join(OUT_ROOT, "measurements.csv")
METHODS_MD = os.path.join(OUT_ROOT, "methods_note.md")

IMAGE_GLOB = [
    "V01_straight_clean.jpg",
    "V02_noisy_extra.jpg",
    "V03_straight_calibrator.jpg",
    "V09AI_straight_clean.jpg",
    "V09AI_straight_clean_darker.jpg",
]

AVERAGE_NAIL_WIDTH_MM = 13.0
MIN_COMPONENT_AREA_PX = 150
TARGET_NAILS_PER_IMAGE = 4

def ensure_dirs():
    for d in (OUT_ROOT, MASK_DIR, OVERLAY_DIR):
        os.makedirs(d, exist_ok=True)

def connected_components(mask: np.ndarray):
    binary = (mask > 127).astype(np.uint8)
    num, labels = cv2.connectedComponents(binary)
    comps = []
    for i in range(1, num):
        comp_mask = (labels == i).astype(np.uint8)
        area = comp_mask.sum()
        if area < MIN_COMPONENT_AREA_PX:
            continue
        comps.append((area, comp_mask))
    comps.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in comps]

def measure_component(comp_mask: np.ndarray, mm_per_px: float):
    return min_area_rect_metrics(comp_mask, mm_per_px)

def fallback_segment(img_bgr: np.ndarray) -> np.ndarray:
    """Crude fallback segmentation if ONNX inference unavailable.
    Uses adaptive threshold + morphology to approximate nail regions so downstream
    measurement logic can proceed. Not production-quality; flagged in methods note."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Otsu threshold (nails often lighter)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Invert if foreground proportion too high
    fg_ratio = th.mean()/255.0
    if fg_ratio > 0.65:
        th = 255 - th
    kernel = np.ones((5,5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    return th

def draw_overlay(img_bgr: np.ndarray, comp_masks, measurements, mm_per_px: float):
    overlay = img_bgr.copy()
    for idx, (mask, m) in enumerate(zip(comp_masks, measurements), start=1):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        text = f"{idx}: {m['length_mm']:.1f}x{m['width_mm']:.1f}mm"
        x = int(m['cx']); y = int(m['cy'])
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,255), 1, cv2.LINE_AA)
    scale_text = f"scale {mm_per_px*1000:.2f} um/px" if mm_per_px else "scale unknown"
    cv2.putText(overlay, scale_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
    return overlay

def compute_scale(img_bgr: np.ndarray | None, filename: str, global_scale_mm_per_px: float | None):
    if img_bgr is None:
        return None, "image_load_fail", 0.0
    scale_source = "reuse"
    mm_per_px = global_scale_mm_per_px
    conf = 1.0 if mm_per_px is not None else 0.0
    if "calibrator" in filename.lower():
        mm_per_px, conf = mm_per_pixel_from_aruco(img_bgr)
        if mm_per_px is None or conf < 0.5:
            mm_per_px = None
            conf = 0.0
        else:
            scale_source = "aruco"
    if mm_per_px is None:
        scale_source = "heuristic_avg_width"
        conf = 0.2
    return mm_per_px, scale_source, conf

def heuristic_scale_from_components(comp_masks):
    widths = []
    for m in comp_masks:
        ys, xs = np.where(m>0)
        if xs.size == 0:
            continue
        width_px = xs.max() - xs.min() + 1
        widths.append(width_px)
    if not widths:
        return None
    widest_px = max(widths)
    return AVERAGE_NAIL_WIDTH_MM / widest_px

def find_latest_metrics():
    candidates = glob.glob('runs/seg/**/metrics_summary.json', recursive=True)
    if not candidates:
        return None, None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    path = candidates[0]
    try:
        with open(path,'r',encoding='utf-8') as f:
            data = json.load(f)
        return data, path
    except Exception:
        return None, path

def write_methods_note(mm_per_px: float | None, used_fallback: bool):
    assumed = f"{mm_per_px:.4f} mm/px" if mm_per_px else "(derived per-image or heuristic)"
    metrics, metrics_path = find_latest_metrics()
    metrics_section = "No training metrics available (model not trained this session)."
    if metrics:
        # Display key segmentation metrics if present
        keys = [k for k in metrics.keys() if 'mAP' in k or 'precision' in k or 'recall' in k]
        lines = [f"- {k}: {metrics[k]:.4f}" for k in keys]
        metrics_section = "Training Metrics (from latest run):\n" + "\n".join(lines) + (f"\nSource: {metrics_path}" if metrics_path else "")
    # Try to find escalation summary near metrics path if exists
    escalation_note = ""
    if metrics_path:
        esc_candidate = os.path.join(os.path.dirname(metrics_path), 'escalation_summary.json')
        if os.path.exists(esc_candidate):
            try:
                with open(esc_candidate,'r',encoding='utf-8') as f:
                    esc = json.load(f)
                improved = esc.get('improved')
                escalation_note = f"\nEscalation: primary mAP50={esc.get('primary_map50'):.3f} secondary mAP50={esc.get('secondary_map50'):.3f} -> {'improved' if improved else 'no significant gain'}."
            except Exception:
                pass
    # Include args_used.json if present
    args_meta = ''
    if metrics_path:
        args_candidate = os.path.join(os.path.dirname(metrics_path), 'args_used.json')
        if os.path.exists(args_candidate):
            try:
                with open(args_candidate,'r',encoding='utf-8') as f:
                    a = json.load(f)
                exposed = {k: a.get(k) for k in ['epochs','batch','imgsz','close_mosaic','patience','cache','workers','seed'] if k in a}
                args_meta = "\nTraining Args: " + ", ".join(f"{k}={v}" for k,v in exposed.items())
            except Exception:
                pass
    fallback_note = "ONNX model missing; fallback threshold segmentation used for some images." if used_fallback else "Primary ONNX model used for all images."
    text = f"""# Handsy Phase 0 Mini-Task Methods

Date: {datetime.utcnow().isoformat()}Z

## Overview
Pipeline performs nail segmentation on 5 provided hand images, outputs binary masks, overlays with contour annotations, and per-nail measurements (length, width, area in mm).

## Model
Inference uses an ONNX-exported YOLO segmentation model (`model/best.onnx`). If unavailable, a crude adaptive threshold fallback is applied (non-production) to allow measurement prototyping. {fallback_note}

## Segmentation & Post-processing
1. Run ONNX model (size 640) OR fallback threshold segmentation.
2. Connected components filtering (area ≥ {MIN_COMPONENT_AREA_PX} px) → largest up to 4 retained.
3. Measure each via `cv2.minAreaRect` (major axis = length, minor = width).

## Scaling
Primary: Detect ArUco 4x4_50 marker (20 mm side) in calibrator image to compute global mm_per_px.
Computed global scale: {assumed}
If ArUco fails or absent, heuristic uses average adult fingernail width {AVERAGE_NAIL_WIDTH_MM} mm vs widest detected nail span.

## Measurements
For each nail: length_mm, width_mm, area_mm2 = area_px * (mm_per_px^2). Fewer than 4 nails → only detected components recorded.

## Training
{metrics_section}{escalation_note}{args_meta}

## Limitations
- Global scale assumes minimal perspective distortion.
- Heuristic scale fallback may bias absolute scale.
- Fallback segmentation (if used) may include background noise or miss distal edges.
- Partial occlusion reduces length accuracy.

## Reproducibility
Script: `scripts/handsy_phase0.py`
Run: `python scripts/handsy_phase0.py`

"""
    with open(METHODS_MD, 'w', encoding='utf-8') as f:
        f.write(text)

def main():
    ensure_dirs()
    global_scale = None
    rows = []
    # Pass to find global scale
    for fname in IMAGE_GLOB:
        path = os.path.join(HANDSY_DIR, fname)
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        mm_per_px, source, conf = compute_scale(img, fname, global_scale)
        if source == "aruco" and mm_per_px:
            global_scale = mm_per_px
            break
    # Process images
    used_fallback_any = False
    for fname in IMAGE_GLOB:
        path = os.path.join(HANDSY_DIR, fname)
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        mm_per_px, scale_source, scale_conf = compute_scale(img, fname, global_scale)
        if img is None:
            print(f"Failed to read image {fname}")
            continue
        mask = None
        try:
            mask = infer_mask(img)
        except Exception as e:
            print(f"Inference failed for {fname}: {e}; using fallback segmentation.")
            mask = fallback_segment(img)
            used_fallback_any = True
        if mask is None:
            print(f"No mask produced for {fname}")
            continue
        mask_path = os.path.join(MASK_DIR, fname.replace('.jpg', '_mask.png'))
        cv2.imwrite(mask_path, mask)
        comps = connected_components(mask)
        if not comps:
            print(f"No components in {fname}")
            continue
        if mm_per_px is None:
            mm_per_px = heuristic_scale_from_components(comps)
            scale_source = "heuristic_avg_width"
            scale_conf = 0.2 if mm_per_px is not None else 0.0
        if mm_per_px is None:
            print(f"Could not determine scale for {fname}; skipping measurements")
            continue
        comps = comps[:TARGET_NAILS_PER_IMAGE]
        measurements = []
        for c in comps:
            m = measure_component(c, mm_per_px)
            if m is not None:
                measurements.append(m)
        overlay = draw_overlay(img, comps, measurements, mm_per_px)
        overlay_path = os.path.join(OVERLAY_DIR, fname.replace('.jpg', '_overlay.jpg'))
        cv2.imwrite(overlay_path, overlay)
        for idx, m in enumerate(measurements, start=1):
            rows.append({
                'image_id': fname,
                'nail_id': idx,
                'length_mm': f"{m['length_mm']:.2f}",
                'width_mm': f"{m['width_mm']:.2f}",
                'area_mm2': f"{m['area_mm2']:.2f}",
                'mm_per_px': f"{mm_per_px:.5f}",
                'scale_source': scale_source,
                'scale_confidence': f"{scale_conf:.2f}"
            })
    if rows:
        with open(MEASURE_CSV, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['image_id','nail_id','length_mm','width_mm','area_mm2','mm_per_px','scale_source','scale_confidence']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    write_methods_note(global_scale, used_fallback_any)
    # Package ZIP
    zip_name = 'handsy_phase0_submission.zip'
    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT_ROOT):
            for f in files:
                fp = os.path.join(root, f)
                arc = os.path.relpath(fp, '.')
                zf.write(fp, arc)
    print(f"Done. Outputs in {OUT_ROOT}; packaged as {zip_name}")

if __name__ == "__main__":
    main()
