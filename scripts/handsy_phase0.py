import os
import csv
import cv2
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
        return None, "image_load_fail"
    scale_source = "reuse"
    mm_per_px = global_scale_mm_per_px
    if "calibrator" in filename.lower():
        mm_per_px, conf = mm_per_pixel_from_aruco(img_bgr)
        if mm_per_px is None or conf < 0.5:
            mm_per_px = None
        else:
            scale_source = "aruco"
    if mm_per_px is None:
        scale_source = "heuristic_avg_width"
    return mm_per_px, scale_source

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

def write_methods_note(mm_per_px: float | None):
    assumed = f"{mm_per_px:.4f} mm/px" if mm_per_px else "(derived per-image or heuristic)"
    text = f"""# Handsy Phase 0 Mini-Task Methods

Date: {datetime.utcnow().isoformat()}Z

## Overview
Pipeline performs nail segmentation on 5 provided hand images, outputs binary masks, overlays with contour annotations, and per-nail measurements (length, width, area in mm).

## Model
Inference uses an ONNX-exported YOLO segmentation model (`model/best.onnx`). The model produces multi-object nail masks which are post-processed into connected components (top 4 per image by area).

## Segmentation & Post-processing
1. Run ONNX model (size 640) on each image.
2. Reconstruct mask proto → merged binary mask.
3. Connected components filtering (area ≥ {MIN_COMPONENT_AREA_PX} px) → take largest 4.
4. Measure each via `cv2.minAreaRect` (major axis = length, minor = width).

## Scaling
Primary: Detect ArUco 4x4_50 marker (20 mm side) in calibrator image to compute global mm_per_px.
Computed global scale: {assumed}
If ArUco fails or absent in other images, fallback uses heuristic: assumed average adult fingernail width {AVERAGE_NAIL_WIDTH_MM} mm vs widest detected nail pixel span in that image.

## Measurements
For each nail: length_mm, width_mm from rectangle, area_mm2 = area_px * (mm_per_px^2).
If fewer than 4 nails detected, CSV contains only detected components.

## Runtime
Per image expected < 2s CPU.

## Limitations
- Global scale assumes minimal perspective distortion.
- Heuristic fallback may bias absolute scale.
- No quality filtering beyond area threshold.
- Partial occlusion may reduce length estimate.

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
        mm_per_px, source = compute_scale(img, fname, global_scale)
        if source == "aruco" and mm_per_px:
            global_scale = mm_per_px
            break
    # Process images
    for fname in IMAGE_GLOB:
        path = os.path.join(HANDSY_DIR, fname)
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        mm_per_px, scale_source = compute_scale(img, fname, global_scale)
        if img is None:
            print(f"Failed to read image {fname}")
            continue
        try:
            mask = infer_mask(img)
        except Exception as e:
            print(f"Inference failed for {fname}: {e}")
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
                'scale_source': scale_source
            })
    if rows:
        with open(MEASURE_CSV, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['image_id','nail_id','length_mm','width_mm','area_mm2','mm_per_px','scale_source']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    write_methods_note(global_scale)
    print(f"Done. Outputs in {OUT_ROOT}")

if __name__ == "__main__":
    main()
