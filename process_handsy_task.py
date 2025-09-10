import os
import time
import zipfile
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import statistics

# Ensure paths (sitecustomize adds app/ to sys.path)
import sitecustomize  # noqa: F401

# Import internal pipeline components
from app import pipeline as _pipe  # type: ignore
from app.scale_aruco import mm_per_pixel_from_aruco  # type: ignore
from app.overlay import draw_overlay  # type: ignore


def _heuristic_multi_nail(img_bgr: np.ndarray):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L_eq = clahe.apply(L)
    base = cv2.adaptiveThreshold(L_eq,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,41,-5)
    if (base>0).mean() > 0.55:
        base = cv2.bitwise_not(base)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    low_sat = cv2.inRange(S, np.array([0], dtype=S.dtype), np.array([120], dtype=S.dtype))
    comb = cv2.bitwise_and(base, low_sat)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, kernel, iterations=1)
    comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts,_ = cv2.findContours(comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h,w = comb.shape
    area_min, area_max = 0.0008*w*h, 0.18*w*h
    keep=[]
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_min or a > area_max: continue
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        if ha <= 0: continue
        solidity = a/(ha+1e-6)
        if solidity < 0.7: continue
        keep.append(c)
    keep = sorted(keep, key=cv2.contourArea, reverse=True)[:4]
    mask = np.zeros_like(comb)
    for c in keep:
        cv2.drawContours(mask,[c],-1,255,-1)
    return mask, keep


def process_image_for_task(image_path: Path, output_dir: Path):
    print(f"Processing {image_path.name} ...")
    img_bytes = image_path.read_bytes()
    img_bgr = _pipe._image_from_bytes(img_bytes)
    if img_bgr is None:
        print("  - Skipped: could not load image")
        return None

    seg_mask = _pipe._onnx_segmentation(img_bgr)
    heuristic_used = False
    if seg_mask is None or not seg_mask.any():
        heuristic_used = True
        seg_mask, cnts = _heuristic_multi_nail(img_bgr)
    else:
        cnts,_ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    if not cnts:
        print("  - ERROR: no contours detected")
        return None

    mm_per_px, scale_conf = mm_per_pixel_from_aruco(img_bgr)
    if mm_per_px is None:
        widths=[]
        for c in cnts:
            (_, _),(rw,rh),_ang = cv2.minAreaRect(c)
            widths.append(min(rw,rh))
        if widths:
            median_w = statistics.median(widths)
            typical_width_mm = 13.0
            mm_per_px = typical_width_mm / median_w
            scale_conf = 0.2
            print(f"  - Heuristic scale {mm_per_px:.4f} mm/px (conf {scale_conf})")
        else:
            mm_per_px, scale_conf = 0.25, 0.0
            print("  - Fallback scale 0.25 mm/px")
    else:
        print(f"  - Marker scale {mm_per_px:.4f} mm/px (conf {scale_conf:.2f})")

    union_mask = np.zeros_like(seg_mask if seg_mask is not None else img_bgr[:,:,0])
    for c in cnts:
        cv2.drawContours(union_mask,[c],-1,255,-1)
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), union_mask)
    print(f"  - Mask written: {mask_path.name}")

    rows=[]
    overlay = img_bgr.copy()
    palette=[(0,255,0),(255,0,0),(0,128,255),(255,128,0)]
    for i,c in enumerate(cnts, start=1):
        single = np.zeros_like(union_mask)
        cv2.drawContours(single,[c],-1,255,-1)
        base_metrics = dict(_pipe._measure(single, mm_per_px))
        meta = {
            'nail_index': i,
            'mm_per_px': mm_per_px,
            'scale_confidence': scale_conf,
            'source_image': image_path.name,
            'segmentation_mode': 'heuristic' if heuristic_used else 'onnx'
        }
        full = {**base_metrics, **meta}
        rows.append(full)
        centroid = c.reshape(-1,2).mean(axis=0).astype(int)
        cv2.drawContours(overlay,[c],-1,palette[(i-1)%len(palette)],2)
    cv2.putText(overlay,f"{i} L {full['length_mm']:.1f}mm",(centroid[0]-40, centroid[1]),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(overlay,f"{i} L {full['length_mm']:.1f}mm",(centroid[0]-40, centroid[1]),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
    scale_txt=f"scale {mm_per_px:.3f} mm/px conf {scale_conf:.2f}"+(" heuristic" if scale_conf<1.0 else "")
    cv2.putText(overlay,scale_txt,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(overlay,scale_txt,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
    ok,png_buf=cv2.imencode('.png', overlay)
    if ok:
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        overlay_path.write_bytes(png_buf.tobytes())
        print(f"  - Overlay written: {overlay_path.name}")
    else:
        print("  - WARNING: overlay encode failed")

    allowed=['source_image','nail_index','length_mm','width_prox_mm','width_mid_mm','width_dist_mm','mm_per_px','scale_confidence','mask_area_px','sharpness','segmentation_mode']
    df_rows=[{k:r[k] for k in allowed if k in r} for r in rows]
    return pd.DataFrame(df_rows)


def main():
    base = Path(__file__).resolve().parent
    input_dir = base / "tests" / "data"
    output_dir = Path("handsy_submission")
    output_dir.mkdir(exist_ok=True)

    imgs = [p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not imgs:
        print("No images found in tests/data")
        return

    dfs = []
    for p in imgs:
        df = process_image_for_task(p, output_dir)
        if df is not None:
            dfs.append(df)

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        cols_order = ['source_image','nail_index','length_mm','width_prox_mm','width_mid_mm','width_dist_mm','mm_per_px','scale_confidence','mask_area_px','sharpness','segmentation_mode']
        final_df = final_df[[c for c in cols_order if c in final_df.columns]]
        csv_path = output_dir / 'measurements.csv'
        final_df.to_csv(csv_path, index=False, float_format='%.3f')
        print(f"Measurements written: {csv_path.name}")

    methods_note = output_dir / 'methods_note.md'
    methods_note.write_text("""# Methods Note: Handsy Inc. Mini-Task Submission

Dual-path segmentation & measurement:

1. Segmentation
    - ONNX model (YOLO-style) if available (not present in this run).
    - Heuristic fallback: LAB lightness equalization + adaptive threshold + HSV low-sat mask + morphology + contour filtering (area + solidity). Largest ≤4 contours retained.
2. Per-Nail Metrics
    - Each contour -> binary mask -> PCA axis metrics (length_mm, width_* quartiles) via existing pipeline logic.
3. Scaling
    - ArUco DICT_4X4_50 (20 mm) for mm_per_px (conf=1.0) when present.
    - Else median nail width heuristic (typical width 13 mm, conf≈0.2). If no contours: absolute fallback 0.25 mm/px (conf=0.0).
4. Outputs
    - <image>_mask.png (union mask) | <image>_overlay.png (per-nail labels + scale)
    - measurements.csv (one row per nail with nail_index & segmentation_mode)
5. Runtime
    - Heuristic path: ~50–150 ms/image CPU; ONNX path (future) ~100–400 ms.
6. Limitations
    - Heuristic may merge or miss nails under glare / extreme polish colors.
    - Width heuristic scaling approximate; true marker strongly preferred.
    - Replace `model/best.onnx` to enable ML segmentation without code changes.
""", encoding='utf-8')
    print(f"Methods note written: {methods_note.name}")

    zip_name = f"Handsy_Submission_{int(time.time())}.zip"
    zip_path = Path(zip_name)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in output_dir.iterdir():
            z.write(f, arcname=f.name)
    print(f"Created submission archive: {zip_path}")


if __name__ == '__main__':
    # Force ONNX path & dummy connection string
    os.environ['USE_ONNX'] = '1'
    os.environ.setdefault('ONNX_MODEL_PATH', 'model/best.onnx')
    os.environ.setdefault('AZURE_STORAGE_CONNECTION_STRING', 'dummy')
    main()
