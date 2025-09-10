# scripts\handsy_phase0.py
from __future__ import annotations
import os, sys, csv, cv2, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.onnx_infer import infer_mask
from app.scale_aruco import mm_per_pixel_from_aruco
from app.overlay import draw_overlay
from app.to_3d import outline_to_stl_bytes

ROOT = Path(__file__).resolve().parents[1]
IN_DIR  = ROOT / "handsy_in"
OUT_DIR = ROOT / "handsy_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_ID1_FALLBACK = os.environ.get("USE_ID1_FALLBACK", "0") == "1"

def try_id1_scale(img_bgr):
    if not USE_ID1_FALLBACK: return (None, 0.0)
    try:
        from transformers import OwlViTProcessor, OwlViTForObjectDetection  # type: ignore
        import torch
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H,W = rgb.shape[:2]
        proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval()
        inputs = proc(text=[["credit card","gift card","debit card","bank card"]], images=rgb, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        results = proc.post_process_object_detection(outputs, threshold=0.25, target_sizes=torch.tensor([[H,W]]) )[0]
        if len(results["scores"]) == 0: return (None, 0.0)
        i = int(results["scores"].argmax())
        x0,y0,x1,y1 = results["boxes"][i].cpu().numpy()
        longer_px = max(abs(x1-x0), abs(y1-y0))
        longer_mm = 85.60  # ISO/IEC 7810 ID-1
        return float(longer_mm / max(1.0, longer_px)), 0.7
    except ImportError:
        return (None, 0.0)
    except Exception:
        return (None, 0.0)

def clean_mask_keep_top_k(mask, k=4):
    m = (mask>0).astype(np.uint8)
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:k]
    out = np.zeros_like(m)
    for c in cnts:
        if cv2.contourArea(c) < 50: continue
        cv2.drawContours(out, [c], -1, (255,255,255), -1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=2)
    return out

def measure(mask, mmpp):
    ys,xs = np.where(mask>0)
    if xs.size==0: return None
    pts = np.column_stack([xs,ys]).astype(np.float32)
    mean = np.mean(pts, axis=0)
    mean, eig = cv2.PCACompute(pts, mean=mean.reshape(1, -1))
    axis, ortho = eig[0], eig[1]
    centered = pts - mean
    proj = centered @ axis; ort = centered @ ortho
    lo, hi = float(proj.min()), float(proj.max())
    Lpx = max(1.0, (hi-lo))
    Lmm = Lpx * mmpp
    def width_at(fr):
        t = lo + fr*(hi-lo)
        band = np.abs(proj - t) < 0.01*Lpx + 1.0
        if not np.any(band): return 0.0
        span = ort[band]; return float((span.max()-span.min()) * mmpp)
    return dict(length_mm=Lmm,
                width_prox_mm=width_at(0.25),
                width_mid_mm=width_at(0.50),
                width_dist_mm=width_at(0.75),
                mask_area_px=int((mask>0).sum()))

def process_one(img_path: Path):
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise ValueError(f"read fail: {img_path}")
    mmpp, conf = mm_per_pixel_from_aruco(img)
    method = "aruco"
    if mmpp is None:
        mmpp, conf = try_id1_scale(img)
        method = "id1_card" if mmpp is not None else "default"
        if mmpp is None: mmpp, conf = 0.25, 0.2
    raw = infer_mask(img)
    mask = clean_mask_keep_top_k(raw, k=4)
    base = img_path.stem
    # save mask
    cv2.imencode(".png", mask)[1].tofile(str(OUT_DIR / f"{base}_mask.png"))
    # overlay
    m = measure(mask, mmpp) or {}
    m.update({"mm_per_px": mmpp, "scale_confidence": conf})
    overlay_png = draw_overlay(img, mask, m)
    (OUT_DIR / f"{base}_overlay.png").write_bytes(overlay_png)
    # (optional) STL for largest contour
    return base, m, mask

def per_contour_rows(mask_clean, mmpp, scale_conf, image_base):
    import cv2, numpy as np  # local to keep top-level lean
    rows = []
    cnts, _ = cv2.findContours((mask_clean>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    for idx, c in enumerate(cnts):
        single = np.zeros_like(mask_clean, dtype=np.uint8)
        cv2.drawContours(single, [c], -1, (255, 255, 255), -1)
        m = measure(single, mmpp)
        if not m:
            continue
        rows.append({
            "image": image_base,
            "nail_index": idx,
            "length_mm": m["length_mm"],
            "width_prox_mm": m["width_prox_mm"],
            "width_mid_mm": m["width_mid_mm"],
            "width_dist_mm": m["width_dist_mm"],
            "mask_area_px": m["mask_area_px"],
            "mm_per_px": mmpp,
            "scale_confidence": scale_conf,
        })
    return rows

def main():
    paths = [p for p in IN_DIR.glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
    if not paths:
        print(f"No images in {IN_DIR}"); return
    csv_path = OUT_DIR / "measurements.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image","nail_index","length_mm","width_prox_mm","width_mid_mm","width_dist_mm",
            "mask_area_px","mm_per_px","scale_confidence"
        ])
        if write_header:
            w.writeheader()
        for p in paths:
            base, metrics, mask_clean = process_one(p)
            mmpp = metrics.get("mm_per_px")
            scl = metrics.get("scale_confidence")
            rows = per_contour_rows(mask_clean, mmpp, scl, base)
            for r in rows:
                w.writerow(r)
    print(f"✅ Wrote outputs to {OUT_DIR}")

if __name__ == "__main__":
    main()
