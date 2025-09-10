import cv2
import numpy as np

def pca_axis_metrics(mask: np.ndarray, mm_per_px: float):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    contour = np.column_stack((xs, ys)).astype(np.float32)
    mean_init = np.array([], dtype=np.float32)
    mean, eigenvectors = cv2.PCACompute(contour, mean_init)
    axis_vec = eigenvectors[0]
    ortho_vec = eigenvectors[1]
    centered = contour - mean
    proj = centered @ axis_vec
    ortho = centered @ ortho_vec
    min_p, max_p = proj.min(), proj.max()
    length_px = float(max_p - min_p)
    length_mm = length_px * mm_per_px
    def width_at(frac):
        target = min_p + frac * (max_p - min_p)
        band_mask = np.abs(proj - target) < 0.01 * length_px + 1.0
        if not np.any(band_mask):
            return 0.0
        span = ortho[band_mask]
        return (span.max() - span.min()) * mm_per_px
    width_prox_mm = width_at(0.25)
    width_mid_mm = width_at(0.50)
    width_dist_mm = width_at(0.75)
    mask_area_px = int((mask > 0).sum())
    lap = cv2.Laplacian(mask, cv2.CV_32F)
    sharpness = float(lap.var())
    return dict(length_mm=length_mm, width_prox_mm=width_prox_mm, width_mid_mm=width_mid_mm,
                width_dist_mm=width_dist_mm, mask_area_px=mask_area_px, sharpness=sharpness,
                axis_origin=mean.flatten().tolist(), axis_vec=axis_vec.tolist())

def min_area_rect_metrics(mask: np.ndarray, mm_per_px: float):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    length_px = max(w, h)
    width_px = min(w, h)
    area_px = float(len(pts))
    length_mm = length_px * mm_per_px
    width_mm = width_px * mm_per_px
    area_mm2 = area_px * (mm_per_px ** 2)
    return dict(length_mm=length_mm, width_mm=width_mm, area_mm2=area_mm2, length_px=length_px,
                width_px=width_px, area_px=area_px, angle=angle, cx=cx, cy=cy)
