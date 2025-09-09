import cv2, numpy as np

def draw_overlay(img_bgr, mask, metrics):
    vis = img_bgr.copy()
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0,255,0), 2)
    txt = (f"L={metrics['length_mm']:.2f} mm | Wp={metrics['width_prox_mm']:.2f} | "
           f"Wm={metrics['width_mid_mm']:.2f} | Wd={metrics['width_dist_mm']:.2f}")
    cv2.putText(vis, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise RuntimeError("Failed to encode overlay PNG")
    return png.tobytes()
