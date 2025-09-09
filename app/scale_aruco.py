import cv2, numpy as np

def mm_per_pixel_from_aruco(img_bgr, marker_mm=20.0):
    # OpenCV ArUco detection
    aruco = cv2.aruco
    dict_ = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(img_bgr, dict_, parameters=params)
    if ids is None or len(corners) == 0:
        return None, 0.0
    c = corners[0].reshape(-1, 2)
    side_px = (cv2.norm(c[0]-c[1]) + cv2.norm(c[1]-c[2]) +
               cv2.norm(c[2]-c[3]) + cv2.norm(c[3]-c[0])) / 4.0
    return float(marker_mm / side_px), 1.0
