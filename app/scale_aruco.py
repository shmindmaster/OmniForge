import cv2

def mm_per_pixel_from_aruco(img_bgr, marker_mm=20.0):
    try:
        aruco = cv2.aruco
        if not hasattr(aruco, 'getPredefinedDictionary'):
            return None, 0.0
        dict_ = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        if hasattr(aruco, 'DetectorParameters'):
            params = aruco.DetectorParameters()
        else:
            params = None
        corners = ids = None
        # Prefer detector class if available
        if hasattr(aruco, 'ArucoDetector'):
            detector = aruco.ArucoDetector(dict_, params) if params else aruco.ArucoDetector(dict_)
            res = detector.detectMarkers(img_bgr)
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                corners, ids = res[0], res[1]
        # Skip legacy detectMarkers if missing to avoid attribute issues
        if ids is None or not corners:
            return None, 0.0
        c = corners[0].reshape(-1, 2)
        side_px = (cv2.norm(c[0]-c[1]) + cv2.norm(c[1]-c[2]) + cv2.norm(c[2]-c[3]) + cv2.norm(c[3]-c[0])) / 4.0
        if side_px <= 0:
            return None, 0.0
        return float(marker_mm / side_px), 1.0
    except Exception:
        return None, 0.0
