import os
import numpy as np
import cv2
import types

from app import pipeline as pipe

def synthetic_image(with_marker=True):
    img = np.full((200, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (80,60), (220,160), (0,0,0), -1)  # nail-ish region
    if with_marker:
        # simple square to simulate marker (not true ArUco but our mock will force scale)
        cv2.rectangle(img, (10,10), (50,50), (0,0,0), -1)
    return img

def test_pipeline_with_mocked_components(monkeypatch):
    # Force USE_ONNX=True path but mock infer
    monkeypatch.setenv("USE_ONNX", "1")
    # Mock onnx inference to return central rectangle mask
    def fake_infer(img):
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(m, (80,60), (220,160), 255, -1)
        return m
    import app.onnx_infer as oi
    monkeypatch.setattr(oi, "infer_mask", fake_infer)

    # Mock ArUco scale function
    def fake_scale(img):
        return 0.1, 1.0
    import app.scale_aruco as sa
    monkeypatch.setattr(sa, "mm_per_pixel_from_aruco", fake_scale)

    img = synthetic_image().copy()
    ok, buf = cv2.imencode('.png', img)
    assert ok
    res = pipe.run_pipeline(buf.tobytes())
    metrics = res["metrics"]
    assert 0.0 <= metrics["scale_confidence"] <= 1.0
    assert metrics["length_mm"] > 0
    assert len(res["stl_bytes"]) > 100  # some bytes
    csv_text = res["csv_bytes"].decode('utf-8')
    assert 'length_mm' in csv_text
    # numeric check
    import re
    assert re.search(r'length_mm,', csv_text) or 'length_mm' in csv_text
