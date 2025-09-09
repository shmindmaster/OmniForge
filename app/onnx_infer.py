import os
import cv2
import numpy as np
import onnxruntime as ort

_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "model/best.onnx")
_SESSION = None

def _get_session():
    global _SESSION
    if _SESSION is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"ONNX model not found at {_MODEL_PATH}")
        providers = ["CPUExecutionProvider"]
        _SESSION = ort.InferenceSession(_MODEL_PATH, providers=providers)
    return _SESSION

def _preprocess(img_bgr: np.ndarray, size: int = 640):
    h, w = img_bgr.shape[:2]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_resized, (2, 0, 1))
    return img_chw[None, ...], (h, w)

def _postprocess(outputs: dict, orig_hw, input_hw=(640,640)):
    # Ultralytics YOLO ONNX segmentation typical outputs: (1,N,classes+5+mask_dim) and (1,mask_dim,mask_h,mask_w)
    # We attempt to detect shapes heuristically.
    mask_proto = None
    det_out = None
    for out in outputs.values():
        arr = out
        if arr.ndim == 4:  # (1,mask_dim,h,w)
            mask_proto = arr
        elif arr.ndim == 3:  # (1,N,C)
            det_out = arr
    if det_out is None or mask_proto is None:
        raise ValueError("Unexpected ONNX outputs; could not find detection and mask proto tensors")

    det_out = det_out[0]  # (N,C)
    mask_proto = mask_proto[0]  # (mask_dim,h,w)

    # Heuristic: last k values of each det row correspond to mask coefficients; first 4 box, objectness, class scores follow.
    num_cols = det_out.shape[1]
    # Assume 32 mask coeffs (common) - fallback scan
    possible_mask_dims = [32, 64]
    mask_dim = None
    for md in possible_mask_dims:
        if num_cols - md > 5:  # need at least box(4)+obj(1)+cls>=1
            mask_dim = md
            break
    if mask_dim is None:
        raise ValueError("Could not infer mask coefficient dimension")

    coeffs = det_out[:, -mask_dim:]
    scores = det_out[:, 4]
    keep = scores > 0.25
    if not np.any(keep):
        return np.zeros(orig_hw, dtype=np.uint8)
    coeffs = coeffs[keep]

    # Reconstruct masks
    proto_h, proto_w = mask_proto.shape[1:]
    masks = coeffs @ mask_proto.reshape(mask_dim, -1)
    masks = 1 / (1 + np.exp(-masks))  # sigmoid
    masks = masks.reshape(-1, proto_h, proto_w)
    # Upsample to input size
    upsampled = []
    for m in masks:
        upsampled.append(cv2.resize(m, input_hw, interpolation=cv2.INTER_LINEAR))
    upsampled = np.stack(upsampled, axis=0)
    merged = (upsampled.max(axis=0) > 0.5).astype(np.uint8) * 255
    # Resize back to original
    orig_mask = cv2.resize(merged, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_NEAREST)
    return orig_mask

def infer_mask(img_bgr: np.ndarray) -> np.ndarray:
    sess = _get_session()
    inp, orig_hw = _preprocess(img_bgr)
    input_name_map = {i.name: i for i in sess.get_inputs()}
    # Use first input
    first_input = next(iter(input_name_map.keys()))
    feeds = {first_input: inp}
    outputs = sess.run(None, feeds)
    out_names = [o.name for o in sess.get_outputs()]
    outputs_map = {k: v for k, v in zip(out_names, outputs)}
    mask = _postprocess(outputs_map, orig_hw)
    if mask.ndim != 2:
        raise ValueError("Mask dimensionality invalid")
    return mask

