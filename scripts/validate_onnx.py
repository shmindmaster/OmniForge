import argparse, os, sys, json, hashlib
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    print('onnxruntime not installed. Install onnxruntime or onnxruntime-gpu to run this check.')
    sys.exit(1)

import cv2

DEF_MODEL = 'model/best.onnx'

def parse_args():
    ap = argparse.ArgumentParser(description='Quick sanity validation of exported ONNX segmentation model.')
    ap.add_argument('--model', default=DEF_MODEL)
    ap.add_argument('--image', default='data/handsytest/V01_straight_clean.jpg', help='Test image path')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--print-shape', action='store_true')
    return ap.parse_args()


def letterbox(im, new_size=640):
    h, w = im.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print('Model not found:', model_path)
        return 2
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    print(f'Using ONNX model {model_path} sha12={sha}')
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    out_names = [o.name for o in sess.get_outputs()]
    print('Input name:', inp.name, 'shape:', inp.shape, 'dtype:', inp.type)
    print('Output names:', out_names)
    img = cv2.imread(args.image)
    if img is None:
        print('Failed to read test image:', args.image)
        return 3
    lb, scale, left, top = letterbox(img, args.imgsz)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2,0,1))[None]
    ort_out = sess.run(out_names, {inp.name: arr})
    # Heuristic checks: at least one output must have spatial dims 80+ (expected mask proto) or detection dimension > 10
    passed = False
    shape_info = []
    for idx, o in enumerate(ort_out):
        try:
            shp = list(getattr(o, 'shape', []))
        except Exception:
            shp = []
        shape_info.append({'index': idx, 'shape': shp})
        if len(shp) >= 3 and (shp[-1] >= 80 or shp[-2] >= 80):
            passed = True
        if len(shp) == 3 and shp[1] > 10 and shp[2] >= 6:
            passed = True
    result = {
        'model': str(model_path),
        'sha256_12': sha,
        'outputs': shape_info,
        'inference_ok': passed
    }
    if args.print_shape:
        print(json.dumps(result, indent=2))
    else:
        print('Inference OK' if passed else 'Inference suspicious; inspect shapes ->', shape_info)
    # Write report
    with open('model/onnx_validation_report.json','w',encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    return 0 if passed else 4

if __name__ == '__main__':
    raise SystemExit(main())
