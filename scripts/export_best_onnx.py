import hashlib, argparse, os, json
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='Path to best.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--opset', type=int, default=13)
    ap.add_argument('--out', default='model/best.onnx')
    ap.add_argument('--simplify', action='store_true')
    ap.add_argument('--device', default='cpu')
    return ap.parse_args()

def main():
    args = parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    res = model.export(format='onnx', imgsz=args.imgsz, opset=args.opset, simplify=args.simplify, device=args.device, dynamic=False)
    onnx_path = Path(res) if isinstance(res, (str, Path)) else Path(args.out)
    if not onnx_path.exists():
        # fallback rename
        if Path(args.weights).parent.joinpath('weights','best.onnx').exists():
            onnx_path = Path(args.weights).parent.joinpath('weights','best.onnx')
    if onnx_path.exists():
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        if onnx_path.resolve() != target.resolve():
            data = onnx_path.read_bytes()
            target.write_bytes(data)
        sha = hashlib.sha256(target.read_bytes()).hexdigest()[:12]
        meta = {'export_from': str(args.weights), 'onnx_path': str(target), 'sha256_12': sha, 'imgsz': args.imgsz, 'opset': args.opset, 'simplify': args.simplify}
        with open('model/onnx_export_meta.json','w',encoding='utf-8') as f:
            json.dump(meta,f,indent=2)
        print('Exported ONNX ->', target, 'sha12', sha)
    else:
        raise SystemExit('ONNX export failed: file not found')

if __name__ == '__main__':
    main()
