import os, json, time, argparse
from datetime import datetime
from pathlib import Path

TARGET_MAP50 = 0.88

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/yolo/data.yaml')
    ap.add_argument('--model', default='yolov11s-seg.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--project', default='runs/seg')
    ap.add_argument('--name', default='nails_v11s')
    ap.add_argument('--device', default='cpu')
    return ap.parse_args()

def main():
    args = parse_args()
    from ultralytics import YOLO
    model = YOLO(args.model)
    start = time.time()
    res = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=True,
    )
    duration = time.time() - start
    raw_metrics = getattr(res, 'results_dict', {}) or {}
    metrics = {}
    for k,v in raw_metrics.items():
        try:
            metrics[k] = float(v)
        except Exception:
            continue
    metrics['train_duration_sec'] = float(duration)
    metrics['timestamp_iso'] = datetime.utcnow().isoformat() + 'Z'
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics_summary.json','w',encoding='utf-8') as f:
        json.dump(metrics,f,indent=2)
    map50 = metrics.get('metrics/mAP50(B)', metrics.get('metrics/mAP50', 0.0))
    print(f"Validation mAP50: {map50:.3f}")
    if map50 < TARGET_MAP50 - 0.03:
        print('WARNING: mAP50 below threshold; consider escalating to yolov11m-seg or auditing dataset.')
    print('Training complete. Best weights at:', out_dir / 'weights' / 'best.pt')

if __name__ == '__main__':
    main()
