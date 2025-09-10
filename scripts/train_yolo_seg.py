import os, json, time, argparse
from datetime import datetime
from pathlib import Path

TARGET_MAP50 = 0.88

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/yolo/data.yaml')
    ap.add_argument('--model', default='yolo11s-seg.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--project', default='runs/seg')
    ap.add_argument('--name', default='nails_v11s')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--close_mosaic', type=int, default=None, help='Disable mosaic augmentation for final N epochs (ultralytics close_mosaic).')
    ap.add_argument('--cache', action='store_true', help='Cache images to RAM to accelerate training.')
    ap.add_argument('--workers', type=int, default=8, help='Dataloader workers.')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    ap.add_argument('--auto_escalate', action='store_true', help='If enabled, retrain with yolo11m-seg.pt when primary run underperforms threshold margin.')
    ap.add_argument('--escalate_margin', type=float, default=0.03, help='Margin below TARGET_MAP50 that triggers escalation when --auto_escalate is set.')
    return ap.parse_args()

def main():
    args = parse_args()
    from ultralytics import YOLO
    # Torch / CUDA environment report
    try:
        import torch
        print('[ENV] Torch version:', torch.__version__)
        print('[ENV] CUDA available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('[ENV] CUDA device count:', torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print(f"[ENV] Device {i}: {torch.cuda.get_device_name(i)}")
        if str(args.device) not in ('cpu', 'CPU') and not torch.cuda.is_available():
            raise SystemExit('Requested GPU device but CUDA not available in current torch build. Install a CUDA-enabled torch wheel.')
    except Exception as e:
        print('[ENV] Torch inspection failed:', e)
    # Allow passing a local path or a model name; if path missing, use name directly
    candidate_models = [args.model, 'yolo11s-seg.pt', 'yolo11m-seg.pt']
    model_loaded = None
    for cand in candidate_models:
        try:
            print(f"Trying model base '{cand}'...")
            model_loaded = YOLO(cand)
            print(f"Loaded model '{cand}'")
            break
        except Exception as e:
            print(f"Failed to load '{cand}': {e}")
    if model_loaded is None:
        raise SystemExit('Unable to load any candidate segmentation model weights.')
    model = model_loaded
    # Ensure reproducibility if seed provided
    try:
        if args.seed is not None:
            import random, numpy as np, torch
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass
    base_train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        project=args.project,
        verbose=True,
        workers=args.workers,
        cache=args.cache,
    )
    if args.close_mosaic is not None:
        base_train_kwargs['close_mosaic'] = args.close_mosaic

    def run_training(model_obj, run_name):
        train_kwargs = dict(base_train_kwargs)
        train_kwargs['name'] = run_name
        print('\n=== Training run:', run_name, '===')
        print('Training configuration:', json.dumps({k: v for k,v in train_kwargs.items()}, indent=2))
        start_local = time.time()
        res_local = model_obj.train(**train_kwargs)
        duration_local = time.time() - start_local
        raw_metrics_local = getattr(res_local, 'results_dict', {}) or {}
        metrics_local = {}
        for k,v in raw_metrics_local.items():
            try:
                metrics_local[k] = float(v)
            except Exception:
                continue
        metrics_local['train_duration_sec'] = float(duration_local)
        metrics_local['timestamp_iso'] = datetime.utcnow().isoformat() + 'Z'
        out_dir_local = Path(args.project) / run_name
        out_dir_local.mkdir(parents=True, exist_ok=True)
        with open(out_dir_local / 'metrics_summary.json','w',encoding='utf-8') as f:
            json.dump(metrics_local,f,indent=2)
        with open(out_dir_local / 'args_used.json','w',encoding='utf-8') as f:
            json.dump({k: getattr(args,k) for k in vars(args)}, f, indent=2)
        map50_local = metrics_local.get('metrics/mAP50(B)', metrics_local.get('metrics/mAP50', 0.0))
        print(f"Validation mAP50 ({run_name}): {map50_local:.3f}")
        print('Best weights at:', out_dir_local / 'weights' / 'best.pt')
        return map50_local, out_dir_local, metrics_local

    primary_map50, primary_out_dir, primary_metrics = run_training(model, args.name)
    escalate_performed = False
    escalation_record = {}
    if args.auto_escalate and '11m' not in args.model and primary_map50 < TARGET_MAP50 - args.escalate_margin:
        print(f"Auto-escalation: mAP50 {primary_map50:.3f} below target-{args.escalate_margin:.3f}; switching to yolo11m-seg.pt")
        try:
            from ultralytics import YOLO as _YOLO
            model_m = _YOLO('yolo11m-seg.pt')
            secondary_name = args.name + '_m'
            secondary_map50, secondary_out_dir, secondary_metrics = run_training(model_m, secondary_name)
            escalation_record = {
                'primary_run': args.name,
                'primary_map50': primary_map50,
                'secondary_run': secondary_name,
                'secondary_map50': secondary_map50,
                'target_map50': TARGET_MAP50,
                'escalate_margin': args.escalate_margin,
                'improved': secondary_map50 > primary_map50 + 1e-6
            }
            with open(secondary_out_dir / 'escalation_summary.json','w',encoding='utf-8') as f:
                json.dump(escalation_record,f,indent=2)
            escalate_performed = True
        except Exception as e:
            print('Escalation attempt failed:', e)
    if not escalate_performed and primary_map50 < TARGET_MAP50 - args.escalate_margin:
        print('WARNING: mAP50 below threshold; consider manual escalation or dataset audit.')
    print('\nTraining pipeline complete.')
    if escalate_performed:
        print('Escalation summary:', json.dumps(escalation_record, indent=2))

if __name__ == '__main__':
    main()
