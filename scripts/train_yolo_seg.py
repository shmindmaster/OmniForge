import os, json, time, argparse
from datetime import datetime
from pathlib import Path

TARGET_MAP50 = 0.90  # tighten target for nails; adjust if needed

def parse_args():
    ap = argparse.ArgumentParser()
    # Data / model
    ap.add_argument('--data', default='data/yolo/data.yaml')
    ap.add_argument('--model', default='yolo11m-seg.pt')  # start light; escalate if needed
    # Phase 1 (generalization)
    ap.add_argument('--imgsz', type=int, default=832)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--patience', type=int, default=20)
    # Phase 2 (fine-tune for crisp masks)
    ap.add_argument('--ft_epochs', type=int, default=30)
    ap.add_argument('--ft_imgsz', type=int, default=1024)
    ap.add_argument('--ft_lr_scale', type=float, default=0.5)  # shrink LR for FT
    # Loader / compute
    ap.add_argument('--batch', type=int, default=-1, help='-1 = Ultralytics auto-batch')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    # Optim & schedule
    ap.add_argument('--optimizer', default='AdamW')
    ap.add_argument('--cos_lr', action='store_true', default=True)
    ap.add_argument('--lr0', type=float, default=0.003)
    ap.add_argument('--lrf', type=float, default=0.01)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--warmup_epochs', type=float, default=3.0)
    # Augs (phase 1 defaults tuned for nails)
    ap.add_argument('--mosaic', type=float, default=0.70)
    ap.add_argument('--close_mosaic', type=int, default=15, help='disable mosaic in last N epochs (P1)')
    ap.add_argument('--mixup', type=float, default=0.10)
    ap.add_argument('--copy_paste', type=float, default=0.0)
    ap.add_argument('--degrees', type=float, default=5.0)
    ap.add_argument('--translate', type=float, default=0.05)
    ap.add_argument('--scale', type=float, default=0.10)
    ap.add_argument('--shear', type=float, default=0.0)
    ap.add_argument('--perspective', type=float, default=0.0)
    ap.add_argument('--fliplr', type=float, default=0.5)
    ap.add_argument('--flipud', type=float, default=0.0)
    ap.add_argument('--hsv_h', type=float, default=0.01)
    ap.add_argument('--hsv_s', type=float, default=0.40)
    ap.add_argument('--hsv_v', type=float, default=0.40)
    ap.add_argument('--erasing', type=float, default=0.20)
    # caching
    ap.add_argument('--cache', choices=['ram','disk','off'], default='off')
    # Project / naming
    ap.add_argument('--project', default='runs/seg')
    ap.add_argument('--name', default='nails_v11s')
    # Escalation
    ap.add_argument('--auto_escalate', action='store_true', default=True)
    ap.add_argument('--escalate_margin', type=float, default=0.03)
    # Export
    ap.add_argument('--export_onnx', action='store_true', default=True)
    ap.add_argument('--onnx_out', default='model/best.onnx')
    # Data fraction (for smoke tests / rapid iteration)
    ap.add_argument('--fraction', type=float, default=1.0, help='Subset of dataset to use (0< fraction <=1]. Use e.g. 0.05 for 5% quick check.')
    return ap.parse_args()

def _ultra_env_report(device):
    try:
        import torch
        print('[ENV] Torch:', torch.__version__,
              ' CUDA:', torch.cuda.is_available(),
              ' Devices:', torch.cuda.device_count())
        if str(device).lower() not in ('cpu',) and not torch.cuda.is_available():
            raise SystemExit('GPU requested but CUDA not available in this torch build.')
    except Exception as e:
        print('[ENV] Torch inspection failed:', e)

def _seed_everything(seed):
    if seed is None: return
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _metrics_map50(d):
    # Ultralytics uses slightly different keys across versions; grab any mAP50 we find
    for k in ('metrics/seg_mAP50(B)', 'metrics/mAP50(B)', 'metrics/mAP50'):
        if k in d: return float(d[k])
    # fallback: first key containing 'mAP50'
    for k,v in d.items():
        if 'mAP50' in k:
            try: return float(v)
            except: pass
    return 0.0

def _cache_arg(x):
    return {'ram': 'ram', 'disk':'disk', 'off': False}[x]

def run_training(model, run_name, overrides):
    print('\n=== Training run:', run_name, '===')
    print(json.dumps(overrides, indent=2))
    t0 = time.time()
    res = model.train(**overrides)
    dur = time.time() - t0
    rd = getattr(res, 'results_dict', {}) or {}
    # coerce floats
    md = {k: (float(v) if isinstance(v,(int,float,str)) and str(v).replace('.','',1).isdigit() else v) for k,v in rd.items()}
    md['train_duration_sec'] = float(dur)
    md['timestamp_iso'] = datetime.utcnow().isoformat() + 'Z'
    outdir = Path(overrides['project']) / overrides['name']
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'metrics_summary.json').write_text(json.dumps(md, indent=2), 'utf-8')
    (outdir / 'args_used.json').write_text(json.dumps(overrides, indent=2), 'utf-8')
    m50 = _metrics_map50(md)
    print(f"Validation mAP50 ({run_name}): {m50:.3f}")
    print('Best weights at:', outdir / 'weights' / 'best.pt')
    return m50, outdir

def main():
    args = parse_args()
    from ultralytics import YOLO
    _ultra_env_report(args.device); _seed_everything(args.seed)

    # load base model (fallback to m if s missing)
    for cand in (args.model, 'yolo11m-seg.pt', 'yolo11m-seg.pt'):
        try:
            model = YOLO(cand); print(f"Loaded model: {cand}"); break
        except Exception as e:
            print(f"Failed to load {cand}: {e}")
    else:
        raise SystemExit('Unable to load any segmentation model weights.')

    base_aug = dict(
        mosaic=args.mosaic, mixup=args.mixup, copy_paste=args.copy_paste,
        degrees=args.degrees, translate=args.translate, scale=args.scale,
        shear=args.shear, perspective=args.perspective,
        fliplr=args.fliplr, flipud=args.flipud,
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
        erasing=args.erasing,
    )

    base_opt = dict(
        optimizer=args.optimizer, cos_lr=args.cos_lr,
        lr0=args.lr0, lrf=args.lrf, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
    )

    common = dict(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, patience=args.patience,
        batch=args.batch, device=args.device, workers=args.workers,
        cache=_cache_arg(args.cache), project=args.project, verbose=True,
        fraction=args.fraction,
        **base_aug, **base_opt
    )
    if args.close_mosaic: common['close_mosaic'] = args.close_mosaic

    # === Phase 1 ===
    p1_name = args.name + '_p1'
    p1_map, p1_dir = run_training(model, p1_name, {**common, 'name': p1_name})

    # === Phase 2 (fine-tune) ===
    best_p1 = p1_dir / 'weights' / 'best.pt'
    model_ft = YOLO(str(best_p1))
    # turn off heavy augs; bump resolution; reduce LR
    ft_overrides = {
        **common, 'name': args.name + '_p2',
        'imgsz': args.ft_imgsz, 'epochs': args.ft_epochs,
        'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0,
        'close_mosaic': 0,
        'lr0': max(1e-5, args.lr0 * args.ft_lr_scale),
        'erasing': args.erasing * 0.5,
    }
    p2_map, p2_dir = run_training(model_ft, ft_overrides['name'], ft_overrides)

    final_dir = p2_dir if p2_map >= p1_map else p1_dir
    final_pt = final_dir / 'weights' / 'best.pt'
    (final_dir / 'FINAL_SELECTED.txt').write_text(str(final_pt), 'utf-8')
    print(f"\nFinal selection: {final_pt} (mAP50={max(p1_map,p2_map):.3f})")

    # === Optional escalation to 11m if needed ===
    if args.auto_escalate and (max(p1_map, p2_map) < TARGET_MAP50 - args.escalate_margin) and '11m' not in str(args.model):
        print(f"Auto-escalate: best mAP50 {max(p1_map,p2_map):.3f} < target-{args.escalate_margin:.3f}. Trying yolo11m-seg.pt")
        try:
            model_m = YOLO('yolo11m-seg.pt')
            m_name = args.name + '_m_p1'
            m1_map, m1_dir = run_training(model_m, m_name, {**common, 'name': m_name})
            model_m_ft = YOLO(str(m1_dir / 'weights' / 'best.pt'))
            m2_over = {**ft_overrides, 'name': args.name + '_m_p2'}
            m2_map, m2_dir = run_training(model_m_ft, m2_over['name'], m2_over)
            if max(m2_map, m1_map) > max(p1_map, p2_map):
                final_dir, final_pt = (m2_dir if m2_map >= m1_map else m1_dir), (m2_dir if m2_map >= m1_map else m1_dir) / 'weights' / 'best.pt'
                print(f"Escalation improved to {max(m1_map,m2_map):.3f}; selecting {final_pt}")
        except Exception as e:
            print('Escalation failed:', e)

    # === Export final to ONNX ===
    if args.export_onnx:
        try:
            from ultralytics import YOLO as _Y
            exp_model = _Y(str(final_pt))
            onnx_path = Path(args.onnx_out)
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Exporting ONNX → {onnx_path}")
            exp_model.export(format='onnx', dynamic=True, simplify=True, opset=13, imgsz=args.ft_imgsz)
            # Move/rename if needed
            # Ultralytics writes to runs/…; copy best to desired path
            auto_out = next(final_dir.glob('weights/*.onnx'), None)
            if auto_out:
                onnx_path.write_bytes(auto_out.read_bytes())
                print(f"ONNX saved at {onnx_path}")
        except Exception as e:
            print('ONNX export failed:', e)

    print('\nTraining pipeline complete.')

if __name__ == '__main__':
    main()
