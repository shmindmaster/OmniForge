import os, random, yaml, cv2, glob, json
import numpy as np

DATA_CFG = 'data/yolo/data.yaml'
OUT_DIR = 'dataset_inspect'
SAMPLES = 12

def load_cfg():
    with open(DATA_CFG, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def read_seg_label(txt_path):
    masks = []
    if not os.path.exists(txt_path):
        return masks
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls = int(parts[0])
            # Remaining are polygon pairs x y normalized (YOLO seg format: cls x y x y ...)
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0: continue
            xs = coords[0::2]
            ys = coords[1::2]
            masks.append({'cls': cls, 'xs': xs, 'ys': ys})
    return masks

def polygon_to_mask(seg, img_w, img_h):
    pts = np.stack([np.array(seg['xs'])*img_w, np.array(seg['ys'])*img_h], axis=1).astype(np.int32)
    m = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return m

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = load_cfg()
    base = cfg.get('path','.')
    train_dir = os.path.join(base, cfg['train'])
    label_dir = train_dir.replace('images','labels')
    images = glob.glob(os.path.join(train_dir, '*.jpg')) + glob.glob(os.path.join(train_dir, '*.png'))
    random.shuffle(images)
    sample = images[:SAMPLES]
    stats = {'total_images': len(images), 'examined': len(sample), 'missing_label':0, 'empty_polygons':0, 'instances':0}
    for img_path in sample:
        img = cv2.imread(img_path)
        if img is None: continue
        h,w = img.shape[:2]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base_name + '.txt')
        segs = read_seg_label(label_path)
        if not os.path.exists(label_path):
            stats['missing_label'] += 1
        overlay = img.copy()
        color = (0,255,0)
        inst_masks=[]
        for seg in segs:
            if not seg['xs']:
                stats['empty_polygons'] += 1
                continue
            m = polygon_to_mask(seg, w, h)
            inst_masks.append(m)
            cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, color, 2)
            stats['instances'] += 1
        out_path = os.path.join(OUT_DIR, base_name + '_overlay.jpg')
        cv2.imwrite(out_path, overlay)
    with open(os.path.join(OUT_DIR,'stats.json'),'w',encoding='utf-8') as f:
        json.dump(stats,f,indent=2)
    print('Inspection complete. Stats:', stats)

if __name__ == '__main__':
    main()
