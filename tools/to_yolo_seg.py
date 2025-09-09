import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# This script converts a dataset of RGB images and corresponding binary masks into YOLOv8/YOLO11
# segmentation label format (class_id x1 y1 x2 y2 ... normalized polygon coordinates).
# It supports nested structures like: <raw>/train/{images,masks}, <raw>/val/{images,masks}, <raw>/test/{images,masks}
# and will aggregate all discovered pairs then perform a fresh train/val split (default 90/10) unless
# --respect-existing-splits is provided, in which case the existing train/val are kept and 'test' (if present)
# is merged with val.

RNG_SEED = 42
TRAIN_SPLIT = 0.9
MIN_POLY_POINTS = 3  # YOLO requires at least 3 points

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTS = {".png", ".bmp", ".jpg"}
MASK_NAME_HINTS = ["mask", "seg", "label", "nail"]

RAW_DIR: Path = Path('.')  # placeholder, set from args
YOLO_DIR: Path = Path('.')  # placeholder, set from args
CLASS_NAMES: list = ["nail"]
RESPECT_SPLITS = False


def find_image_mask_pairs():
    """Discover image/mask pairs using directory roles and heuristics."""
    image_candidates = []
    mask_candidates = []
    for root, _, files in os.walk(RAW_DIR):
        root_path = Path(root)
        root_parts_lower = {p.lower() for p in root_path.parts}
        for f in files:
            fp = root_path / f
            suffix = fp.suffix.lower()
            stem_lower = fp.stem.lower()
            is_image = suffix in IMAGE_EXTS and ('images' in root_parts_lower)
            is_mask = suffix in MASK_EXTS and ('masks' in root_parts_lower)
            if not is_image and suffix in IMAGE_EXTS:
                is_image = True
            if not is_mask and suffix in MASK_EXTS and any(h in stem_lower for h in MASK_NAME_HINTS):
                is_mask = True
            if is_image:
                image_candidates.append(fp)
            if is_mask:
                mask_candidates.append(fp)

    mask_index = {}
    for m in mask_candidates:
        stem = m.stem.lower()
        variants = {stem}
        for hint in MASK_NAME_HINTS:
            variants.add(stem.replace(f"_{hint}", "").replace(f"-{hint}", ""))
        for v in variants:
            mask_index.setdefault(v, []).append(m)

    pairs = []
    for img in image_candidates:
        stem = img.stem.lower()
        candidate_lists = [mask_index.get(stem, [])]
        for hint in MASK_NAME_HINTS:
            candidate_lists.append(mask_index.get(f"{stem}_{hint}", []))
            candidate_lists.append(mask_index.get(f"{stem}-{hint}", []))
        flat = [m for sub in candidate_lists for m in sub]
        if not flat:
            continue
        chosen = sorted(flat, key=lambda p: p.stat().st_size)[0]
        pairs.append((img, chosen))
    print(f"Discovered {len(pairs)} image/mask pairs (images found={len(image_candidates)}, masks found={len(mask_candidates)})")
    return pairs


def mask_to_polygons(mask_array):
    import cv2  # local import to avoid dependency issues earlier
    # mask_array is 2D binary
    mask_bin = (mask_array > 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < MIN_POLY_POINTS:
            continue
        # Flatten and normalize
        cnt = cnt.squeeze(1)  # Nx2
        if cnt.ndim != 2:
            continue
        polygons.append(cnt)
    return polygons


def simplify_polygon(points, epsilon_ratio=0.002):
    import cv2
    if len(points) < MIN_POLY_POINTS:
        return points
    # Perimeter for dynamic epsilon
    peri = cv2.arcLength(points.reshape(-1,1,2).astype(np.float32), True)
    epsilon = epsilon_ratio * peri
    approx = cv2.approxPolyDP(points.reshape(-1,1,2).astype(np.float32), epsilon, True)
    return approx.reshape(-1,2).astype(np.int32)


def polygon_to_yolo_line(poly, w, h, class_id=0):
    # Normalize coordinates
    norm = []
    for x, y in poly:
        nx = max(0.0, min(1.0, x / w))
        ny = max(0.0, min(1.0, y / h))
        norm.extend([f"{nx:.6f}", f"{ny:.6f}"])
    return f"{class_id} " + " ".join(norm)


def write_split_file_list(pairs):
    random.seed(RNG_SEED)
    random.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_SPLIT)
    return pairs[:n_train], pairs[n_train:]


def ensure_dirs():
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            (YOLO_DIR / split / sub).mkdir(parents=True, exist_ok=True)


def create_data_yaml():
    yaml_path = YOLO_DIR / 'data.yaml'
    class_lines = "\n".join([f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)])
    content = f"""path: {YOLO_DIR.as_posix()}
train: train/images
val: val/images
names:
{class_lines}
"""
    yaml_path.write_text(content, encoding='utf-8')


def process():
    pairs = find_image_mask_pairs()
    if not pairs:
        print("No image-mask pairs found in raw directory.")
        return

    if RESPECT_SPLITS:
        train_pairs = [p for p in pairs if 'train' in {q.lower() for q in p[0].parts}]
        val_pairs = [p for p in pairs if 'val' in {q.lower() for q in p[0].parts} or 'test' in {q.lower() for q in p[0].parts}]
    else:
        train_pairs, val_pairs = write_split_file_list(pairs)

    ensure_dirs()

    stats = {"train": 0, "val": 0, "labels_train": 0, "labels_val": 0}

    for split, spairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, mask_path in spairs:
            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                mask = Image.open(mask_path).convert('L')
                mask_arr = np.array(mask)
                polygons = mask_to_polygons(mask_arr)
                lines = []
                for poly in polygons:
                    poly = simplify_polygon(poly)
                    if len(poly) < MIN_POLY_POINTS:
                        continue
                    line = polygon_to_yolo_line(poly, w, h, 0)
                    if line.count(' ') < 6:
                        continue
                    lines.append(line)
                if not lines:
                    ys, xs = np.where(mask_arr > 0)
                    if xs.size and ys.size:
                        bbox_poly = np.array([
                            [xs.min(), ys.min()],
                            [xs.max(), ys.min()],
                            [xs.max(), ys.max()],
                            [xs.min(), ys.max()],
                        ], dtype=np.int32)
                        line = polygon_to_yolo_line(bbox_poly, w, h, 0)
                        lines.append(line)
                    else:
                        continue

                out_img = YOLO_DIR / split / 'images' / img_path.name
                img.save(out_img, quality=95)
                label_name = img_path.with_suffix('.txt').name
                out_label = YOLO_DIR / split / 'labels' / label_name
                out_label.write_text('\n'.join(lines), encoding='utf-8')
                stats[split] += 1
                stats['labels_' + split] += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    create_data_yaml()

    print("Conversion complete.")
    print(json.dumps(stats, indent=2))
    example_labels = list((YOLO_DIR / 'train' / 'labels').glob('*.txt'))
    if example_labels:
        print("Example label file (first lines):")
        print(example_labels[0].read_text().splitlines()[:3])


def parse_args():
    parser = argparse.ArgumentParser(description="Convert binary masks to YOLO segmentation format")
    parser.add_argument('--raw-dir', required=True, help='Root raw dataset directory (contains train/val/test or images/masks)')
    parser.add_argument('--out-dir', required=True, help='Output YOLO dataset directory')
    parser.add_argument('--class-name', default='nail', help='Single class name')
    parser.add_argument('--train-split', type=float, default=TRAIN_SPLIT, help='Train split ratio (ignored if --respect-existing-splits)')
    parser.add_argument('--respect-existing-splits', action='store_true', help='Use existing train/val/test splits instead of re-splitting')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    RAW_DIR = Path(args.raw_dir).resolve()
    YOLO_DIR = Path(args.out_dir).resolve()
    CLASS_NAMES = [args.class_name]
    TRAIN_SPLIT = args.train_split
    RESPECT_SPLITS = args.respect_existing_splits
    process()
