import os
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
TARGETS = {
    'hero': (DOCS / 'banners' / 'hero-1200x630.webp', (1200, 630)),
    'triptych': (DOCS / 'showcase' / 'triptych-2700x900.webp', (2700, 900)),
    'aruco': (DOCS / 'showcase' / 'aruco-calibration-1200x800.webp', (1200, 800)),
    'seg': (DOCS / 'showcase' / 'segmentation-visual-1200x1200.webp', (1200, 1200)),
    'stl': (DOCS / 'showcase' / 'stl-hero-1200x900.webp', (1200, 900)),
}

# crude mapping of best-looking sources based on filenames
CANDIDATES = {
    'hero': [
        '1. Hero Image.png',
        '01. Hero (futuristic hand + 3D nail).png',
        '01.Hero (futuristic hand  3D nail).png',
    ],
    'triptych': [
        '2. Input → Output Triptych.png',
        '02. Input → Output Triptych.png',
        '2. Input & Output Diagram.png',
    ],
    'aruco': [
        '3. ArUco Marker Calibration.png',
        '3. ArUco Calibration (accuracy story).png',
        '3. ArUco Calibration (accuracy story)1.png',
    ],
    'seg': [
        '4. Segmentation Visualization (clarity).png',
        '4. Segmentation Visualization (clarity)1.png',
        '4. Segmentation Visualization.png',
    ],
    'stl': [
        '5. 3D Model Output STL-OBJ hero.png',
        '5. 3D Model Output STL-OBJ hero1.png',
        '5. 3D Model Output.png',
    ],
}


def pick_first_existing(names):
    for n in names:
        p = ROOT / n
        if p.exists():
            return p
    return None


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def convert_to_webp(src: Path, dest: Path, size: tuple[int, int], quality=90):
    ensure_dir(dest)
    img = Image.open(src).convert('RGB')
    img = img.resize(size, Image.LANCZOS)
    img.save(dest, 'WEBP', quality=quality, method=6)
    # ensure <= 300 KB by reducing quality iteratively
    q = quality
    while dest.stat().st_size > 300_000 and q > 60:
        q -= 5
        img.save(dest, 'WEBP', quality=q, method=6)


def main():
    try:
        import PIL  # noqa
    except Exception:
        raise SystemExit('Pillow is required: pip install pillow')

    produced = []
    for key, (dest, size) in TARGETS.items():
        src = pick_first_existing(CANDIDATES[key])
        if not src:
            continue
        convert_to_webp(src, dest, size)
        produced.append(dest)

    # remove stray root images not in docs or app/static
    keep = set(str(p.resolve()) for p in produced)
    keep.add(str((DOCS / 'diagrams' / 'architecture.svg').resolve()))
    keep.add(str((ROOT / 'app' / 'static' / 'markers' / 'aruco_20mm.png').resolve()))

    for p in ROOT.iterdir():
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.svg'} and str(p.resolve()) not in keep:
            try:
                p.unlink()
            except Exception:
                pass

if __name__ == '__main__':
    main()
