import argparse
import os
import sys
import zipfile
from pathlib import Path
import urllib.request

# Minimal Roboflow dataset downloader (avoids installing roboflow package)
# Expects an export URL pattern the user can supply (private keys not embedded here).
# For a private project, construct a download URL from Roboflow UI (Export > YOLOv8 PyTorch) and
# set it via --url or ROBOFLOW_EXPORT_URL env var.
# Example public-style URL (placeholder):
# https://universe.roboflow.com/<workspace>/<project>/dataset/<version>/download/yolov8

def download_zip(url: str, dest_zip: Path):
    print(f"Downloading dataset zip from {url} ...")
    urllib.request.urlretrieve(url, dest_zip)
    print("Download complete.")


def extract_zip(zip_path: Path, dest_dir: Path):
    print(f"Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    print("Extraction complete.")


def main():
    parser = argparse.ArgumentParser(description="Download Roboflow dataset export (YOLO format)")
    parser.add_argument('--url', default=os.getenv('ROBOFLOW_EXPORT_URL'), help='Direct Roboflow export download URL')
    parser.add_argument('--out', default='data/roboflow', help='Destination root directory')
    args = parser.parse_args()

    if not args.url:
        print("ERROR: Provide --url or set ROBOFLOW_EXPORT_URL env var to a Roboflow export link.")
        sys.exit(1)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    zip_path = out_root / 'rf_export.zip'
    download_zip(args.url, zip_path)
    extract_zip(zip_path, out_root)
    try:
        zip_path.unlink()
    except OSError:
        pass

    # Attempt to detect nested folder and flatten one level if needed
    subdirs = [p for p in out_root.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / 'train').exists():
        inner = subdirs[0]
        print(f"Flattening inner directory {inner.name}...")
        for item in inner.iterdir():
            target = out_root / item.name
            if not target.exists():
                item.replace(target)
        try:
            inner.rmdir()
        except OSError:
            pass

    print("Done. Verify structure matches data/roboflow/data.yaml (train/ valid/ test/).")

if __name__ == '__main__':
    main()
