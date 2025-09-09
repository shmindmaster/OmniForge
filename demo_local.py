import os
import time
import threading
import sys
import json
from pathlib import Path

import requests
def ensure_server():
    """Ensure uvicorn app.main:app --host 0.0.0.0 --port 8000 is running."""
    import socket
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", 8000))
        s.close()
        print("✓ Server already running on port 8000")
        return
    except Exception:
        pass
    
    print("Starting server...")
    def run():
        os.system("uvicorn app.main:app --host 0.0.0.0 --port 8000")
    t = threading.Thread(target=run, daemon=True)
    t.start()
    
    for _ in range(50):
        try:
            s = socket.socket(); s.connect(("127.0.0.1", 8000)); s.close(); 
            print("✓ Server started successfully")
            break
        except Exception:
            time.sleep(0.2)

def test_upload(img_path: str):
    """Send POST /process with a sample image and print returned JSON."""
    ensure_server()
    
    print(f"\n--- Testing upload: {img_path} ---")
    
    with open(img_path, 'rb') as f:
        files = {'file': (Path(img_path).name, f, 'image/png')}
        r = requests.post('http://127.0.0.1:8000/process', files=files)
    
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        result = r.json()
        print("✓ Upload successful!")
        print("\nReturned JSON:")
        print(json.dumps(result, indent=2))
        
        print(f"\n--- URLs + Metrics ---")
        print(f"Overlay: {result.get('overlay_url', 'N/A')}")
        print(f"CSV: {result.get('csv_url', 'N/A')}")
        print(f"STL: {result.get('stl_url', 'N/A')}")
        print(f"Backend: {result.get('backend', 'N/A')}")
        print(f"Scale Confidence: {result.get('scale_confidence', 'N/A')}")
        
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\n--- Measurements ---")
            print(f"Length: {metrics.get('length_mm', 'N/A'):.1f} mm")
            print(f"Width (prox/mid/dist): {metrics.get('width_prox_mm', 0):.1f} / {metrics.get('width_mid_mm', 0):.1f} / {metrics.get('width_dist_mm', 0):.1f} mm")
            print(f"mm/px: {metrics.get('mm_per_px', 'N/A')}")
            print(f"Sharpness: {metrics.get('sharpness', 'N/A')}")
    else:
        print("❌ Upload failed")
        print(r.text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_local.py <image_path>")
        print("\nThis script:")
        print("• Ensures uvicorn app.main:app --host 0.0.0.0 --port 8000 is running")
        print("• Sends POST /process with sample images (with and without ArUco)")
        print("• Prints returned JSON (URLs + metrics)")
        sys.exit(1)
    
    test_upload(sys.argv[1])
