import os, time, threading, requests, sys
from pathlib import Path

def ensure_server():
    import socket
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", 8000))
        s.close()
        return
    except Exception:
        pass
    def run():
        os.system("uvicorn app.main:app --host 0.0.0.0 --port 8000")
    t = threading.Thread(target=run, daemon=True)
    t.start()
    for _ in range(50):
        try:
            s = socket.socket(); s.connect(("127.0.0.1", 8000)); s.close(); break
        except Exception:
            time.sleep(0.2)

def main(img_path: str):
    ensure_server()
    with open(img_path, 'rb') as f:
        files = {'file': (Path(img_path).name, f, 'image/png')}
        r = requests.post('http://127.0.0.1:8000/process', files=files)
    print("Status:", r.status_code)
    print(r.text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_local.py <image>")
        sys.exit(1)
    main(sys.argv[1])
