import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_last_line(p: Path) -> str | None:
    try:
        with p.open('r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            return None
        return lines[-1].strip()
    except FileNotFoundError:
        return None

def main():
    ap = argparse.ArgumentParser(description="Tail YOLO results.csv for a run")
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--interval', type=float, default=60.0, help='Seconds between checks')
    ap.add_argument('--once', action='store_true', help='Print single snapshot and exit')
    args = ap.parse_args()

    results = ROOT / 'runs' / 'segment' / args.run_name / 'results.csv'
    if not results.exists():
        print(f"results.csv not found yet at {results}")
        return

    header_printed = False
    while True:
        txt = results.read_text(encoding='utf-8').strip().splitlines()
        if not txt:
            print('empty results.csv')
        else:
            if not header_printed and len(txt) > 1:
                print('HEADER:', txt[0])
                header_printed = True
            print('LAST  :', txt[-1])
        if args.once:
            break
        time.sleep(args.interval)

if __name__ == '__main__':
    main()
