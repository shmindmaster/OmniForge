import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]):
    print(f"[run] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(r.returncode)
    if r.stdout:
        print(r.stdout)
    if r.stderr:
        print(r.stderr, file=sys.stderr)


def detect_latest_run(segment_dir: Path) -> str | None:
    if not segment_dir.exists():
        return None
    runs = [p for p in segment_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    # sort by modified time desc
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name


def main():
    ap = argparse.ArgumentParser(description="Run Phase-0 evaluation (masks, overlays, CSV, methods note)")
    ap.add_argument("--run-name", help="YOLO training run name (folder under runs/segment). If omitted, picks most recent.", default=None)
    ap.add_argument("--handsy-in", default="handsy_in", help="Input images directory")
    ap.add_argument("--handsy-out", default="handsy_out", help="Output directory for artifacts")
    args = ap.parse_args()

    run_name = args.run_name or detect_latest_run(ROOT / "runs" / "segment")
    if not run_name:
        raise SystemExit("Could not determine run name; specify --run-name")

    # Phase-0 artifact generation (handsy_phase0.py already writes measurements + overlays + masks)
    run([sys.executable, "scripts/handsy_phase0.py"])  # handsy_in -> handsy_out

    # Methods note (enhanced script)
    run([sys.executable, "scripts/write_methods_note.py", "--run-name", run_name, "--handsy-out", args.handsy_out])

    print(f"✅ Evaluation complete for run '{run_name}'. See {args.handsy_out}.")


if __name__ == "__main__":
    main()
