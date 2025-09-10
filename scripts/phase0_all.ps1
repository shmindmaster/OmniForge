param(
  [string]$RunName = "im2fit_final_seg",
  [string]$Model = "yolo11m-seg.pt",
  [switch]$Quick
)

# Root & venv resolution
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..") | Select-Object -ExpandProperty Path
$VenvPath = Join-Path $RepoRoot ".venv"
if (-not (Test-Path $VenvPath)) {
  Write-Host "[ENV] Creating virtual environment .venv" -ForegroundColor Cyan
  python -m venv $VenvPath
}
$Python = Join-Path $VenvPath "Scripts/python.exe"
$Pip = Join-Path $VenvPath "Scripts/pip.exe"
if (-not (Test-Path $Python)) { throw "Virtual environment python not found at $Python" }

Write-Host "--- Phase-0: Environment Hardening (venv) ---"
& $Python -m pip install --upgrade pip | Out-Null
& $Pip install -U ultralytics | Out-Null
& $Pip install onnxruntime-gpu "transformers[torch]" opencv-contrib-python pandas | Out-Null
& $Pip install -r (Join-Path $RepoRoot 'app/requirements.txt') | Out-Null

Write-Host "--- Phase-0: Enabling TF32 + cuDNN benchmark ---"
@'
import torch
try:
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = True
  print('TF32:', torch.backends.cuda.matmul.allow_tf32, 'cuDNN benchmark:', torch.backends.cudnn.benchmark)
except Exception as e:
  print('TF32 setup skipped:', e)
'@ | & $Python -

$DATA_YAML = (Resolve-Path (Join-Path $RepoRoot "data/roboflow/data.yaml")).Path

# Training configuration (base)
$TrainArgs = @{
  "model" = $Model
  "data" = $DATA_YAML
  "imgsz" = 640
  "epochs" = 40
  "patience" = 10
  "batch" = -1
  "workers" = 10
  "cache" = 'disk'
  "optimizer" = 'AdamW'
  "cos_lr" = 'True'
  "close_mosaic" = 15
  "mixup" = 0.10
  "erasing" = 0.20
  "seed" = 42
  "plots" = 'True'
  "name" = $RunName
  "project" = 'runs/segment'
}

if ($Quick) {
  Write-Host "[Quick] Enabling reduced settings for fast CPU smoke run" -ForegroundColor Yellow
  $TrainArgs["epochs"] = 5
  $TrainArgs["batch"] = 8
  $TrainArgs["workers"] = 0
  $TrainArgs["cos_lr"] = 'False'
  $TrainArgs["cache"] = 'ram'
  $TrainArgs["fraction"] = 0.02
  $TrainArgs["amp"] = 'False'
  $TrainArgs["plots"] = 'False'
}

# Convert hashtable to CLI string
$ArgString = ($TrainArgs.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' '

Write-Host "--- Phase-0: Training ($RunName) ---"
& $Python -m ultralytics segment train $ArgString
if ($LASTEXITCODE -ne 0) { throw "Training failed (exit $LASTEXITCODE)" }

Write-Host "--- Phase-0: Validation (val + test) ---"
& $Python -m ultralytics segment val model="runs/segment/$RunName/weights/best.pt" data="$DATA_YAML" retina_masks=True
& $Python -m ultralytics segment val model="runs/segment/$RunName/weights/best.pt" data="$DATA_YAML" split=test retina_masks=True

Write-Host "--- Phase-0: Export ONNX ---"
& $Python -m ultralytics export model="runs/segment/$RunName/weights/best.pt" format=onnx dynamic=True simplify=True opset=14
if (-not (Test-Path (Join-Path $RepoRoot 'model'))) { New-Item -ItemType Directory -Path (Join-Path $RepoRoot 'model') | Out-Null }
Copy-Item -Path "runs/segment/$RunName/weights/best.onnx" -Destination "model/best.onnx" -Force

Write-Host "--- Phase-0: ONNX SHA256 & Providers ---"
@'
from hashlib import sha256
import onnxruntime as ort, pathlib
p=pathlib.Path('model/best.onnx')
if not p.exists():
  raise SystemExit('Missing ONNX: model/best.onnx')
h=sha256(p.read_bytes()).hexdigest()
print('ONNX_SHA256='+h)
providers=[('CUDAExecutionProvider',{'device_id':0}), 'CPUExecutionProvider']
try:
  sess=ort.InferenceSession(str(p), providers=providers)
  print('Providers=', sess.get_providers())
except Exception as e:
  print('Provider init issue:', e)
open('model/onnx_sha256.txt','w',encoding='utf-8').write(h+'\n')
'@ | & $Python -

Write-Host "--- Phase-0: Evaluation & Artifact Generation ---"
$env:USE_ID1_FALLBACK = "1"
& $Python scripts/run_handsy_evaluation.py --run-name $RunName
& $Python scripts/write_methods_note.py --run-name $RunName

Write-Host "--- Phase-0: QC Checks ---"
@'
from pathlib import Path
import csv
out=Path('handsy_out')
if not out.exists():
  raise SystemExit('handsy_out missing')
masks=list(out.glob('*_mask.png'))
over=list(out.glob('*_overlay.png'))
assert len(masks)==5, f"Expected 5 masks got {len(masks)}"
assert len(over)==5, f"Expected 5 overlays got {len(over)}"
with open(out/'measurements.csv','r',encoding='utf-8') as f:
  rows=list(csv.reader(f))
header=rows[0]
needed=['image','nail_index','length_mm','width_prox_mm','width_mid_mm','width_dist_mm','mask_area_px','mm_per_px','scale_confidence','scale_method']
missing=[c for c in needed if c not in header]
assert not missing, f"Missing columns {missing}"
assert len(rows)-1>=20, 'Insufficient data rows'
print('QC PASS: artifact set complete')
'@ | & $Python -

Write-Host "--- Phase-0: COMPLETE ---"
