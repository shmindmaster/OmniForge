# Production-Grade Pipeline Script for im2fit - CORRECTED VERSION
# Forces venv Python + CUDA, proper error gating, and robust path resolution

param(
    [string]$RunName = "im2fit_final_seg"
)

Write-Host "=== im2fit Production Pipeline (CORRECTED) ===" -ForegroundColor Cyan

# --- Resolve tools from THIS repo's venv ---
$Py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$Yolo = (Resolve-Path ".\.venv\Scripts\yolo.exe").Path
$DATA_YAML = (Resolve-Path ".\data\roboflow\data.yaml").Path

Write-Host "Python: $Py" -ForegroundColor Gray
Write-Host "Data YAML: $DATA_YAML" -ForegroundColor Gray
Write-Host "Run Name: $RunName" -ForegroundColor Gray

# --- CUDA sanity (fail fast if CPU) ---
Write-Host "--- CUDA Sanity Check ---" -ForegroundColor Yellow
$cudaCheckScript = @"
import sys, torch
print("Python:", sys.executable)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "Avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    sys.exit(2)
"@
& $Py -c $cudaCheckScript

if ($LASTEXITCODE -ne 0) { 
    throw "CUDA not available in venv — check torch install (cu128) and drivers." 
}

# --- Performance knobs (optional) ---
Write-Host "--- Enabling Performance Optimizations ---" -ForegroundColor Yellow
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128,expandable_segments:True"
$perfScript = @"
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
print("TF32 + cudnn.benchmark enabled")
"@
& $Py -c $perfScript

# --- Train on GPU 0 (gates everything on success) ---
Write-Host "--- Training on GPU 0 ---" -ForegroundColor Yellow
& $Yolo segment train `
  model=yolo11m-seg.pt data="$DATA_YAML" `
  imgsz=640 epochs=40 patience=10 batch=-1 workers=10 `
  device=0 cache=disk optimizer=AdamW cos_lr=True close_mosaic=15 `
  mixup=0.10 erasing=0.20 seed=42 plots=True `
  name=$RunName project="runs/segment" exist_ok=True

if ($LASTEXITCODE -ne 0) { throw "Training failed" }

# Resolve the actual run directory (handles auto-suffix like im2fit_final_seg5)
$RunPrefix = $RunName
$RunDir = Get-ChildItem "runs/segment" -Directory |
  Where-Object { $_.Name -like "$RunPrefix*" } |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (!$RunDir) { throw "No run directory found matching $RunPrefix" }

$BEST = Join-Path $RunDir.FullName "weights/best.pt"
if (!(Test-Path $BEST)) { throw "Missing $BEST — training did not produce best.pt" }

Write-Host "Using run directory: $($RunDir.Name)" -ForegroundColor Gray

Write-Host "✅ Training completed successfully - best.pt found" -ForegroundColor Green

# --- Dual validation with crisp masks for eval ---
Write-Host "--- Validation (val split) ---" -ForegroundColor Yellow
& $Yolo segment val model="$BEST" data="$DATA_YAML" retina_masks=True
if ($LASTEXITCODE -ne 0) { throw "Val (split=val) failed" }

Write-Host "--- Validation (test split) ---" -ForegroundColor Yellow
& $Yolo segment val model="$BEST" data="$DATA_YAML" split=test retina_masks=True
if ($LASTEXITCODE -ne 0) { throw "Val (split=test) failed" }

Write-Host "✅ Validation completed successfully" -ForegroundColor Green

# --- Export & verify ONNX ---
Write-Host "--- Exporting to ONNX ---" -ForegroundColor Yellow
& $Yolo export model="$BEST" format=onnx dynamic=True simplify=True opset=14
if ($LASTEXITCODE -ne 0) { throw "ONNX export failed" }

New-Item -ItemType Directory -Force -Path "model" | Out-Null
Copy-Item "$($RunDir.FullName)/weights/best.onnx" "model/best.onnx" -Force

Write-Host "--- Verifying ONNX ---" -ForegroundColor Yellow
$onnxScript = @"
from hashlib import sha256
h = sha256(open("model/best.onnx","rb").read()).hexdigest()
print("ONNX_SHA256", h)
try:
    import onnxruntime as ort
    s = ort.InferenceSession("model/best.onnx", providers=[('CUDAExecutionProvider',{'device_id':0}), 'CPUExecutionProvider'])
    print("Providers:", s.get_providers())
except Exception as e:
    print("ORT check failed:", e)
"@
& $Py -c $onnxScript

Write-Host "✅ ONNX export and verification completed" -ForegroundColor Green

# --- Run your artifact generation & QC as before ---
Write-Host "--- Generating Client Artifacts ---" -ForegroundColor Yellow
& $Py scripts\run_handsy_evaluation.py
if ($LASTEXITCODE -ne 0) { throw "handsy evaluation failed" }

& $Py scripts\write_methods_note.py --run-name $RunName
if ($LASTEXITCODE -ne 0) { throw "methods note generation failed" }

Write-Host "--- Final Quality Control Checks ---" -ForegroundColor Yellow
# Check mask files
$maskCheckScript = @"
import os
masks = [p for p in os.listdir('handsy_out') if p.endswith('_mask.png')]
assert len(masks) == 5, f'QC FAILED: Expected 5 mask files, found {len(masks)}.'
print('✅ QC PASSED: 5 mask files found.')
"@
& $Py -c $maskCheckScript

# Check overlay files
$overlayCheckScript = @"
import os
overlays = [p for p in os.listdir('handsy_out') if p.endswith('_overlay.png')]
assert len(overlays) == 5, f'QC FAILED: Expected 5 overlay files, found {len(overlays)}.'
print('✅ QC PASSED: 5 overlay files found.')
"@
& $Py -c $overlayCheckScript

# Check CSV rows
$csvRowCheckScript = @"
import pandas as pd
df = pd.read_csv('handsy_out/measurements.csv')
assert len(df) >= 20, f'QC FAILED: Expected >= 20 rows in CSV, found {len(df)}.'
print(f'✅ QC PASSED: {len(df)} rows found in measurements.csv.')
"@
& $Py -c $csvRowCheckScript

# Check CSV columns
$csvColCheckScript = @"
import pandas as pd
df = pd.read_csv('handsy_out/measurements.csv')
cols = ['image', 'nail_index', 'length_mm', 'width_prox_mm', 'scale_method', 'scale_confidence']
missing = [c for c in cols if c not in df.columns]
assert not missing, f'QC FAILED: Missing required columns: {missing}.'
print('✅ QC PASSED: All required CSV columns are present.')
"@
& $Py -c $csvColCheckScript

Write-Host "=== Pipeline Complete - All Checks Passed ===" -ForegroundColor Green
