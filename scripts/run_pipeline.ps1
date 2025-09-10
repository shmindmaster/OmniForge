# Production-Grade Pipeline Script for im2fit
# Equivalent to the Makefile but optimized for Windows PowerShell

param(
    [string]$RunName = "im2fit_final_seg",
    [switch]$Setup,
    [switch]$Train,
    [switch]$Validate,
    [switch]$ExportOnnx,
    [switch]$RunEvaluation,
    [switch]$QC,
    [switch]$All
)

# Set performance environment variables
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128,expandable_segments:True"

# Root directory resolution
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$DATA_YAML = (Resolve-Path (Join-Path $RepoRoot "data/roboflow/data.yaml")).Path

Write-Host "=== im2fit Production Pipeline ===" -ForegroundColor Cyan
Write-Host "Repository Root: $RepoRoot" -ForegroundColor Gray
Write-Host "Data YAML: $DATA_YAML" -ForegroundColor Gray
Write-Host "Run Name: $RunName" -ForegroundColor Gray

# Function to run setup
function Invoke-Setup {
    Write-Host "--- Upgrading Dependencies & Verifying Environment ---" -ForegroundColor Yellow
    python -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True; print('TF32+benchmark enabled')"
    pip install -U ultralytics
    pip install onnxruntime-gpu opencv-contrib-python pandas "transformers[torch]"
}

# Function to run training
function Invoke-Train {
    Write-Host "--- Starting Final Optimized Training Run: $RunName ---" -ForegroundColor Yellow
    yolo segment train model=yolov11m-seg.pt data="$DATA_YAML" `
        imgsz=640 epochs=40 patience=10 batch=-1 device=0 workers=10 `
        cache=disk optimizer=AdamW cos_lr=True close_mosaic=15 deterministic=False `
        mixup=0.10 erasing=0.20 seed=42 plots=True name=$RunName project="runs/segment"
    
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }
}

# Function to run validation
function Invoke-Validate {
    Write-Host "--- Validating model on 'val' and 'test' splits ---" -ForegroundColor Yellow
    yolo segment val model="runs/segment/$RunName/weights/best.pt" data="$DATA_YAML" split=val retina_masks=True
    yolo segment val model="runs/segment/$RunName/weights/best.pt" data="$DATA_YAML" split=test retina_masks=True
}

# Function to export ONNX
function Invoke-ExportOnnx {
    Write-Host "--- Exporting model to ONNX and Verifying CUDA Provider ---" -ForegroundColor Yellow
    yolo export model="runs/segment/$RunName/weights/best.pt" format=onnx dynamic=True simplify=True opset=14
    
    if (-not (Test-Path "model")) {
        New-Item -ItemType Directory -Path "model" | Out-Null
    }
    
    Copy-Item "runs\segment\$RunName\weights\best.onnx" "model\best.onnx" -Force
    
    python -c @"
from hashlib import sha256
import onnxruntime as ort
p = 'model/best.onnx'
print(f'ONNX_SHA256= {sha256(open(p, "rb").read()).hexdigest()}')
sess = ort.InferenceSession(p, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print(f'Providers: {sess.get_providers()}')
"@
}

# Function to run evaluation
function Invoke-RunEvaluation {
    Write-Host "--- Generating client artifacts in 'handsy_out' folder ---" -ForegroundColor Yellow
    python scripts/run_handsy_evaluation.py
    python scripts/write_methods_note.py --run-name $RunName
}

# Function to run QC checks
function Invoke-QC {
    Write-Host "--- Performing Final Quality Control Checks ---" -ForegroundColor Yellow
    
    # Check mask files
    python -c @"
import os
masks = [p for p in os.listdir('handsy_out') if p.endswith('_mask.png')]
assert len(masks) == 5, f'QC FAILED: Expected 5 mask files, found {len(masks)}.'
print('✅ QC PASSED: 5 mask files found.')
"@
    
    # Check overlay files
    python -c @"
import os
overlays = [p for p in os.listdir('handsy_out') if p.endswith('_overlay.png')]
assert len(overlays) == 5, f'QC FAILED: Expected 5 overlay files, found {len(overlays)}.'
print('✅ QC PASSED: 5 overlay files found.')
"@
    
    # Check CSV rows
    python -c @"
import pandas as pd
df = pd.read_csv('handsy_out/measurements.csv')
assert len(df) >= 20, f'QC FAILED: Expected >= 20 rows in CSV, found {len(df)}.'
print(f'✅ QC PASSED: {len(df)} rows found in measurements.csv.')
"@
    
    # Check CSV columns
    python -c @"
import pandas as pd
df = pd.read_csv('handsy_out/measurements.csv')
cols = ['image', 'nail_index', 'length_mm', 'width_prox_mm', 'scale_method', 'scale_confidence']
missing = [c for c in cols if c not in df.columns]
assert not missing, f'QC FAILED: Missing required columns: {missing}.'
print('✅ QC PASSED: All required CSV columns are present.')
"@
}

# Main execution logic
try {
    if ($All -or (-not $Setup -and -not $Train -and -not $Validate -and -not $ExportOnnx -and -not $RunEvaluation -and -not $QC)) {
        Write-Host "Running complete pipeline..." -ForegroundColor Green
        Invoke-Setup
        Invoke-Train
        Invoke-Validate
        Invoke-ExportOnnx
        Invoke-RunEvaluation
        Invoke-QC
    } else {
        if ($Setup) { Invoke-Setup }
        if ($Train) { Invoke-Train }
        if ($Validate) { Invoke-Validate }
        if ($ExportOnnx) { Invoke-ExportOnnx }
        if ($RunEvaluation) { Invoke-RunEvaluation }
        if ($QC) { Invoke-QC }
    }
    
    Write-Host "=== Pipeline Complete ===" -ForegroundColor Green
} catch {
    Write-Host "=== Pipeline Failed ===" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
