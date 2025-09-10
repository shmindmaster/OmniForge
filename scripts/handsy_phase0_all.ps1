<#
Runs the full Phase-0 pipeline:
  1. (Optional) Train model if no ONNX present
  2. Validate (if training performed)
  3. Export ONNX and place at model/best.onnx
  4. Batch process handsy_in images -> handsy_out artifacts
  5. Generate methods note

Usage:
  pwsh scripts/handsy_phase0_all.ps1 [-ForceTrain] [-Device cpu|0]
  # Example (CPU only):
  pwsh scripts/handsy_phase0_all.ps1 -Device cpu
#>
Param(
  [switch]$ForceTrain,
  [string]$Device = 'cpu'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Cmd {
  param([string]$Cmd)
  Write-Host "→ $Cmd" -ForegroundColor Yellow
  Invoke-Expression $Cmd
}

if (-Not (Test-Path 'model')) { New-Item -ItemType Directory -Path model | Out-Null }

$onnx = 'model/best.onnx'
if ($ForceTrain -or -Not (Test-Path $onnx)) {
  Write-Host "[Train] Starting training because $onnx missing or -ForceTrain supplied" -ForegroundColor Cyan
  if (-Not (Test-Path 'data/roboflow/data.yaml')) {
    Write-Error 'Expected dataset config at data/roboflow/data.yaml'; exit 1
  }
  Invoke-Cmd "yolo segment train model=yolo11m-seg.pt data='data/roboflow/data.yaml' imgsz=640 epochs=100 batch=16 patience=25 device=$Device name=im2fit_rf_y11m"
  Invoke-Cmd "yolo segment val model='runs/segment/im2fit_rf_y11m/weights/best.pt' data='data/roboflow/data.yaml'"
  Invoke-Cmd "yolo export model='runs/segment/im2fit_rf_y11m/weights/best.pt' format=onnx dynamic=True simplify=True opset=13"
  $exported = 'runs/segment/im2fit_rf_y11m/weights/best.onnx'
  if (-Not (Test-Path $exported)) { Write-Error "Exported ONNX not found at $exported"; exit 1 }
  Copy-Item $exported $onnx -Force
  Write-Host "Copied ONNX to $onnx" -ForegroundColor Green
}
else {
  Write-Host "[Skip Train] Found existing $onnx" -ForegroundColor Green
}

if (-Not (Test-Path 'handsy_in')) { Write-Error 'Input folder handsy_in missing'; exit 1 }

Write-Host "[Phase0] Running batch processor" -ForegroundColor Cyan
Invoke-Cmd 'python scripts/handsy_phase0.py'

Write-Host "[Methods] Generating methods note" -ForegroundColor Cyan
Invoke-Cmd 'python scripts/write_methods_note.py'

Write-Host "Done. Review handsy_out folder." -ForegroundColor Green
