<#
Convenience wrapper to launch full GPU training with escalation and export.
Usage: .\scripts\train_handsy_gpu.ps1 [-Epochs 50] [-Batch 16] [-Patience 10] [-CloseMosaic 10] [-Name handsy_final_model] [-NoEscalate]
#>
param(
  [int]$Epochs = 50,
  [int]$Batch = 16,
  [int]$Patience = 10,
  [int]$CloseMosaic = 10,
  [string]$Name = 'handsy_final_model',
  [switch]$NoEscalate
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path .venv)) { throw 'Virtual environment (.venv) missing. Run scripts/setup_cuda_torch.ps1 first.' }
. .\.venv\Scripts\Activate.ps1

Write-Host '[train] Verifying torch CUDA build'
python -c "import torch,sys;print('torch',torch.__version__,'cuda_available',torch.cuda.is_available());\n\n(sys.exit(1) if not torch.cuda.is_available() else None)"

$esc = if ($NoEscalate) { '' } else { '--auto_escalate' }
Write-Host '[train] Launching primary training run'
python scripts/train_yolo_seg.py --model yolo11s-seg.pt --data data/yolo/data.yaml `
  --epochs $Epochs --patience $Patience --imgsz 640 --device 0 --close_mosaic $CloseMosaic `
  --batch $Batch --cache --workers 8 --seed 42 --name $Name $esc

if (-not (Test-Path "runs/seg/$Name/weights/best.pt")) { throw 'Primary training did not produce best.pt' }

Write-Host '[train] Exporting ONNX'
python scripts/export_best_onnx.py --weights "runs/seg/$Name/weights/best.pt" --simplify --device 0

Write-Host '[train] Validating ONNX'
python scripts/validate_onnx.py --model model/best.onnx

Write-Host '[train] Generating deliverables'
python scripts/handsy_phase0.py

Write-Host '[train] Complete.' -ForegroundColor Green
