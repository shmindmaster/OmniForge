<#
Synopsis: Prepare CUDA-enabled PyTorch inside local venv for this project.
Usage:   .\scripts\setup_cuda_torch.ps1 [-Force] [-CudaTag cu128]
Notes:   Requires NVIDIA driver supporting target CUDA (driver shows 12.9 → cu128 wheel is compatible).
#>
param(
  [switch]$Force,
  [string]$CudaTag = 'cu128'
)

$ErrorActionPreference = 'Stop'
Write-Host "[setup] Starting CUDA PyTorch setup (target tag: $CudaTag)" -ForegroundColor Cyan

if (-not (Test-Path .venv)) {
  Write-Host '[setup] Creating virtual environment .venv'
  python -m venv .venv
}

Write-Host '[setup] Activating venv'
. .\.venv\Scripts\Activate.ps1

function Current-Torch {
  $code = @'
import json
try:
    import torch
    info = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    }
except Exception as e:
    info = {"error": str(e)}
print(json.dumps(info))
'@
  python -c $code
}

if (-not $Force) {
  $info = Current-Torch | ConvertFrom-Json
  if ($info.version -and $info.version -match "\+$CudaTag" -and $info.cuda_available) {
    Write-Host "[setup] CUDA build already present ($($info.version)). Use -Force to reinstall." -ForegroundColor Green
    exit 0
  }
}

Write-Host '[setup] Uninstalling existing torch packages'
python -m pip uninstall -y torch torchvision torchaudio 2>$null | Out-Null

Write-Host "[setup] Installing torch torchvision torchaudio ($CudaTag) from PyTorch index"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CudaTag

Write-Host '[setup] Verifying installation'
$post = Current-Torch | ConvertFrom-Json
if (-not $post.version) { throw 'Torch import failed after install.' }
Write-Host "[setup] Torch version: $($post.version)"
Write-Host "[setup] CUDA available: $($post.cuda_available)"
if (-not $post.cuda_available) { throw 'CUDA not available; verify driver and wheel selection.' }
Write-Host "[setup] Device count: $($post.device_count) -> $($post.devices -join ', ')" -ForegroundColor Green
Write-Host '[setup] Completed successfully.' -ForegroundColor Green
