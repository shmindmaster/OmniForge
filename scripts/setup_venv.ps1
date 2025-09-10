Param(
  [string]$Python = 'python',
  [string]$VenvDir = '.venv'
)

if (-Not (Get-Command $Python -ErrorAction SilentlyContinue)) {
  Write-Error "Python not found on PATH. Supply -Python <path> if installed elsewhere."; exit 1
}

if (-Not (Test-Path $VenvDir)) {
  & $Python -m venv $VenvDir
}

Write-Host "Activating venv $VenvDir" -ForegroundColor Cyan
$activate = Join-Path $VenvDir 'Scripts' 'Activate.ps1'
if (-Not (Test-Path $activate)) { Write-Error "Activation script missing: $activate"; exit 1 }
. $activate

Write-Host "Upgrading pip" -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "Installing core project dependencies" -ForegroundColor Cyan
pip install -r app/requirements.txt

if ($env:ENABLE_TRANSFORMERS -eq '1') {
  Write-Host "Installing optional transformers + pillow for ID-1 fallback" -ForegroundColor Cyan
  pip install transformers pillow
}

Write-Host "Environment ready. To activate later: `n  . $activate" -ForegroundColor Green
