$ErrorActionPreference = "Stop"

Write-Host "=== Toxic Comment Detector Demo Verification ===" -ForegroundColor Cyan

$py = "c:/Users/richa/Downloads/files (1)/.venv/Scripts/python.exe"
if (-not (Test-Path $py)) {
  Write-Host "Python virtual environment not found at .venv." -ForegroundColor Red
  Write-Host "Create it first or run configure Python environment in VS Code."
  exit 1
}

Write-Host "1/4 Installing dependencies..." -ForegroundColor Yellow
& $py -m pip install -r requirements.txt

Write-Host "2/4 Training in fast mode..." -ForegroundColor Yellow
& $py train.py --fast

Write-Host "3/4 Evaluating model..." -ForegroundColor Yellow
& $py evaluate.py

Write-Host "4/4 Running demo predictions..." -ForegroundColor Yellow
& $py predict.py --demo

Write-Host "All checks passed." -ForegroundColor Green
Write-Host "Launch UI with:"
Write-Host "$py -m streamlit run app.py"
