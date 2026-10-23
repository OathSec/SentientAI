# Builds dist\FuzzyWorker.exe (one file, no Python needed to run it).
# Run from this folder:   powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Stop"
python -m venv .buildenv
.\.buildenv\Scripts\python -m pip install --quiet -r requirements.txt pyinstaller
.\.buildenv\Scripts\pyinstaller --noconfirm --onefile --windowed --name FuzzyWorker `
    --collect-submodules fuzzy_worker fuzzy_worker\__main__.py
Write-Host "Built: $PWD\dist\FuzzyWorker.exe"
