$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    throw "Ambienti .venv mungon. Ekzekuto setup_project.ps1 një herë."
}

& ".venv\Scripts\python.exe" -m streamlit run app.py

