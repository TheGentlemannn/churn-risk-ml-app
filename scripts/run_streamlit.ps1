$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& ".\.venv\Scripts\python.exe" -m streamlit run "app\streamlit_app.py" `
    --server.port 8502 `
    --server.headless true `
    --browser.gatherUsageStats false
