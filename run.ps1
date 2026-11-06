# Evaluate the queue on this machine, one model at a time.
# Usage: powershell -File run.ps1        (from the repo root, or double-click)
# Ctrl+C interrupts; run it again later and it resumes after the last finished model.
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# Load .env (KEY=value per line, e.g. HF_TOKEN=...)
if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
        $k, $v = $_ -split '=', 2
        Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
    }
}

while ($true) {
    & $py eval\run_queue.py --status
    & $py eval\run_queue.py
    if ($LASTEXITCODE -eq 7) { break }
}
Write-Host "Done: results are in eval\results\<model>\"
