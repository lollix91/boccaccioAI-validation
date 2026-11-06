# Paired significance tests of the distilled BoccaccioAI against every other model.
# Usage: powershell -ExecutionPolicy Bypass -File run_significance.ps1   (from any folder)
#
# 1. eval\collect_samples.py  re-evaluates only the tasks whose per-item samples are missing
#    (distilled: all 9 tasks, ~2h40; base-p2: ARC/HellaSwag/Belebele, ~10 min) into a temporary
#    folder, checks that the metrics are identical to the existing results, then copies the
#    samples. Existing files are never overwritten or deleted; the re-run stays in
#    eval\results\<name>__samples\ (delete by hand if you want). Ctrl+C is safe: re-run to resume.
# 2. eval\significance.py     McNemar / permutation tests + bootstrap CIs (CPU, ~1 min).
# 3. eval\paper_assets.py     regenerates tables and macros for the paper (dagger in Table 4).
#
# Optional: powershell -File run_significance.ps1 -BatchSize 24
#   uses a fixed batch size instead of lm-eval "auto" (safer if you use the GPU for other apps).
param([int]$BatchSize = 0)
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
        $k, $v = $_ -split '=', 2
        Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
    }
}

$start = Get-Date
$extra = @()
if ($BatchSize -gt 0) { $extra += @("--batch-size", "$BatchSize") }
& $py eval\collect_samples.py @extra
if ($LASTEXITCODE -ne 0) { Write-Host "collect_samples failed: see messages above"; exit 1 }
& $py eval\significance.py
if ($LASTEXITCODE -ne 0) { Write-Host "significance failed"; exit 1 }
& $py eval\paper_assets.py
if ($LASTEXITCODE -ne 0) { Write-Host "paper_assets failed"; exit 1 }
Write-Host ("Done in {0:hh\:mm\:ss}. Results: eval\results\significance.json; paper assets regenerated." -f ((Get-Date) - $start))
