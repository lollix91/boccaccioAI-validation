# Bits per byte of every model on held-out Italian text (Wikipedia articles created after 2025-06-01).
# Usage: powershell -ExecutionPolicy Bypass -File run_bpb.ps1   (from any folder)
#
# 1. eval\build_heldout.py  builds data\heldout\wiki_it_2025.jsonl if missing (Wikipedia API +
#    13-gram overlap check against the 2023 dump used in our pretraining; network + CPU).
# 2. eval\bpb.py            one run per model of eval\model_meta.json (GPU); models that already
#    have eval\results\<name>\bpb_wiki-it-2025.json are skipped, so Ctrl+C + re-run resumes.
# 3. eval\paper_assets.py   adds BPB and bytes/token to Table B and the \bres macros.
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$corpus = "data\heldout\wiki_it_2025.jsonl"
$dataset = "wiki-it-2025"

if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
        $k, $v = $_ -split '=', 2
        Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
    }
}

$start = Get-Date
if (-not (Test-Path $corpus)) {
    & $py -W ignore eval\build_heldout.py
    if ($LASTEXITCODE -ne 0) { Write-Host "build_heldout failed"; exit 1 }
}

$models = & $py -c "import json; [print(m['name'] + '|' + m['hf']) for m in json.load(open('eval/model_meta.json'))['models']]"
foreach ($line in $models) {
    $name, $hf = $line -split '\|', 2
    if (Test-Path "eval\results\$name\bpb_$dataset.json") { Write-Host "== $name already done"; continue }
    # Gemma-3 has a 262k vocabulary: batch 1 keeps the fp32 logits within 12 GB.
    $bs = if ($name -like "gemma*") { 1 } else { 2 }
    Write-Host "== $name (batch $bs)"
    & $py eval\bpb.py --model $hf --name $name --text-file $corpus --field text --dataset-name $dataset --batch-size $bs
    if ($LASTEXITCODE -ne 0) { Write-Host "bpb failed on $name"; exit 1 }
}

& $py eval\paper_assets.py
if ($LASTEXITCODE -ne 0) { Write-Host "paper_assets failed"; exit 1 }
Write-Host ("Done in {0:hh\:mm\:ss}. Results: eval\results\<model>\bpb_$dataset.json; paper assets regenerated." -f ((Get-Date) - $start))
