# Detached SWE-bench Pro batch runner — launched via Start-Process so it outlives the Claude session.
# Outer loop: a network outage makes run_batch burn through its todo list with PULL-FAIL/errors and
# exit; we re-run it (resume skips done instances) until no round makes progress several times in a row.
# Run-id is a parameter so a fresh sanitized-protocol run (pro2) never sinks into an old contaminated run.
param([string]$RunId = 'pro2')
$ErrorActionPreference = 'Continue'
$root = '<PROJECT_ROOT>'
$env:PYTHONUTF8 = '1'
$env:MSWEA_SILENT_STARTUP = '1'
$env:MSWEA_COST_TRACKING = 'ignore_errors'
$env:PYTHONPATH = "$root\_wincompat"
$env:DATASET = 'pro'
$log = "$root\dataset\batch_$RunId.log"

function Count-Done {
    $n = 0
    Get-ChildItem -Path "$root\dataset\raw\*\$RunId\outcome.json" -ErrorAction SilentlyContinue | ForEach-Object {
        if (Test-Path (Join-Path $_.DirectoryName "relevant_code.json")) { $n += 1 }
    }
    $n
}

# target = however many instances the (frozen) split file actually holds — upstream drift shrank it
# from 731 to 666 once already; a hardcoded number would make the loop wait forever for ghosts
$target = (Get-Content "$root\dataset\splits\pro.json" -Raw | ConvertFrom-Json).Count
$stale = 0
while ($true) {
    $before = Count-Done
    if ($before -ge $target) { Add-Content $log "[loop] all $target done, exiting"; break }
    Add-Content $log "[loop] round start: $before/$target done"
    & "$root\.venv\Scripts\python.exe" "$root\src\run_batch.py" --run-id $RunId --dataset pro --instances-file "$root\dataset\splits\pro.json" --workers 2 *>> $log
    $after = Count-Done
    if ($after -gt $before) {
        $stale = 0          # made progress -> immediately retry the remaining (errored) instances
    } else {
        $stale += 1
        Add-Content $log "[loop] no progress this round (stale $stale/6), sleeping 15 min"
        if ($stale -ge 6) { Add-Content $log "[loop] 6 stale rounds - remaining instances look permanently broken, exiting"; break }
        Start-Sleep -Seconds 900
    }
}
