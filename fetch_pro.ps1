# Detached fetch_code runner for Pro — fills relevant_code.json for every collected pro1 instance
# (fix samples need the gold-patch source regions; fetch_code skips instances that already have it).
# Outer loop: re-runs until no instance is missing (also picks up newly collected ones), exits when
# caught up or after 4 stale rounds (network outage that never recovers).
$ErrorActionPreference = 'Continue'
$root = '<PROJECT_ROOT>'
$env:PYTHONUTF8 = '1'
$env:MSWEA_SILENT_STARTUP = '1'
$env:MSWEA_COST_TRACKING = 'ignore_errors'
$env:PYTHONPATH = "$root\_wincompat"

function Count-Missing {
    $done = (Get-ChildItem -Path "$root\dataset\raw\*\pro1\outcome.json" -ErrorAction SilentlyContinue | Measure-Object).Count
    $have = (Get-ChildItem -Path "$root\dataset\raw\*\pro1\relevant_code.json" -ErrorAction SilentlyContinue | Measure-Object).Count
    $done - $have
}

$stale = 0
while ($true) {
    $before = Count-Missing
    if ($before -le 0) { Add-Content "$root\dataset\fetch_pro.log" "[fetch-loop] caught up, exiting"; break }
    Add-Content "$root\dataset\fetch_pro.log" "[fetch-loop] round start: $before missing"
    & "$root\.venv\Scripts\python.exe" "$root\src\fetch_code.py" --run-id pro1 --dataset pro *>> "$root\dataset\fetch_pro.log"
    $after = Count-Missing
    if ($after -lt $before) {
        $stale = 0
    } else {
        $stale += 1
        Add-Content "$root\dataset\fetch_pro.log" "[fetch-loop] no progress (stale $stale/4), sleeping 15 min"
        if ($stale -ge 4) { Add-Content "$root\dataset\fetch_pro.log" "[fetch-loop] 4 stale rounds, exiting"; break }
        Start-Sleep -Seconds 900
    }
}
