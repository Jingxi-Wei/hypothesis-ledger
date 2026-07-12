# Detached batch runner — launched via Start-Process so it outlives the Claude session.
$ErrorActionPreference = 'Continue'
$root = '<PROJECT_ROOT>'
$env:PYTHONUTF8 = '1'
$env:MSWEA_SILENT_STARTUP = '1'
$env:MSWEA_COST_TRACKING = 'ignore_errors'
$env:PYTHONPATH = "$root\_wincompat"
& "$root\.venv\Scripts\python.exe" "$root\src\run_batch.py" --run-id r1 --difficulty "1-4 hours,>4 hours" --workers 2 *>> "$root\dataset\batch_r1.log"
