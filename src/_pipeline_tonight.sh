#!/bin/bash
# Overnight Stage-A driver (2026-07-09, user: "当前数据全跑完"): wait for the parallel audit, then run the
# entire local data side per rmscaffold/HANDOFF.md §2 — full export, SFT prep, resample pairs (train+eval
# tags, the only proxy steps), RM pair prep + gates, held-out eval pairs, eval items, accounting report.
# Detached via nohup so it survives the Claude session. All output -> dataset/pipeline_tonight.log
cd "<PROJECT_ROOT>" || exit 1
export PYTHONUTF8=1 MSWEA_SILENT_STARTUP=1 MSWEA_COST_TRACKING=ignore_errors PYTHONPATH="$PWD/_wincompat"
PY=".venv/Scripts/python.exe"
LOG_AUDIT="dataset/audit_par.log"

step() { echo ""; echo "########## [$(date '+%H:%M:%S')] $1 ##########"; }

step "S0: wait for audit (resumable self-heal if it stalls)"
RESTARTS=0
while true; do
  if grep -q "audit-par DONE" "$LOG_AUDIT" 2>/dev/null; then
    echo "audit DONE marker found"; break
  fi
  # stale check: log untouched for >20 min while no DONE = audit died -> restart (skips finished ones)
  if [ -f "$LOG_AUDIT" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$LOG_AUDIT") ))
    if [ "$age" -gt 1200 ]; then
      if [ "$RESTARTS" -ge 3 ]; then echo "audit stalled 4x — giving up on the remainder, continuing with what we have"; break; fi
      RESTARTS=$((RESTARTS+1))
      echo "audit log stale ${age}s -> restarting _audit_parallel (attempt $RESTARTS)"
      $PY src/_audit_parallel.py >> "$LOG_AUDIT" 2>&1
    fi
  fi
  sleep 60
done
echo "audited: $(ls dataset/raw/*/pro2/audit.json 2>/dev/null | wc -l) / 136"

step "S1: full export (SFT samples + natural preference pairs)"
$PY src/export.py --run-id pro2 --dataset pro

step "S2: prep_sft (merged LLaMA-Factory training file -> dataset/sft/)"
$PY src/prep_sft.py

step "S3: gen_pairs --list-only (cost estimate, zero calls)"
$PY rmscaffold/gen_pairs.py --run-id pro2 --dataset pro --list-only

step "S4: gen_pairs TRAIN tag (resample pairs, proxy)"
$PY rmscaffold/gen_pairs.py --run-id pro2 --dataset pro

step "S5: gen_pairs EVAL tag (held-out nodes, proxy)"
$PY rmscaffold/gen_pairs.py --run-id pro2 --dataset pro \
    --instances-file dataset/splits/pro_test.json --include-holdout --tag eval

step "S6: prep_rm (merge + gates; kept>=300 gate per HANDOFF)"
$PY rmscaffold/prep_rm.py

step "S7: build_rm_eval (held-out quality-check pairs, both channels)"
$PY rmscaffold/build_rm_eval.py --run-id pro2

step "S8: eval_build (held-out process-eval items, zero proxy)"
$PY src/eval_build.py

step "S9: calibration accounting report"
$PY src/_calib_accounting.py

step "ALL DONE — weekend GPU checklist: upload rmscaffold/ + sft_adapter/ + items.jsonl, run HANDOFF §3"
