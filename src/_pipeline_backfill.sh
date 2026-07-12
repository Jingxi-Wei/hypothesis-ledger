#!/bin/bash
# Backfill driver (2026-07-09): 16/136 pro2 audits failed overnight on a TRANSIENT connection error
# ("Streaming POST failed" — network blip, NOT the context-overflow poison signature), so tonight's
# _pipeline_tonight.sh moved on to export with only 120/136 audited. This script waits for that driver
# to finish (so it never competes for the ~2-concurrent proxy slots), retries the 16, and — if any
# recover — folds them into the zero-proxy downstream steps only (export/prep_sft/prep_rm/build_rm_eval/
# eval_build/accounting). Deliberately does NOT re-run gen_pairs (S4/S5, the proxy-heavy resample step):
# any newly-recovered instances will simply lack resample-tag RM pairs until a future gen_pairs run —
# an acceptable, documented gap for a pilot, not worth doubling proxy spend tonight.
cd "<PROJECT_ROOT>" || exit 1
export PYTHONUTF8=1 MSWEA_SILENT_STARTUP=1 MSWEA_COST_TRACKING=ignore_errors PYTHONPATH="$PWD/_wincompat"
PY=".venv/Scripts/python.exe"
MAIN_LOG="dataset/pipeline_tonight.log"
LOG="dataset/pipeline_backfill.log"

step() { echo ""; echo "########## [$(date '+%H:%M:%S')] $1 ##########"; }

step "B0: wait for tonight's main driver to finish (never share the proxy with it)"
while ! grep -q "ALL DONE" "$MAIN_LOG" 2>/dev/null; do
  # driver died without reaching ALL DONE (stale >20min, no PID) -> stop waiting, proceed anyway
  age=$(( $(date +%s) - $(stat -c %Y "$MAIN_LOG" 2>/dev/null || date +%s) ))
  if [ "$age" -gt 2400 ]; then echo "main log stale ${age}s with no ALL DONE — proceeding anyway"; break; fi
  sleep 60
done
echo "main driver finished (or stale) — proceeding with backfill"

step "B1: retry the 16 audits that failed on a transient connection error"
$PY - <<'PYEOF'
import json, sys
sys.path.insert(0, "src")
from collect import audit_all, load_instances

RAW_ROOT = __import__("pathlib").Path("dataset/raw")
ds = load_instances("pro")
retried = recovered = 0
for p in sorted(RAW_ROOT.iterdir()):
    rd = p / "pro2"
    if not (rd / "outcome.json").exists() or (rd / "audit.json").exists():
        continue
    if not (rd / "ledger.json").exists() or p.name not in ds:
        continue
    retried += 1
    try:
        inst = ds[p.name]
        traj = json.loads((rd / "trajectory.json").read_text(encoding="utf-8"))
        outcome = json.loads((rd / "outcome.json").read_text())["outcome"]
        cards = json.loads((rd / "ledger.json").read_text(encoding="utf-8")).get("cards", [])
        audit = audit_all(traj["messages"], inst["problem_statement"], inst.get("patch") or "", outcome, cards=cards)
        (rd / "audit.json").write_text(audit, encoding="utf-8")
        recovered += 1
        print(f"[backfill] RECOVERED {p.name[:55]}", flush=True)
    except Exception as e:
        print(f"[backfill] still failing {p.name[:55]}: {repr(e)[:120]}", flush=True)
print(f"[backfill] retried={retried} recovered={recovered}")
PYEOF
echo "audited now: $(ls dataset/raw/*/pro2/audit.json 2>/dev/null | wc -l) / 136"

step "B1.5: relative re-judge (mechanism groups + ranking) over stored candidates — proxy, ~1 call/node"
# user design correction 2026-07-09: absolute gold-anchored labels flatten no-hint candidates into 'partial';
# the deploy-faithful question is RELATIVE (rank imperfect candidates). Reuses stored candidates, no regen.
$PY rmscaffold/rejudge_rank.py --run-id pro2 --tag train
$PY rmscaffold/rejudge_rank.py --run-id pro2 --tag eval || echo "(eval tag candidates not present — fine)"

step "B2: fold any newly-audited instances into the zero-proxy downstream (export/prep_sft/RM prep/eval_build)"
$PY src/export.py --run-id pro2 --dataset pro
$PY src/prep_sft.py
# redirect-anchored pairs (real stage-best hypothesis vs worse oracle-less resamples — the pair channel the
# user's resample design actually intended; zero proxy, regenerates the FULL file now that S4/S5 are done)
$PY rmscaffold/rederive_real_pairs.py --run-id pro2 --tag train
$PY rmscaffold/rederive_real_pairs.py --run-id pro2 --tag eval || echo "(eval tag files not present yet — fine)"
# cross-group sibling pairs from the relative re-judge (zero proxy; clones never pair)
$PY rmscaffold/derive_rank_pairs.py --run-id pro2 --tag train
$PY rmscaffold/derive_rank_pairs.py --run-id pro2 --tag eval || echo "(eval rankings not present — fine)"
$PY rmscaffold/prep_rm.py
$PY rmscaffold/build_rm_eval.py --run-id pro2
$PY src/eval_build.py

step "B3: refreshed calibration accounting report"
$PY src/_calib_accounting.py

step "BACKFILL DONE"
