#!/bin/bash
# TB collection driver (2026-07-09, user: "做完resample串行terminal数据采集"). Chains AFTER the backfill
# (not merely after S4b/S5): backfill's B1 (16 audit retries) + B1.5 (~192-node tier re-judge) also burn the
# serial proxy, so TB waits for the BACKFILL DONE marker. Collection itself follows run_loop_pro's pattern
# (resumable rounds until no progress) — run_batch skips completed instances, skip.json quarantines poison.
# Target = 89 TB tasks minus 4 held-out (tb_test.json) = 85. Images are per-task dockerhub (some multi-GB:
# caffe/doom/povray) — pull_timeout 7200 is set in collect_one; prune-images keeps disk bounded.
cd "<PROJECT_ROOT>" || exit 1
export PYTHONUTF8=1 MSWEA_SILENT_STARTUP=1 MSWEA_COST_TRACKING=ignore_errors PYTHONPATH="$PWD/_wincompat"
PY=".venv/Scripts/python.exe"
BF_LOG="dataset/pipeline_backfill.log"

step() { echo ""; echo "########## [$(date '+%H:%M:%S')] $1 ##########"; }

step "T0: wait for backfill (proxy must be free before agents start solving)"
# liveness by PROCESS, not log mtime — backfill is SILENT while it waits in B0, so a stale log means
# nothing (first version of this check fired instantly and nearly collided TB with S4b, 2026-07-09).
alive_backfill() {
  powershell.exe -NoProfile -Command \
    '(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "bash.exe" -and $_.CommandLine -match "_pipeline_backfill" } | Measure-Object).Count' \
    2>/dev/null | tr -d '[:space:]'
}
RELAUNCHES=0
while ! grep -q "BACKFILL DONE" "$BF_LOG" 2>/dev/null; do
  n=$(alive_backfill)
  if [ "${n:-0}" -eq 0 ] 2>/dev/null; then
    if [ "$RELAUNCHES" -lt 2 ]; then
      RELAUNCHES=$((RELAUNCHES+1))
      echo "[T0] backfill process gone without DONE marker -> relaunching (attempt $RELAUNCHES; it is resumable)"
      nohup bash src/_pipeline_backfill.sh >> "$BF_LOG" 2>&1 &
    else
      echo "[T0] backfill died twice — falling back: wait for continuation's ALL DONE, then proceed without backfill"
      while ! grep -q "ALL DONE" dataset/pipeline_tonight.log 2>/dev/null; do sleep 120; done
      echo "[T0] continuation finished; proxy free; TB proceeds (run backfill manually later — data-side only)"
      break
    fi
  fi
  sleep 120
done

count_done() { ls dataset/raw/tb__*/tb1/outcome.json 2>/dev/null | wc -l; }

# TB target (user 2026-07-10, revised): do ALL hard tasks (the reverse/attack-class ones are where gpt-5.5
# may actually mis-hypothesize and get pushed into oracle-correction — the data we actually want), then fill
# to 50 with medium/easy. Ordering (tb_hard_first.json) front-loads reverse/attack hard. Loop exits when
# both conditions hold: >=50 total AND all 30 hard attempted (so hard is never left unfinished by the count).
TB_TARGET=50
hard_done() { python - <<'PY' 2>/dev/null
import json, glob, os, sys
sys.path.insert(0, "src")
import tb
insts = {i["instance_id"]: i["difficulty"] for i in tb.load_instances()}
done = {os.path.basename(os.path.dirname(os.path.dirname(f))) for f in glob.glob("dataset/raw/tb__*/tb1/outcome.json")}
print(sum(1 for i in done if insts.get(i)=="hard"))
PY
}

step "T1: Terminal-Bench collection rounds (run tb1, target $TB_TARGET, 2 workers)"
STALE=0
while true; do
  BEFORE=$(count_done)
  # 49 = all 29 achievable hard (model-extraction-relu-logits skip-listed: proxy content-policy refusal)
  # + 19 medium + 1 easy, already at/over the original 50 intent. Total-only gate (hard_done's inline python
  # was unreliable under Git Bash); 49 already implies hard is exhausted.
  if [ "$BEFORE" -ge 49 ]; then echo "[tb-loop] done: $BEFORE total (all achievable hard collected)"; break; fi
  echo "[tb-loop] round start: $BEFORE/$TB_TARGET total, ${HD:-?}/30 hard"
  # --limit 20: fresh python per <=20 instances — bounds subprocess churn per process (the 0xC0000142
  # poisoning needs ~10h/10k spawns to develop; short-lived workers never get there). Resume makes the
  # rounds seamless; LauncherBroken (rc>255 streak) additionally hard-exits a poisoned worker mid-round.
  # hard-first ordering (2026-07-10): TB's own hard label showed 0/5 correction so far vs medium 3/15 —
  # n tiny; front-load the remaining 25 hard tasks to test the slice and maximize correction odds.
  $PY src/run_batch.py --dataset tb --run-id tb1 --workers 2 --limit 20 \
      --instances-file dataset/splits/tb_hard_first.json
  AFTER=$(count_done)
  if [ "$AFTER" -gt "$BEFORE" ]; then
    STALE=0
  else
    STALE=$((STALE+1))
    echo "[tb-loop] no progress this round (stale $STALE/3), sleeping 10 min"
    if [ "$STALE" -ge 3 ]; then echo "[tb-loop] 3 stale rounds — remaining tasks look permanently broken, exiting"; break; fi
    sleep 600
  fi
done

step "T2: TB DONE ($(count_done)/$TB_TARGET) — backfill the 19 unaudited self_solved (pro1 18 + r1 1)"
# pro1's existing 85 audits verified CURRENT (card-aligned, full v3 fields) — NO reaudit needed; only the
# never-audited stragglers get processed. compress is deterministic/idempotent, audit_run skips existing.
$PY src/compress.py --run-id pro1
$PY src/compress.py --run-id r1
$PY src/audit_run.py --run-id pro1 --dataset pro
$PY src/audit_run.py --run-id r1 --dataset verified

step "T3: refresh exports (pro1 + r1 + pro2) and the merged SFT training file"
# self_solved data is clean by construction (never saw oracle/feedback — user-confirmed convention);
# r1's raw_leak correction cards are auto-blocked per-card by export's protocol gate, so exporting r1
# is safe: only its clean samples come through.
$PY src/export.py --run-id pro1 --dataset pro
$PY src/export.py --run-id r1 --dataset verified
$PY src/export.py --run-id pro2 --dataset pro
$PY src/prep_sft.py

step "T4: LiveCodeBench collection — EASY ONLY (2026-07-10 observed: med/hard 70-100% chaotic on xhigh —
      algorithm-design failures that direction-level oracle can't rescue, ~23min each to produce a
      no-terminal chaotic. 14 med/hard chaotic already banked = difficulty-ceiling evidence. Only easy
      yields correction (~50%) at reasonable cost. Target = the 38 non-holdout easy.)"
lcb_done() { ls dataset/raw/lcb__*/lcb1/outcome.json 2>/dev/null | wc -l; }
easy_done() { $PY -c "import json,glob,os; e=set(json.load(open('dataset/splits/lcb_easy.json'))); print(sum(1 for f in glob.glob('dataset/raw/lcb__*/lcb1/outcome.json') if os.path.basename(os.path.dirname(os.path.dirname(f))) in e))" 2>/dev/null; }
LSTALE=0
while true; do
  LB=$(easy_done)
  if [ "${LB:-0}" -ge 38 ]; then echo "[lcb-loop] all 38 easy done"; break; fi
  echo "[lcb-loop] round start: $LB/38 easy done"
  $PY src/run_batch.py --dataset lcb --run-id lcb1 --workers 2 --limit 20 \
      --instances-file dataset/splits/lcb_easy.json
  LA=$(lcb_done)
  echo "[lcb-loop] outcome mix so far:"
  $PY -c "
import json, glob
from collections import Counter
c = Counter()
for f in glob.glob('dataset/raw/lcb__*/lcb1/outcome.json'):
    c[json.load(open(f, encoding='utf-8'))['outcome']] += 1
print('   ', dict(c))"
  if [ "$LA" -gt "$LB" ]; then LSTALE=0; else
    LSTALE=$((LSTALE+1))
    if [ "$LSTALE" -ge 3 ]; then echo "[lcb-loop] 3 stale rounds, exiting"; break; fi
    sleep 600
  fi
done

step "CHAIN DONE — TB $(count_done), LCB $(lcb_done)"
