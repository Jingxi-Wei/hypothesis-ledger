#!/bin/bash
# Continuation driver (2026-07-09, user: "直接按新的跑,旧的保留"): the original S4 was killed at 191/404 so
# the REMAINING nodes get the patched COMBINED judge (absolute labels + groups/tiers/ranking in one call)
# instead of old-judge-now + full re-judge later. The 191 already-judged nodes keep their absolute labels
# (still fuel the ladder channel + stats); only THEY need the tier re-judge, which backfill B1.5 does — its
# resume logic skips every node the combined judge already wrote into rankings_*.jsonl.
# Output APPENDS to pipeline_tonight.log: backfill's B0 watches that file (mtime staleness + "ALL DONE" grep).
cd "<PROJECT_ROOT>" || exit 1
export PYTHONUTF8=1 MSWEA_SILENT_STARTUP=1 MSWEA_COST_TRACKING=ignore_errors PYTHONPATH="$PWD/_wincompat"
PY=".venv/Scripts/python.exe"

step() { echo ""; echo "########## [$(date '+%H:%M:%S')] $1 ##########"; }

step "S4b: gen_pairs TRAIN resume with COMBINED judge (labels+groups+tiers+ranking, one call/node)"
$PY rmscaffold/gen_pairs.py --run-id pro2 --dataset pro

step "S5: gen_pairs EVAL tag (held-out nodes, combined judge)"
$PY rmscaffold/gen_pairs.py --run-id pro2 --dataset pro \
    --instances-file dataset/splits/pro_test.json --include-holdout --tag eval

step "ALL DONE — continuation finished; backfill takes over (B1 audits -> B1.5 tier re-judge of the pre-patch nodes -> B2 derivations)"
