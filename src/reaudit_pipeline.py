"""Detached one-shot orchestrator: re-audit flagged instances -> rewrite_oracle -> export -> prep_sft -> leak check.

Launched detached (survives session/SSH boundary) and logs to dataset/reaudit_pipeline.log. Each step is a
subprocess so a crash is contained + the log shows exactly where it stopped. Re-audit is resumable, so re-launching
this after a crash continues from where it left off.

  python src/reaudit_pipeline.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
ENV = {**os.environ, "PYTHONUTF8": "1", "MSWEA_SILENT_STARTUP": "1",
       "MSWEA_COST_TRACKING": "ignore_errors", "HF_HUB_OFFLINE": "1"}

STEPS = [
    # compress FIRST: the v3 audit is card-aligned, so the (deduped) ledger must exist before auditing
    ("compress r1", [PY, "src/compress.py", "--run-id", "r1"], True),
    ("compress pro1", [PY, "src/compress.py", "--run-id", "pro1"], True),
    ("re-audit ALL (card-aligned v3)", [PY, "src/reaudit.py", "--all"], True),
    # rewrite + scrub are FATAL: they are the contamination gates — skipping them and still printing
    # PIPELINE DONE would let resume_after_reaudit ship an unscrubbed train package (2026-07-07 review)
    ("rewrite_oracle r1", [PY, "src/rewrite_oracle.py", "--run-id", "r1"], True),
    ("rewrite_oracle pro1", [PY, "src/rewrite_oracle.py", "--run-id", "pro1"], True),
    ("scrub F2P names", [PY, "src/scrub_f2p.py"], True),         # closes agent-hypothesis + audit-prose name leaks
    ("export r1", [PY, "src/export.py", "--run-id", "r1", "--dataset", "verified"], True),
    ("export pro1", [PY, "src/export.py", "--run-id", "pro1", "--dataset", "pro"], True),
    ("prep_sft", [PY, "src/prep_sft.py"], True),
    ("leak check", [PY, "src/_check_leakage.py"], False),
]


def main() -> None:
    for name, cmd, fatal in STEPS:
        print(f"\n===== {name} =====", flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT), env=ENV).returncode
        print(f"[{name}] exit {rc}", flush=True)
        if rc != 0 and fatal:
            print(f"[ABORT] fatal step '{name}' failed (exit {rc}) — fix and re-launch (re-audit resumes)", flush=True)
            return
    print("\n===== PIPELINE DONE =====", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
