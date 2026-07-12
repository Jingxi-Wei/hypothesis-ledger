"""Full processing pipeline: collected trajectories -> SFT-ready training file. Run AFTER collection.

  python src/process_all.py --run-id r1

Steps (each is also runnable standalone):
  1. compress       Layer0 trajectory -> Layer1 ledger (grep RESULTS + distractor reads compressed)
  2. rewrite_oracle de-oracle audits made before the prompt fix (deterministic; self-contained reasoning)
  3. fetch_code     read the gold-patch source regions from each container (so the fix sample isn't blind)
  4. export         Layer3 training views: audit / propose (grounded in evidence) / fix (with code), source-tagged
  5. prep_sft       combine the 3 views -> one LLaMA-Factory sharegpt file
"""
import os
import subprocess
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
app = typer.Typer(add_completion=False)

# (script, passes --run-id?, passes --dataset?)
STEPS: list[tuple[str, bool, bool]] = [
    ("compress.py", True, False),
    ("rewrite_oracle.py", True, False),
    ("scrub_f2p.py", True, False),   # AFTER compress (ledgers regenerated), BEFORE export: strips graded-test names
    ("fetch_code.py", True, True),   # from audit prose + the agent's own hypothesis lines (harness-feedback leak)
    ("export.py", True, True),
    ("prep_sft.py", False, False),
]


@app.command()
def main(run_id: str = typer.Option("r1", "--run-id"),
         dataset: str = typer.Option("verified", "--dataset", help="verified | full | pro (forwarded to fetch_code + export)"),
         skip: str = typer.Option("", "--skip", help="comma-separated script names to skip, e.g. 'fetch_code.py'")) -> None:
    env = {**os.environ, "PYTHONUTF8": "1", "MSWEA_SILENT_STARTUP": "1", "MSWEA_COST_TRACKING": "ignore_errors",
           "PYTHONPATH": str(ROOT / "_wincompat")}
    skips = {s.strip() for s in skip.split(",") if s.strip()}
    for script, takes_run_id, takes_dataset in STEPS:
        if script in skips:
            print(f"\n===== {script} (skipped) =====", flush=True)
            continue
        print(f"\n===== {script} =====", flush=True)
        args = [PY, str(ROOT / "src" / script)] + (["--run-id", run_id] if takes_run_id else []) \
            + (["--dataset", dataset] if takes_dataset else [])
        if subprocess.run(args, env=env).returncode != 0:
            print(f"\n[process_all] {script} FAILED — stopping.", flush=True)
            raise typer.Exit(1)
    print("\n[process_all] pipeline complete -> dataset/sft/train_sharegpt.jsonl", flush=True)


if __name__ == "__main__":
    app()
