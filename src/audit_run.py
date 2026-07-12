"""Standalone, re-runnable posthoc audit over collected trajectories.

Decoupled from collection (which needs docker): the audit is LLM-only, so the prompt can be
iterated and re-run over saved trajectories cheaply.

  python src/audit_run.py --run-id r1                 # (re)audit every collected trajectory
  python src/audit_run.py --run-id r1 -i <instance>   # one instance
"""
import json
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import audit_all, load_instances  # noqa: E402  (load_instances dispatches HF vs tb/lcb adapters)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_id: str = typer.Option("r1", "--run-id"),
    instance: str = typer.Option("", "-i", "--instance", help="single instance id (default: all collected)"),
    dataset: str = typer.Option("verified", "--dataset", help="verified | full | pro | tb | lcb (or HF path)"),
    split: str = typer.Option("test", "--split"),
    overwrite: bool = typer.Option(False, "--overwrite", help="re-audit even if audit.json already exists"),
) -> None:
    # lcb_filtered=False: the audit must cover EVERY collected trajectory — a narrower LCB_AFTER_DATE at audit
    # time than at collection time would otherwise silently skip collected instances (2026-07-07 review)
    ds = load_instances(dataset, split, lcb_filtered=False)
    if instance:
        targets = [instance]
    else:
        targets = sorted(p.name for p in RAW.iterdir() if (p / run_id / "trajectory.json").exists())
    done = skipped = 0
    for iid in targets:
        run_dir = RAW / iid / run_id
        traj_p = run_dir / "trajectory.json"
        if not traj_p.exists():
            continue
        if iid not in ds:  # distinguishable from "no trajectory": a collected instance missing from the loaded
            # dataset means a loader/source mismatch (wrong --dataset flag, changed source file) — say so
            print(f"[skip] {iid}: collected but NOT in the loaded '{dataset}' dataset (check --dataset / source file)", flush=True)
            continue
        if (run_dir / "audit.json").exists() and not overwrite:
            skipped += 1
            continue
        inst = ds[iid]
        traj = json.loads(traj_p.read_text(encoding="utf-8"))
        out_p = run_dir / "outcome.json"
        outcome = json.loads(out_p.read_text())["outcome"] if out_p.exists() else "unknown"
        led_p = run_dir / "ledger.json"  # card-aligned audit needs the ledger -> run compress first
        if not led_p.exists():
            print(f"[skip] {iid}: no ledger.json (run compress first — the audit is card-aligned)", flush=True)
            continue
        cards = json.loads(led_p.read_text(encoding="utf-8")).get("cards", [])
        audit = audit_all(traj["messages"], inst["problem_statement"], inst.get("patch") or "", outcome, cards=cards)
        (run_dir / "audit.json").write_text(audit, encoding="utf-8")
        done += 1
        print(f"[audit] {iid} ({outcome}) audited", flush=True)
    print(f"[audit done] audited={done} skipped={skipped}", flush=True)


if __name__ == "__main__":
    app()
