"""Behavior metrics over rollout runs (zero proxy, deterministic). Run compress on each run-id first.

Reads dataset/raw/<iid>/<run_id>/{outcome.json, ledger.json, trajectory.json} and reports, per run:
  resolve            resolved_first / resolved_retry / resolved_nosubmit / failed_* rates
  hypotheses         distinct HYPOTHESIS cards per episode (does it explore the hypothesis space at all)
  redirect_rate      binary arm: fraction of failed attempts followed by a NEW hypothesis card
                     (the trained move) rather than more of the same
  thrash_rate        unchanged/empty resubmits per failed attempt (the untrained move)
  premature_submit   episodes whose FIRST submit happened with zero self-run tests before it
  self_tests         the agent's own test executions per episode (test-grounding behavior)

Compare arms side by side:  python src/eval_rollout_grade.py --runs rollB_seen_base,rollB_seen_sft
"""
import json
import re
import statistics
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
app = typer.Typer(add_completion=False)

_SUBMIT_MARKS = ("Submission recorded", "All tests pass. Task solved.", "Rollout feedback (binary)",
                 "Out of attempts")


def _msg_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(_msg_text(x) for x in c)
    if isinstance(c, dict):
        return str(c.get("text") or c.get("content") or "")
    return "" if c is None else str(c)


def grade_run(run_id: str) -> dict:
    rows = []
    for p in sorted(RAW.iterdir()):
        rd = p / run_id
        op = rd / "outcome.json"
        if not op.exists():
            continue
        o = json.loads(op.read_text(encoding="utf-8"))
        row = {"iid": p.name, "outcome": o.get("outcome", "?"), "attempts": len(o.get("evals", [])),
               "unchanged": o.get("unchanged_resubmits", 0), "hyps": None, "self_tests": None,
               "premature": None, "redirects": None, "failed_attempts": None}
        lp = rd / "ledger.json"
        tp = rd / "trajectory.json"
        if lp.exists() and tp.exists():
            cards = json.loads(lp.read_text(encoding="utf-8")).get("cards", [])
            msgs = json.loads(tp.read_text(encoding="utf-8")).get("messages", [])
            row["hyps"] = len(cards)
            row["self_tests"] = sum(len(c.get("tests", [])) for c in cards)
            # message indices where a submit verdict came back / where each card was born
            submit_idx = [i for i, m in enumerate(msgs) if any(k in _msg_text(m.get("content")) for k in _SUBMIT_MARKS)]
            card_idx = []
            for c in cards:
                m = re.match(r"msg#(\d+)", str(c.get("raw_ref") or ""))
                card_idx.append(int(m.group(1)) if m else -1)
            if submit_idx:
                first = submit_idx[0]
                tests_before = sum(len(c.get("tests", [])) for c, ci in zip(cards, card_idx) if 0 <= ci < first)
                row["premature"] = tests_before == 0
            fails = [i for i in submit_idx if "Rollout feedback (binary)" in _msg_text(msgs[i].get("content"))]
            row["failed_attempts"] = len(fails)
            row["redirects"] = sum(1 for f in fails if any(ci > f for ci in card_idx))
        rows.append(row)
    n = len(rows) or 1
    out = {"run": run_id, "n": len(rows),
           "outcomes": {k: sum(1 for r in rows if r["outcome"] == k) for k in sorted({r["outcome"] for r in rows})},
           "resolve_rate": round(sum(1 for r in rows if str(r["outcome"]).startswith("resolved")) / n, 3),
           "mean_attempts": round(statistics.fmean(r["attempts"] for r in rows), 2) if rows else 0}
    led = [r for r in rows if r["hyps"] is not None]
    if led:
        out["mean_hypotheses"] = round(statistics.fmean(r["hyps"] for r in led), 2)
        out["mean_self_tests"] = round(statistics.fmean(r["self_tests"] for r in led), 2)
        prem = [r for r in led if r["premature"] is not None]
        if prem:
            out["premature_submit_rate"] = round(sum(r["premature"] for r in prem) / len(prem), 3)
        fa = sum(r["failed_attempts"] or 0 for r in led)
        if fa:
            out["redirect_after_failure_rate"] = round(sum(r["redirects"] or 0 for r in led) / fa, 3)
            out["thrash_rate"] = round(sum(r["unchanged"] for r in led) / fa, 3)
    else:
        out["note"] = "no ledgers — run compress --run-id " + run_id + " first for behavior metrics"
    out["rows"] = rows
    return out


@app.command()
def main(runs: str = typer.Option(..., "--runs", help="comma-separated run-ids to grade (e.g. base,sft arms)"),
         out: str = typer.Option("dataset/eval/rollout_report.json", "--out")) -> None:
    reports = [grade_run(r.strip()) for r in runs.split(",") if r.strip()]
    keys = ["n", "resolve_rate", "mean_attempts", "mean_hypotheses", "mean_self_tests",
            "premature_submit_rate", "redirect_after_failure_rate", "thrash_rate"]
    print(f"{'metric':28s}" + "".join(f"{r['run']:>24s}" for r in reports))
    for k in keys:
        print(f"{k:28s}" + "".join(f"{str(r.get(k, '—')):>24s}" for r in reports))
    for r in reports:
        print(f"  {r['run']} outcomes: {r['outcomes']}")
    op = ROOT / out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved -> {op}]")


if __name__ == "__main__":
    app()
