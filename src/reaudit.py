"""Re-audit ONLY the flagged instances with the hardened audit prompt (collect.audit_all forbids hidden-test refs).

Scope = definitive-leak instances (from dataset/_checks/reaudit_targets.json) UNION cannot-derive instances
(correction-round propose whose HYPOTHESIS anchors are ALL ungrounded in the input — recomputed from the current
samples). 2 workers (codex-proxy tolerates ~2 concurrent; audit is LLM-only, no docker). Resumable via a
`.reaudited_v2` marker per instance, so a re-run skips finished ones and retries failures.

  python src/reaudit.py            # re-audit the flagged set (overwrites their audit.json)
  python src/reaudit.py --list     # just print the targets and exit (no proxy calls)
"""
import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path

import typer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import audit_all  # noqa: E402  (LLM-only; pulls ORACLE_MODEL / AUDIT_EFFORT=xhigh)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
SAMP = ROOT / "dataset" / "samples"
DS = {"r1": "princeton-nlp/SWE-bench_Verified", "pro1": "ScaleAI/SWE-bench_Pro"}
app = typer.Typer(add_completion=False)

_BT = re.compile(r"`([^`\n]{2,80})`")
_PA = re.compile(r"\b[\w./\\-]*\w\.(?:py|go|js|ts|tsx|jsx|c|h|cpp|rb|java)\b")
_SN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CA = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_STOP = {"fail_to_pass", "pass_to_pass", "self_rescue", "why_proposed"}


def _anchors(t: str) -> set[str]:
    a = {m.strip() for m in _BT.findall(t) if len(m.strip()) >= 3}
    a |= set(_PA.findall(t))
    a |= {m for m in _SN.findall(t) if len(m) >= 6 and m not in _STOP}
    a |= {m for m in _CA.findall(t) if len(m) >= 6}
    return a


def compute_targets() -> list[tuple[str, str]]:
    tg = json.loads((ROOT / "dataset" / "_checks" / "reaudit_targets.json").read_text(encoding="utf-8"))
    targets = {(r, i) for r, ss in tg["definitive"].items() for i in ss}  # 17 definitive-leak instances
    for run in ("r1", "pro1"):  # + cannot-derive: correction-round hypo with NO anchor grounded in its input
        f = SAMP / run / "propose.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            if s.get("source") not in ("self_rescue", "oracle"):
                continue
            ui, ti = s["messages"][0]["content"], s["messages"][1]["content"]
            h = re.search(r"HYPOTHESIS: (.*?)(?:\nREASONING:|\Z)", ti, re.S)
            a = _anchors(h.group(1) if h else ti)
            if a and sum(1 for t in a if t in ui) == 0:
                targets.add((run, s["instance_id"]))
    return sorted(targets)


@app.command()
def main(list_only: bool = typer.Option(False, "--list", help="print targets and exit"),
         all_instances: bool = typer.Option(False, "--all", help="re-audit EVERY collected instance (card-aligned v3 pass)"),
         workers: int = typer.Option(2, "--workers", help="concurrent proxy calls (proxy tolerates ~2)")) -> None:
    marker = ".reaudited_v3" if all_instances else ".reaudited_v2"
    if all_instances:  # full card-aligned generation: every instance with a trajectory
        targets = sorted({(run, p.name) for p in RAW.iterdir() for run in DS
                          if (p / run / "trajectory.json").exists()})
    else:
        targets = compute_targets()
    by_run = {r: sum(1 for rr, _ in targets if rr == r) for r in DS}
    print(f"[reaudit] {len(targets)} target instances {by_run} (marker {marker})", flush=True)
    if list_only:
        for r, i in targets:
            print(f"  {r}: {i}")
        return

    ds = {run: {i["instance_id"]: i for i in load_dataset(name, split="test")} for run, name in DS.items()}

    def do(job: tuple[str, str]) -> tuple[str, str]:
        run, iid = job
        rd = RAW / iid / run
        if not (rd / "trajectory.json").exists() or iid not in ds[run]:
            return (iid, "SKIP-missing")
        if (rd / marker).exists():
            return (iid, "skip-done")
        if not (rd / "ledger.json").exists():
            return (iid, "SKIP-no-ledger (run compress first)")
        try:
            inst = ds[run][iid]
            traj = json.loads((rd / "trajectory.json").read_text(encoding="utf-8"))
            op = rd / "outcome.json"
            outcome = json.loads(op.read_text())["outcome"] if op.exists() else "unknown"
            cards = json.loads((rd / "ledger.json").read_text(encoding="utf-8")).get("cards", [])
            audit = audit_all(traj["messages"], inst["problem_statement"], inst["patch"], outcome, cards=cards)
            (rd / "audit.json").write_text(audit, encoding="utf-8")
            (rd / marker).write_text(marker, encoding="utf-8")  # marker ONLY on success -> retryable
            return (iid, "reaudited")
        except Exception as e:  # one instance's failure must not kill the batch; no marker -> retried next run
            return (iid, f"ERROR {type(e).__name__}: {str(e)[:120]}")

    n = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in cf.as_completed([ex.submit(do, j) for j in targets]):
            iid, status = fut.result()
            n += 1
            print(f"[{n}/{len(targets)}] {status}: {iid}", flush=True)
    print(f"[reaudit done] processed {n}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    app()
