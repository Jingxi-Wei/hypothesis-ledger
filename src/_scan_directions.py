"""Posthoc leak scan over INJECTED direction text (zero proxy, read-only).

The generation-time redaction net (collect._redact_hidden) was a silent no-op on Pro until 2026-07-07;
this scans what actually reached the agent: every injected feedback/oracle message in collected
trajectories, checked against the instance's hidden-test tokens (Pro-aware derivation). Run it any time
— e.g. after a batch round that still ran pre-fix code — to turn "probably fine" into a number.

  python src/_scan_directions.py --run-id pro2
"""
import json
import re
from pathlib import Path

import typer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
MARKERS = ("Sanitized test-feedback direction", "guidance round")
app = typer.Typer(add_completion=False)


def _jl(x):
    if isinstance(x, list):
        return x
    try:
        return json.loads(x)
    except Exception:
        import ast
        return ast.literal_eval(x)


def _tokens(names) -> list[str]:
    toks: set[str] = set()
    for t in names:
        t = str(t)
        toks.update(re.findall(r"\btest_[A-Za-z0-9_]{3,}\b", t))
        toks.update(re.findall(r"\bTest[A-Z][A-Za-z0-9_]{2,}\b", t))
        s = t.strip()
        if " " in s and len(s) >= 15:
            toks.add(s)
    return sorted(toks, key=len, reverse=True)


@app.command()
def main(run_id: str = typer.Option("pro2", "--run-id"),
         dataset: str = typer.Option("ScaleAI/SWE-bench_Pro", "--dataset")) -> None:
    ds = {i["instance_id"]: i for i in load_dataset(dataset, split="test")}
    n_traj = n_inj = 0
    hits = []
    for p in sorted(RAW.iterdir()):
        tp = p / run_id / "trajectory.json"
        if not tp.exists() or p.name not in ds:
            continue
        n_traj += 1
        inst = ds[p.name]
        tok_sets = [("F2P", _tokens(_jl(inst.get("fail_to_pass") or inst.get("FAIL_TO_PASS") or "[]"))),
                    ("P2P", _tokens(_jl(inst.get("pass_to_pass") or inst.get("PASS_TO_PASS") or "[]")))]
        for m in json.loads(tp.read_text(encoding="utf-8")).get("messages", []):
            c = m.get("content") or ""
            if not isinstance(c, str) or not any(mk in c for mk in MARKERS):
                continue
            n_inj += 1
            for kind, toks in tok_sets:
                for tk in toks:
                    pat = (r"\b" if tk[:1].isalnum() else "") + re.escape(tk) + (r"\b" if tk[-1:].isalnum() else "")
                    if re.search(pat, c):
                        hits.append((p.name, kind, tk))
    print(f"[scan_directions] {run_id}: {n_traj} trajectories, {n_inj} injected direction messages, "
          f"{len(hits)} hidden-test token hits")
    for h in hits[:20]:
        print("  LEAK:", h)
    if hits:
        print("  -> contaminated trajectories should be RE-COLLECTED (a conditioned trajectory cannot be scrubbed)")


if __name__ == "__main__":
    app()
