"""Deterministic last-mile gate: strip hidden F2P test NAMES from audit prose AND ledger hypotheses.

A graded-test NAME can only come from the harness feedback, so any occurrence in trainable prose is a leak
regardless of who wrote it. Two channels: (a) the auditor's prose (the hardened prompt forbids it, but
compliance is a coin flip), and (b) the AGENT'S OWN stated HYPOTHESIS lines — after the self-rescue feedback
lists failing tests, the agent writes e.g. "TestLoad fails because ..." and compress carries that verbatim
into the ledger, whence it reaches the RULED OUT chain, audit-sample hist, and propose targets. Re-auditing
can never fix (b). This scrub closes both mechanically: per instance, every F2P-derived token is replaced
with a neutral noun in audit prose fields and in ledger cards[].hypothesis. Idempotent, no LLM, safe to run
beside the batch. MUST run AFTER compress (compress regenerates ledgers from the raw trajectory) and BEFORE
export.

  python src/scrub_f2p.py                # scrub every collected instance (r1 + pro1)
"""
import json
import re
import sys
from pathlib import Path

import typer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
DS = {"r1": "princeton-nlp/SWE-bench_Verified", "pro1": "ScaleAI/SWE-bench_Pro",
      "pro2": "ScaleAI/SWE-bench_Pro"}


def _ds_for(run: str) -> str:
    """Unknown run-ids must not KeyError-or-skip (pro2 was silently unscrubable before 2026-07-07):
    fall back by prefix, loudly, so a new run-id is never a silent no-op."""
    if run in DS:
        return DS[run]
    guess = "ScaleAI/SWE-bench_Pro" if run.startswith("pro") else "princeton-nlp/SWE-bench_Verified"
    print(f"[scrub] run '{run}' not in DS map — using {guess} by prefix (add it to DS to pin)", flush=True)
    return guess
_PROSE = ("why_proposed", "flaw_given_info_at_the_time", "should_have_turned", "hypothesis", "how_to_check")
REPL = "the affected behavior"
app = typer.Typer(add_completion=False)


def _jl(x):
    if isinstance(x, list):
        return x
    try:
        return json.loads(x)
    except Exception:
        import ast
        return ast.literal_eval(x)


def f2p_tokens(f2p: list[str]) -> list[str]:
    """Same token derivation as _check_leakage: py test_* names, go Test* names, js sentence titles."""
    toks: set[str] = set()
    for t in f2p:
        toks.update(re.findall(r"\btest_[A-Za-z0-9_]{3,}\b", t))
        toks.update(re.findall(r"\bTest[A-Z][A-Za-z0-9_]{2,}\b", t))
        s = t.strip()
        if " " in s and len(s) >= 15:
            toks.add(s)
    return sorted(toks, key=len, reverse=True)  # longest first so subset tokens don't clip supersets


def _scrub_text(text: str, toks: list[str]) -> tuple[str, int]:
    n = 0
    for tok in toks:
        pat = re.compile((r"\b" if tok[:1].isalnum() else "") + re.escape(tok) + (r"\b" if tok[-1:].isalnum() else ""))
        text, k = pat.subn(REPL, text)
        n += k
    return text, n


@app.command()
def main(run_id: str = typer.Option("", "--run-id", help="r1 | pro1 | (empty = both)")) -> None:
    runs = [run_id] if run_id else list(DS)
    files_changed = fields_changed = 0
    for run in runs:
        ds = {i["instance_id"]: i for i in load_dataset(_ds_for(run), split="test")}
        for p in sorted(RAW.iterdir()):
            if p.name not in ds:
                continue
            toks = f2p_tokens(_jl(ds[p.name].get("FAIL_TO_PASS") or ds[p.name].get("fail_to_pass") or "[]"))
            if not toks:
                continue
            # (a) audit prose
            ap = p / run / "audit.json"
            if ap.exists():
                raw = ap.read_text(encoding="utf-8")
                txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
                try:
                    a = json.loads(txt)
                except Exception:
                    m = re.search(r"\{.*\}", txt, re.DOTALL)  # same tolerance as export._parse_json
                    a = json.loads(m.group(0)) if m else None
                    if a is None:
                        print(f"[scrub] WARN unparseable audit: {run}/{p.name}", flush=True)
                if a is not None:
                    n = 0
                    if isinstance(a.get("overall"), str):
                        a["overall"], k = _scrub_text(a["overall"], toks)
                        n += k
                    for h in a.get("per_hypothesis", []):
                        if not isinstance(h, dict):
                            continue
                        for f in _PROSE:
                            if isinstance(h.get(f), str):
                                h[f], k = _scrub_text(h[f], toks)
                                n += k
                    if n:
                        ap.write_text(json.dumps(a, indent=2, ensure_ascii=False), encoding="utf-8")
                        files_changed += 1
                        fields_changed += n
                        print(f"[scrub] {run}/{p.name}: audit {n} occurrence(s)", flush=True)
            # (b) the agent's own hypothesis lines in the ledger
            lp = p / run / "ledger.json"
            if lp.exists():
                led = json.loads(lp.read_text(encoding="utf-8"))
                n = 0
                for c in led.get("cards", []):
                    if isinstance(c.get("hypothesis"), str):
                        c["hypothesis"], k = _scrub_text(c["hypothesis"], toks)
                        n += k
                if n:
                    lp.write_text(json.dumps(led, indent=2, ensure_ascii=False), encoding="utf-8")
                    files_changed += 1
                    fields_changed += n
                    print(f"[scrub] {run}/{p.name}: ledger {n} occurrence(s)", flush=True)
    print(f"[scrub done] {files_changed} file(s), {fields_changed} occurrence(s)", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    app()
