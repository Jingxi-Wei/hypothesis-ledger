"""Neutralize 'oracle'/'hidden guidance'/'reviewer' references in EXISTING audits.

Deterministic, no LLM: re-attributes eliminations/directions to the FAILURE DIAGNOSIS of the prior attempt —
the one attribution that is SELF-CONSISTENT at inference: the exporter chains each prior hypothesis's audit
FLAW (its diagnosis) into later inputs, so "the failure diagnosis ruled out / pointed to" cites something the
model actually has in context. NOT 'tests' (no test demonstrated the wrong direction), NOT 'oracle' (a learner
has none), and NOT 'reviewer' (the v1 rewrite's framing — also absent at inference; v1 residue baked into
audit.json files is converted here too). Only the PROSE fields are touched (why_proposed / flaw /
should_have_turned / overall / hypothesis) — never the `phase` label or enums.

  python src/rewrite_oracle.py --run-id r1
"""
import json
import re
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
app = typer.Typer(add_completion=False)

# Ordered specific -> general. Each maps an oracle/guidance/reviewer phrase to the diagnosis-frame attribution
# (the prior attempt's failure diagnosis ruled it out / pointed there). Applied case-insensitively with word
# boundaries, leading case preserved.
RULES: list[tuple[str, str]] = [
    # oracle-frame -> diagnosis-frame (one hop)
    ("the first hidden correction", "the first failure diagnosis"),
    ("the second hidden correction", "the second failure diagnosis"),
    ("hidden correction", "failure diagnosis"),
    ("hidden guidance", "the failure diagnosis"),
    ("the oracle guidance", "the failure diagnosis"),
    ("oracle guidance", "failure diagnosis"),
    ("the oracle's wording", "the diagnosis wording"),
    ("oracle's wording", "the diagnosis wording"),
    ("oracle-assisted", "diagnosis-assisted"),
    ("oracle-driven", "diagnosis-driven"),
    ("the oracle had ruled out", "the earlier attempt's failure had ruled out"),
    ("the oracle ruled out", "the earlier attempt's failure ruled out"),
    ("the oracle had already", "the failure diagnosis had already"),
    ("the oracle explicitly", "the failure diagnosis explicitly"),
    ("the oracle told the agent", "the failure diagnosis made clear"),
    ("the oracle pointed", "the failure diagnosis pointed"),
    ("the oracle", "the failure diagnosis"),
    ("redirected the agent", "steered the agent"),
    ("the redirect", "the failure diagnosis"),
    ("a redirect", "a failure diagnosis"),
    ("oracle", "the failure diagnosis"),  # catch-all for any straggler
    # v1 reviewer-frame residue (baked into audit.json by the old rules) -> diagnosis-frame
    ("the reviewer's feedback", "the diagnosis of the failed attempt"),
    ("reviewer's feedback", "the diagnosis of the failed attempt"),
    ("reviewer feedback", "the failure diagnosis"),
    ("a maintainer's review", "the failure diagnosis"),
    ("maintainer's review", "failure diagnosis"),
    ("the reviewer's note", "the failure diagnosis"),
    ("a reviewer's note", "a failure diagnosis"),
    ("reviewer note", "failure-diagnosis note"),
    ("the reviewer's wording", "the diagnosis wording"),
    ("the reviewer had", "the failure diagnosis had"),
    ("the reviewer explicitly", "the failure diagnosis explicitly"),
    ("the reviewer told the agent", "the failure diagnosis made clear"),
    ("the reviewer pointed", "the failure diagnosis pointed"),
    ("the reviewer", "the failure diagnosis"),
    ("review-assisted", "diagnosis-assisted"),
    ("review-driven", "diagnosis-driven"),
    ("reviewer", "diagnosis"),  # catch-all for any straggler
]
_PROSE = ("why_proposed", "flaw_given_info_at_the_time", "should_have_turned", "hypothesis", "how_to_check")


# the bare catch-alls must not mangle legitimate product names: "Oracle database/Linux" prose was really
# corrupted in the corpus (django-11138/15503, vuls — 2026-07-07 review). Guard with a product-context lookahead.
_GUARDED = {
    "oracle": r"\boracle\b(?!\s+(?:database|db|linux|sql|backend|driver|cloud|corporation|java|jdk|instant\s*client|autonomous))",
}


def _apply(text: str) -> str:
    for pat, repl in RULES:
        def f(m: re.Match) -> str:
            return repl[:1].upper() + repl[1:] if m.group(0)[:1].isupper() else repl
        rx = _GUARDED.get(pat, r"\b" + re.escape(pat) + r"\b")  # \b: don't mangle e.g. 'reviewers'
        text = re.sub(rx, f, text, flags=re.IGNORECASE)
    return text


def _clean_audit(raw_text: str) -> tuple[str, int]:
    txt = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    a = json.loads(txt)
    n = 0
    if isinstance(a.get("overall"), str):
        new = _apply(a["overall"])
        n += new != a["overall"]
        a["overall"] = new
    for h in a.get("per_hypothesis", []):
        for k in _PROSE:
            if isinstance(h.get(k), str):
                new = _apply(h[k])
                n += new != h[k]
                h[k] = new
    return json.dumps(a, indent=2, ensure_ascii=False), n


@app.command()
def main(run_id: str = typer.Option("r1", "--run-id"), instance: str = typer.Option("", "-i", "--instance")) -> None:
    targets = [instance] if instance else sorted(p.name for p in RAW.iterdir() if (p / run_id / "audit.json").exists())
    files_changed = fields_changed = 0
    for iid in targets:
        ap = RAW / iid / run_id / "audit.json"
        try:
            cleaned, n = _clean_audit(ap.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] {iid}: {e!r}")
            continue
        if n:
            ap.write_text(cleaned, encoding="utf-8")
            files_changed += 1
            fields_changed += n
            print(f"[rewrite] {iid}: {n} field(s) de-oracled")
    print(f"[rewrite done] {files_changed} files, {fields_changed} fields neutralized")


if __name__ == "__main__":
    app()
