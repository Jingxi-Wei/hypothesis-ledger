"""Grade the two arms on the held-out items: mechanical metrics (free, deterministic) + a BLIND fixed-rubric
LLM grader (gpt5.5). Same items, same grader, same rubric for both arms; the grader never sees arm identity.

  python src/eval_grade.py                      # mechanical only (safe while the batch owns the proxy)
  python src/eval_grade.py --llm                # + blind rubric grading (run in a proxy gap: SERIAL calls)
  python src/eval_grade.py --llm --limit 40     # grade a subsample first

THE SCORE = the [llm] content grading, judged against the reference with formatting ignored (free prose ok):
  audit   : judgment_correct (stance agrees with reference verdict), flaw_match (names the SAME problem),
            clear (says plainly what is wrong)
  propose : cause_match (points at the same cause; for expected=probe: admits insufficiency instead of guessing),
            clear
Everything regex-based (format%, verdict_agree, CHECK-line, grounding, premature_*) is DIAGNOSTIC only —
format is worth zero (user decision 2026-07-07).
"""
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "dataset" / "eval"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import _anchors  # noqa: E402  (same anchor definition as the training gate)

app = typer.Typer(add_completion=False)

VERDICT_RE = re.compile(r"VERDICT:\s*(good|weak|wrong)", re.I)
DECLINE_RE = re.compile(r"STILL MISSING|not (?:yet )?enough evidence|evidence (?:is|remains) (?:too )?(?:thin|insufficient)"
                        r"|cannot (?:yet )?(?:name|single out|pin)|insufficient evidence|WHERE TO PROBE", re.I)
HYP_RE = re.compile(r"HYPOTHESIS:\s*(.+?)(?:\nREASONING:|\nCHECK:|\Z)", re.S)

# CONTENT-ONLY grading (user decision 2026-07-07): format is worth zero — free prose is fine. Correct is
# correct: does the answer point at the same problem the reference names, and say it clearly. Graded against
# the reference, blind to which arm produced the text.
RUBRIC_PROPOSE = (
    "You grade ONE candidate answer for a bug-cause task against the ground-truth REFERENCE. Judge CONTENT "
    "ONLY — ignore formatting and section labels entirely; free prose is fine. Answer STRICT JSON "
    "{\"cause_match\": 0 or 1, \"clear\": 0 or 1}. cause_match=1 iff the candidate points at the same "
    "underlying cause/mechanism as REFERENCE.hypothesis (same defect, any wording or granularity); 0 if it "
    "names a different mechanism/location or commits to no cause. EXCEPTION: if expected is 'probe', "
    "cause_match=1 iff the candidate declines to commit and says what is missing / where to look instead of "
    "guessing. clear=1 iff from the candidate text alone a reader can tell exactly WHERE the problem is "
    "claimed to be and WHY. JSON only."
)
RUBRIC_AUDIT = (
    "You grade ONE candidate audit of a hypothesis against the ground-truth REFERENCE. Judge CONTENT ONLY — "
    "ignore formatting and labels; free prose is fine. Answer STRICT JSON {\"judgment_correct\": 0 or 1, "
    "\"flaw_match\": 0 or 1, \"clear\": 0 or 1}. judgment_correct=1 iff the candidate's overall stance "
    "(hypothesis is sound vs flawed/insufficient) agrees with REFERENCE.verdict (good=sound; "
    "weak/wrong=flawed). flaw_match=1 iff the specific problem the candidate names is the same problem "
    "REFERENCE.flaw describes (when REFERENCE.verdict is good: flaw_match=1 iff the candidate does NOT "
    "invent a material flaw). clear=1 iff the candidate states plainly what is wrong (or why it is sound), "
    "so a reader gets the exact issue without guessing. JSON only."
)


def _llm(prompt_sys: str, prompt_user: str) -> dict | None:
    import litellm
    r = litellm.completion(model="openai/gpt5.5", api_base="http://127.0.0.1:8080/v1", api_key="pwd",
                           extra_body={"reasoning_effort": "medium"},
                           messages=[{"role": "system", "content": prompt_sys},
                                     {"role": "user", "content": prompt_user}])
    t = r.choices[0].message.content or ""
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


@app.command()
def main(llm: bool = typer.Option(False, "--llm", help="run the blind rubric grader (serial proxy calls)"),
         limit: int = typer.Option(0, "--limit", help="grade only the first N items with the LLM"),
         outputs: str = typer.Option("", "--outputs", help="path to eval_outputs.jsonl (default dataset/eval/)")) -> None:
    items = {json.loads(l)["item_id"]: json.loads(l)
             for l in (EVAL / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    op = Path(outputs) if outputs else EVAL / "eval_outputs.jsonl"
    # artifact suffix per outputs file: consecutive gradings (rm_vs_random, rm_vs_first, sft arms...)
    # must not clobber each other's rubric_scores.json / mechanical_metrics.json
    tag = f"_{op.stem}" if outputs else ""
    outs = [json.loads(l) for l in op.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"items={len(items)} outputs={len(outs)}")

    M = {arm: defaultdict(list) for arm in ("base", "sft")}
    for o in outs:
        it = items.get(o["item_id"])
        if not it:
            continue
        sch = it.get("schema")           # None = canonical section labels
        xf = "_xfer" if it.get("schema_transfer") else ""
        for arm in ("base", "sft"):
            txt = o.get(f"output_{arm}") or ""
            m = M[arm]
            if it["task"] == "format_bleed":  # OOD prompt: emitting any of OUR section labels = capability regression
                m["field_bleed"].append(bool(re.search(
                    r"\b(VERDICT|HYPOTHESIS|FLAW|NEXT CHECK|SUPPORT|REASONING|STILL MISSING|WHERE TO PROBE NEXT):", txt)))
                continue
            if it["task"] == "audit":
                vlab = sch["VERDICT"] if sch else "VERDICT"  # transfer twin requests a renamed verdict label
                vm = re.search(rf"{re.escape(vlab)}:\s*(good|weak|wrong)", txt, re.I)
                m["audit_format" + xf].append(bool(vm))
                if vm:
                    mv, rv = vm.group(1).lower(), (it["reference"]["verdict"] or "").lower()
                    if rv in ("good", "weak", "wrong"):
                        m["verdict_agree" + xf].append(mv == rv)
                        if not xf:  # reference-severity metrics only on canonical items
                            if rv == "good":
                                m["premature_refute"].append(mv in ("weak", "wrong"))
                            else:
                                m["missed_flaw"].append(mv == "good")
            else:
                hm = HYP_RE.search(txt)
                declined = bool(DECLINE_RE.search(txt)) and not hm
                m["propose_format"].append(bool(hm) or declined)
                m["check_line"].append("CHECK:" in txt or bool(re.search(r"WHERE TO PROBE", txt)))
                if it.get("expected") == "probe":
                    m["premature_guess"].append(bool(hm) and not declined)
                else:
                    m["declined_when_derivable"].append(declined)
                    if hm:  # grounding of the guessed hypothesis against the input it saw
                        A = _anchors(hm.group(1))
                        # zero-anchor = vacuously grounded (nothing to smuggle) — same semantics as
                        # export._grounded; excluding those rows shrank the denominator and biased the metric
                        m["hyp_grounded"].append(True if not A else (
                            all(t in it["input"] for t in A) or
                            sum(1 for t in A if t in it["input"]) / len(A) >= 0.5))

    def pct(v):
        return f"{100*sum(v)/len(v):5.1f}% (n={len(v)})" if v else "   —"
    print("\n=== diagnostics (format-dependent — NOT the score; content grading below is the score) ===")
    keys = ["audit_format", "verdict_agree", "premature_refute", "missed_flaw",
            "propose_format", "check_line", "premature_guess", "declined_when_derivable", "hyp_grounded",
            "audit_format_xfer", "verdict_agree_xfer", "field_bleed"]  # schema-transfer + format-bleed sanity
    for k in keys:
        print(f"  {k:24s} base {pct(M['base'][k])}   sft {pct(M['sft'][k])}")

    if llm:
        print("\n=== content grading (blind, format ignored; gpt5.5, serial) ===", flush=True)
        R = {arm: defaultdict(list) for arm in ("base", "sft")}
        graded = 0
        for o in outs:
            it = items.get(o["item_id"])
            if not it:
                continue
            if it["task"] == "format_bleed" or it.get("schema_transfer"):
                continue  # no rubric for OOD sanity items or transfer twins (twin content == its canonical item)
            if limit and graded >= limit:
                break
            graded += 1
            for arm in ("base", "sft"):  # graded independently; the grader never sees the arm label
                txt = o.get(f"output_{arm}") or ""
                rub = RUBRIC_AUDIT if it["task"] == "audit" else RUBRIC_PROPOSE
                s = _llm(rub, f"# INPUT the candidate saw\n{it['input'][:6000]}\n\n"
                              f"# REFERENCE (grading key — the candidate never saw this)\n"
                              f"{json.dumps(it.get('reference', {}), ensure_ascii=False)[:2000]}\n"
                              f"expected: {it.get('expected', 'propose')}\n\n"
                              f"# CANDIDATE RESPONSE\n{txt[:3000]}")
                if s:
                    for k, v in s.items():
                        if isinstance(v, (int, float)):
                            R[arm][f"{it['task']}:{k}"].append(v)
            if graded % 10 == 0:
                print(f"  graded {graded}", flush=True)
        print("\n  rubric means (1-5):")
        for k in sorted(set(R["base"]) | set(R["sft"])):
            b, s = R["base"].get(k, []), R["sft"].get(k, [])
            fb = f"{statistics.mean(b):.2f} (n={len(b)})" if b else "—"
            fs = f"{statistics.mean(s):.2f} (n={len(s)})" if s else "—"
            print(f"    {k:28s} base {fb}   sft {fs}")
        (EVAL / f"rubric_scores{tag}.json").write_text(json.dumps({a: {k: v for k, v in R[a].items()} for a in R},
                                                                  indent=2), encoding="utf-8")
    (EVAL / f"mechanical_metrics{tag}.json").write_text(
        json.dumps({a: {k: [int(x) for x in v] for k, v in M[a].items()} for a in M}, indent=2), encoding="utf-8")
    print(f"\n[saved -> {EVAL} (suffix '{tag}')]" if tag else f"\n[saved -> {EVAL}]")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    app()
