"""Deterministic SFT-data integrity check (no LLM, no proxy — safe to run beside the batch).

Two questions (2026-07-03):
 1. propose: does the INPUT carry the information needed to DERIVE the target
    (GAP + HYPOTHESIS + REASONING/why_proposed)?  -> anchor grounding + leak-channel scan
 2. self-correct rounds: is the target the POST-feedback corrected hypothesis, with the
    test feedback itself absent from the input?   -> ledger cross-check + crutch/F2P scans

Leak channels: the harness feedback text never enters the ledger (compress.py skips it), but the
posthoc AUDITOR saw it — so its paraphrase can leak through audit prose into (a) the input's
"diagnosis:" lines (RULED OUT chain) and (b) the target's GAP/REASONING/FLAW. Hard evidence of
that channel = a hidden F2P test name appearing there while absent from issue+evidence.

  python src/_check_leakage.py            # full report
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMP = ROOT / "dataset" / "samples"
RAW = ROOT / "dataset" / "raw"
OUTDIR = ROOT / "dataset" / "_checks"

RUNS = {"r1": "princeton-nlp/SWE-bench_Verified", "pro1": "ScaleAI/SWE-bench_Pro"}

# verbatim crutch signatures (must be 0 in inputs; any hit in targets also reported)
CRUTCH = ["tests FAILED", "FAIL_TO_PASS", "PASS_TO_PASS", "graded test",
          "Reconsider your current HYPOTHESIS", "guidance round", "WHERE your reasoning is wrong",
          "Your submitted patch", "hidden guidance", "oracle", "reviewer"]
# paraphrase references to hidden-test RESULTS (bad in diagnosis/target; prescriptive "run tests" is fine)
RESULT_REF = re.compile(
    r"hidden tests?|project'?s tests?|tests? (?:still |now )?fail(?:ed|s)\b|test failures?"
    r"|fail(?:ed|s)? the (?:hidden |project )?tests?|failing tests?", re.I)

BACKTICK = re.compile(r"`([^`\n]{2,80})`")
PATH = re.compile(r"\b[\w./\\-]*\w\.(?:py|go|js|jsx|ts|tsx|c|h|cpp|hpp|rs|rb|java|cfg|ini|toml|ya?ml|json)\b")
SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
LINENO = re.compile(r"\bline (\d+)\b")
SNAKE_STOP = {"fail_to_pass", "pass_to_pass", "self_rescue", "why_proposed"}


def _jl(x):
    if isinstance(x, list):
        return x
    try:
        return json.loads(x)
    except Exception:
        import ast
        return ast.literal_eval(x)


def f2p_tokens(f2p: list[str]) -> set[str]:
    """Distinctive tokens that identify a hidden graded test: py test_* names, go Test* names,
    js/ts sentence-style titles (matched as whole substrings)."""
    toks: set[str] = set()
    for t in f2p:
        for m in re.findall(r"\btest_[A-Za-z0-9_]{3,}\b", t):
            toks.add(m)
        for m in re.findall(r"\bTest[A-Z][A-Za-z0-9_]{2,}\b", t):
            toks.add(m)
        s = t.strip()
        if " " in s and len(s) >= 15:  # js "describe > it should ..." style
            toks.add(s)
    return toks


# ---------- input/target section parsing (mirrors export.py assembly) ----------

def split_input(typ: str, ui: str) -> dict:
    z = {"issue": "", "evidence": "", "diagnosis": ""}
    if typ == "propose":
        m = re.split(r"\n\nEVIDENCE GATHERED SO FAR:\n", ui, maxsplit=1)
        z["issue"] = m[0]
        rest = m[1] if len(m) > 1 else ""
        dm = re.split(r"Hypotheses already tried and RULED OUT[^\n]*\n", rest, maxsplit=1)
        z["evidence"] = dm[0]
        z["diagnosis"] = dm[1] if len(dm) > 1 else ""
    elif typ == "audit":
        m = re.split(r"\n\n(?:Investigation so far:\n|CURRENT HYPOTHESIS)", ui, maxsplit=1)
        z["issue"] = m[0]
        hm = re.search(r"Investigation so far:\n(.*?)\n\nCURRENT HYPOTHESIS", ui, re.S)
        z["diagnosis"] = hm.group(1) if hm else ""  # hist carries (diagnosis: ...) lines
        em = re.search(r"EVIDENCE available at this point:\n(.*?)\n\nAudit this", ui, re.S)
        z["evidence"] = (em.group(1) if em else "")
        cm = re.search(r"CURRENT HYPOTHESIS[^\n]*:\n(.*?)\n\nEVIDENCE", ui, re.S)
        z["evidence"] += "\n" + (cm.group(1) if cm else "")
    else:  # fix
        m = re.split(r"\n\nROOT CAUSE \(confirmed\):\n", ui, maxsplit=1)
        z["issue"] = m[0]
        rest = m[1] if len(m) > 1 else ""
        rm = re.split(r"\n\nRELEVANT CODE[^\n]*\n", rest, maxsplit=1)
        z["diagnosis"] = rm[0]  # the confirmed cause (audit-prose channel)
        z["evidence"] = rm[1] if len(rm) > 1 else ""
    return z


def split_target(typ: str, ti: str) -> dict:
    z = {}
    if typ == "propose":
        g = re.search(r"GAP IN THE PREVIOUS ATTEMPT: (.*?)(?:\nHYPOTHESIS:)", ti, re.S)
        h = re.search(r"HYPOTHESIS: (.*?)(?:\nREASONING:|\Z)", ti, re.S)
        r = re.search(r"REASONING: (.*)\Z", ti, re.S)
        z = {"GAP": g.group(1) if g else "", "HYPOTHESIS": h.group(1) if h else ti,
             "REASONING": r.group(1) if r else ""}
    elif typ == "audit":
        for k, pat in (("FLAW", r"FLAW: (.*?)(?:\nNEXT CHECK:|\Z)"), ("NEXT", r"NEXT CHECK: (.*)\Z")):
            m = re.search(pat, ti, re.S)
            z[k] = m.group(1) if m else ""
    else:
        z = {"PATCH": ""}  # gold patch — not scanned for anchors
    return z


def anchors(text: str) -> set[str]:
    a: set[str] = set()
    for m in BACKTICK.findall(text):
        if len(m.strip()) >= 3:
            a.add(m.strip())
    for m in PATH.findall(text):
        a.add(m)
    for m in SNAKE.findall(text):
        if len(m) >= 6 and m not in SNAKE_STOP:
            a.add(m)
    for m in CAMEL.findall(text):
        if len(m) >= 6:
            a.add(m)
    for m in LINENO.findall(text):
        a.add(f"line:{m}")
    return a


def grounded(tok: str, inp: str) -> bool:
    if tok.startswith("line:"):
        return re.search(rf"\b{re.escape(tok[5:])}\b", inp) is not None
    return tok in inp


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    from datasets import load_dataset
    meta: dict[str, dict] = {}
    for run, name in RUNS.items():
        if not (SAMP / run).exists():
            continue
        for row in load_dataset(name, split="test"):
            f2p = row.get("FAIL_TO_PASS") or row.get("fail_to_pass") or "[]"
            meta[row["instance_id"]] = {"f2p": f2p_tokens(_jl(f2p))}

    crutch_hits, result_hits, f2p_hits = [], [], []
    ground_rows, struct_bad, fix_bad = [], [], []
    n = Counter()
    led_cache: dict[tuple, dict] = {}

    def ledger(run, iid):
        key = (run, iid)
        if key not in led_cache:
            p = RAW / iid / run / "ledger.json"
            led_cache[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return led_cache[key]

    for run in RUNS:
        for typ in ("audit", "propose", "fix"):
            f = SAMP / run / f"{typ}.jsonl"
            if not f.exists():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                s = json.loads(line)
                iid, src = s["instance_id"], s.get("source")
                ui, ti = s["messages"][0]["content"], s["messages"][1]["content"]
                n[typ] += 1
                zin, zout = split_input(typ, ui), split_target(typ, ti)

                # 1) verbatim crutch signatures
                for sig in CRUTCH:
                    for where, txt in (("input", ui), ("target", ti if typ != "fix" else "")):
                        for m in re.finditer(re.escape(sig), txt, re.I):
                            c = txt[max(0, m.start() - 70):m.end() + 90].replace("\n", " ")
                            crutch_hits.append({"run": run, "type": typ, "iid": iid, "hid": s.get("hypothesis_id"),
                                                "sig": sig, "where": where, "ctx": c})

                # 2) paraphrased test-RESULT references in audit-prose channels (diagnosis-in-input + target fields)
                chans = [("input.diagnosis", zin["diagnosis"])] + [(f"target.{k}", v) for k, v in zout.items() if k != "PATCH"]
                for ch, txt in chans:
                    for m in RESULT_REF.finditer(txt or ""):
                        c = txt[max(0, m.start() - 70):m.end() + 90].replace("\n", " ")
                        result_hits.append({"run": run, "type": typ, "iid": iid, "hid": s.get("hypothesis_id"),
                                            "src": src, "chan": ch, "ctx": c})

                # 3) hidden F2P test-name leakage: in diagnosis/target but NOT in issue/evidence
                toks = meta.get(iid, {}).get("f2p", set())
                for tok in toks:
                    seen_ok = tok in zin["issue"] or tok in zin["evidence"]
                    for ch, txt in chans:
                        if txt and tok in txt and not seen_ok:
                            f2p_hits.append({"run": run, "type": typ, "iid": iid, "hid": s.get("hypothesis_id"),
                                             "src": src, "chan": ch, "tok": tok[:90]})

                # 4) propose target grounding (anchor containment in input)
                if typ == "propose":
                    per = {}
                    for k in ("GAP", "HYPOTHESIS", "REASONING"):
                        aset = anchors(zout.get(k, ""))
                        ung = sorted(t for t in aset if not grounded(t, ui))
                        per[k] = {"n": len(aset), "ung": ung}
                    issue_only = "(only the issue so far)" in ui
                    ground_rows.append({"run": run, "iid": iid, "hid": s.get("hypothesis_id"), "src": src,
                                        "issue_only": issue_only, **{k: per[k] for k in per}})

                # 5) self-correct structure: target = the post-feedback card; prev refuted hyp visible in input
                if typ == "propose" and src in ("self_rescue", "oracle"):
                    led = ledger(run, iid)
                    cards = {c["hypothesis_id"]: (i, c) for i, c in enumerate(led.get("cards", []))}
                    prob = []
                    hid = s.get("hypothesis_id")
                    if hid in cards:
                        idx, card = cards[hid]
                        if card["trigger"] != src:
                            prob.append(f"trigger mismatch {card['trigger']}!={src}")
                        if not zout["HYPOTHESIS"].strip().startswith(card["hypothesis"][:80]):
                            prob.append("target hyp != ledger card hyp")
                        if idx > 0:
                            prev = led["cards"][idx - 1]["hypothesis"][:80]
                            if prev not in ui:
                                prob.append("prev refuted hyp NOT in input RULED OUT")
                        else:
                            prob.append("self_rescue/oracle card is H1 (no prior)")
                        if not zout["GAP"]:
                            prob.append("no GAP section")
                        elif zout["GAP"].strip()[:60] not in ui:
                            prob.append("GAP text not mirrored in input diagnosis")
                    else:
                        prob.append("hypothesis_id missing from ledger")
                    if prob:
                        struct_bad.append({"run": run, "iid": iid, "hid": hid, "src": src, "prob": prob})

                # 6) fix gating: only solved outcomes; cause == last card's hypothesis
                if typ == "fix":
                    od = RAW / iid / run / "outcome.json"
                    outc = json.loads(od.read_text(encoding="utf-8")).get("outcome") if od.exists() else None
                    prob = []
                    if outc not in ("self_solved", "self_corrected", "oracle_redirected"):
                        prob.append(f"outcome={outc}")
                    led = ledger(run, iid)
                    if led.get("cards"):
                        last = led["cards"][-1]["hypothesis"][:80]
                        if last not in ui and zin["diagnosis"][:80] not in (last,):
                            # cause may be the audit's (de-oracled) copy — accept if 60% prefix matches
                            if zin["diagnosis"].strip()[:40] != last[:40]:
                                prob.append("ROOT CAUSE != last card hyp (check audit copy)")
                    if prob:
                        fix_bad.append({"run": run, "iid": iid, "outcome": outc, "prob": prob})

    # ---------- report ----------
    print(f"samples scanned: {dict(n)}  (total {sum(n.values())})")
    print("\n=== [1] verbatim crutch signatures ===")
    print(f"hits: {len(crutch_hits)}")
    for k, c in Counter((h['sig'], h['type'], h['where']) for h in crutch_hits).most_common():
        print(f"  {c:4d}  sig={k[0]!r:<38} type={k[1]:<7} in={k[2]}")

    print("\n=== [2] paraphrased test-RESULT references (audit-prose channels) ===")
    print(f"hits: {len(result_hits)}")
    for k, c in Counter((h['type'], h['chan']) for h in result_hits).most_common():
        print(f"  {c:4d}  type={k[0]:<7} chan={k[1]}")

    print("\n=== [3] hidden F2P test-name leaks (not in issue/evidence) ===")
    print(f"hits: {len(f2p_hits)} across {len({(h['iid'], h['hid']) for h in f2p_hits})} samples")
    for k, c in Counter((h['type'], h['chan']) for h in f2p_hits).most_common():
        print(f"  {c:4d}  type={k[0]:<7} chan={k[1]}")

    print("\n=== [4] propose target grounding (anchors present in input?) ===")
    for scope, rows in (("ALL", ground_rows),
                        ("issue-only-input", [r for r in ground_rows if r["issue_only"]]),
                        ("with-evidence", [r for r in ground_rows if not r["issue_only"]])):
        if not rows:
            continue
        ta = sum(r[k]["n"] for r in rows for k in ("GAP", "HYPOTHESIS", "REASONING"))
        tu = sum(len(r[k]["ung"]) for r in rows for k in ("GAP", "HYPOTHESIS", "REASONING"))
        full = sum(1 for r in rows if all(not r[k]["ung"] for k in ("GAP", "HYPOTHESIS", "REASONING")))
        print(f"  [{scope}] samples={len(rows)} fully-grounded={full} ({full/len(rows):.0%}) "
              f"anchors={ta} ungrounded={tu} ({(tu/ta if ta else 0):.1%})")
    by_src = defaultdict(list)
    for r in ground_rows:
        by_src[r["src"]].append(r)
    for srck, rows in sorted(by_src.items()):
        ta = sum(r[k]["n"] for r in rows for k in ("GAP", "HYPOTHESIS", "REASONING"))
        tu = sum(len(r[k]["ung"]) for r in rows for k in ("GAP", "HYPOTHESIS", "REASONING"))
        print(f"    source={srck:<12} samples={len(rows)} ungrounded-rate={(tu/ta if ta else 0):.1%}")
    worst = sorted(ground_rows, key=lambda r: -sum(len(r[k]["ung"]) for k in ("GAP", "HYPOTHESIS", "REASONING")))[:8]
    print("  worst samples (iid/hid, src, #ungrounded, first few):")
    for r in worst:
        u = [t for k in ("GAP", "HYPOTHESIS", "REASONING") for t in r[k]["ung"]]
        print(f"    {r['iid'][:44]:<44} {r['hid']:<4} {r['src'] or '-':<11} {len(u):3d}  {u[:5]}")

    print("\n=== [5] self-correct structure violations ===")
    print(f"bad: {len(struct_bad)} / {sum(1 for r in ground_rows if r['src'] in ('self_rescue', 'oracle'))}")
    for b in struct_bad[:15]:
        print(f"  {b['iid'][:44]:<44} {b['hid']:<4} {b['src']:<11} {b['prob']}")

    print("\n=== [6] fix gating violations ===")
    print(f"bad: {len(fix_bad)} / {n['fix']}")
    for b in fix_bad[:15]:
        print(f"  {b['iid'][:44]:<44} outcome={b['outcome']} {b['prob']}")

    dump = {"crutch": crutch_hits, "result_ref": result_hits, "f2p": f2p_hits,
            "grounding_worst": worst, "struct_bad": struct_bad, "fix_bad": fix_bad}
    (OUTDIR / "leakage_report.json").write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[full detail -> {OUTDIR / 'leakage_report.json'}]")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
