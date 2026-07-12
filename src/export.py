"""Layer 1 ledger + Layer 2 audit -> Layer 3 training samples (audit / propose / fix / probe) + RM preference pairs.

Deterministic assembly, NO LLM / proxy. Crutch-free by construction: the ledger stores the agent's
own actions + a trigger LABEL, never the injected test/oracle text, so inputs never leak the crutch.

  audit   : issue + a hypothesis + evidence-at-the-time   -> verdict + flaw + support calibration   (ALL hypotheses)
  propose : issue + history (NO current hypothesis)        -> a GOOD hypothesis + why_proposed        (verdict==good only)
  fix     : issue + a confirmed good hypothesis            -> the correct patch (gold)                 (instances with a good hypothesis)

Chaining: each prior hypothesis is carried into later inputs WITH its audit FLAW (diagnosis). At inference the
model's own self-audit precedes its next proposal, so propose must be trained conditioned on the diagnosis —
that is what makes the target hypothesis reachable instead of a lucky leap (the oracle's information survives
only as diagnosis quality, never as a direction in the input).

_walk() is the single source of truth for per-card node state (the propose-input/decision-point context is built
for EVERY card, not just verdict==good): export_instance (SFT samples), export_preference (RM pairs) and
rmscaffold/gen_pairs.py (RM resample nodes) all consume the same walker, so their inputs can never drift apart.
"""
import json
import re
from pathlib import Path

import typer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
OUT = ROOT / "dataset" / "samples"
app = typer.Typer(add_completion=False)

# The unified propose/probe ask. A NAMED constant because THREE surfaces must stay byte-identical:
# _walk's propose_input (SFT samples + natural pairs), eval_build's item inputs, and gen_pairs'
# raw-masked resample contexts — a reworded copy would silently split the RM training distribution
# from the scoring distribution.
UNIFIED_ASK = ("From this evidence, state a specific, checkable hypothesis about the CAUSE of this bug. If — and only "
               "if — the evidence does not yet single out a cause, instead say what is still missing and where to probe "
               "next, rather than guessing.")


def _soften_redirect(s: str) -> str:
    """'redirect' is on the audit's own ban list (it names the oracle mechanism). The auditor sometimes uses it
    as an innocent verb of ATTENTION in the direction fields ('should have redirected attention to X') — harmless
    (evidence-grounded, never references the oracle) but off the ban list, and this field is exported. Replace ONLY
    those unambiguous attention-verb / perfect-tense collocations with 'turn'; leave technical NOUN uses (an HTTP
    redirect handler / URL / response — common in web repos like navidrome/element-web) untouched so real content
    is never corrupted."""
    if not s or "redirect" not in s.lower():
        return s
    forms = {"": "turn", "ed": "turned", "ing": "turning", "s": "turns"}
    s = re.sub(r"\bredirect(ed|ing|s)?\b(?=\s+(?:attention|focus|away|back|toward|towards))",
               lambda m: forms[m.group(1) or ""], s, flags=re.I)
    s = re.sub(r"\b((?:should|would|could)\s+have\s+|have\s+)redirect(ed)?\b",
               lambda m: m.group(1) + ("turned" if m.group(2) else "turn"), s, flags=re.I)
    return s


def _parse_json(txt: str) -> dict | None:
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else None
        except Exception:
            return None


def _evidence(card: dict) -> str:
    parts = []
    for t in card.get("tests", []):
        r = t.get("result", "")
        parts.append(f"- ran test `{t['cmd']}`" + (f" -> {r[:160]}" if r else ""))
    for e in card.get("edits", []):
        parts.append(f"- edited: {e[:600]}")  # enough of the change that 'where the code falls short' is judgeable
    for h in card.get("found", []):
        parts.append(f"- found: {h}")
    if card.get("inspected"):
        parts.append(f"- inspected: {', '.join(card['inspected'])}")
    return "\n".join(parts) or "(exploration / reading only)"


def _relevant_code(rd: Path) -> str:
    """The current source of the files the fix touches (from fetch_code.py) — so the fix sample isn't blind."""
    p = rd / "relevant_code.json"
    if not p.exists():
        return "(source not captured)"
    rc = json.loads(p.read_text(encoding="utf-8"))
    blocks = [f"# {f} (lines {r['lines']}):\n{r['code']}" for f, regions in rc.items() for r in regions]
    return ("\n\n".join(blocks))[:6000] or "(source not captured)"


# --- derivability check: does a hypothesis name code/identifiers that are actually present in the input? ---
_BT = re.compile(r"`([^`\n]{2,80})`")
_PA = re.compile(r"\b[\w./\\-]*\w\.(?:py|go|js|ts|tsx|jsx|c|h|cpp|rb|java)\b")
_SN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CA = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_ANCHOR_STOP = {"fail_to_pass", "pass_to_pass", "self_rescue", "why_proposed"}


def _anchors(t: str) -> set[str]:
    """Concrete code anchors in a hypothesis: backtick spans, file paths, snake_case, CamelCase (len>=6)."""
    a = {m.strip() for m in _BT.findall(t) if len(m.strip()) >= 3}
    a |= set(_PA.findall(t))
    a |= {m for m in _SN.findall(t) if len(m) >= 6 and m not in _ANCHOR_STOP}
    a |= {m for m in _CA.findall(t) if len(m) >= 6}
    return a


def _grounded(text: str, prompt: str) -> bool:
    """eval_grade's hyp_grounded rule: every anchor visible in the prompt, or at least half of them.
    A statement with no code anchors is vacuously grounded (it cannot smuggle unseen specifics)."""
    A = _anchors(text)
    if not A:
        return True
    return all(t in prompt for t in A) or sum(1 for t in A if t in prompt) / len(A) >= 0.5


def _norm_line(t: str) -> str:
    """One-line hypothesis statement: collapse internal whitespace/newlines (RM response normalization)."""
    return re.sub(r"\s+", " ", (t or "").strip())


def _refuted_match(h: str, refuted_norms: set[str]) -> bool:
    """Truncation-tolerant membership against outcome.json's refuted list. The two channels truncate
    DIFFERENTLY (collect._current_hypothesis stores the LAST regex match capped at 300 chars; compress
    stores the FIRST match capped at 400), so exact equality silently misses any hypothesis > 300 chars —
    which would let an oracle-refuted 'winner' through as chosen. Prefix containment (either direction) is
    allowed only when the shorter side is >= 250 chars, i.e. exactly the truncation regime."""
    nh = _norm_line(h).casefold()
    if not nh:
        return False
    for r in refuted_norms:
        if nh == r:
            return True
        if min(len(nh), len(r)) >= 250 and (nh.startswith(r) or r.startswith(nh)):
            return True
    return False


def _protocol(rd: Path) -> str:
    """Which self-rescue protocol produced this trajectory's correction rounds:
      'sanitized' — new: the auditor gives a direction with hidden-test details withheld = clean.
      'raw_leak'  — old: raw pytest/test output was shown to the agent = correction cards are contaminated
                    (the agent reward-hacked toward the leaked expected values) -> re-collect, don't train on them.
      'none'      — no rescue/oracle ever triggered (self_solved / explore-only) = protocol-irrelevant, always clean.
    Detected from the fed-back observations in the trajectory."""
    tp = rd / "trajectory.json"
    if not tp.exists():
        return "none"
    txt = tp.read_text(encoding="utf-8", errors="ignore")
    if "Sanitized test-feedback direction" in txt:
        return "sanitized"
    # legacy oracle marker included: compress._trigger knows it, and a trajectory carrying ONLY that
    # wording would otherwise be classified 'none' (= clean) and its correction pairs kept
    if "tests FAILED" in txt or "guidance round" in txt or "WHERE your reasoning is wrong" in txt:
        return "raw_leak"
    return "none"


def _walk(iid: str, run_id: str, gold: dict):
    """Walk a trajectory's cards once, yielding per-card node state (nodes, meta) — or None if not exportable.

    Each node carries the card, its audit entry, and the INPUT CONTEXTS built from the evidence accumulated
    BEFORE this card (temporal fidelity: a card's own findings ground later hypotheses, never its own).
    The propose_input (decision-point context) is built for EVERY card so RM pair/resample builders can use
    any card as a decision point; export_instance only emits propose/probe samples at verdict==good cards."""
    rd = RAW / iid / run_id
    if not (rd / "ledger.json").exists() or not (rd / "audit.json").exists():
        return None
    ledger = json.loads((rd / "ledger.json").read_text(encoding="utf-8"))
    audit = _parse_json((rd / "audit.json").read_text(encoding="utf-8"))
    if not audit:
        return None
    audit = _clean_audit(audit)  # scrub the audit's own banned vocabulary (esp. 'redirect') at the source
    cards = ledger["cards"]
    proto = _protocol(rd)  # trajectory-level protocol; correction cards from a raw_leak run are contaminated
    # card-number alignment: v3 audits tag each entry with its ledger card number — positional zip broke
    # whenever the auditor deduplicated/split hypotheses (62/250 instances had #entries != #cards, mispairing
    # verdict/flaw/why). Legacy audits without "card" fall back to position.
    by_card: dict[int, dict] = {}
    for j, e in enumerate(audit.get("per_hypothesis", [])):
        if not isinstance(e, dict):
            continue
        try:
            k = int(e.get("card"))
        except (TypeError, ValueError):
            k = j + 1
        by_card.setdefault(k, e)
    issue = (gold.get("problem_statement") or "")[:3000]
    op = rd / "outcome.json"
    oj = json.loads(op.read_text(encoding="utf-8")) if op.exists() else {}
    # found_so starts at the PREAMBLE (hits found before the first hypothesis) — the temporal cut matters:
    # a card's own found/inspected came AFTER its hypothesis was stated (the hypothesis directed that search),
    # so it may ground LATER hypotheses but never its own propose input (at inference it wouldn't exist yet).
    nodes, prior = [], []  # prior: (hypothesis, diagnosis, should_have_turned) per earlier card — diagnosis = audit FLAW
    found_so, insp_so = list(ledger.get("preamble_found", [])), []
    views_so: dict[str, str] = dict(ledger.get("preamble_views") or {})  # file -> latest content snapshot the
    # agent READ (only files some card edits); seeded with pre-H1 reads — legitimately known before any proposal
    edits_so: list[tuple[str, str]] = []  # (hypothesis_id, edit cmd) — cumulative code state at submit time
    for i, card in enumerate(cards):
        a = by_card.get(i + 1, {})
        hyp, ev, verdict = card["hypothesis"], _evidence(card), a.get("verdict", "")
        hist = ("Investigation so far:\n" + "\n".join(f"- {h}" + (f"\n  (diagnosis: {fl})" if fl else "")
                                                      for h, fl, _t in prior) + "\n\n") if prior else ""
        audit_input = (f"ISSUE:\n{issue}\n\n{hist}CURRENT HYPOTHESIS (the agent's conjecture about the cause):\n{hyp}\n\n"
                       f"EVIDENCE available at this point:\n{ev}\n\n"
                       "Audit this hypothesis: given only the information available, is it a sound conjecture, and what is wrong / insufficient about it?")
        # decision-point context (the propose input). The INPUT carries the gathered evidence (grep/read findings,
        # what the failed attempt changed + the source it touched, prior ruled-out attempts) — NO test verdict / oracle.
        evid = ""
        if found_so:
            evid += "Relevant code located (file:line):\n" + "\n".join(f"- {h}" for h in found_so[-12:]) + "\n"
        if insp_so:
            evid += "Files inspected: " + ", ".join(insp_so[-10:]) + "\n"
        if card.get("trigger") in ("self_rescue", "oracle") and edits_so:
            # correction round: give (a) what the failed attempt changed and (b) the touched files as the agent last
            # saw them — both the agent's OWN actions, so the target is derivable from code it actually saw, not a
            # test verdict.
            evid += ("EDITS the failed attempt had made (cumulative changes at submit time):\n"
                     + "\n".join(f"- [{h}] {e}" for h, e in edits_so[-8:]) + "\n")
            if views_so:
                evid += ("SOURCE FILES as the agent last read them:\n" + "\n\n".join(
                    f"# {f}\n{v}" for f, v in list(views_so.items())[-3:]) + "\n")
        if prior:  # carry each elimination WITH its diagnosis (the model's own self-audit reproduces this at inference)
            evid += ("Hypotheses already tried and RULED OUT (with the diagnosis of what was wrong):\n"
                     + "\n".join(f"- {h}" + (f"\n  diagnosis: {fl}" if fl else "") for h, fl, _t in prior) + "\n")
        # UNIFIED ask (same for propose & probe) so the model learns to DECIDE which is warranted — propose when the
        # evidence supports a cause, probe when it does not, instead of always guessing (the premature-guess metric).
        propose_input = (f"ISSUE:\n{issue}\n\nEVIDENCE GATHERED SO FAR:\n{evid or '(only the issue so far)'}\n"
                         + UNIFIED_ASK)
        corr = card.get("trigger") in ("self_rescue", "oracle") and prior and prior[-1][1]
        hyp_anchors = _anchors(hyp)
        # cannot-derive: a correction-round hypothesis that names code NOT in the input (the agent found it by
        # searching AFTER this point) = a leap from THIS input.
        probe_gated = bool(corr and hyp_anchors and not any(t in propose_input for t in hyp_anchors))
        nodes.append({"i": i, "card": card, "a": a, "hyp": hyp, "ev": ev, "verdict": verdict,
                      "audit_input": audit_input, "propose_input": propose_input, "corr": corr,
                      "anchors": hyp_anchors, "probe_gated": probe_gated,
                      "prior_last": prior[-1] if prior else None})
        # accumulate THIS card's evidence only now — it grounds later hypotheses, never its own (temporal fidelity)
        for h in card.get("found", []):
            if h not in found_so:
                found_so.append(h)
        for f in card.get("inspected", []):
            if f not in insp_so:
                insp_so.append(f)
        for f, v in (card.get("read_snippets") or {}).items():
            views_so.pop(f, None)  # re-insert so [-3:] slices the most RECENTLY read files
            views_so[f] = v
        for e in card.get("edits", []):
            edits_so.append((card["hypothesis_id"], e))
        prior.append((hyp, (a.get("flaw_given_info_at_the_time") or "").strip()[:240],
                      _soften_redirect((a.get("should_have_turned") or "").strip()[:240])))
    meta = {"cards": cards, "by_card": by_card, "proto": proto, "issue": issue,
            "outcome": oj.get("outcome"), "oj": oj, "rd": rd}
    return nodes, meta


_REDIR = re.compile(r"\bredirect(ed|ing|ion)?\b", re.I)
_REDIR_MAP = {"": "turn", "ed": "turned", "ing": "turning", "ion": "turn"}
_AUDIT_PROSE = ("why_proposed", "flaw_given_info_at_the_time", "should_have_turned", "hypothesis", "overall")


def _deoracle(s: str) -> str:
    """Neutralize audit vocabulary that names the (hidden) redirect mechanism. 'redirect(ed)' reads as an
    ordinary verb but is on the audit prompt's own ban-list and reaches BOTH the training input (via a prior
    hypothesis's flaw/direction) and the target (NEXT CHECK / propose direction); normalize to 'turn'. 'oracle'
    should never appear — if it does, that IS a real leak, so neutralize it too. 2026-07-09 (3-sample spot-check)."""
    if not isinstance(s, str):
        return s
    s = _REDIR.sub(lambda m: _REDIR_MAP[(m.group(1) or "").lower()], s)
    s = re.sub(r"\bthe oracle\b", "the analysis", s, flags=re.I)
    s = re.sub(r"\boracle\b", "analysis", s, flags=re.I)
    return s


def _clean_audit(audit: dict) -> dict:
    """Scrub every prose field of the audit at the SOURCE, so every input/target built from it is clean."""
    if isinstance(audit.get("overall"), str):
        audit["overall"] = _deoracle(audit["overall"])
    for e in audit.get("per_hypothesis", []):
        if isinstance(e, dict):
            for k in _AUDIT_PROSE:
                if k in e:
                    e[k] = _deoracle(e[k])
    return audit


def export_instance(iid: str, run_id: str, gold: dict) -> list[dict]:
    w = _walk(iid, run_id, gold)
    if w is None:
        return []
    nodes, meta = w
    cards, by_card, proto, rd = meta["cards"], meta["by_card"], meta["proto"], meta["rd"]
    out = []
    for n in nodes:
        card, a, hyp, verdict = n["card"], n["a"], n["hyp"], n["verdict"]
        if a:  # audit sample (every hypothesis)
            # turning point = a hypothesis that needed a redirect (weak/wrong), vs an ordinary on-track audit (good)
            tp = verdict in ("weak", "wrong") or a.get("support_calibration") in ("weak_support", "misleading_support", "refute")
            ti = (f"VERDICT: {verdict}\nSUPPORT: {a.get('support_calibration', '')}\n"
                  f"FLAW: {a.get('flaw_given_info_at_the_time', '')}\nNEXT CHECK: {_soften_redirect(a.get('should_have_turned', ''))}")
            # source = the trigger of the card: oracle (direction the agent couldn't find = highest-value Δ) >
            # self_rescue (a missed edge/boundary condition = smaller Δ) > explore (found it directly).
            out.append({"type": "audit", "turning_point": bool(tp), "source": card.get("trigger"),
                        "instance_id": iid, "hypothesis_id": card["hypothesis_id"],
                        "messages": [{"role": "user", "content": n["audit_input"]}, {"role": "assistant", "content": ti}]})
        if verdict == "good":  # propose OR probe sample at this decision point
            ui = n["propose_input"]
            if n["probe_gated"]:
                # Don't teach the leap — teach the honest move: name the prior attempt's gap + where to look next.
                # Turns hallucinated-direction negatives into recognize-insufficiency positives (nothing dropped).
                # Direction = the prior attempt's audited should_have_turned (a where, not the answer).
                _ph, prev_flaw, prev_turn = n["prior_last"]
                ti = (f"GAP IN THE PREVIOUS ATTEMPT: {prev_flaw}\n"
                      "STILL MISSING: the evidence gathered does not yet single out the responsible code path — a specific "
                      "cause cannot be named with confidence yet.\n"
                      f"WHERE TO PROBE NEXT: {prev_turn or 'inspect the code path the failed change (above) landed in.'}")
                out.append({"type": "probe", "turning_point": True, "source": card.get("trigger"),
                            "instance_id": iid, "hypothesis_id": card["hypothesis_id"],
                            "messages": [{"role": "user", "content": ui}, {"role": "assistant", "content": ti}]})
            else:
                ti = f"HYPOTHESIS: {hyp}\nREASONING: {a.get('why_proposed', '')}"
                if a.get("how_to_check"):  # a hypothesis is only useful with its next verification ACTION —
                    # without this line the model learned to call things "falsifiable" without saying how (P1/P3)
                    ti += f"\nCHECK: {a['how_to_check']}"
                if n["corr"]:  # model the derivation: the failed attempt's diagnosis, then the repair (gap -> new hypo)
                    ti = f"GAP IN THE PREVIOUS ATTEMPT: {n['prior_last'][1]}\n{ti}"
                # oracle-sourced 'good' hypotheses are the gold: they teach the DIRECTION the model couldn't find alone.
                out.append({"type": "propose", "source": card.get("trigger"), "instance_id": iid, "hypothesis_id": card["hypothesis_id"],
                            "messages": [{"role": "user", "content": ui}, {"role": "assistant", "content": ti}]})
    # fix sample: only for SOLVED trajectories, and the confirmed cause is the LAST hypothesis — the one whose
    # patch actually passed the hidden tests. Correctness comes from the OUTCOME, not the audit verdict (a correct
    # leap is judged 'weak' yet is still the true cause; an earlier 'good'-verdict hypothesis may be well-reasoned
    # but REFUTED — pairing that with the gold patch would mislabel ~26% of fix samples). Chaotic runs have no
    # verified cause -> no fix sample (they still feed audit/propose).
    outcome = meta["outcome"]
    if (outcome in ("self_solved", "self_corrected", "oracle_redirected") and cards and gold.get("patch")
            and not iid.startswith(("tb__", "lcb__"))):  # TB/LCB gold is a script/none, not a source diff — the
        # "produce the fix as a patch" framing doesn't fit; TB/LCB contribute audit/propose/probe only (v1)
        li = len(cards) - 1
        cause = by_card.get(li + 1, {}).get("hypothesis") or cards[li]["hypothesis"]
        ui = (f"ISSUE:\n{meta['issue']}\n\nROOT CAUSE (confirmed):\n{cause}\n\n"
              f"RELEVANT CODE (current source of the files to change):\n{_relevant_code(rd)}\n\n"
              "Produce the fix as a patch to the source files.")
        out.append({"type": "fix", "source": cards[li].get("trigger"), "instance_id": iid, "hypothesis_id": "final",
                    "messages": [{"role": "user", "content": ui}, {"role": "assistant", "content": gold["patch"]}]})
    for s in out:  # protocol tag: a correction sample (self_rescue/oracle) from an OLD raw-leak trajectory is
        # contaminated (the agent saw raw hidden-test output) -> droppable downstream; everything else is clean,
        # including every self_solved/explore sample (no feedback ever triggered -> protocol-irrelevant).
        s["protocol"] = "raw_leak" if (proto == "raw_leak" and s.get("source") in ("self_rescue", "oracle")) else "clean"
    return out


def export_preference(iid: str, run_id: str, gold: dict) -> list[dict]:
    """Within-instance preference pairs for a reward model, v2 (decision-point construction).

    Information-parity rules (why this shape):
      * chosen eligibility = OUTCOME-VERIFIED only: the final hypothesis of a solved trajectory (its patch
        passed the hidden tests). Audit verdicts are NOT enough — a 'good'-verdict conjecture can be
        well-reasoned yet refuted; using it as chosen would teach the RM to prefer refuted directions.
      * decision pairs: prompt = the SOLVING card's propose input (same _walk constructor as SFT/eval items,
        so RM training distribution == BoN scoring distribution). chosen = the verified winner, gated on
        _grounded(prompt) — every code anchor it names must be visible in the prompt, so the pair can never
        reward citing unseen specifics. rejected = each earlier refuted / wrong / weak hypothesis, whose text
        (with diagnosis) sits in the prompt's RULED OUT block: re-proposing it at this point is objectively
        the worse answer. Fairness is judged against the PROMPT, not against when each text was authored.
      * issue_only pairs (secondary; matches the H1-type BoN items): prompt = bare issue; chosen must be
        grounded in the ISSUE text itself (most late winners fail this gate — correct conservatism).
      * responses are one-line normalized hypothesis statements on BOTH sides (no format shortcut).
      * rejected texts come from the audit restatement / ledger card only (both F2P-scrubbed channels);
        the raw outcome.json refuted list is used ONLY to mark which cards were oracle-refuted.
    Tagged with protocol (trajectory-level; prep_rm drops raw_leak) + pair_type/strength/rejected_source."""
    w = _walk(iid, run_id, gold)
    if w is None:
        return []
    nodes, meta = w
    cards, by_card, proto = meta["cards"], meta["by_card"], meta["proto"]
    if meta["outcome"] not in ("self_solved", "self_corrected", "oracle_redirected") or len(cards) < 2:
        return []  # chosen must be outcome-verified; single-hypothesis runs have no within-instance comparison
    refuted_norms = {_norm_line(r).casefold() for r in meta["oj"].get("refuted_hypotheses", []) if r and r.strip()}

    def restate(j: int) -> str:  # audit restatement preferred (scrubbed, cleaner), ledger card as fallback
        return _norm_line(by_card.get(j + 1, {}).get("hypothesis") or cards[j]["hypothesis"])

    chosen = restate(len(cards) - 1)
    if not chosen or _refuted_match(cards[-1]["hypothesis"], refuted_norms) or _refuted_match(chosen, refuted_norms):
        return []  # a refuted 'winner' means the outcome/refutation records disagree — don't emit
    rejected = []  # (text, source) from every non-final card that is a bad choice at the final decision point
    for j, n in enumerate(nodes[:-1]):
        txt = restate(j)
        if not txt or txt == chosen:
            continue
        if _refuted_match(n["hyp"], refuted_norms) or _refuted_match(txt, refuted_norms):
            src = "oracle_refuted"
        elif n["verdict"] == "wrong":
            src = "verdict_wrong"
        elif n["verdict"] == "weak":
            src = "verdict_weak"
        else:
            continue  # an earlier good, never-refuted step is not a negative
        rejected.append((txt, src))
    if not rejected:
        return []
    final = nodes[-1]
    pairs, seen = [], set()

    def emit(pair_type: str, prompt: str) -> None:
        if not _grounded(chosen, prompt) or (pair_type == "decision" and final["probe_gated"]):
            return  # chosen must be derivable from THIS prompt, else the pair rewards unseen specifics
        for txt, src in rejected:
            key = (pair_type, chosen[:200], txt[:200])
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"type": "preference", "pair_type": pair_type, "instance_id": iid,
                          "protocol": proto, "strength": "verified", "rejected_source": src,
                          "chosen_grounded": True, "rejected_grounded": _grounded(txt, prompt),
                          "node": cards[-1].get("hypothesis_id", f"H{len(cards)}"),
                          "prompt": prompt, "chosen": chosen, "rejected": txt})

    emit("decision", final["propose_input"])
    emit("issue_only", f"ISSUE:\n{meta['issue']}\n\nState the single most likely hypothesis about the CAUSE of this bug.")
    return pairs


DS_NAME = {"verified": "princeton-nlp/SWE-bench_Verified", "full": "princeton-nlp/SWE-bench",
           "pro": "ScaleAI/SWE-bench_Pro"}


@app.command()
def main(run_id: str = typer.Option("r1", "--run-id"),
         dataset: str = typer.Option("verified", "--dataset", help="verified | full | pro | tb | lcb (gold lookup source)"),
         preference_only: bool = typer.Option(False, "--preference-only",
                                              help="refresh ONLY preference.jsonl (RM pairs) — leaves the "
                                                   "audit/propose/fix/probe sample files untouched")) -> None:
    if dataset in ("tb", "lcb"):  # local adapters carry gold = {problem_statement, patch(=solve.sh / None)}
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parent))
        import tb as _tb, lcb as _lcb
        _adapter = {"tb": _tb, "lcb": _lcb}[dataset]
        ds = {i["instance_id"]: {"problem_statement": i["problem_statement"], "patch": i.get("patch")}
              for i in _adapter.load_instances()}
    else:
        ds = {i["instance_id"]: i for i in load_dataset(DS_NAME[dataset], split="test")}
    # held-out wall: Pro test instances are NEVER exported to training (reserved for the process-Δ eval).
    # Auto-loaded so a routine re-export can never contaminate. Verified is already clean (collection was train-split-only).
    # FAIL LOUD when the wall file is missing — a silent empty set would export held-out instances into
    # training on any fresh checkout / renamed-splits machine (2026-07-07 review).
    holdout: set[str] = set()
    hp = ROOT / "dataset" / "splits" / f"{dataset}_test.json"
    if dataset == "pro":
        if not hp.exists():
            raise SystemExit(f"[export] HOLDOUT WALL MISSING: {hp} not found — refusing to export Pro training "
                             "data without the pro_test split (a silent empty wall would train on held-out instances).")
        holdout = set(json.loads(hp.read_text(encoding="utf-8")))
    elif dataset in ("tb", "lcb") and hp.exists():  # TB/LCB reserve a held-out eval split too, once one is created
        holdout = set(json.loads(hp.read_text(encoding="utf-8")))
    out_dir = OUT / run_id  # namespace by run so multiple runs (r1 / pro1 / ...) coexist; prep_sft merges them
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {} if preference_only else {t: (out_dir / f"{t}.jsonl").open("w", encoding="utf-8")
                                        for t in ("audit", "propose", "fix", "probe")}
    pref_f = (out_dir / "preference.jsonl").open("w", encoding="utf-8")  # RM pairs (separate schema; prep_sft ignores it)
    counts = {"audit": 0, "propose": 0, "fix": 0, "probe": 0}
    n_inst = n_held = n_pref = 0
    for iid in sorted(p.name for p in RAW.iterdir() if (p / run_id / "ledger.json").exists()):
        if iid not in ds:
            continue
        if iid in holdout:
            n_held += 1
            continue
        n_inst += 1
        if not preference_only:
            for s in export_instance(iid, run_id, ds[iid]):
                files[s["type"]].write(json.dumps(s, ensure_ascii=False) + "\n")
                counts[s["type"]] += 1
        for pp in export_preference(iid, run_id, ds[iid]):
            pref_f.write(json.dumps(pp, ensure_ascii=False) + "\n")
            n_pref += 1
    for f in files.values():
        f.close()
    pref_f.close()
    if preference_only:
        print(f"[export] {n_inst} instances -> {n_pref} preference pairs in {out_dir} (samples untouched)"
              + (f" | {n_held} held-out instance(s) excluded" if holdout else ""))
    else:
        print(f"[export] {n_inst} instances -> samples {counts} (total {sum(counts.values())}) + {n_pref} preference pairs in {out_dir}"
              + (f" | {n_held} held-out instance(s) excluded" if holdout else ""))


if __name__ == "__main__":
    app()
