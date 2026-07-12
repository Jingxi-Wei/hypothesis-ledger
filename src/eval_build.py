"""Build the HELD-OUT process-eval items (deterministic, NO proxy).

The eval mirrors the training views exactly (same input construction as export.py) on pro_test instances
that training NEVER saw (export auto-excludes them; collection + the v3 card-aligned audit cover them, so
their audits serve as REFERENCE anchors for free). Two task types:

  audit   : issue + history + current hypothesis + evidence  -> VERDICT/SUPPORT/FLAW/NEXT CHECK
            reference = the v3 auditor's entry (verdict / flaw / support / should_have_turned)
  propose : issue + gathered evidence (+EDITS/SOURCE on correction rounds) -> hypothesis or probe
            expected = "probe" when the reference hypothesis names code absent from the input (the same
            cannot-derive gate export uses), else "propose"

References NEVER enter the input. Items are stable given the same raw data (sorted iteration, no RNG).

  python src/eval_build.py            # -> dataset/eval/items.jsonl
"""
import json
import re
import sys
from pathlib import Path

import typer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import _anchors, _evidence, _parse_json, _protocol, UNIFIED_ASK  # noqa: E402  (same gates/rendering as training)
import hashlib  # noqa: E402
import _schema  # noqa: E402  — schema-transfer twins + format-bleed items (surface-invariance eval)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
OUT = ROOT / "dataset" / "eval"
app = typer.Typer(add_completion=False)


def build_instance(iid: str, run_id: str, gold: dict) -> list[dict]:
    """Mirror of export.export_instance, emitting (input, reference) eval items instead of training targets."""
    rd = RAW / iid / run_id
    if not (rd / "ledger.json").exists() or not (rd / "audit.json").exists():
        return []
    ledger = json.loads((rd / "ledger.json").read_text(encoding="utf-8"))
    audit = _parse_json((rd / "audit.json").read_text(encoding="utf-8"))
    if not audit:
        return []
    cards = ledger["cards"]
    proto = _protocol(rd)  # correction cards of an old raw_leak trajectory are contaminated references:
    # the agent saw raw hidden-test output there — export drops those samples from training, and the
    # held-out eval must not measure against them either (holdout instances are protocol-mixed too)
    by_card: dict[int, dict] = {}
    for j, e in enumerate(audit.get("per_hypothesis", [])):
        if isinstance(e, dict):
            try:
                k = int(e.get("card"))
            except (TypeError, ValueError):
                k = j + 1
            by_card.setdefault(k, e)
    issue = (gold.get("problem_statement") or "")[:3000]
    items, prior = [], []
    found_so, insp_so = list(ledger.get("preamble_found", [])), []
    views_so: dict[str, str] = dict(ledger.get("preamble_views") or {})
    edits_so: list[tuple[str, str]] = []
    for i, card in enumerate(cards):
        a = by_card.get(i + 1, {})
        hyp, ev = card["hypothesis"], _evidence(card)
        if proto == "raw_leak" and card.get("trigger") in ("self_rescue", "oracle"):
            # skip emitting ITEMS for contaminated correction cards, but keep walking so evidence/prior
            # accumulation stays identical (later explore cards still need the true history)
            for h in card.get("found", []):
                if h not in found_so:
                    found_so.append(h)
            for f in card.get("inspected", []):
                if f not in insp_so:
                    insp_so.append(f)
            for f, v in (card.get("read_snippets") or {}).items():
                views_so.pop(f, None)
                views_so[f] = v
            for e in card.get("edits", []):
                edits_so.append((card["hypothesis_id"], e))
            prior.append((hyp, (a.get("flaw_given_info_at_the_time") or "").strip()[:240],
                          (a.get("should_have_turned") or "").strip()[:240]))
            continue
        hist = ("Investigation so far:\n" + "\n".join(f"- {h}" + (f"\n  (diagnosis: {fl})" if fl else "")
                                                      for h, fl, _t in prior) + "\n\n") if prior else ""
        if a:  # ---- audit item ----
            ui = (f"ISSUE:\n{issue}\n\n{hist}CURRENT HYPOTHESIS (the agent's conjecture about the cause):\n{hyp}\n\n"
                  f"EVIDENCE available at this point:\n{ev}\n\n"
                  "Audit this hypothesis: given only the information available, is it a sound conjecture, and what is wrong / insufficient about it?")
            items.append({"item_id": f"{iid}::{card['hypothesis_id']}::audit", "task": "audit",
                          "instance_id": iid, "hypothesis_id": card["hypothesis_id"], "source": card.get("trigger"),
                          "input": ui,
                          "reference": {"verdict": a.get("verdict", ""),
                                        "support": a.get("support_calibration", ""),
                                        "flaw": a.get("flaw_given_info_at_the_time", ""),
                                        "next_check": a.get("should_have_turned", "")}})
        if a.get("verdict") == "good":  # ---- propose/probe item (same construction + gate as export) ----
            evid = ""
            if found_so:
                evid += "Relevant code located (file:line):\n" + "\n".join(f"- {h}" for h in found_so[-12:]) + "\n"
            if insp_so:
                evid += "Files inspected: " + ", ".join(insp_so[-10:]) + "\n"
            if card.get("trigger") in ("self_rescue", "oracle") and edits_so:
                evid += ("EDITS the failed attempt had made (cumulative changes at submit time):\n"
                         + "\n".join(f"- [{h}] {e}" for h, e in edits_so[-8:]) + "\n")
                if views_so:
                    evid += ("SOURCE FILES as the agent last read them:\n" + "\n\n".join(
                        f"# {f}\n{v}" for f, v in list(views_so.items())[-3:]) + "\n")
            if prior:
                evid += ("Hypotheses already tried and RULED OUT (with the diagnosis of what was wrong):\n"
                         + "\n".join(f"- {h}" + (f"\n  diagnosis: {fl}" if fl else "") for h, fl, _t in prior) + "\n")
            ui = (f"ISSUE:\n{issue}\n\nEVIDENCE GATHERED SO FAR:\n{evid or '(only the issue so far)'}\n"
                  + UNIFIED_ASK)  # shared constant: eval items must stay byte-identical with training prompts
            is_corr = card.get("trigger") in ("self_rescue", "oracle") and prior and prior[-1][1]
            hyp_anchors = _anchors(hyp)
            expected = "probe" if (is_corr and hyp_anchors and not any(t in ui for t in hyp_anchors)) else "propose"
            items.append({"item_id": f"{iid}::{card['hypothesis_id']}::propose", "task": "propose",
                          "instance_id": iid, "hypothesis_id": card["hypothesis_id"], "source": card.get("trigger"),
                          "expected": expected, "input": ui,
                          "reference": {"hypothesis": hyp, "why": a.get("why_proposed", ""),
                                        "check": a.get("how_to_check", ""),
                                        "gap": (prior[-1][1] if is_corr else "")}})
        for h in card.get("found", []):
            if h not in found_so:
                found_so.append(h)
        for f in card.get("inspected", []):
            if f not in insp_so:
                insp_so.append(f)
        for f, v in (card.get("read_snippets") or {}).items():
            views_so.pop(f, None)
            views_so[f] = v
        for e in card.get("edits", []):
            edits_so.append((card["hypothesis_id"], e))
        prior.append((hyp, (a.get("flaw_given_info_at_the_time") or "").strip()[:240],
                      (a.get("should_have_turned") or "").strip()[:240]))
    return items


@app.command()
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    holdout = set(json.loads((ROOT / "dataset" / "splits" / "pro_test.json").read_text()))
    ds = {i["instance_id"]: i for i in load_dataset("ScaleAI/SWE-bench_Pro", split="test")}
    items, n_inst = [], 0
    for iid in sorted(holdout):
        if (RAW / iid / "pro1" / "ledger.json").exists() and iid in ds:
            got = build_instance(iid, "pro1", ds[iid])
            if got:
                n_inst += 1
                items.extend(got)
    from collections import Counter
    c = Counter(it["task"] for it in items)
    ce = Counter(it.get("expected") for it in items if it["task"] == "propose")
    # schema-transfer twins (audit only, 1/4 of audits): input requests a schema NEVER seen in training, same
    # reference — measures whether the audit skill is surface-invariant (the payoff of the training-side skins).
    xfer = [{**it, "item_id": it["item_id"] + "::xfer",
             "input": it["input"] + _schema.note(_schema.TASK_LABELS["audit"], _schema.SCHEMA_TRANSFER),
             "schema": _schema.SCHEMA_TRANSFER, "schema_transfer": True}
            for it in items if it["task"] == "audit"
            and int(hashlib.sha1(it["item_id"].encode()).hexdigest(), 16) % 4 == 0]
    items.extend(xfer)
    items.extend(_schema.FORMAT_BLEED_ITEMS)  # out-of-domain prompts: the SFT model must NOT emit our fields here
    with (OUT / "items.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[eval_build] {n_inst} held-out instances -> {len(items)} items "
          f"(audit {c.get('audit',0)}, propose {c.get('propose',0)} [expected: {dict(ce)}], "
          f"schema-transfer {len(xfer)}, format-bleed {len(_schema.FORMAT_BLEED_ITEMS)}) -> {OUT/'items.jsonl'}")
    print("[note] the eval set GROWS as the batch collects more pro_test instances — rebuild before each eval run;"
          " items are deterministic per instance, so arms built from the same file are always comparable.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    app()
