"""Pro downstream-calibration accounting. Reads ledgers+audits+outcomes+trajectories + the exported samples
and prints the sample-level picture the collection plan asks for at the stop point:
  - outcome distribution + correction fraction (instance level)
  - card verdict/support calibration (is the auditor's judgment spread sane, or collapsed to one label?)
  - exported sample counts by type/source + correction_frac / redirect% / RM clean pairs
  - METRIC 1 (escalating-oracle): deep-round (oracle) probe-gate rate + why_proposed grounding by oracle depth
  - METRIC 2: post-feedback exploration-step fingerprint distribution (thin = direction did the locating)
Run AFTER audits + export are done."""
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "dataset" / "raw"
SAMP = ROOT / "dataset" / "samples" / "pro2"
RUN = "pro2"

import compress as C  # noqa: E402
import export as E  # noqa: E402
from datasets import load_dataset  # noqa: E402


def bar(counter, total=None):
    total = total or sum(counter.values())
    return "  ".join(f"{k}={v}({100*v/total:.0f}%)" for k, v in sorted(counter.items(), key=lambda x: -x[1]))


# ---------- A. instance-level outcome distribution ----------
outc = Counter()
have_audit = []
for f in sorted(glob.glob(str(RAW / "*" / RUN / "outcome.json"))):
    d = Path(f).parent
    o = json.loads(Path(f).read_text())
    outc[o["outcome"]] += 1
    if (d / "audit.json").exists() and (d / "ledger.json").exists():
        have_audit.append(d.parent.name)
n_inst = sum(outc.values())
corr_inst = outc.get("oracle_redirected", 0) + outc.get("self_corrected", 0)
print("=" * 78)
print(f"A. OUTCOMES ({n_inst} complete instances) | audited+compressed: {len(have_audit)}")
print(f"   {bar(outc)}")
print(f"   correction instances (oracle_redirected + self_corrected): {corr_inst} = {100*corr_inst/n_inst:.0f}%")

# ---------- B. card verdict / support calibration ----------
vt = defaultdict(Counter)  # trigger -> verdict counts
support = Counter()
ncards_trig = Counter()
for iid in have_audit:
    rd = RAW / iid / RUN
    ledger = json.loads((rd / "ledger.json").read_text(encoding="utf-8"))
    audit = E._parse_json((rd / "audit.json").read_text(encoding="utf-8")) or {}
    by_card = {}
    for j, e in enumerate(audit.get("per_hypothesis", [])):
        if isinstance(e, dict):
            try:
                by_card.setdefault(int(e.get("card")), e)
            except (TypeError, ValueError):
                by_card.setdefault(j + 1, e)
    for i, card in enumerate(ledger["cards"]):
        trig = card.get("trigger", "?")
        ncards_trig[trig] += 1
        a = by_card.get(i + 1, {})
        vt[trig][a.get("verdict", "?")] += 1
        support[a.get("support_calibration", "?")] += 1
allv = Counter()
for t in vt:
    allv.update(vt[t])
print("=" * 78)
print(f"B. CARDS ({sum(ncards_trig.values())} total) by trigger: {bar(ncards_trig)}")
print(f"   verdict (all): {bar(allv)}")
for t in ("explore", "self_rescue", "oracle"):
    if vt[t]:
        print(f"     {t:12}: {bar(vt[t])}")
print(f"   support_calibration: {bar(support)}")

# ---------- C. exported samples ----------
print("=" * 78)
print("C. EXPORTED SAMPLES (dataset/samples/pro2/)")
if not SAMP.exists():
    print("   (not exported yet — run: DATASET=pro python src/export.py --dataset pro --run-id pro2)")
else:
    by_type = Counter()
    by_source = Counter()
    tp = Counter()
    corr_samples = 0
    for t in ("audit", "propose", "fix", "probe"):
        fp = SAMP / f"{t}.jsonl"
        if not fp.exists():
            continue
        for line in fp.read_text(encoding="utf-8").splitlines():
            s = json.loads(line)
            by_type[t] += 1
            by_source[s.get("source", "?")] += 1
            if s.get("turning_point"):
                tp[t] += 1
            if s.get("source") in ("self_rescue", "oracle"):
                corr_samples += 1
    tot = sum(by_type.values())
    pref = []
    pf = SAMP / "preference.jsonl"
    if pf.exists():
        pref = [json.loads(l) for l in pf.read_text(encoding="utf-8").splitlines()]
    clean_pref = sum(1 for p in pref if p.get("chosen_grounded"))
    print(f"   SFT samples ({tot}): {bar(by_type)}")
    print(f"   by source: {bar(by_source)}")
    if tot:
        print(f"   correction_frac (self_rescue+oracle / all samples): {100*corr_samples/tot:.0f}%")
        print(f"   redirect%       (oracle-sourced / all samples):     {100*by_source.get('oracle',0)/tot:.0f}%")
    print(f"   turning-point samples by type: {dict(tp)}")
    print(f"   RM preference pairs: {len(pref)} | chosen-grounded (clean): {clean_pref}")
    if pref:
        print(f"     pair_type: {bar(Counter(p['pair_type'] for p in pref))}")
        print(f"     rejected_source: {bar(Counter(p['rejected_source'] for p in pref))}")

# ---------- D. METRIC 1: deep-round probe-gate + grounding (via export walker) ----------
print("=" * 78)
print("D. METRIC 1 — oracle-round probe-gate + grounding (deep rounds should gate MORE = correctly refuse to teach a leap)")
ds = {i["instance_id"]: i for i in load_dataset(E.DS_NAME["pro"], split="test")}
holdout = set(json.loads((ROOT / "dataset" / "splits" / "pro_test.json").read_text()))
depth_stats = defaultdict(lambda: {"good": 0, "gated": 0})  # oracle-depth -> counts among good cards
for iid in have_audit:
    if iid not in ds or iid in holdout:
        continue
    w = E._walk(iid, RUN, ds[iid])
    if not w:
        continue
    nodes, _ = w
    odepth = 0
    for n in nodes:
        trig = n["card"].get("trigger")
        if trig == "oracle":
            odepth += 1
        if n["verdict"] == "good" and trig in ("self_rescue", "oracle"):
            key = f"oracle_{odepth}" if trig == "oracle" else "self_rescue"
            depth_stats[key]["good"] += 1
            if n["probe_gated"]:
                depth_stats[key]["gated"] += 1
for key in sorted(depth_stats):
    s = depth_stats[key]
    g = s["good"]
    rate = 100 * s["gated"] / g if g else 0
    print(f"   {key:12}: {g:3} good cards, probe-gated {s['gated']:3} = {rate:.0f}%  (grounded/propose = {100-rate:.0f}%)")

# ---------- E. METRIC 2: post-feedback exploration fingerprint ----------
print("=" * 78)
print("E. METRIC 2 — exploration steps between a feedback and the next edit (thin => direction did the locating)")
fingerprints = []
for iid in have_audit:
    rd = RAW / iid / RUN
    try:
        msgs = json.loads((rd / "trajectory.json").read_text(encoding="utf-8"))["messages"]
    except Exception:
        continue

    def txt(m):
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        return c or ""

    post, streak = False, 0
    for m in msgs:
        t = txt(m)
        if m.get("role") != "assistant":
            if "Sanitized test-feedback" in t or "guidance round" in t:
                post, streak = True, 0
            continue
        for a in m.get("extra", {}).get("actions", []):
            cmd = a.get("command", "") if isinstance(a, dict) else str(a)
            kind = C._classify(cmd, "swe")
            if post:
                if kind == "read":
                    streak += 1
                elif kind == "edit":
                    fingerprints.append(streak)
                    post, streak = False, 0
hist = Counter(min(f, 5) for f in fingerprints)  # bucket 5+ together
if fingerprints:
    import statistics
    print(f"   n={len(fingerprints)} feedback->edit transitions | mean={statistics.mean(fingerprints):.1f} median={statistics.median(fingerprints):.0f}")
    print("   distribution (steps-before-first-edit): " + "  ".join(f"{k}{'+' if k==5 else ''}={v}" for k, v in sorted(hist.items())))
    thin = sum(v for k, v in hist.items() if k <= 1)
    print(f"   THIN (<=1 explore step) = {100*thin/len(fingerprints):.0f}%  (high => oracle direction is doing the locating, not the agent)")
print("=" * 78)
