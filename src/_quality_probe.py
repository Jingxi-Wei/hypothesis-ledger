"""Five data-quality probes (2026-07-03, user-specified). READ-ONLY diagnosis — changes nothing.

 1. audit verdict distribution (P4 suspect): good share overall + by trajectory outcome
 2. TEST/check-step presence in propose targets (P1/P3 suspect)
 3. REASONING evidence-citation density vs connective-word density (rationalization vs reasoning)
 4. adjacent-hypothesis similarity within a trajectory (spin detection)
 5. redirect trigger purity: does a turn follow NEW evidence, or is it positional?

Writes sample dumps for eyeballing to dataset/_checks/quality_probe/ ; prints the numbers.
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMP = ROOT / "dataset" / "samples"
RAW = ROOT / "dataset" / "raw"
OUT = ROOT / "dataset" / "_checks" / "quality_probe"
RUNS = ("r1", "pro1")

_BT = re.compile(r"`([^`\n]{2,80})`")
_PA = re.compile(r"\b[\w./\\-]*\w\.(?:py|go|js|ts|tsx|jsx|c|h|cpp|rb|java)\b")
_SN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CA = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_STOP = {"fail_to_pass", "pass_to_pass", "self_rescue", "why_proposed"}


def anchors(t: str) -> set[str]:
    a = {m.strip() for m in _BT.findall(t) if len(m.strip()) >= 3}
    a |= set(_PA.findall(t))
    a |= {m for m in _SN.findall(t) if len(m) >= 6 and m not in _STOP}
    a |= {m for m in _CA.findall(t) if len(m) >= 6}
    return a


def load(typ: str) -> list[dict]:
    rows = []
    for run in RUNS:
        f = SAMP / run / f"{typ}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                s = json.loads(line)
                s["_run"] = run
                rows.append(s)
    return rows


def outcome_of(run: str, iid: str, cache={}) -> str:
    k = (run, iid)
    if k not in cache:
        p = RAW / iid / run / "outcome.json"
        cache[k] = json.loads(p.read_text(encoding="utf-8")).get("outcome", "?") if p.exists() else "?"
    return cache[k]


def ledgers() -> dict[tuple, dict]:
    out = {}
    for run in RUNS:
        for p in RAW.iterdir():
            lp = p / run / "ledger.json"
            if lp.exists():
                out[(run, p.name)] = json.loads(lp.read_text(encoding="utf-8"))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit, propose = load("audit"), load("propose")
    leds = ledgers()

    # ---------- 1. verdict distribution ----------
    print("=" * 30, "[1] audit VERDICT distribution", "=" * 30)
    vd = Counter()
    by_outcome = defaultdict(Counter)
    sc = Counter()
    for s in audit:
        m = re.search(r"VERDICT: (\w+)", s["messages"][1]["content"])
        v = m.group(1) if m else "?"
        vd[v] += 1
        by_outcome[outcome_of(s["_run"], s["instance_id"])][v] += 1
        m2 = re.search(r"SUPPORT: (\w+)", s["messages"][1]["content"])
        sc[m2.group(1) if m2 else "?"] += 1
    tot = sum(vd.values())
    for v, c in vd.most_common():
        print(f"  {v:8s} {c:4d}  ({c/tot:.0%})")
    print(f"  -> good share = {vd.get('good',0)/tot:.0%}  (病灶阈值 <25%)")
    print("  support_calibration:", dict(sc.most_common()))
    print("  by trajectory outcome (rows: outcome, cols: verdict %):")
    for o, cs in sorted(by_outcome.items()):
        t = sum(cs.values())
        row = "  ".join(f"{v}={cs.get(v,0)/t:.0%}" for v in ("good", "weak", "wrong"))
        print(f"    {o:18s} n={t:4d}  {row}")

    # ---------- 2. TEST/check-step presence in propose targets ----------
    print("\n" + "=" * 30, "[2] check-step presence in propose targets", "=" * 30)
    OP = re.compile(r"\b(?:re-?run|run(?:ning)?|compare|comparing|check(?:ing)? (?:whether|if|that)|verify|measure|"
                    r"inspect(?:ing)?|add(?:ing)? a (?:test|print|assert)|write a (?:test|script)|reproduce|"
                    r"repro(?:duction)? (?:script|case)|try (?:it|the)|toggle|swap|instrument)\b", re.I)
    ADJ = re.compile(r"\b(?:falsifiable|checkable|testable|verifiable)\b", re.I)
    has_op = has_adj = both = neither = 0
    dump2 = []
    rng = random.Random(7)
    for s in propose:
        ti = s["messages"][1]["content"]
        reasoning = (re.search(r"REASONING: (.*)\Z", ti, re.S) or [None, ""])[1] if "REASONING:" in ti else ti
        o, a2 = bool(OP.search(reasoning)), bool(ADJ.search(reasoning))
        has_op += o; has_adj += a2; both += o and a2; neither += (not o) and (not a2)
        dump2.append((s["_run"], s["instance_id"], s["hypothesis_id"], s.get("source"), o, a2, ti))
    n = len(propose)
    print(f"  propose n={n}")
    print(f"  操作性检验动词出现:  {has_op:4d} ({has_op/n:.0%})   ← 上界(动词出现≠给了检验步骤)")
    print(f"  'falsifiable'类形容词: {has_adj:4d} ({has_adj/n:.0%})   ← 坏信号计数")
    print(f"  两者皆无:            {neither:4d} ({neither/n:.0%})")
    print("  NOTE: propose 的 target 模板 = HYPOTHESIS+REASONING(why_proposed),设计上就没有 TEST/检验字段(audit 的 NEXT CHECK 才有)")
    sample2 = rng.sample(dump2, min(30, len(dump2)))
    (OUT / "p2_propose_targets_sample30.txt").write_text(
        "\n\n".join(f"### {r}/{i} {h} src={src} op={o} adj={a}\n{t}" for r, i, h, src, o, a, t in sample2), encoding="utf-8")

    # ---------- 3. REASONING citation density vs connectives ----------
    print("\n" + "=" * 30, "[3] REASONING evidence-citation vs connectives", "=" * 30)
    CONN = re.compile(r"\b(?:therefore|thus|hence|so|because|since|consequently)\b", re.I)
    rows3 = []
    for s in propose:
        ui, ti = s["messages"][0]["content"], s["messages"][1]["content"]
        m = re.search(r"REASONING: (.*)\Z", ti, re.S)
        if not m:
            continue
        reasoning = m.group(1)
        # evidence zone = input minus the ISSUE section (citing the issue is fine but weaker than citing gathered evidence)
        ev_zone = ui.split("EVIDENCE GATHERED SO FAR:", 1)[-1]
        cites = sum(1 for t in anchors(reasoning) if t in ev_zone)
        words = max(1, len(reasoning.split()))
        conn = len(CONN.findall(reasoning))
        rows3.append((cites, conn, words, s))
    c0 = sum(1 for c, _, _, _ in rows3 if c == 0)
    avg_c = sum(c for c, _, _, _ in rows3) / len(rows3)
    avg_conn = sum(cn / w * 100 for _, cn, w, _ in rows3) / len(rows3)
    print(f"  n={len(rows3)}  平均证据锚点引用/条={avg_c:.2f}   零引用条数={c0} ({c0/len(rows3):.0%})")
    print(f"  连接词密度(每百词)均值={avg_conn:.2f}")
    by_src3 = defaultdict(list)
    for c, cn, w, s in rows3:
        by_src3[s.get("source")].append(c)
    for src, cs in sorted(by_src3.items()):
        print(f"    source={src:12s} n={len(cs):4d}  平均引用={sum(cs)/len(cs):.2f}  零引用={sum(1 for x in cs if x==0)/len(cs):.0%}")
    samp3 = rng.sample(rows3, min(50, len(rows3)))
    (OUT / "p3_reasoning_sample50.txt").write_text(
        "\n\n".join(f"### {s['_run']}/{s['instance_id']} {s['hypothesis_id']} src={s.get('source')} cites={c} conn={cn}\n"
                    f"REASONING: {re.search(r'REASONING: (.*)', s['messages'][1]['content'], re.S).group(1)[:800]}"
                    for c, cn, w, s in samp3), encoding="utf-8")

    # ---------- 4. adjacent-hypothesis similarity (spin) ----------
    print("\n" + "=" * 30, "[4] adjacent hypothesis similarity (打转)", "=" * 30)
    pairs = []
    for (run, iid), led in leds.items():
        cards = led.get("cards", [])
        for a2, b in zip(cards, cards[1:]):
            r = SequenceMatcher(None, a2["hypothesis"].lower(), b["hypothesis"].lower()).ratio()
            pairs.append((r, run, iid, a2["hypothesis_id"], b["hypothesis_id"], b.get("trigger"), a2["hypothesis"], b["hypothesis"]))
    if pairs:
        rs = sorted(p[0] for p in pairs)
        import statistics
        hi = [p for p in pairs if p[0] > 0.55]
        print(f"  相邻 hypo 对 n={len(pairs)}  ratio 中位数={statistics.median(rs):.2f}  p90={rs[int(.9*len(rs))]:.2f}")
        print(f"  ratio>0.55(疑似换措辞不换实质): {len(hi)} 对 ({len(hi)/len(pairs):.0%})")
        for p in sorted(hi, reverse=True)[:8]:
            print(f"    {p[0]:.2f} {p[2][:40]} {p[3]}->{p[4]} [{p[5]}]")
        (OUT / "p4_similar_pairs.txt").write_text(
            "\n\n".join(f"### ratio={p[0]:.2f} {p[1]}/{p[2]} {p[3]}->{p[4]} trigger={p[5]}\nA: {p[6]}\nB: {p[7]}"
                        for p in sorted(hi, reverse=True)), encoding="utf-8")

    # ---------- 5. redirect trigger purity ----------
    print("\n" + "=" * 30, "[5] redirect trigger purity (转向前一卡的新信息量)", "=" * 30)
    stats = defaultdict(Counter)
    no_trig = []
    for (run, iid), led in leds.items():
        cards = led.get("cards", [])
        for i in range(1, len(cards)):
            prev, cur = cards[i - 1], cards[i]
            new_info = len(prev.get("found", [])) + len([t for t in prev.get("tests", []) if t.get("result")]) + len(prev.get("edits", []))
            trig = cur.get("trigger")
            changed = SequenceMatcher(None, prev["hypothesis"].lower(), cur["hypothesis"].lower()).ratio() < 0.5
            bucket = "with-new-evidence" if new_info > 0 else "NO-new-evidence"
            stats[trig][bucket] += 1
            if new_info == 0 and changed:
                no_trig.append((run, iid, cur["hypothesis_id"], trig, prev["hypothesis"][:90], cur["hypothesis"][:90]))
    for trig, cs in sorted(stats.items()):
        t = sum(cs.values())
        print(f"  trigger={trig:12s} n={t:4d}  转向前一卡有新信息={cs.get('with-new-evidence',0)/t:.0%}  无新信息={cs.get('NO-new-evidence',0)/t:.0%}")
    print(f"  无触发转向(前卡零新信息且假设实质变了): {len(no_trig)}")
    (OUT / "p5_no_trigger_turns.txt").write_text(
        "\n\n".join(f"### {r}/{i} -> {h} [{tg}]\nPREV: {a2}\nCUR:  {b}" for r, i, h, tg, a2, b in no_trig), encoding="utf-8")

    print(f"\n[dumps -> {OUT}]")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
