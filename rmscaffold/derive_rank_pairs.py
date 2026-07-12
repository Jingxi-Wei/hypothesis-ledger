"""Cross-group sibling pairs from the relative re-judge (rankings_<run>_<tag>.jsonl).

Pairs candidates ACROSS mechanism groups only: chosen from a better-ranked group, rejected from a worse one.
Clone nodes (single group) yield nothing — ranking rewordings of one idea would teach style preference, not
mechanism preference (the user's "重采不是所有重采都类似" point). Guards, consistent with the other channels:
  * chosen must be _grounded in the masked prompt (never reward citing unseen specifics);
  * label-consistency: skip combos where the S4 ABSOLUTE label contradicts the rank order
    (rejected labeled strictly better than chosen), belt to the ranking's braces;
  * widest rank gaps first, cap per node.
  python rmscaffold/derive_rank_pairs.py --run-id pro2 --tag train
Output: rmscaffold/resample/pairs_rankgap-<run>_<tag>.jsonl  (matches prep_rm's pairs_*_train.jsonl glob)
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import typer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from export import _grounded, _norm_line  # noqa: E402

RANK = {"wrong": 0, "partial": 1, "correct": 2}
app = typer.Typer(add_completion=False)


@app.command()
def main(run_id: str = typer.Option("pro2", "--run-id"),
         tag: str = typer.Option("train", "--tag"),
         cap: int = typer.Option(4, "--cap-per-node"),
         seed: int = typer.Option(42, "--seed")) -> None:
    cand_p = HERE / "resample" / f"candidates_{run_id}_{tag}.jsonl"
    lab_p = HERE / "resample" / f"labels_{run_id}_{tag}.jsonl"
    rank_p = HERE / "resample" / f"rankings_{run_id}_{tag}.jsonl"
    out_p = HERE / "resample" / f"pairs_rankgap-{run_id}_{tag}.jsonl"
    for p in (cand_p, lab_p, rank_p):
        if not p.exists():
            raise SystemExit(f"[rankpairs] missing {p.name}")

    def rows(p):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    meta = {r["node_id"]: r for r in rows(cand_p)}
    labs = {r["node_id"]: r for r in rows(lab_p)}
    rng = random.Random(seed)
    st = Counter()
    out = []
    for rk in rows(rank_p):
        nid = rk["node_id"]
        node, lab_row = meta.get(nid), labs.get(nid)
        if node is None or lab_row is None:
            st["orphan"] += 1
            continue
        st["nodes"] += 1
        groups, ranking, cands = rk["groups"], rk["ranking"], rk["candidates"]
        tiers, real = rk.get("tiers"), rk.get("real_hyp")
        # absolute labels for THESE candidates: labels file rows are uniq(+real); map by text
        lab_uniq = lab_row["candidates"][:-1] if lab_row.get("real_label") is not None else lab_row["candidates"]
        lab_vals = lab_row["labels"][:-1] if lab_row.get("real_label") is not None else lab_row["labels"]
        label_of = {_norm_line(c).casefold(): l for c, l in zip(lab_uniq, lab_vals)}
        prompt = node["prompt"]
        base = {"type": "preference", "instance_id": node["instance_id"], "protocol": "sanitized",
                "strength": "judged", "node": nid, "context_mode": node.get("context_mode", "ledger"),
                "gen_model": node.get("gen_model", ""), "post_feedback": node.get("post_feedback", False),
                "prompt": prompt}

        # ---- channel A: 'above'-tier candidate BEATS the guided real hypothesis (rare, precious:
        # a no-hint candidate judged strictly closer to gold than what the agent reached with the hint) ----
        if tiers and real:
            st[f"tiers_{'/'.join(sorted(set(tiers)))}"] += 0  # touch for stats key visibility
            for gi, t in enumerate(tiers):
                if t != "above":
                    continue
                for a in groups[gi]:
                    ca = cands[a - 1]
                    if not _grounded(ca, prompt):
                        st["excellent_ungrounded_skipped"] += 1
                        continue
                    if _norm_line(ca).casefold() == _norm_line(real).casefold():
                        continue
                    st["pairs_excellent_over_real"] += 1
                    out.append({**base, "pair_type": "resample_excellent", "rejected_source": "real_below_candidate",
                                "chosen_label": label_of.get(_norm_line(ca).casefold()), "rejected_label": "real_hyp",
                                "chosen_grounded": True, "rejected_grounded": _grounded(real, prompt),
                                "chosen": ca, "rejected": real})

        # ---- channel B: cross-TIER candidate pairs (above>par, above>below, par>below); when tiers are
        # absent (no real anchor at this node) fall back to pure cross-group ranking ----
        if len(groups) < 2:
            st["single_group_skipped"] += 1  # clones — by design, no candidate-vs-candidate pairs
            continue
        TIER_RANK = {"below": 0, "par": 1, "above": 2}
        pos = {gi: r for r, gi in enumerate(ranking)}  # group index -> rank position (0 = best)
        combos = []
        for gi_a, ga in enumerate(groups):
            for gi_b, gb in enumerate(groups):
                if tiers:
                    gap = TIER_RANK[tiers[gi_a]] - TIER_RANK[tiers[gi_b]]
                else:
                    gap = pos[gi_b] - pos[gi_a]
                if gap <= 0:
                    continue
                for a in ga:
                    for b in gb:
                        ca, cb = cands[a - 1], cands[b - 1]
                        la = label_of.get(_norm_line(ca).casefold())
                        lb = label_of.get(_norm_line(cb).casefold())
                        if la is not None and lb is not None and RANK[lb] > RANK[la]:
                            st["label_contradiction_skipped"] += 1  # absolute label outranks the rank order
                            continue
                        if not _grounded(ca, prompt):
                            st["chosen_ungrounded_skipped"] += 1
                            continue
                        combos.append((gap, ca, cb, la, lb))
        rng.shuffle(combos)
        combos.sort(key=lambda x: -x[0])  # widest tier/mechanism gaps carry the cleanest signal
        for gap, ca, cb, la, lb in combos[:cap]:
            st["pairs_rank"] += 1
            out.append({**base, "pair_type": "resample_rank", "rejected_source": "judge_rank",
                        "rank_gap": gap, "chosen_label": la, "rejected_label": lb,
                        "chosen_grounded": True, "rejected_grounded": _grounded(cb, prompt),
                        "chosen": ca, "rejected": cb})
    out_p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"[rankpairs] {out_p.name}: " + " ".join(f"{k}={v}" for k, v in sorted(st.items())))


if __name__ == "__main__":
    app()
