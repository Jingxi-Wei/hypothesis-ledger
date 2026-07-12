"""Redirect-anchored resample pairs: REAL trajectory hypothesis (chosen) x worse fresh candidates (rejected).

WHY (user design intent, 2026-07-09): the whole point of masking the oracle out of the resample context is
that candidates born WITHOUT the redirect are worse than the trajectory's stage-best hypothesis born WITH it.
That asymmetry IS the training signal — the RM learns to rank "where the corrective signal pushes" above
"what you'd propose without it", which is exactly the deploy-time BoN role (externalized oracle).
gen_pairs.py deliberately kept real_hyp out of pairs ("different birth" purism, line ~424); this script adds
the missing pair channel WITHOUT re-calling the proxy: everything needed (real hyp text, judge labels,
prompts) is already in the durable resample files, so derivation is pure post-processing.

Honesty guards kept from derive_pairs:
  * chosen (the real hyp) must be _grounded in the masked prompt — never reward citing unseen specifics
    (a thin post-feedback hop may cite code only seen via the direction; those are DROPPED and counted);
  * rejected must be judged STRICTLY worse on the correct>partial>wrong ladder;
  * pair records carry chosen_label / rejected_source / post_feedback for slicing (correction-round
    echo residue stays quantifiable, per HANDOFF).

  python rmscaffold/rederive_real_pairs.py --run-id pro2 --tag train
Output: rmscaffold/resample/pairs_realanchor-<run>_<tag>.jsonl
  (name matches prep_rm's pairs_*_train.jsonl glob -> merged automatically; deterministic, safe to re-run)
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
from export import _grounded, _norm_line  # noqa: E402  (same guards as gen_pairs.derive_pairs)

RANK = {"wrong": 0, "partial": 1, "correct": 2}
app = typer.Typer(add_completion=False)


@app.command()
def main(run_id: str = typer.Option("pro2", "--run-id"),
         tag: str = typer.Option("train", "--tag"),
         cap: int = typer.Option(4, "--cap-per-node"),
         seed: int = typer.Option(42, "--seed")) -> None:
    cand_p = HERE / "resample" / f"candidates_{run_id}_{tag}.jsonl"
    lab_p = HERE / "resample" / f"labels_{run_id}_{tag}.jsonl"
    out_p = HERE / "resample" / f"pairs_realanchor-{run_id}_{tag}.jsonl"
    if not cand_p.exists() or not lab_p.exists():
        raise SystemExit(f"[rederive] missing {cand_p.name} / {lab_p.name}")

    def rows(p):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # in-progress writer may leave a torn last line; skip

    meta = {r["node_id"]: r for r in rows(cand_p)}
    # tier-consistency filter (2026-07-09): when the relative re-judge (rejudge_rank.py) has run, candidates
    # it placed in 'par'/'above' groups are NOT worse than the real hypothesis by direct judgment — never use
    # them as rejected here, even if the S4 absolute-label ladder disagrees (judges clash at boundaries).
    rank_p = HERE / "resample" / f"rankings_{run_id}_{tag}.jsonl"
    not_worse: dict[str, set] = {}
    if rank_p.exists():
        for r in rows(rank_p):
            if r.get("tiers"):
                keep = set()
                for gi, t in enumerate(r["tiers"]):
                    if t in ("par", "above"):
                        for a in r["groups"][gi]:
                            keep.add(r["candidates"][a - 1])
                not_worse[r["node_id"]] = keep
    rng = random.Random(seed)
    st = Counter()
    out = []
    for lab in rows(lab_p):
        nid = lab["node_id"]
        node = meta.get(nid)
        real_label = lab.get("real_label")
        st["nodes"] += 1
        if node is None or real_label is None:
            st["no_real"] += 1
            continue
        st[f"real_{real_label}"] += 1
        cands, labels = lab["candidates"][:-1], lab["labels"][:-1]  # last slot = the real hyp itself
        real = lab["candidates"][-1]
        prompt = node["prompt"]
        if not _grounded(real, prompt):
            st["chosen_ungrounded_dropped"] += 1  # direction-informed hop citing unseen code: not rewardable
            continue
        nw = not_worse.get(nid, set())
        worse = [(c, l) for c, l in zip(cands, labels)
                 if RANK.get(l, 0) < RANK[real_label] and _norm_line(c).casefold() != _norm_line(real).casefold()
                 and c not in nw]
        st["tier_filtered"] += sum(1 for c, l in zip(cands, labels)
                                   if RANK.get(l, 0) < RANK[real_label] and c in nw)
        rng.shuffle(worse)
        for c, l in worse[:cap]:
            st[f"pair_{real_label}_over_{l}"] += 1
            out.append({"type": "preference", "pair_type": "resample_real", "instance_id": node["instance_id"],
                        "protocol": "sanitized",  # pro2 collection ran entirely on the sanitized protocol
                        "strength": "judged", "rejected_source": f"judge_{l}", "chosen_label": real_label,
                        "chosen_grounded": True, "rejected_grounded": _grounded(c, prompt), "node": nid,
                        "context_mode": node.get("context_mode", "ledger"), "gen_model": node.get("gen_model", ""),
                        "post_feedback": node.get("post_feedback", False),
                        "prompt": prompt, "chosen": real, "rejected": c})
    out_p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    st["pairs_total"] = len(out)
    print(f"[rederive] {out_p.name}: " + " ".join(f"{k}={v}" for k, v in sorted(st.items())))


if __name__ == "__main__":
    app()
