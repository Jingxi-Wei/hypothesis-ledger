"""Relative re-judge of stored resample candidates: mechanism GROUPS + group RANKING (one call per node).

WHY (user design correction, 2026-07-09): the S4 judge graded candidates ABSOLUTELY against gold
("correct = same mechanism as the gold patch"). But every candidate was born WITHOUT the oracle hint —
if a no-hint hypothesis could reach gold level, the agent wouldn't have needed the oracle at all. On hard
nodes everything lands in 'partial' and intra-candidate differences vanish (47/154 all-partial nodes, no
pairs). Deployment (BoN) is exactly the opposite situation: the RM must pick the most promising among N
imperfect no-hint candidates. So the judge must answer the RELATIVE question.

And the resamples are NOT all alike (the user's second point): some nodes yield 5-6 rewordings of one
mechanism (clones — ranking them would teach the RM style preferences, i.e. noise), others yield genuinely
different mechanisms (real signal). Hence two-step judging:
  (1) GROUP candidates asserting the same causal mechanism (rewordings -> one group);
  (2) RANK the groups by closeness to the true cause (judge sees gold; outputs labels only — established
      privileged-judge trust model).
Pairs are later derived ACROSS groups only (derive_rank_pairs.py); single-group (clone) nodes yield none.

Reuses stored candidates (zero generation cost). Resumable: skips node_ids already in the rankings file.
  python rmscaffold/rejudge_rank.py --run-id pro2 --tag train
Output: rmscaffold/resample/rankings_<run>_<tag>.jsonl  {node_id, groups, ranking}
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import typer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

GPT_MODEL_PREFIX = os.environ.get("GPT_MODEL_PREFIX", "openai/gpt-5.5")
GPT_SPEED = os.environ.get("GPT_SPEED", "fast")
JUDGE_EFFORT = os.environ.get("JUDGE_EFFORT", "high")

_SYS_TIER = ("You compare candidate root-cause hypotheses for a known bug. You are given the ground truth (gold "
             "patch and, when available, the verified cause), and a REFERENCE hypothesis — the one the agent "
             "actually reached at this point WITH a guidance hint. The candidates were written WITHOUT that hint, "
             "so most are imperfect — absolute correctness is NOT the question. Do three things: "
             "(1) GROUP candidates that assert the same causal mechanism at the same location — rewordings of one "
             "idea belong in ONE group; split only on genuine mechanism/location differences. "
             "(2) For each group, judge its TIER against the REFERENCE, where closer to the true cause is better: "
             "'below' = farther from the true cause than the reference; 'par' = the same level (same mechanism or "
             "an equally close one); 'above' = strictly closer to the true cause than the reference. "
             "(3) RANK the groups from closest to the true cause to farthest (no ties — grouping absorbs "
             "equivalence; the ranking must be consistent with the tiers). "
             "Judge mechanism proximity only — not style, not length, not confidence. "
             'Output STRICT JSON only: {"groups": [[<1-based candidate numbers>], ...], '
             '"tiers": ["below"|"par"|"above" per group], '
             '"ranking": [<group indices into "groups", 0-based, best first>]}')

_SYS_PLAIN = ("You compare candidate root-cause hypotheses for a known bug. You are given the ground truth (gold "
              "patch and, when available, the verified cause). The candidates were written WITHOUT any hint, so "
              "most are imperfect — absolute correctness is NOT the question. Do two things: "
              "(1) GROUP candidates that assert the same causal mechanism at the same location — rewordings of one "
              "idea belong in ONE group; split only on genuine mechanism/location differences. "
              "(2) RANK the groups from closest to the true cause to farthest (no ties — grouping absorbs "
              "equivalence). Judge mechanism proximity only — not style, not length, not confidence. "
              'Output STRICT JSON only: {"groups": [[<1-based candidate numbers>], ...], '
              '"ranking": [<group indices into "groups", 0-based, best first>]}')


def _kw(effort: str) -> dict:
    suffix = f"-{GPT_SPEED}" if GPT_SPEED else ""
    return dict(model=f"{GPT_MODEL_PREFIX}-{effort}{suffix}",
                api_base="http://127.0.0.1:8080/v1", api_key="pwd")


def _llm(sys_p: str, user_p: str, retries: int = 2) -> str:
    import litellm
    for att in range(retries + 1):
        try:
            r = litellm.completion(messages=[{"role": "system", "content": sys_p},
                                             {"role": "user", "content": user_p}], **_kw(JUDGE_EFFORT))
            return r.choices[0].message.content or ""
        except Exception:
            if att == retries:
                raise
            time.sleep(5 * (att + 1))
    return ""


def _parse(txt: str, k: int, want_tiers: bool):
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        groups = [[int(x) for x in g] for g in d["groups"]]
        ranking = [int(x) for x in d["ranking"]]
        tiers = [str(t) for t in d["tiers"]] if want_tiers else None
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None
    flat = sorted(x for g in groups for x in g)
    if flat != list(range(1, k + 1)):            # exact partition of 1..k
        return None
    if sorted(ranking) != list(range(len(groups))):  # exact permutation of group indices
        return None
    if want_tiers and (len(tiers) != len(groups) or any(t not in ("below", "par", "above") for t in tiers)):
        return None
    out = {"groups": groups, "ranking": ranking}
    if want_tiers:
        out["tiers"] = tiers
    return out


app = typer.Typer(add_completion=False)


@app.command()
def main(run_id: str = typer.Option("pro2", "--run-id"),
         tag: str = typer.Option("train", "--tag"),
         limit: int = typer.Option(0, "--limit", help="max NEW nodes this session (0 = all)")) -> None:
    cand_p = HERE / "resample" / f"candidates_{run_id}_{tag}.jsonl"
    lab_p = HERE / "resample" / f"labels_{run_id}_{tag}.jsonl"
    out_p = HERE / "resample" / f"rankings_{run_id}_{tag}.jsonl"
    if not cand_p.exists() or not lab_p.exists():
        raise SystemExit(f"[rejudge] missing {cand_p.name} / {lab_p.name}")

    def rows(p):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    meta = {r["node_id"]: r for r in rows(cand_p)}
    labs = {r["node_id"]: r for r in rows(lab_p)}               # carries the real hyp (last slot) = the anchor
    done = {r["node_id"] for r in rows(out_p)} if out_p.exists() else set()
    todo = [n for n in labs if n in meta and n not in done]
    print(f"[rejudge] {len(todo)} nodes to rank ({len(done)} already done)", flush=True)
    if limit:
        todo = todo[:limit]

    from collect import load_instances  # candidates file carries no gold; fetch per instance from the dataset
    ds = load_instances("pro")

    st = Counter()
    with out_p.open("a", encoding="utf-8") as f:
        for i, nid in enumerate(sorted(todo), 1):
            node = meta[nid]
            # candidates WITHOUT the real hyp (labels file appends real at the end; candidates file has gen only)
            cands = list(dict.fromkeys(c for c in node["candidates"] if c))
            if len(cands) < 2:
                st["too_few"] += 1
                continue
            gold = (ds.get(node["instance_id"], {}).get("patch") or "")
            lab_row = labs[nid]
            real = lab_row["candidates"][-1] if lab_row.get("real_label") is not None else None
            ref_block = (f"\n\n# REFERENCE hypothesis (reached WITH a guidance hint — the tier anchor)\n{real}"
                         if real else "")
            sys_p = _SYS_TIER if real else _SYS_PLAIN
            user_p = ("# Context the candidates saw (may be truncated)\n" + node["prompt"][-8000:]
                      + "\n\n# GROUND TRUTH (privileged — output labels only)\n" + (gold[:4000] or "(gold patch unavailable — rank by internal consistency with the context evidence)")
                      + ref_block
                      + "\n\n# Candidates (written WITHOUT the hint)\n" + "\n".join(f"{j + 1}. {c}" for j, c in enumerate(cands)))
            try:
                parsed = _parse(_llm(sys_p, user_p), len(cands), want_tiers=bool(real))
            except Exception as e:
                st["errors"] += 1
                print(f"  [{i}/{len(todo)}] {nid[-45:]}: ERROR {type(e).__name__}: {str(e)[:100]}", flush=True)
                continue
            if parsed is None:
                st["unparseable"] += 1
                print(f"  [{i}/{len(todo)}] {nid[-45:]}: unparseable, skipped", flush=True)
                continue
            f.write(json.dumps({"node_id": nid, "candidates": cands, "real_hyp": real, **parsed},
                               ensure_ascii=False) + "\n")
            f.flush()
            st["ranked"] += 1
            st[f"n_groups_{len(parsed['groups'])}"] += 1
            for t in parsed.get("tiers", []):
                st[f"tier_{t}"] += 1  # how often no-hint candidates reach/beat the guided bar = direction value
            print(f"  [{i}/{len(todo)}] {nid[-45:]}: {len(parsed['groups'])} group(s) tiers={parsed.get('tiers')}", flush=True)
    print(f"[rejudge done] {dict(st)}", flush=True)


if __name__ == "__main__":
    app()
