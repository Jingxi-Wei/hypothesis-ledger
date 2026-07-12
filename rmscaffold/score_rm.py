"""RM quality check (GPU side): chosen > rejected accuracy on the HELD-OUT pairs — the RM headline number.

Run AFTER train_rm.sh (and after check_template.py PASSES), with rm_eval_pairs.jsonl built locally
by build_rm_eval.py and uploaded here.

  python score_rm.py --smoke     # 5 pairs, raw scores — MUST be non-constant (value-head sanity)
  python score_rm.py             # full held-out accuracy + slices -> rm_eval_scores.json

PICK MODE — the distribution-matched BoN number (upload the two resample eval files first):
  python score_rm.py --pick-candidates resample/candidates_pro2_eval.jsonl \
                     --pick-labels resample/labels_pro2_eval.jsonl
At every judged held-out node the RM scores ALL candidates; report P(picked is judge-correct) for
RM-pick vs random-pick (= per-node correct fraction) vs first-pick, with a bootstrap CI on the
RM-minus-random delta over MIXED nodes (nodes that have both correct and wrong candidates — the only
ones where selection can matter). This is BoN on the exact training/deployment context distribution,
no policy model load needed (candidates were pre-generated).

READ THE SLICES, not just the headline:
  * pair_type=resample is the number that predicts BoN (fresh-candidate discrimination, same distribution
    BoN scores). decision/issue_only measure don't-repeat-refuted / issue-level ranking.
  * strength: verified = outcome-verified labels (hard); judged = gold-anchored judge labels (softer).
  * length bias: score↔length correlation ≈ classic RM pathology ("longer = better"). If acc on the
    rejected-is-longer subset collapses versus the chosen-is-longer subset, the RM learned length, not content.

Honest gate: if overall accuracy ~50% (coin flip), the RM learned nothing discriminative — do NOT run BoN;
go back to the pair data (volume? label noise? prompt too thin?) before burning GPU time.
"""
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import typer

import rm_lib

HERE = Path(__file__).resolve().parent
app = typer.Typer(add_completion=False)


def _boot_ci(flags: list[bool], n: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    accs = sorted(sum(rng.choices(flags, k=len(flags))) / len(flags) for _ in range(n))
    return accs[int(0.025 * n)], accs[int(0.975 * n)]


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def _pick_eval(cand_p: Path, lab_p: Path) -> None:
    cands = {json.loads(l)["node_id"]: json.loads(l)
             for l in cand_p.read_text(encoding="utf-8").splitlines() if l.strip()}
    labs = [json.loads(l) for l in lab_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    model, tok = rm_lib.load_rm()
    rows = []
    for i, lab in enumerate(labs):
        node = cands.get(lab["node_id"])
        if not node or "prompt" not in node:
            continue  # old candidates file without stored prompt — regenerate with current gen_pairs
        judged, labels = lab["candidates"], lab["labels"]
        if lab.get("real_label") is not None:  # trajectory's real hypothesis was judged for calibration only
            judged, labels = judged[:-1], labels[:-1]
        if len(judged) < 2:
            continue
        scores = [rm_lib.score(model, tok, node["prompt"], c) for c in judged]
        pick = max(range(len(judged)), key=lambda j: scores[j])
        corr = [l == "correct" for l in labels]
        # mixed = has BOTH a judge-correct AND a judge-WRONG candidate (the approved headline definition).
        # 'partial' counts as neither: a correct-vs-partial-only node is a discrimination the RM was never
        # trained on (partial is excluded from both training sides) and must not dilute the headline.
        mixed = any(l == "correct" for l in labels) and any(l == "wrong" for l in labels)
        rows.append({"node_id": lab["node_id"], "rm_correct": corr[pick], "first_correct": corr[0],
                     "frac_correct": sum(corr) / len(corr), "mixed": mixed,
                     "post_feedback": node.get("post_feedback", False),
                     "context_mode": node.get("context_mode", "?"), "scores": scores, "labels": labels})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(labs)} nodes", flush=True)
    if not rows:
        print("[pick] no scorable nodes (candidates file must carry 'prompt' — regenerate with current gen_pairs)")
        return

    def _rate(rs, key):
        return sum(r[key] for r in rs) / len(rs)

    def _report(name, rs):
        if not rs:
            return None
        rm, rnd, fst = _rate(rs, "rm_correct"), _rate(rs, "frac_correct"), _rate(rs, "first_correct")
        deltas = [r["rm_correct"] - r["frac_correct"] for r in rs]
        lo, hi = _boot_ci(deltas)  # one CI implementation for the whole tool (works on float deltas too)
        print(f"  [{name}] n={len(rs)}  RM-pick {rm:.3f} | random {rnd:.3f} | first {fst:.3f} | "
              f"Δ(RM-random) {rm - rnd:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        return {"n": len(rs), "rm_pick": round(rm, 4), "random_pick": round(rnd, 4), "first_pick": round(fst, 4),
                "delta_rm_minus_random": round(rm - rnd, 4), "delta_ci95": [round(lo, 4), round(hi, 4)]}

    # BoN CURVE (the "real exam" — a single-N accuracy alone gets discounted by practitioners): P(RM-pick is
    # judge-correct) as N grows, vs the N-independent random baseline (= mean frac_correct). Exact enumeration
    # over all C(k, n) candidate subsets per node (k <= ~7, trivial). A rise-then-flat/-fall shape and where it
    # turns is the overoptimization-budget readout.
    from itertools import combinations
    curve = {}
    max_k = max((len(r["labels"]) for r in rows), default=0)
    for n_pick in range(1, min(max_k, 8) + 1):
        vals = []
        for r in rows:
            k = len(r["labels"])
            if k < n_pick:
                continue
            corr = [l == "correct" for l in r["labels"]]
            hits = tot = 0
            for sub in combinations(range(k), n_pick):
                best = max(sub, key=lambda j: r["scores"][j])
                hits += corr[best]
                tot += 1
            vals.append(hits / tot)
        if vals:
            curve[n_pick] = round(sum(vals) / len(vals), 4)
    print("  BoN curve (RM-pick correct-rate by N; random baseline = frac_correct, N-independent):")
    print("   ", "  ".join(f"N={n}:{v:.3f}" for n, v in curve.items()))

    print("[pick] BoN-style selection on judged held-out nodes (correct-by-judge = the success criterion):")
    out = {"bon_curve_rm_pick": curve,
           "all_nodes": _report("all nodes", rows),
           "mixed_nodes": _report("mixed nodes (selection can matter)", [r for r in rows if r["mixed"]]),
           "mixed_pre_feedback": _report("mixed & pre-feedback", [r for r in rows if r["mixed"] and not r["post_feedback"]]),
           "mixed_post_feedback": _report("mixed & post-feedback (echo residue possible)",
                                          [r for r in rows if r["mixed"] and r["post_feedback"]]),
           "rows": rows}
    (HERE / "rm_pick_scores.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("  mixed-nodes Δ is the headline; CI crossing 0 = an honest null. -> rm_pick_scores.json")
    if rm_lib.OVERFLOWS:
        print(f"  WARNING: {rm_lib.OVERFLOWS} scored sequences exceeded max_len (head+tail spliced) — "
              "check prompt budgets before trusting affected scores.")


@app.command()
def main(pairs_file: str = typer.Option("rm_eval_pairs.jsonl", "--pairs"),
         smoke: bool = typer.Option(False, "--smoke", help="score only 5 pairs and print raw values"),
         pick_candidates: str = typer.Option("", "--pick-candidates",
                                             help="resample candidates_*.jsonl -> run PICK MODE instead"),
         pick_labels: str = typer.Option("", "--pick-labels", help="matching labels_*.jsonl for pick mode"),
         limit: int = typer.Option(0, "--limit")) -> None:
    if bool(pick_candidates) != bool(pick_labels):
        raise typer.BadParameter("pick mode needs BOTH --pick-candidates and --pick-labels — with only one, "
                                 "the tool would silently run the ordinary pair eval for hours instead.")
    if pick_candidates and pick_labels:
        _pick_eval(HERE / pick_candidates, HERE / pick_labels)  # pathlib '/' keeps an absolute right side as-is
        return
    pairs = [json.loads(l) for l in (HERE / pairs_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if smoke:
        pairs = pairs[:5]
    elif limit:
        pairs = pairs[:limit]
    if not pairs:  # BEFORE the model load — an empty file must not cost minutes of 27B loading + a stack trace
        print(f"[score_rm] no pairs in {pairs_file} — build rm_eval_pairs.jsonl locally first (build_rm_eval.py)")
        return
    print(f"[score_rm] scoring {len(pairs)} pairs")
    model, tok = rm_lib.load_rm()
    rows = []
    for i, p in enumerate(pairs):
        sc = rm_lib.score(model, tok, p["prompt"], p["chosen"])
        sr = rm_lib.score(model, tok, p["prompt"], p["rejected"])
        rows.append({"instance_id": p["instance_id"], "pair_type": p.get("pair_type", "?"),
                     "strength": p.get("strength", "?"), "rejected_source": p.get("rejected_source", "?"),
                     "len_chosen": len(p["chosen"]), "len_rejected": len(p["rejected"]),
                     "score_chosen": sc, "score_rejected": sr, "correct": sc > sr})
        if smoke:
            print(f"  chosen={sc:+.4f}  rejected={sr:+.4f}  {'OK' if sc > sr else 'MISS'}")
        elif (i + 1) % 20 == 0:
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"  {i + 1}/{len(pairs)}  running acc={acc:.3f}", flush=True)
    if smoke:
        vals = [r["score_chosen"] for r in rows] + [r["score_rejected"] for r in rows]
        spread = max(vals) - min(vals)
        print(f"[smoke] score spread = {spread:.5f} -> "
              f"{'LOOKS ALIVE' if spread > 1e-3 else 'CONSTANT — value head NOT loaded, stop'}")
        return
    if not rows:
        print("[score_rm] no pairs — build rm_eval_pairs.jsonl locally first (build_rm_eval.py)")
        return
    flags = [r["correct"] for r in rows]
    acc = sum(flags) / len(flags)
    lo, hi = _boot_ci(flags)
    margin = statistics.fmean(r["score_chosen"] - r["score_rejected"] for r in rows)

    slices = {}
    for key in ("pair_type", "strength", "rejected_source"):
        for val, grp in sorted(_group(rows, key).items()):
            f = [r["correct"] for r in grp]
            slices[f"{key}={val}"] = {"n": len(f), "acc": round(sum(f) / len(f), 3)}
    # length-bias diagnostics
    all_scores = [r["score_chosen"] for r in rows] + [r["score_rejected"] for r in rows]
    all_lens = [float(r["len_chosen"]) for r in rows] + [float(r["len_rejected"]) for r in rows]
    corr = _pearson(all_scores, all_lens)
    rej_longer = [r["correct"] for r in rows if r["len_rejected"] > r["len_chosen"]]
    cho_longer = [r["correct"] for r in rows if r["len_chosen"] > r["len_rejected"]]
    lb = {"score_length_corr": round(corr, 3),
          "acc_when_rejected_longer": {"n": len(rej_longer),
                                       "acc": round(sum(rej_longer) / len(rej_longer), 3) if rej_longer else None},
          "acc_when_chosen_longer": {"n": len(cho_longer),
                                     "acc": round(sum(cho_longer) / len(cho_longer), 3) if cho_longer else None}}
    out = {"n_pairs": len(rows), "accuracy": round(acc, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "mean_margin": round(margin, 4), "slices": slices, "length_bias": lb, "rows": rows}
    (HERE / "rm_eval_scores.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[score_rm] held-out chosen>rejected accuracy = {acc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
          f"(n={len(rows)}, mean margin {margin:+.4f})")
    for k, v in slices.items():
        print(f"    {k:32s} acc={v['acc']:.3f} (n={v['n']})")
    print(f"    length bias: corr={lb['score_length_corr']}  rej-longer acc={lb['acc_when_rejected_longer']}  "
          f"cho-longer acc={lb['acc_when_chosen_longer']}")
    print("  ~0.5 overall = RM learned nothing (do NOT proceed to BoN); report the number either way.\n"
          "  resample slice ≈ the BoN predictor; a large gap rej-longer vs cho-longer = length hack, not content.")
    if rm_lib.OVERFLOWS:
        print(f"  WARNING: {rm_lib.OVERFLOWS} scored sequences exceeded max_len (head+tail spliced) — "
              "check pair length gates before trusting affected scores.")


def _group(rows: list[dict], key: str) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[r.get(key, "?")].append(r)
    return g


if __name__ == "__main__":
    app()
