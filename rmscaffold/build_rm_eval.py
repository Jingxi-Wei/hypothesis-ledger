"""Held-out RM quality-check pairs (deterministic, zero proxy, local).

Training pairs come only from non-holdout instances (export.py main() skips pro_test). For the RM headline
("chosen > rejected on held-out pairs: X%") we need pairs from the HELD-OUT instances, from BOTH channels:

  * natural pairs   built here by calling the SAME constructor (export.export_preference) on
                    splits/pro_test.json instances — requires ledger.json + audit.json (posthoc
                    compress+audit must have covered the holdout under this run-id).
  * resample pairs  rmscaffold/resample/pairs_<run>_eval.jsonl — produced by
                    `gen_pairs.py --instances-file dataset/splits/pro_test.json --include-holdout --tag eval`
                    (proxy, batch paused). Merged here if present. This slice is the number that actually
                    predicts BoN: fresh-candidate discrimination, same distribution BoN scores.

Full pair records are kept (pair_type/strength/... tags) so score_rm.py can slice accuracy per type.

    python rmscaffold/build_rm_eval.py --run-id pro2
"""
import hashlib
import json
import sys
from pathlib import Path

import typer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from export import export_preference  # noqa: E402  — same constructor as training pairs (no drift)

import prep_rm  # noqa: E402  — reuse the F2P leak scan (sibling module)

app = typer.Typer(add_completion=False)


@app.command()
def main(run_id: str = typer.Option("pro2", "--run-id"),
         max_per_instance: int = typer.Option(12, "--max-per-instance",
                                              help="cap correlated pairs per instance (they share chosen/prompt "
                                                   "text; uncapped, one long chain dominates the headline and the "
                                                   "pair-level bootstrap CI comes out far too narrow)"),
         seed: int = typer.Option(42, "--seed"),
         keep_raw_leak: bool = typer.Option(False, "--keep-raw-leak",
                                            help="DIAGNOSTIC ONLY — a raw-leak eval pair mismeasures the RM"),
         no_leak_scan: bool = typer.Option(False, "--no-leak-scan", help="skip the F2P token scan (offline)")) -> None:
    hp = ROOT / "dataset" / "splits" / "pro_test.json"
    if not hp.exists():
        raise SystemExit(f"[build_rm_eval] holdout split missing: {hp}")
    holdout = json.loads(hp.read_text(encoding="utf-8"))
    ds = {i["instance_id"]: i for i in load_dataset("ScaleAI/SWE-bench_Pro", split="test")}
    tokmap = prep_rm.load_token_map(no_leak_scan)
    pairs, n_inst, dropped = [], 0, {"raw_leak": 0, "leak": 0, "dup": 0}
    seen = set()

    def push(p: dict) -> None:
        if not keep_raw_leak and p.get("protocol") == "raw_leak":
            dropped["raw_leak"] += 1
            return
        toks = tokmap.get(p["instance_id"])
        if toks and prep_rm.leak_hits(p, toks):
            dropped["leak"] += 1
            return
        key = hashlib.sha1("\x1f".join((p["prompt"], p["chosen"], p["rejected"])).encode("utf-8")).hexdigest()
        if key in seen:
            dropped["dup"] += 1
            return
        seen.add(key)
        pairs.append(p)

    for iid in sorted(holdout):
        if iid not in ds:
            continue
        got = export_preference(iid, run_id, ds[iid])
        if got:
            n_inst += 1
        for p in got:
            push(p)
    n_nat = len(pairs)
    # resample eval pairs, BOTH channels: candidate-sibling pairs (gen_pairs) and redirect-anchored
    # real-vs-candidate pairs (rederive_real_pairs.py) — records carry pair_type, score_rm slices on it
    for rp in (HERE / "resample" / f"pairs_{run_id}_eval.jsonl",
               HERE / "resample" / f"pairs_realanchor-{run_id}_eval.jsonl"):
        if rp.exists():
            for l in rp.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    push(json.loads(l))
    from collections import Counter, defaultdict
    import random
    n_resample = len(pairs) - n_nat
    per_inst = defaultdict(list)
    for p in pairs:
        per_inst[p["instance_id"]].append(p)
    rng, capped, n_cap_dropped = random.Random(seed), [], 0
    for iid in sorted(per_inst):
        grp = per_inst[iid]
        if len(grp) > max_per_instance:
            n_cap_dropped += len(grp) - max_per_instance
            grp = rng.sample(grp, max_per_instance)
        capped += grp
    pairs = capped
    ct = Counter(p.get("pair_type", "?") for p in pairs)
    (HERE / "rm_eval_pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs),
                                              encoding="utf-8")
    print(f"[build_rm_eval] {len(pairs)} held-out pairs (natural {n_nat}, resample {n_resample}, from "
          f"{n_inst} instances, cap dropped {n_cap_dropped}) | types {dict(ct)} | dropped {dropped} -> rm_eval_pairs.jsonl")
    if not pairs:
        print("  (empty = holdout instances have no ledger/audit under this run-id yet — run the posthoc "
              "compress+audit pass for the holdout first, and gen_pairs --tag eval for the resample slice)")
    elif not rp.exists():
        print(f"  (no resample eval slice yet — run gen_pairs.py --run-id {run_id} --instances-file "
              "dataset/splits/pro_test.json --include-holdout --tag eval while the batch is paused; that slice "
              "is the distribution-matched predictor of BoN)")


if __name__ == "__main__":
    app()
