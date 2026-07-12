"""Rollout-eval instance sets (deterministic, zero proxy).

  seen.json    N instances the SFT data was actually built from (samples exist) — absorption check.
               Success here != capability (their fix samples contain the gold patch); FAILURE here is the
               loud signal, and behavior metrics are the primary read.
  unseen.json  M instances from the pro_test holdout — the generalization tier.

Stratified so the seen tier is not all-easy: half from solved-source instances, half from
oracle/chaotic-source ones (the hard tail where self-correction actually gets exercised).

  python src/select_rollout_sets.py            # writes dataset/splits/rollout_{seen,unseen}.json
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
SAMP = ROOT / "dataset" / "samples"
SPLITS = ROOT / "dataset" / "splits"
app = typer.Typer(add_completion=False)


def _sha(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


@app.command()
def main(n_seen: int = typer.Option(24, "--n-seen"),
         n_unseen: int = typer.Option(24, "--n-unseen"),
         runs: str = typer.Option("pro1,pro2", "--runs", help="training runs the SFT data came from (r1 excluded: old protocol)")) -> None:
    # ---- seen tier: instances that actually contributed SFT samples, tagged easy/hard by source ----
    src_by_iid: dict[str, set] = defaultdict(set)
    for run in (r.strip() for r in runs.split(",") if r.strip()):
        for t in ("audit", "propose", "fix", "probe"):
            p = SAMP / run / f"{t}.jsonl"
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                s = json.loads(line)
                src_by_iid[s["instance_id"]].add(s.get("source") or "explore")
    holdout = set(json.loads((SPLITS / "pro_test.json").read_text(encoding="utf-8")))
    pool = {i: srcs for i, srcs in src_by_iid.items() if i not in holdout}
    hard = sorted(i for i, srcs in pool.items() if "oracle" in srcs or "self_rescue" in srcs)
    easy = sorted(i for i in pool if i not in set(hard))
    hard.sort(key=lambda i: _sha("rollout-seen::" + i))
    easy.sort(key=lambda i: _sha("rollout-seen::" + i))
    half = n_seen // 2
    seen = hard[:half] + easy[:n_seen - half]
    if len(seen) < n_seen:  # pool smaller than asked — top up from whichever side has leftovers
        rest = [i for i in hard[half:] + easy[n_seen - half:] if i not in seen]
        seen += rest[:n_seen - len(seen)]

    # ---- unseen tier: deterministic sample of the holdout wall ----
    uns = sorted(holdout, key=lambda i: _sha("rollout-unseen::" + i))[:n_unseen]

    SPLITS.mkdir(parents=True, exist_ok=True)
    (SPLITS / "rollout_seen.json").write_text(json.dumps(sorted(seen), indent=1), encoding="utf-8")
    (SPLITS / "rollout_unseen.json").write_text(json.dumps(sorted(uns), indent=1), encoding="utf-8")
    print(f"[rollout-sets] seen {len(seen)} (hard {sum(1 for i in seen if i in set(hard))}, "
          f"easy {len(seen) - sum(1 for i in seen if i in set(hard))}) from pool {len(pool)} | "
          f"unseen {len(uns)} of {len(holdout)} holdout -> dataset/splits/rollout_*.json")


if __name__ == "__main__":
    app()
