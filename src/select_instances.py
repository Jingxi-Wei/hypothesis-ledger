"""Select SWE-bench Verified hard-tier instances and write instance-id splits.

Hard tier = difficulty >= 15 min (excludes '<15 min fix'), where the agent is
likely to fail first -> the valuable self_corrected / oracle_redirected traces.
Splits are by instance_id, deterministic (sha1 bucket), disjoint.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

HARD = {"15 min - 1 hour", "1-4 hours", ">4 hours"}
OUT = Path(__file__).resolve().parent.parent / "dataset" / "splits"


def bucket(instance_id: str) -> str:
    h = int(hashlib.sha1(instance_id.encode()).hexdigest(), 16) % 100
    if h < 70:
        return "train"
    if h < 80:
        return "dev"
    return "test"


def main(multi_hunk_only: bool = False) -> None:
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    rows = [r for r in ds if r["difficulty"] in HARD]
    if multi_hunk_only:
        rows = [r for r in rows if r["patch"].count("@@") >= 4]
    rows.sort(key=lambda r: r["instance_id"])

    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for r in rows:
        splits[bucket(r["instance_id"])].append(r["instance_id"])

    OUT.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (OUT / f"{name}.json").write_text(json.dumps(ids, indent=2))

    print(f"hard-tier instances: {len(rows)} (multi_hunk_only={multi_hunk_only})")
    print("difficulty:", dict(Counter(r["difficulty"] for r in rows)))
    print("split sizes:", {k: len(v) for k, v in splits.items()})
    all_ids = [i for v in splits.values() for i in v]
    assert len(all_ids) == len(set(all_ids)), "split overlap!"
    print("splits disjoint: OK ->", OUT)


if __name__ == "__main__":
    main(multi_hunk_only="--multi-hunk" in sys.argv)
