"""Carve the SWE-bench Pro held-out test split (deterministic sha1 bucket, same method as select_instances.py).

~10% of the 731 instances -> dataset/splits/pro_test.json. These are RESERVED for the held-out process-Δ eval:
export.py auto-excludes them from training samples (--dataset pro). Collection still runs on ALL instances —
the eval needs trajectories on held-out instances too; the wall is between raw data and TRAINING exports.
"""
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "dataset" / "splits"


def is_test(instance_id: str) -> bool:
    return int(hashlib.sha1(instance_id.encode()).hexdigest(), 16) % 100 >= 90


def main() -> None:
    ids = sorted(i["instance_id"] for i in load_dataset("ScaleAI/SWE-bench_Pro", split="test"))
    test = [i for i in ids if is_test(i)]
    (OUT / "pro_test.json").write_text(json.dumps(test, indent=2))
    print(f"pro total={len(ids)} -> held-out test={len(test)} ({100 * len(test) // len(ids)}%) -> {OUT / 'pro_test.json'}")
    from collections import Counter
    ds = {i["instance_id"]: i for i in load_dataset("ScaleAI/SWE-bench_Pro", split="test")}
    print("test lang dist:", dict(Counter(ds[i]["repo_language"] for i in test)))


if __name__ == "__main__":
    main()
