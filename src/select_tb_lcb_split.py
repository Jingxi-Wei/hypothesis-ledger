"""Carve deterministic held-out test splits for TB and LCB (same sha1 bucket as select_pro_split.py).

~10% of each -> dataset/splits/{tb,lcb}_test.json. RESERVED for the held-out process-Δ eval: export.py
auto-excludes them from training (--dataset tb|lcb) and run_batch keeps them OUT of collection. Deterministic,
so the wall is reproducible on any machine.

  python src/select_tb_lcb_split.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb
import lcb

OUT = Path(__file__).resolve().parent.parent / "dataset" / "splits"


def is_test(instance_id: str) -> bool:
    return int(hashlib.sha1(instance_id.encode()).hexdigest(), 16) % 100 >= 90


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, ids in (("tb", sorted(i["instance_id"] for i in tb.load_instances())),
                      ("lcb", sorted(i["instance_id"] for i in lcb.load_instances()))):
        test = [i for i in ids if is_test(i)]
        (OUT / f"{name}_test.json").write_text(json.dumps(test, indent=2))
        print(f"{name}: total={len(ids)} -> held-out test={len(test)} ({100 * len(test) // max(1, len(ids))}%) "
              f"-> {OUT / f'{name}_test.json'}")


if __name__ == "__main__":
    main()
