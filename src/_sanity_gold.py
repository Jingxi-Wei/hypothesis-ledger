"""Eval sanity: a correct eval MUST (a) fail F2P on the unpatched container and (b) resolve once the
GOLD source patch is applied. Run: python src/_sanity_gold.py <instance_id>"""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset

import collect
from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model  # noqa: F401  (keep import side-effects parity)
from minisweagent.run.benchmarks.swebench import get_sb_environment


def main():
    iid = sys.argv[1] if len(sys.argv) > 1 else "django__django-10973"
    ds = {i["instance_id"]: i for i in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    inst = ds[iid]
    config = get_config_from_spec(collect.CONFIG)
    env = get_sb_environment(config, inst)

    before = collect.eval_in_container(env, inst)
    print(f"[BEFORE gold] f2p_pass={before['f2p_pass']} p2p_pass={before['p2p_pass']} resolved={before['resolved']}")

    b64 = base64.b64encode(inst["patch"].encode()).decode()
    collect._sh(env, f"printf %s '{b64}' | base64 -d > /tmp/gold.patch")
    ap = collect._sh(env, "cd /testbed && (git apply -v /tmp/gold.patch || patch -p1 --fuzz=5 < /tmp/gold.patch)")
    print(f"[gold apply] rc={ap['returncode']}")

    after = collect.eval_in_container(env, inst)
    print(f"[AFTER gold]  f2p_pass={after['f2p_pass']} p2p_pass={after['p2p_pass']} resolved={after['resolved']}")

    ok = (not before["f2p_pass"]) and after["resolved"]
    print(f"[VERDICT] eval discriminates correctly: {ok}")


if __name__ == "__main__":
    main()
