"""Pro eval sanity: empty patch -> FAIL_TO_PASS must fail (bug present); gold patch -> resolved.
  python src/_sanity_pro.py [instance_id]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset

import collect
import pro
from minisweagent.config import get_config_from_spec
from minisweagent.run.benchmarks.swebench import get_sb_environment


def main():
    iid = sys.argv[1] if len(sys.argv) > 1 else "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan"
    ds = {i["instance_id"]: i for i in load_dataset("ScaleAI/SWE-bench_Pro", split="test")}
    inst = pro.normalize_pro(ds[iid])
    config = get_config_from_spec(collect.CONFIG)
    env_cfg = config.setdefault("environment", {})
    env_cfg["cwd"] = "/app"
    env_cfg["timeout"] = 1800
    env_cfg["run_args"] = ["--rm", "--entrypoint", ""]  # Pro images ENTRYPOINT=/bin/bash breaks `sleep 2h`; reset it
    env = get_sb_environment(config, inst)
    print(f"image: {inst['image_name']} | repo: {inst['repo']} | lang: {inst.get('repo_language')}", flush=True)
    try:
        before = pro.eval_pro(env, inst, "", collect._sh)
        print(f"[EMPTY] applied={before['applied']} f2p_pass={before['f2p_pass']} p2p_pass={before['p2p_pass']} resolved={before['resolved']}", flush=True)
        after = pro.eval_pro(env, inst, inst["patch"], collect._sh)
        print(f"[GOLD]  applied={after['applied']} f2p_pass={after['f2p_pass']} p2p_pass={after['p2p_pass']} resolved={after['resolved']}", flush=True)
        print(f"[VERDICT] eval discriminates: {(not before['f2p_pass']) and after['resolved']}", flush=True)
        if not after["resolved"]:
            print("GOLD feedback:\n", after["feedback"][:1200], flush=True)
    finally:
        try:
            cid = getattr(env, "container_id", None)
            if cid:
                import subprocess
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)
        except Exception:
            pass


if __name__ == "__main__":
    main()
