"""Debug the Pro eval: apply gold, run the instance run_script, dump raw stdout/stderr/output + env probes."""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset

import collect
import pro
from minisweagent.config import get_config_from_spec
from minisweagent.run.benchmarks.swebench import get_sb_environment


def sh(env, c, t=1800):
    return collect._sh(env, c, timeout=t)


def main():
    iid = sys.argv[1] if len(sys.argv) > 1 else "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan"
    ds = {i["instance_id"]: i for i in load_dataset("ScaleAI/SWE-bench_Pro", split="test")}
    inst = pro.normalize_pro(ds[iid])
    config = get_config_from_spec(collect.CONFIG)
    ec = config.setdefault("environment", {}); ec["cwd"] = "/app"; ec["timeout"] = 1800
    env = get_sb_environment(config, inst)
    sd = pro.RUN_SCRIPTS / iid
    try:
        print("=== ENV probes ===", flush=True)
        print("SETUP:", sh(env, "echo $SETUP")["output"][:120], flush=True)
        print("node/npm:", sh(env, "which node npm python 2>&1")["output"][:200], flush=True)
        print("ls /app (head):", sh(env, "ls /app | head -20")["output"][:300], flush=True)
        print("ls /app/install:", sh(env, "ls /app/install 2>&1 | head")["output"][:200], flush=True)
        # apply gold
        base = inst["base_commit"]
        sh(env, "mkdir -p /workspace")
        for n, c in [("run_script.sh", (sd/"run_script.sh").read_text(encoding="utf-8")),
                     ("parser.py", (sd/"parser.py").read_text(encoding="utf-8")),
                     ("patch.diff", inst["patch"])]:
            b64 = base64.b64encode(c.encode()).decode()
            sh(env, f"printf %s '{b64}' | base64 -d > /workspace/{n}")
        print("reset+apply:", sh(env, f"cd /app && git reset --hard {base} && git checkout {base} && git apply -v /workspace/patch.diff && echo APPLIED_OK")["output"][-200:], flush=True)
        tf = ",".join(pro._jl(inst["selected_test_files_to_run"]))
        before_last = inst["before_repo_set_cmd"].strip().split("\n")[-1]
        print("running run_script...", flush=True)
        sh(env, f"cd /app && ({before_last}) || true; bash /workspace/run_script.sh {tf} > /workspace/stdout.log 2> /workspace/stderr.log; python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json 2>&1 | tail -5", 1800)
        print("=== STDOUT.log (first 1500) ===", flush=True)
        print(sh(env, "head -c 1500 /workspace/stdout.log")["output"], flush=True)
        print("=== STDERR.log (last 1500) ===", flush=True)
        print(sh(env, "tail -c 1500 /workspace/stderr.log")["output"], flush=True)
        print("=== output.json ===", flush=True)
        print(sh(env, "cat /workspace/output.json 2>&1 | head -c 600")["output"], flush=True)
    finally:
        cid = getattr(env, "container_id", None)
        if cid:
            import subprocess
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)


if __name__ == "__main__":
    main()
