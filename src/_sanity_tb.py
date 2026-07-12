"""Terminal-Bench eval sanity (docker, NO proxy). EMPTY (untouched container) must FAIL the verifier;
running the GOLD solve.sh must make it RESOLVE. Also answers the research's open question: does the TB
harness run at all under this Windows + Docker-Desktop host?
  python src/_sanity_tb.py [task_name]   # default: a light task
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect
import tb
from minisweagent.config import get_config_from_spec
from minisweagent.run.benchmarks.swebench import get_sb_environment


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "cancel-async-tasks"
    inst = tb.load_task(tb.TASKS_DIR / name)
    if inst is None:
        sys.exit(f"no such TB task: {name}")
    inst = tb.normalize_tb(inst)
    config = get_config_from_spec(collect.CONFIG)
    ec = config.setdefault("environment", {})
    ec["cwd"], ec["timeout"], ec["run_args"] = "/app", 1800, ["--rm", "--entrypoint", ""]
    env = get_sb_environment(config, inst)
    print(f"task: {name} | image: {inst['docker_image']} | solve.sh {len(inst['patch'].splitlines())} lines", flush=True)
    try:
        before = tb.eval_tb(env, inst, collect._sh)               # untouched container
        print(f"[EMPTY] resolved={before['resolved']}  (must be False)  feedback={before['feedback'][:120]!r}", flush=True)
        after = tb.run_oracle_solution(env, inst, collect._sh)    # run solve.sh, then verify
        print(f"[GOLD]  resolved={after['resolved']}  (must be True)", flush=True)
        if not after["resolved"]:
            print("GOLD feedback:\n", after["feedback"][:1500], flush=True)
        print(f"[VERDICT] eval discriminates: {(not before['resolved']) and after['resolved']}", flush=True)
    finally:
        try:
            cid = getattr(env, "container_id", None)
            if cid:
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)
        except Exception:
            pass


if __name__ == "__main__":
    main()
