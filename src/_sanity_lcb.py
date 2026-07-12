"""LCB eval sanity (docker, NO proxy). Two checks:
  (1) SYNTHETIC instance (we control the tests) — a KNOWN-correct solution must resolve, a wrong one must not:
      the clean GOLD-resolves / WRONG-fails discriminator (LCB ships no gold solution, so we make one).
  (2) a REAL problem with a deliberately-wrong solution must fail — validates decode + run on real data.
  python src/_sanity_lcb.py
"""
import base64
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect
import lcb
from minisweagent.config import get_config_from_spec
from minisweagent.run.benchmarks.swebench import get_sb_environment


def _write_solution(env, code: str) -> None:
    b64 = base64.b64encode(code.encode()).decode()
    collect._sh(env, f"printf %s '{b64}' | base64 -d > {lcb.SOLUTION_PATH}")


def _env():
    config = get_config_from_spec(collect.CONFIG)
    ec = config.setdefault("environment", {})
    ec["cwd"], ec["timeout"], ec["run_args"] = "/workspace", 120, ["--rm"]
    config.setdefault("run", {})["env_startup_command"] = "mkdir -p /workspace"
    return config


def main():
    synth = {"instance_id": "lcb__synthetic", "image_name": lcb.LCB_IMAGE, "_func_name": None,
             "_public": [{"input": "3\n", "output": "9", "testtype": "stdin"}],
             "_private": [{"input": "5\n", "output": "25", "testtype": "stdin"},
                          {"input": "10\n", "output": "100", "testtype": "stdin"}]}
    config = _env()
    env = get_sb_environment(config, synth)
    print(f"image: {lcb.LCB_IMAGE}", flush=True)
    try:
        _write_solution(env, "print(0)")                          # WRONG: always 0
        wrong = lcb.eval_lcb(env, synth, "", collect._sh)
        print(f"[synthetic WRONG] resolved={wrong['resolved']}  feedback={wrong['feedback'][:100]!r}", flush=True)
        print(f"[synthetic WRONG] oracle_gold set? {bool(wrong.get('oracle_gold'))}", flush=True)

        _write_solution(env, "n=int(input())\nprint(n*n)")        # CORRECT: n^2
        right = lcb.eval_lcb(env, synth, "", collect._sh)
        print(f"[synthetic GOLD]  resolved={right['resolved']}  feedback={right['feedback'][:60]!r}", flush=True)

        no_sol = lcb.eval_lcb(env, {**synth, "_public": synth["_public"], "_private": synth["_private"]}, "", collect._sh)  # solution.py still present
        print(f"[VERDICT synthetic] discriminates: {right['resolved'] and not wrong['resolved']}", flush=True)

        # (2) real problem, wrong solution must fail (validates real decode + run path)
        real = lcb.load_instances()[0]
        real = {**real, "image_name": lcb.LCB_IMAGE}
        _write_solution(env, "import sys\nsys.exit(0)")           # produces nothing
        rf = lcb.eval_lcb(env, real, "", collect._sh)
        print(f"[real {real['instance_id']}] wrong-solution resolved={rf['resolved']} (must be False)", flush=True)
    finally:
        try:
            cid = getattr(env, "container_id", None)
            if cid:
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)
        except Exception:
            pass


if __name__ == "__main__":
    main()
