"""Post-fix TB verification (docker, NO proxy) — validates the 2026-07-07 review fixes on real containers:
  (a) restore: eval_tb must delete verifier droppings (cancel-async-tasks' test.sh does `cp test.py /app/`)
  (b) anti-loop: the submit-time state fingerprint must be IDENTICAL across two no-op evals
  (c) gold still discriminates after the restore step (EMPTY fails -> solve.sh resolves)
  (d) binary fixtures survive _copy_tests byte-exact (read_bytes fix) — checked in a slim python container
  python src/_sanity_tb_fixes.py
"""
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect
import tb
from minisweagent.config import get_config_from_spec
from minisweagent.run.benchmarks.swebench import get_sb_environment

FP = r"find /app -type f -not -path '*/.git/*' -exec stat -c '%s %Y %n' {} + 2>/dev/null | sort | md5sum"


def _mkenv(inst, cwd="/app"):
    config = get_config_from_spec(collect.CONFIG)
    ec = config.setdefault("environment", {})
    ec["cwd"], ec["timeout"], ec["run_args"] = cwd, 1800, ["--rm", "--entrypoint", ""]
    return get_sb_environment(config, inst)


def _cleanup(env):
    try:
        cid = getattr(env, "container_id", None)
        if cid:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)
    except Exception:
        pass


def main():
    ok = True
    # ---- (a)(b)(c) on the polluting task ----
    inst = tb.normalize_tb(tb.load_task(tb.TASKS_DIR / "cancel-async-tasks"))
    env = _mkenv(inst)
    try:
        fp1 = collect._sh(env, FP)["output"].strip()
        e1 = tb.eval_tb(env, inst, collect._sh)                      # EMPTY eval #1 (test.sh drops /app/test.py)
        leftover = collect._sh(env, "ls /app/test.py 2>/dev/null || echo GONE")["output"].strip()
        print(f"(a) restore: /app/test.py after eval -> {leftover!r}  {'OK' if leftover.endswith('GONE') else 'FAIL'}")
        ok &= leftover.endswith("GONE")
        fp2 = collect._sh(env, FP)["output"].strip()
        print(f"(b) anti-loop fingerprint stable across a no-op eval: {fp1 == fp2}  {'OK' if fp1 == fp2 else 'FAIL'}")
        ok &= (fp1 == fp2)
        after = tb.run_oracle_solution(env, inst, collect._sh)
        print(f"(c) discriminates after restore: EMPTY={e1['resolved']} GOLD={after['resolved']}  "
              f"{'OK' if (not e1['resolved']) and after['resolved'] else 'FAIL'}")
        ok &= (not e1["resolved"]) and after["resolved"]
    finally:
        _cleanup(env)

    # ---- (d) binary fixture integrity in a slim container (no heavy task image needed) ----
    task = tb.TASKS_DIR / "video-processing"
    local = next(f for f in (task / "tests").rglob("*") if f.suffix in (".mp4", ".png", ".jpg", ".pt"))
    lmd5 = hashlib.md5(local.read_bytes()).hexdigest()
    env2 = _mkenv({"instance_id": "tb__fixturecheck", "image_name": "python:3.11-slim"}, cwd="/")
    try:
        collect._sh(env2, "mkdir -p /app")
        tb._copy_tests(env2, task, collect._sh)
        rel = local.relative_to(task / "tests").as_posix()
        cmd5 = collect._sh(env2, f"md5sum /tests/{rel}")["output"].split()[0]
        print(f"(d) binary fixture {local.name}: local {lmd5[:12]} vs container {cmd5[:12]}  "
              f"{'OK' if lmd5 == cmd5 else 'FAIL — bytes corrupted'}")
        ok &= (lmd5 == cmd5)
    finally:
        _cleanup(env2)

    print(f"\n[VERDICT] all post-fix checks: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
