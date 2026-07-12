"""Ground-truth the go teleport gold-apply: which apply tier succeeds, and does cfg land in the source?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset
import collect, pro
from minisweagent.config import get_config_from_spec
from minisweagent.run.benchmarks.swebench import get_sb_environment

IID = "instance_gravitational__teleport-3fa6904377c006497169945428e8197158667910-v626ec2a48416b10a88641359a169d99e935ff037"
ds = {i["instance_id"]: i for i in load_dataset("ScaleAI/SWE-bench_Pro", split="test")}
inst = pro.normalize_pro(ds[IID])
cfg = get_config_from_spec(collect.CONFIG)
ec = cfg.setdefault("environment", {}); ec["cwd"] = "/app"; ec["timeout"] = 1800; ec["run_args"] = ["--rm", "--entrypoint", ""]
env = get_sb_environment(cfg, inst)
sh = collect._sh
try:
    patch = inst["patch"]
    base = inst["base_commit"]
    before_lines = [l for l in inst["before_repo_set_cmd"].strip().split("\n") if l.strip()]
    setup_pre = "\n".join(l for l in before_lines if " -- " not in l)
    test_install = "\n".join(l for l in before_lines if " -- " in l)
    print("PATCH target files:")
    for l in patch.splitlines():
        if l.startswith("+++") or l.startswith("---"):
            print("   ", l[:90])
    sh(env, "mkdir -p /workspace")
    pro._b64_write(sh, env, "/workspace/patch.diff", patch)
    pro._b64_write(sh, env, "/workspace/setup_pre.sh", setup_pre)
    sh(env, f"cd /app && git reset --hard {base} && git checkout {base}", timeout=300)
    sh(env, "cd /app && bash /workspace/setup_pre.sh || true", timeout=300)
    print("\n=== git apply -v (tier1) ===")
    r1 = sh(env, "cd /app && git apply -v /workspace/patch.diff 2>&1; echo RC=$?", timeout=120)
    print(r1["output"][-1500:])
    print("\n=== forwarder.go has 'cfg' field after tier1? ===")
    g = sh(env, "cd /app && grep -n 'cfg ' lib/kube/proxy/forwarder.go | head; echo '---struct---'; grep -n 'cfg\\s*ForwarderConfig\\|cfg ForwarderConfig' lib/kube/proxy/forwarder.go | head")
    print(g["output"])
    print("\n=== git status (what changed) ===")
    print(sh(env, "cd /app && git status --porcelain | head -20")["output"])
finally:
    cid = getattr(env, "container_id", None)
    if cid:
        import subprocess; subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)
