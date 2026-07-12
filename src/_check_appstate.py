"""Confirm each Pro image's /app is a git repo checked out at base_commit and clean at container start,
so (a) the agent sees the BUGGY base and (b) `git diff HEAD` == the agent's own edits relative to base."""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load_dataset
import pro

IIDS = [
    "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan",
    "instance_qutebrowser__qutebrowser-f91ace96223cac8161c16dd061907e138fe85111-v059c6fdc75567943479b23ebca7c07b5e9a7f34c",
    "instance_gravitational__teleport-3fa6904377c006497169945428e8197158667910-v626ec2a48416b10a88641359a169d99e935ff037",
    "instance_tutao__tutanota-da4edb7375c10f47f4ed3860a591c5e6557f7b5c-vbc0d9ba8f0071fbe982809910959a6ff8884dbbf",
]
ds = {i["instance_id"]: i for i in load_dataset("ScaleAI/SWE-bench_Pro", split="test")}


def run_in(img, cmd):
    return subprocess.run(["docker", "run", "--rm", "--entrypoint", "", img, "bash", "-lc", cmd],
                          capture_output=True, text=True, timeout=300).stdout.strip()


for iid in IIDS:
    inst = ds[iid]
    img = pro.pro_image_uri(inst)
    base = inst["base_commit"]
    head = run_in(img, "cd /app && git rev-parse HEAD 2>&1")
    dirty = run_in(img, "cd /app && git status --porcelain 2>&1 | head -5")
    print(f"[{inst['repo_language']:6}] {inst['repo']}")
    print(f"   base={base}")
    print(f"   HEAD={head}  MATCH={head == base}")
    print(f"   dirty={'CLEAN' if not dirty else repr(dirty[:200])}")
