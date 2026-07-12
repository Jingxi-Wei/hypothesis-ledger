"""Fetch the RELEVANT source code for each instance's fix sample: the regions the gold patch modifies, read
from the container at the BASE (buggy) state. Without this the fix sample asks the model to produce a patch
blind (no code => blind guessing). Only the files the gold patch touches are read (irrelevant files skipped).

Container-bound, no proxy. Run as a processing step AFTER collection:
  python src/fetch_code.py --run-id r1
Saves dataset/raw/<iid>/<run>/relevant_code.json = {file: [{"lines": "a-b", "code": "..."}]}.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import typer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402  (sets up swebench + _wincompat; provides CONFIG + _sh)
import pro  # noqa: E402
from minisweagent.config import get_config_from_spec  # noqa: E402
from minisweagent.run.benchmarks.swebench import get_sb_environment, get_swebench_docker_image_name  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
CONTEXT = 18  # lines of context around each hunk
DOCKER_PULL_TIMEOUT = int(os.environ.get("DOCKER_PULL_TIMEOUT", "7200"))
app = typer.Typer(add_completion=False)


def parse_patch_targets(patch: str) -> dict[str, list[tuple[int, int]]]:
    """{file -> [(old_start, old_end)]} line ranges the gold patch modifies."""
    targets: dict[str, list[tuple[int, int]]] = {}
    cur = None
    for ln in patch.splitlines():
        m = re.match(r"^\+\+\+ b/(.+)", ln)
        if m:
            cur = m.group(1).strip()
            continue
        m = re.match(r"^@@ -(\d+)(?:,(\d+))?", ln)
        if m and cur:
            s, length = int(m.group(1)), int(m.group(2) or 1)
            targets.setdefault(cur, []).append((s, s + max(length, 1)))
    return targets


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted((max(1, a - CONTEXT), b + CONTEXT) for a, b in ranges):
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def fetch_one(instance: dict, is_pro: bool = False) -> dict:
    targets = parse_patch_targets(instance["patch"])
    config = get_config_from_spec(collect.CONFIG)
    cwd = "/testbed"
    if is_pro:  # same env setup as collect_one: jefzda image (ENTRYPOINT=/bin/bash must be reset), /app cwd
        instance = pro.normalize_pro(instance)
        ec = config.setdefault("environment", {})
        ec["cwd"], ec["timeout"], ec["run_args"] = "/app", 1800, ["--rm", "--entrypoint", ""]
        ec["pull_timeout"] = 7200
        cwd = "/app"
    env = get_sb_environment(config, instance)
    code: dict[str, list[dict]] = {}
    try:
        for f, ranges in targets.items():
            regions = []
            for a, b in _merge(ranges):
                out = collect._sh(env, f"cd {cwd} && sed -n '{a},{b}p' {f}")
                mo = re.search(r"<output>\n?(.*?)\n?</output>", out["output"], re.S)
                body = (mo.group(1) if mo else out["output"]).rstrip()[:1800]
                if body.strip():
                    regions.append({"lines": f"{a}-{b}", "code": body})
            if regions:
                code[f] = regions
    finally:
        try:
            cid = getattr(env, "container_id", None)
            if cid:
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)
        except Exception:
            pass
    return code


@app.command()
def main(run_id: str = typer.Option("r1", "--run-id"), instance: str = typer.Option("", "-i", "--instance"),
         dataset: str = typer.Option("verified", "--dataset", help="verified | full | pro"),
         prune_images: bool = typer.Option(True, "--prune-images/--keep-images")) -> None:
    ds_name = {"verified": "princeton-nlp/SWE-bench_Verified", "full": "princeton-nlp/SWE-bench",
               "pro": "ScaleAI/SWE-bench_Pro"}[dataset]
    is_pro = dataset == "pro"
    ds = {i["instance_id"]: i for i in load_dataset(ds_name, split="test")}
    image = (lambda i: pro.pro_image_uri(i)) if is_pro else get_swebench_docker_image_name
    targets = [instance] if instance else sorted(p.name for p in RAW.iterdir() if (p / run_id / "outcome.json").exists())
    n = 0
    for iid in targets:
        rd = RAW / iid / run_id
        if iid not in ds or (rd / "relevant_code.json").exists():
            continue
        try:
            subprocess.run(["docker", "pull", image(ds[iid])], capture_output=True, timeout=DOCKER_PULL_TIMEOUT)
            code = fetch_one(ds[iid], is_pro=is_pro)
            (rd / "relevant_code.json").write_text(json.dumps(code, indent=2, ensure_ascii=False), encoding="utf-8")
            n += 1
            print(f"[fetch_code] {iid} -> {sum(len(v) for v in code.values())} region(s) across {len(code)} file(s)")
            if prune_images:
                subprocess.run(["docker", "rmi", "-f", image(ds[iid])], capture_output=True, timeout=120)
        except Exception as e:
            print(f"[fetch_code] ERROR {iid}: {e!r}")
    print(f"[fetch_code done] {n} instances")


if __name__ == "__main__":
    app()
