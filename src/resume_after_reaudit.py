"""Detached CHAIN: wait for the re-audit pipeline to finish, THEN resume Pro batch collection.

The codex-proxy is serial (~2 concurrent max), so the batch must not start while re-audit is still calling it.
This waiter polls reaudit_pipeline.log for the terminal marker and only resumes on a clean PIPELINE DONE
(if the pipeline ABORTed, it leaves the batch stopped for inspection). Launched detached so it outlives the
session; it then runs run_loop_pro.ps1 (the proven batch launcher: sets PYTHONPATH=_wincompat, logs to
batch_pro1.log). run_batch is resumable, so this picks up Pro collection where it left off (~205/731).

  python src/resume_after_reaudit.py
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPE_LOG = ROOT / "dataset" / "reaudit_pipeline.log"


def state() -> str:
    if not PIPE_LOG.exists():
        return "waiting"
    t = PIPE_LOG.read_text(encoding="utf-8", errors="ignore")
    if "PIPELINE DONE" in t:
        return "done"
    if "[ABORT]" in t:
        return "abort"
    return "running"


def refresh_train_package() -> None:
    """Copy the freshly-regenerated (re-audited, probe-gated) SFT data into train_package/ and re-zip, so the
    upload-to-GPU bundle reflects the clean data. Deterministic; runs AFTER PIPELINE DONE => prep_sft has written
    dataset/sft/train_sharegpt.jsonl."""
    sft = ROOT / "dataset" / "sft" / "train_sharegpt.jsonl"
    pkg = ROOT / "train_package"
    if not sft.exists():
        print("[chain] WARN: dataset/sft/train_sharegpt.jsonl missing — skipping train_package refresh", flush=True)
        return
    shutil.copy2(sft, pkg / "train_sharegpt.jsonl")
    z = ROOT / "train_package.zip"
    if z.exists():
        z.unlink()
    shutil.make_archive(str(ROOT / "train_package"), "zip", root_dir=str(pkg))  # contents at archive root
    rows = sum(1 for _ in sft.open(encoding="utf-8"))
    print(f"[chain] train_package refreshed: {rows} rows copied + re-zipped ({z.stat().st_size} bytes)", flush=True)


def main() -> None:
    print("[chain] waiting for re-audit pipeline to finish (polling every 30s)...", flush=True)
    while state() not in ("done", "abort"):
        time.sleep(30)
    st = state()
    print(f"[chain] re-audit pipeline -> {st}", flush=True)
    if st != "done":
        print("[chain] pipeline did NOT finish clean -> NOT resuming batch. Inspect reaudit_pipeline.log.", flush=True)
        return
    print("[chain] refreshing train_package with the clean re-audited + probe-gated data...", flush=True)
    refresh_train_package()
    print("[chain] proxy is free -> resuming Pro batch via run_loop_pro.ps1 (output -> batch_pro1.log)", flush=True)
    rc = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                         "-File", str(ROOT / "run_loop_pro.ps1")], cwd=str(ROOT)).returncode
    print(f"[chain] batch process exited {rc} (all instances done => 0; network death => nonzero, re-launch to continue)",
          flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
