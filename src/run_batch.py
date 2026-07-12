"""Batch collection over SWE-bench Verified train-split instances.

Resumable (skips instances that already have an outcome), serial (docker), cleans up
containers between instances. Built to run for hours/days; just re-run to continue.

  python src/run_batch.py --run-id r1                       # all train instances
  python src/run_batch.py --run-id r1 --difficulty "15 min - 1 hour" --limit 4   # probe the 1h tier
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer
from datasets import load_dataset
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pro  # noqa: E402
import fetch_code  # noqa: E402
from collect import collect_one, _is_pro, load_instances, is_context_overflow, LauncherBroken  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCKER_PULL_TIMEOUT = int(os.environ.get("DOCKER_PULL_TIMEOUT", "7200"))
app = typer.Typer(add_completion=False)

# Persistent poison-pill blacklist: instances that overflow the context window (or are hand-listed as
# permanently broken). Filtered out at load time so a known pill is NEVER re-attempted — one context-overflow
# already burned tokens once; retrying it every loop round is pure waste. Shared across run-ids by design.
SKIP_PATH = ROOT / "dataset" / "splits" / "skip.json"
_skip_lock = threading.Lock()


def _load_skip() -> set:
    try:
        return set(json.loads(SKIP_PATH.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _add_skip(iid: str, reason: str = "") -> None:
    """Append an instance_id to the on-disk skip-list (dedup, thread-safe). Auto-called on context-overflow."""
    with _skip_lock:
        cur = _load_skip()
        if iid in cur:
            return
        cur.add(iid)
        SKIP_PATH.parent.mkdir(parents=True, exist_ok=True)
        SKIP_PATH.write_text(json.dumps(sorted(cur), indent=2), encoding="utf-8")
    print(f"  SKIP-LIST += {iid}" + (f" ({reason})" if reason else ""), flush=True)


def _retire_poison(iid: str, run_id: str, reason: str) -> None:
    """Skip-list a poison-pill instance AND delete its run dir (不要留): the giant orphan trajectory is worthless
    (it never converged) and only wastes disk; the skip-list entry is the durable record of why it's gone."""
    _add_skip(iid, reason)
    shutil.rmtree(_run_dir(iid, run_id), ignore_errors=True)


def _cleanup_containers() -> None:
    try:
        ids = subprocess.run(["docker", "ps", "-aq", "--filter", "name=minisweagent"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        for cid in ids:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)
    except Exception:
        pass


def _image(instance: dict) -> str:
    """Docker image for this instance — jefzda (Pro), the instance-carried image (TB task / LCB sandbox), or
    the computed swebench image (Verified/Full)."""
    if _is_pro():
        return pro.pro_image_uri(instance)
    if os.environ.get("DATASET") in ("tb", "lcb"):
        return instance.get("docker_image") or instance.get("image_name")
    return get_swebench_docker_image_name(instance)


def _prune_image(instance: dict) -> None:
    """Remove the instance's ~GB image to keep disk bounded (resume skips done instances, so no re-pull)."""
    try:
        subprocess.run(["docker", "rmi", "-f", _image(instance)], capture_output=True, timeout=120)
    except Exception:
        pass


def _pull(image: str) -> bool:
    """Pull with a short retry; False means the image is unavailable (network outage or bad tag)."""
    for delay in (0, 30):
        if delay:
            time.sleep(delay)
        try:
            if subprocess.run(["docker", "pull", image], capture_output=True, timeout=DOCKER_PULL_TIMEOUT).returncode == 0:
                return True
        except Exception:
            pass
    return False


def _fetch_relevant_code(inst: dict, run_id: str, dataset: str) -> None:
    """Fetch source context in the same worker before that worker starts the next instance."""
    iid = inst["instance_id"]
    rd = _run_dir(iid, run_id)
    out = rd / "relevant_code.json"
    if out.exists():
        return
    code = fetch_code.fetch_one(inst, is_pro=(dataset == "pro"))
    out.write_text(json.dumps(code, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  fetch_code {iid}: {sum(len(v) for v in code.values())} region(s) across {len(code)} file(s)", flush=True)


def _run_dir(iid: str, run_id: str) -> Path:
    return ROOT / "dataset" / "raw" / iid / run_id


def _is_complete(iid: str, run_id: str) -> bool:
    rd = _run_dir(iid, run_id)
    if not (rd / "outcome.json").exists():
        return False
    if os.environ.get("DATASET") in ("tb", "lcb"):
        return True  # no relevant_code step (TB gold = solve.sh; LCB has no gold) -> outcome alone = complete
    return (rd / "relevant_code.json").exists()


@app.command()
def main(
    run_id: str = typer.Option("r1", "--run-id"),
    dataset: str = typer.Option("verified", "--dataset", help="verified | full | pro | tb | lcb"),
    instances_file: str = typer.Option("", "--instances-file", help="json list of instance_ids (e.g. dataset/splits/full_hard.json); order preserved (hardest-first)"),
    difficulty: str = typer.Option("", "--difficulty", help="verified-only difficulty filter, e.g. '1-4 hours,>4 hours'"),
    limit: int = typer.Option(0, "--limit", help="max NEW instances this batch (0 = all)"),
    workers: int = typer.Option(2, "--workers", help="concurrent collection workers (codex-proxy tolerates ~2)"),
    prune_images: bool = typer.Option(True, "--prune-images/--keep-images", help="rmi each instance image after use (saves disk)"),
) -> None:
    os.environ["DATASET"] = dataset  # activate collect.py per-benchmark mode (env + eval + loader dispatch)
    ds = load_instances(dataset)     # HF for verified/full/pro, local adapters for tb/lcb
    if instances_file:
        ids = json.loads(Path(instances_file).read_text())            # pre-ranked pool -> preserve order
    elif dataset in ("tb", "lcb"):
        ids = sorted(ds)                                              # all tasks/problems...
        hp = ROOT / "dataset" / "splits" / f"{dataset}_test.json"
        if hp.exists():                                              # ...minus a held-out eval split, if one exists
            held = set(json.loads(hp.read_text()))
            ids = [i for i in ids if i not in held]
    else:
        ids = sorted(json.loads((ROOT / "dataset" / "splits" / "train.json").read_text()))
    instances = [ds[iid] for iid in ids if iid in ds]
    if difficulty:  # only meaningful for verified (full has no difficulty field)
        diffs = {d.strip() for d in difficulty.split(",")}
        instances = [i for i in instances if i.get("difficulty") in diffs]
    skip = _load_skip()  # poison-pill blacklist: never re-attempt (context-overflow pills + hand-listed broken)
    if skip:
        n_before = len(instances)
        instances = [i for i in instances if i["instance_id"] not in skip]
        print(f"[batch] skip-list excludes {n_before - len(instances)} instance(s) (dataset/splits/skip.json)", flush=True)
    todo = [i for i in instances if not _is_complete(i["instance_id"], run_id)]
    skipped = len(instances) - len(todo)
    if limit:
        todo = todo[:limit]
    total = len(todo)

    counts: dict[str, int] = {}
    done = errors = 0
    lock = threading.Lock()

    def process(inst: dict, n: int) -> None:
        nonlocal done, errors
        iid = inst["instance_id"]
        print(f"[batch {n}/{total}] {iid} ({inst.get('difficulty', inst.get('repo_language', '?'))}) start", flush=True)
        # pre-pull explicitly: docker-run's implicit pull glitched (125) early in a fresh-daemon batch
        if not _pull(_image(inst)):
            # don't burn a container-start + collect cycle on a missing image (network outage / bad tag)
            with lock:
                errors += 1
            print(f"  PULL-FAIL {iid}: image unavailable, skipping (rerun the batch to retry)", flush=True)
            return
        t = time.time()
        summary = None
        outcome_path = _run_dir(iid, run_id) / "outcome.json"
        if outcome_path.exists():
            summary = json.loads(outcome_path.read_text(encoding="utf-8"))
            print(f"  fetch-only {iid}: outcome already exists, relevant_code missing", flush=True)
        else:
            for attempt in range(2):  # one retry on a transient docker failure
                try:
                    summary = collect_one(inst, run_id=run_id, agent_effort="xhigh")  # teacher > student; oracle has budget to solve
                    break
                except LauncherBroken as e:
                    # the WORKER PROCESS is poisoned (local spawns fail with NTSTATUS codes after hours of
                    # subprocess churn) — every subsequent instance in this process would burn LLM calls on
                    # launch failures too. Exit hard; the outer shell loop respawns a fresh process, which
                    # resumes (this instance has no outcome yet, so it is retried clean). 2026-07-09.
                    print(f"  LAUNCHER-BROKEN {iid}: {e} — exiting so the loop respawns a fresh worker", flush=True)
                    os._exit(97)
                except Exception as e:
                    if is_context_overflow(e):  # poison, not transient: retiring beats retrying (would re-overflow)
                        _retire_poison(iid, run_id, "context-overflow (raised)")
                        with lock:
                            errors += 1
                        print(f"  POISON {iid}: context overflow -> skip-listed + dir removed (no retry)", flush=True)
                        return
                    if attempt == 0:
                        print(f"  retry {iid}: {repr(e)[:120]}", flush=True)
                        time.sleep(10)
                        _pull(_image(inst))  # the image may have been half-pulled or pruned by a racing worker
                        continue
                    with lock:
                        errors += 1
                    print(f"  ERROR {iid}: {repr(e)[:200]}", flush=True)
        # context-overflow that collect() caught cleanly (outcome written, no raise): retire the same way.
        if summary and summary.get("outcome") == "context_overflow":
            _retire_poison(iid, run_id, "context-overflow")
            with lock:
                errors += 1
            print(f"  POISON {iid}: context overflow -> skip-listed + dir removed (no retry)", flush=True)
            return
        # NOTE: a chaotic xhigh trajectory is kept as-is — it still teaches reasoning (failed attempts + oracle directions).
        if summary and dataset not in ("tb", "lcb"):  # no source-context fetch (TB gold=solve.sh, LCB has no gold)
            try:
                _fetch_relevant_code(inst, run_id, dataset)
            except Exception as e:
                with lock:
                    errors += 1
                print(f"  FETCH-WARN {iid}: {repr(e)[:200]}", flush=True)
                return
        if prune_images:
            _prune_image(inst)  # per-instance image (each instance has its own); safe across workers
        if summary:
            with lock:
                counts[summary["outcome"]] = counts.get(summary["outcome"], 0) + 1
                done += 1
                snap = dict(counts)
            print(f"  -> {summary['outcome']} ({iid}) in {int(time.time() - t)}s | running: {snap}", flush=True)

    print(f"[batch] {total} to collect, {skipped} already done, {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process, inst, n) for n, inst in enumerate(todo, 1)]
        for f in futs:
            f.result()  # surface any uncaught exception
    _cleanup_containers()  # final global sweep — safe, nothing running now
    print(f"[batch done] done={done} skipped={skipped} errors={errors} | outcomes={counts}", flush=True)
    # force-exit (2026-07-10): a worker thread wedged on a hung docker exec / API call is non-daemon and
    # keeps the process alive AFTER main() returns and every future resolved — observed repeatedly stalling
    # the outer resume loop for hours past [batch done]. All results are already on disk; skip interpreter
    # teardown and exit hard so the loop respawns the next round immediately.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    app()
