"""Parallel posthoc audit (2 concurrent — proxy tolerates ~2) over all COMPLETE pro2 trajectories.
Skips instances that already have audit.json. Mirrors audit_run.py's per-instance logic; used for the
downstream calibration batch where serial (~1.5min x 133) is too slow. Collection must be PAUSED (owns proxy)."""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from collect import audit_all, load_instances  # noqa: E402

RAW = ROOT / "dataset" / "raw"
RUN_ID = "pro2"
ds = load_instances("pro")
lock = threading.Lock()
done = skipped = failed = 0

targets = []
for p in sorted(RAW.iterdir()):
    rd = p / RUN_ID
    if not (rd / "trajectory.json").exists() or not (rd / "outcome.json").exists():
        continue  # skip orphans (no outcome) — calibration is over the 136 complete
    if not (rd / "ledger.json").exists():
        continue
    if (rd / "audit.json").exists():
        skipped += 1
        continue
    if p.name not in ds:
        continue
    targets.append(p.name)

print(f"[audit-par] {len(targets)} to audit, {skipped} already done", flush=True)
total = len(targets)


def one(iid: str, n: int) -> None:
    global done, failed
    rd = RAW / iid / RUN_ID
    try:
        inst = ds[iid]
        traj = json.loads((rd / "trajectory.json").read_text(encoding="utf-8"))
        outcome = json.loads((rd / "outcome.json").read_text())["outcome"]
        cards = json.loads((rd / "ledger.json").read_text(encoding="utf-8")).get("cards", [])
        audit = audit_all(traj["messages"], inst["problem_statement"], inst.get("patch") or "", outcome, cards=cards)
        (rd / "audit.json").write_text(audit, encoding="utf-8")
        with lock:
            done += 1
            k = done
        print(f"[audit-par {k}/{total}] {iid[:50]} ({outcome})", flush=True)
    except Exception as e:
        with lock:
            failed += 1
        print(f"[audit-par FAIL] {iid[:50]}: {repr(e)[:150]}", flush=True)


with ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(one, iid, n) for n, iid in enumerate(targets, 1)]
    for f in futs:
        f.result()
print(f"[audit-par DONE] audited={done} skipped={skipped} failed={failed}", flush=True)
