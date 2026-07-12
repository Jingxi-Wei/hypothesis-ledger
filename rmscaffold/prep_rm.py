"""Preference pairs -> LLaMA-Factory RM training file (deterministic, zero proxy, local).

Merges TWO pair sources into one ranking dataset:
  * natural pairs   dataset/samples/<run>/preference.jsonl   (export.py::export_preference v2 —
                    decision/issue_only pair_types, chosen = outcome-verified, strength=verified)
  * resample pairs  rmscaffold/resample/pairs_<run>_train.jsonl  (gen_pairs.py — fresh same-node
                    candidates judged against gold, strength=judged)

Filters, in order (all counted in rm_stats.json — LOOK at it before training):
  1. run exclusion      r1 dropped by default (old protocol; Verified will be re-collected) — --exclude-runs
  2. protocol           raw_leak pairs dropped (--keep-raw-leak = DIAGNOSTIC ONLY)
  3. F2P leak scan      any hidden-test identifier in prompt/chosen/rejected -> drop (needs HF dataset cache;
                        --no-leak-scan to skip offline, counted as unscanned)
  4. dedupe             (prompt, chosen, rejected) prefix key
  5. length gate        est. tokens (chars/3.5) > --max-tokens -> drop (a truncated pair trains on half a prompt)
  6. per-instance cap   seeded RANDOM sample (not first-N file order) so one long refutation chain
                        cannot dominate the RM's notion of 'bad'

Writes: rm_pairs.jsonl (sharegpt-ranking) + dataset_info.json (`hl_rm`) + rm_stats.json.
Training gate: <300 kept pairs = don't train yet, keep collecting/judging.

Run:  python rmscaffold/prep_rm.py            # from the project root
"""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
SAMP = ROOT / "dataset" / "samples"
HERE = Path(__file__).resolve().parent
app = typer.Typer(add_completion=False)


def _hidden_test_tokens(instance: dict) -> list[str]:
    """Tokens of the HIDDEN (F2P) tests only, both key casings, scrub_f2p-style token model.
    Why F2P-ONLY here (unlike collect's direction redaction, which scrubs P2P too): P2P tests pre-exist in
    the repo, so their names legitimately appear in the agent's OWN test runs inside raw-masked prompts —
    scanning them would mass-drop honest pairs. Only test_patch-added F2P names are contamination.
    Why both casings + Go/JS rules: Verified uses FAIL_TO_PASS + py test_* names; SWE-bench_Pro uses
    lowercase fail_to_pass with Go Test*/JS sentence titles — the old uppercase-py-only version made this
    scan a silent no-op on every Pro pair (2026-07-07 review's top finding)."""
    names: list[str] = []
    for key in ("FAIL_TO_PASS", "fail_to_pass"):
        val = instance.get(key)
        if val is None:
            continue
        v = val
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception:
                try:
                    import ast
                    v = ast.literal_eval(v)
                except Exception:
                    v = []
        if isinstance(v, (list, tuple)):
            names += [str(t) for t in v]
    toks: set[str] = set()
    for t in names:
        # CASE NAME only: for 'path/test_file.py::test_case' entries the file part pre-exists in the repo
        # (the agent legitimately reads/greps it — only the CASE added by the fix is hidden). Scanning file
        # stems mass-flagged honest self_solved pairs (eyeballed 2026-07-07: test_utils/test_version hits
        # were all the agent's own repo exploration, zero real leaks).
        t = t.split("::")[-1] if "::" in t else t
        toks.update(re.findall(r"\btest_[A-Za-z0-9_]{3,}\b", t))
        toks.update(re.findall(r"\bTest[A-Z][A-Za-z0-9_]{2,}\b", t))
        s = t.strip()
        if " " in s and len(s) >= 15:  # js/ts sentence-style titles
            toks.add(s)
    return sorted(toks, key=len, reverse=True)  # longest first so subset tokens don't clip supersets


def leak_hits(pair: dict, tokens: list[str]) -> list[str]:
    blob = "\n".join((pair.get("prompt", ""), pair.get("chosen", ""), pair.get("rejected", "")))
    hits = []
    for tk in tokens:  # boundary anchors only at alnum edges (sentence titles end in punctuation)
        pat = (r"\b" if tk[:1].isalnum() else "") + re.escape(tk) + (r"\b" if tk[-1:].isalnum() else "")
        if re.search(pat, blob):
            hits.append(tk)
    return hits


def load_token_map(no_leak_scan: bool) -> dict[str, list[str]]:
    """instance_id -> hidden-test tokens, merged across the datasets our runs draw from (HF cache)."""
    if no_leak_scan:
        return {}
    from datasets import load_dataset
    m: dict[str, list[str]] = {}
    for name in ("princeton-nlp/SWE-bench_Verified", "ScaleAI/SWE-bench_Pro"):
        try:
            for inst in load_dataset(name, split="test"):
                m[inst["instance_id"]] = _hidden_test_tokens(inst)
        except Exception as e:
            print(f"  [leak-scan] could not load {name} ({type(e).__name__}) — its instances stay unscanned")
    return m


@app.command()
def main(seed: int = typer.Option(42, "--seed"),
         max_per_instance: int = typer.Option(8, "--max-per-instance"),
         max_tokens: int = typer.Option(8192, "--max-tokens", help="drop pairs longer than this (est., chars/3.5)"),
         chars_per_token: float = typer.Option(3.5, "--chars-per-token"),
         exclude_runs: str = typer.Option("r1", "--exclude-runs", help="comma-separated run-ids to drop entirely"),
         keep_raw_leak: bool = typer.Option(False, "--keep-raw-leak",
                                            help="DIAGNOSTIC ONLY: include old raw-leak-protocol pairs"),
         no_leak_scan: bool = typer.Option(False, "--no-leak-scan", help="skip the F2P token scan (offline)")) -> None:
    excluded = {r.strip() for r in exclude_runs.split(",") if r.strip()}
    rng = random.Random(seed)
    tokmap = load_token_map(no_leak_scan)
    # STRUCTURAL holdout wall (belt to gen_pairs'/export's braces): whatever upstream tagging/CLI mistakes
    # happen, a held-out instance's pair must never reach rm_pairs.jsonl. Fail loud if the wall file is gone.
    hp = ROOT / "dataset" / "splits" / "pro_test.json"
    if not hp.exists():
        raise SystemExit("[prep_rm] HOLDOUT WALL MISSING: dataset/splits/pro_test.json not found — refusing to "
                         "build RM training pairs without it (held-out pairs could silently enter training).")
    holdout = set(json.loads(hp.read_text(encoding="utf-8")))

    # ---- gather (source, run, pair) from both channels ----
    found: list[tuple[str, dict]] = []
    n_run_dropped = 0
    for p in sorted(SAMP.glob("*/preference.jsonl")):
        run = p.parent.name
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if run in excluded:
            n_run_dropped += len(rows)
            continue
        found += [(run, s) for s in rows]
    for p in sorted((HERE / "resample").glob("pairs_*_train.jsonl")) if (HERE / "resample").exists() else []:
        run = p.stem.replace("pairs_", "").replace("_train", "")
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if run in excluded:
            n_run_dropped += len(rows)
            continue
        found += [(run, s) for s in rows]

    stats = Counter(total_read=len(found), run_dropped=n_run_dropped)
    by_type, by_strength, by_run, leak_insts = Counter(), Counter(), Counter(), set()
    per_inst: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for run, s in found:
        if s["instance_id"] in holdout:
            stats["holdout_dropped"] += 1
            continue
        if not keep_raw_leak and s.get("protocol") == "raw_leak":
            stats["raw_leak_dropped"] += 1
            continue
        toks = tokmap.get(s["instance_id"])
        if toks is None and not no_leak_scan:
            stats["unscanned"] += 1
        elif toks and leak_hits(s, toks):
            stats["leak_dropped"] += 1
            leak_insts.add(s["instance_id"])
            continue
        # full-content hash: prompt PREFIXES collide across pair_types (both start with the same ISSUE text)
        key = hashlib.sha1("\x1f".join((s["prompt"], s["chosen"], s["rejected"])).encode("utf-8")).hexdigest()
        if key in seen:
            stats["dup_dropped"] += 1
            continue
        seen.add(key)
        if (len(s["prompt"]) + len(s["chosen"]) + len(s["rejected"])) / chars_per_token > max_tokens > 0:
            stats["len_dropped"] += 1
            continue
        s["_run"] = run
        per_inst[s["instance_id"]].append(s)

    rows = []
    for iid, pairs in sorted(per_inst.items()):
        if len(pairs) > max_per_instance:
            stats["cap_dropped"] += len(pairs) - max_per_instance
            # STRATIFIED, verified-first: outcome-verified pairs are the scarce anchor (~15% of the pool) —
            # a uniform sample across the instance evicts them at the same rate as the abundant judged pairs.
            # Random WITHIN each stratum (not first-N): still no file-order bias.
            ver = [p for p in pairs if p.get("strength") == "verified"]
            jud = [p for p in pairs if p.get("strength") != "verified"]
            if len(ver) >= max_per_instance:
                pairs = rng.sample(ver, max_per_instance)
            else:
                pairs = ver + rng.sample(jud, max_per_instance - len(ver))
        for s in pairs:
            by_type[s.get("pair_type", "legacy")] += 1
            by_strength[s.get("strength", "?")] += 1
            by_run[s["_run"]] += 1
            rows.append({"conversations": [{"from": "human", "value": s["prompt"]}],
                         "chosen": {"from": "gpt", "value": s["chosen"]},
                         "rejected": {"from": "gpt", "value": s["rejected"]}})
    rng.shuffle(rows)

    (HERE / "rm_pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                         encoding="utf-8")  # trailing newline: wc -l must equal the pair count
    (HERE / "dataset_info.json").write_text(json.dumps({
        "hl_rm": {"file_name": "rm_pairs.jsonl", "formatting": "sharegpt", "ranking": True,
                  "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"},
                  "tags": {"role_tag": "from", "content_tag": "value",
                           "user_tag": "human", "assistant_tag": "gpt"}}
    }, indent=2), encoding="utf-8")
    out = dict(stats)
    out.update({"kept": len(rows), "instances": len(per_inst), "by_pair_type": dict(by_type),
                "by_strength": dict(by_strength), "by_run": dict(by_run),
                "leak_instances": sorted(leak_insts)})
    (HERE / "rm_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    verdict = ("OK to train" if len(rows) >= 300 else
               "TOO FEW clean pairs — keep collecting (pro2) / judging (gen_pairs), then re-run")
    print(f"[prep_rm] kept {len(rows)} pairs from {len(per_inst)} instances | types {dict(by_type)} | "
          f"strength {dict(by_strength)} | runs {dict(by_run)}\n"
          f"  dropped: run {n_run_dropped}, holdout {stats['holdout_dropped']}, raw_leak {stats['raw_leak_dropped']}, "
          f"leak {stats['leak_dropped']}, dup {stats['dup_dropped']}, len {stats['len_dropped']}, cap {stats['cap_dropped']}"
          + (f", unscanned {stats['unscanned']}" if stats.get("unscanned") else "")
          + f"\n  -> rm_pairs.jsonl | {verdict}")


if __name__ == "__main__":
    app()
