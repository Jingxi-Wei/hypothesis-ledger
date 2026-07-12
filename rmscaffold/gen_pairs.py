"""Resample pairs: same-node candidate generation + gold-anchored judging (pair_type=resample).

WHY (the gap natural pairs cannot fill): in natural decision pairs the rejected side is always a hypothesis
already printed in the prompt's RULED OUT block — the RM mostly learns "don't repeat refuted directions".
At deployment every candidate is FRESH. This script manufactures that distribution: at each decision point
(propose node) it samples K fresh one-line hypotheses, then one gold-anchored judge call labels them all;
correct×wrong siblings become pairs. Candidates share the exact same information basis (same prompt, same
birth time) — information parity by construction, and the PAIR PROMPT IS the generation context itself, so
everything a candidate cites is checkable against the prompt.

Context = RAW-MASKED by default (user decision 2026-07-07): the RM's real post — gating the propose step of
a LIVE agent — scores candidates inside the agent's raw conversation context, where no oracle exists. So the
deployment-faithful training context is the collected raw prefix with every harness/oracle feedback message
masked out (at deployment those messages simply never happen). --context ledger remains as the ablation arm
matching the static eval-items world (compressed digests).
  Honest residue of masking: the agent's OWN later prose may echo a masked direction ("the feedback points
at X") — that echo is woven into agent text and cannot be masked deterministically. Nodes after the first
feedback are therefore tagged post_feedback=true; slice on it before trusting correction-round numbers.

The judge may use privileged info (gold patch + verified cause) — privilege affects LABEL quality only and
never enters pair text. `grounded` is computed deterministically (export._grounded), not asked of the judge.

  !!! PROXY IS SERIAL: run ONLY while the collection batch is PAUSED (same queue slot as posthoc audit). !!!

  python rmscaffold/gen_pairs.py --run-id pro2 --dataset pro                       # training nodes
  python rmscaffold/gen_pairs.py --run-id pro2 --dataset pro --instances-file dataset/splits/pro_test.json \
         --include-holdout --tag eval                                              # held-out eval nodes
  python rmscaffold/gen_pairs.py --selftest                                        # offline parse/pair check
  python rmscaffold/gen_pairs.py --from-candidates rmscaffold/bon_candidates_base.jsonl \
         --items-file dataset/eval/items.jsonl --run-id pro2 --dataset pro --tag policy_eval
         # judge+pair candidates a GPU policy generated (bon_candidates format) — closes the policy gap later

Env: GEN_EFFORT (default medium), JUDGE_EFFORT (default high), GPT_MODEL_PREFIX, GPT_SPEED — same proxy
model-name-carries-effort convention as src/collect.py.
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from export import _walk, _grounded, _norm_line, _parse_json, _protocol, DS_NAME, RAW, UNIFIED_ASK  # noqa: E402

app = typer.Typer(add_completion=False)

GPT_MODEL_PREFIX = os.environ.get("GPT_MODEL_PREFIX", "openai/gpt-5.5")
GPT_SPEED = os.environ.get("GPT_SPEED", "fast")
GEN_EFFORT = os.environ.get("GEN_EFFORT", "medium")
JUDGE_EFFORT = os.environ.get("JUDGE_EFFORT", "high")

# light instruction perturbations — diversity lever for the K samples (reasoning-model temperature is not
# reliable through the proxy). They nudge the search angle, never the required format or the honesty bar.
_PERTURB = [
    "",
    " Consider a plausible cause DIFFERENT from the most obvious first reading.",
    " Weigh which of the code locations mentioned in the evidence is most likely mis-behaving, and commit to one.",
    " Consider whether the defect is in the handling of an edge/boundary case rather than the main path.",
    " Consider whether an earlier processing/parsing stage (not the place symptoms appear) is responsible.",
    " Consider the interaction between the components mentioned, not a single component in isolation.",
]

_GEN_SYS = ("You are debugging. Read the context and answer with EXACTLY one line of the form\n"
            "HYPOTHESIS: <a single-sentence, specific, checkable conjecture about the CAUSE of the bug>\n"
            "Name only code/identifiers that appear in the context. No preamble, no extra lines.")

_JUDGE_SYS = ("You label and compare candidate root-cause hypotheses for a known bug. You are given the ground "
              "truth (the gold patch, and the verified cause when the bug was actually solved), and possibly a "
              "REFERENCE hypothesis — the one the agent actually reached at this point WITH a guidance hint; "
              "the candidates were written WITHOUT that hint. Tasks: "
              "(1) For EACH candidate, judge direction: 'correct' = it identifies the same causal "
              "mechanism/location the gold patch fixes; 'partial' = overlaps the true cause but misses or "
              "mixes the mechanism; 'wrong' = a different mechanism/location. "
              "(2) GROUP candidates that assert the same causal mechanism at the same location — rewordings "
              "of one idea belong in ONE group; split only on genuine mechanism/location differences. "
              "(3) If a REFERENCE is provided: label it on the same absolute scale (reference_label, judged "
              "on its content alone), and judge each group's TIER against it, where closer to the true cause "
              "is better: 'below' = farther than the reference, 'par' = the same level, 'above' = strictly "
              "closer to the true cause. "
              "(4) RANK the groups from closest to the true cause to farthest (no ties — grouping absorbs "
              "equivalence). Judge ONLY against the ground truth — not style, not length, not confidence. "
              "Output STRICT JSON only.")


def _kw(effort: str) -> dict:
    suffix = f"-{GPT_SPEED}" if GPT_SPEED else ""
    return dict(model=f"{GPT_MODEL_PREFIX}-{effort}{suffix}",
                api_base="http://127.0.0.1:8080/v1", api_key="pwd")


def _llm(sys_p: str, user_p: str, effort: str, retries: int = 2) -> str:
    import litellm
    for att in range(retries + 1):
        try:
            r = litellm.completion(messages=[{"role": "system", "content": sys_p},
                                             {"role": "user", "content": user_p}], **_kw(effort))
            return r.choices[0].message.content or ""
        except Exception as e:
            if att == retries:
                raise
            print(f"    [retry {att + 1}] {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(5 * (att + 1))
    return ""


_HYP_LINE = re.compile(r"HYPOTHESIS:\s*(.+)", re.IGNORECASE)


def _extract_hyp(text: str) -> str:
    """One-line hypothesis from a generation (response normalization — same convention as natural pairs)."""
    m = _HYP_LINE.search(text or "")
    return _norm_line(m.group(1)) if m else _norm_line(text)[:400]


# every harness-injected feedback message (exact marker strings from collect.py _on_submit) — at deployment
# none of these messages exist, so a deployment-faithful context masks them ALL, not just the directions.
_CRUTCH_MARKERS = ("Sanitized test-feedback direction", "guidance round", "tests FAILED",
                   "UNCHANGED from your last refuted attempt", "Resubmitted an unchanged",
                   "Still failing after several rounds", "All tests pass. Task solved.")

def _content_text(c) -> str:
    """Message content as text. Trajectories MAY store list-of-parts content (compress._text defends against
    exactly this); treating a list as a string would crash _norm_line and — worse — make the crutch-marker
    substring check silently False, letting feedback text through the mask."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(_content_text(x) for x in c)
    if isinstance(c, dict):
        return str(c.get("text") or c.get("content") or "")
    return "" if c is None else str(c)


def _raw_masked_context(rd: Path, hyp: str, budget_chars: int = 24000,
                        raw_ref: int | None = None) -> tuple[str, bool] | None:
    """Deployment-faithful context: the raw trajectory prefix up to (EXCLUDING) the message where this card's
    hypothesis was stated, with every harness/oracle feedback message masked. Budget keeps the FIRST messages
    (task + issue) plus the most recent tail — the decision-relevant window.

    Boundary: the card's positional anchor (compress records raw_ref='msg#<i>') when available — exact, and
    immune to the restatement trap (a hypothesis restated after a refutation would FIRST-match its original
    statement far earlier, silently dropping the intervening feedback and mislabeling post_feedback). Fuzzy
    80-char prefix matching is only the legacy-ledger fallback. Returns None when no boundary can be located
    (falling through would leak the node's own answer into its context)."""
    tp = rd / "trajectory.json"
    if not tp.exists():
        return None
    msgs = json.loads(tp.read_text(encoding="utf-8")).get("messages", [])
    if raw_ref is not None and not (0 <= raw_ref < len(msgs)):
        raw_ref = None
    key = _norm_line(hyp)[:80].casefold()
    parts, matched, post_fb = [], False, False
    for i, m in enumerate(msgs):
        c = _content_text(m.get("content"))
        if raw_ref is not None:
            if i >= raw_ref:
                matched = True
                break
        elif m.get("role") == "assistant" and key and key in _norm_line(c).casefold():
            matched = True
            break
        if any(mk in c for mk in _CRUTCH_MARKERS):
            post_fb = True
            c = "(submission feedback withheld — at deployment no oracle / hidden-test verdict exists)"
        parts.append(f"[{m.get('role', '?')}]\n{c}")
    if not matched or not parts:
        return None
    text = "\n\n".join(parts)
    if len(text) > budget_chars:
        n_head = min(2, len(parts))  # keep system AND the first user message — the ISSUE lives there;
        head = "\n\n".join(parts[:n_head])[:min(8000, budget_chars)]  # head must respect small budgets too
        tail: list[str] = []
        used = len(head)
        for p in reversed(parts[n_head:]):
            if used + len(p) > budget_chars:
                break
            tail.append(p)
            used += len(p)
        if not tail and len(parts) > n_head:  # one oversized newest message must not empty the tail —
            tail.append(parts[-1][-max(500, budget_chars - used):])  # include it truncated instead
        text = head + "\n\n[... earlier steps truncated ...]\n\n" + "\n\n".join(reversed(tail))
    return (f"# Agent transcript so far\n{text}\n\n{UNIFIED_ASK}", post_fb)


def _judge(prompt: str, cands: list[str], gold_patch: str, cause: str | None,
           real: str | None = None) -> tuple[list | None, str | None, dict | None]:
    """Combined single-call judge (2026-07-09, user: '直接按新的跑,旧的保留'): absolute labels for the
    candidates PLUS mechanism groups / reference-anchored tiers / ranking — everything the two-pass design
    (old S4 judge + rejudge_rank) produced, in one call. Returns (labels, real_label, extras); extras =
    {groups, tiers?, ranking} in the PRESENTED (shuffled) candidate space, or None when those fields fail
    validation (labels alone still count — the node then falls back to rejudge_rank in the backfill).
    NOTE the reference is now MARKED for the judge (tiers need an anchor), so real_label is no longer blind
    — acceptable for its diagnostic role; instruction says to label it on content alone."""
    lst = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cands))
    k = len(cands)
    schema = ('{"labels": [{"n": 1, "direction": "correct|partial|wrong"}, ...] '
              f"(exactly {k} entries, n = 1..{k} in order)"
              + (', "reference_label": "correct|partial|wrong", '
                 '"groups": [[<1-based candidate numbers>], ...], '
                 '"tiers": ["below"|"par"|"above" per group], '
                 '"ranking": [<0-based group indices, best first>]}' if real else
                 ', "groups": [[<1-based candidate numbers>], ...], '
                 '"ranking": [<0-based group indices, best first>]}'))
    # honest premise: candidates saw this context, PLUS possibly a one-sentence angle nudge with no code
    # content (_PERTURB) — claiming "NOTHING else" would hand the judge a false premise
    user_p = (f"# Context the candidates saw (some also received a one-sentence angle nudge with no code content)\n{prompt}\n\n"
              f"# GOLD patch (ground truth — labels only, never quoted back)\n{gold_patch}\n\n"
              + (f"# Verified cause (this hypothesis actually led to the passing fix)\n{cause}\n\n" if cause else "")
              + (f"# REFERENCE hypothesis (reached WITH a guidance hint — the tier anchor)\n{real}\n\n" if real else "")
              + f"# Candidates (written WITHOUT the hint)\n{lst}\n\n"
              + f"Answer as STRICT JSON: {schema}")
    out = _parse_json(_llm(_JUDGE_SYS, user_p, JUDGE_EFFORT))
    if not out or not isinstance(out.get("labels"), list):
        return None, None, None
    labels = [None] * k
    for e in out["labels"]:
        if not isinstance(e, dict):  # a bare string/int entry must count as unparseable, not crash the node
            continue
        try:
            n = int(e.get("n"))
        except (TypeError, ValueError):
            continue
        if 1 <= n <= k and isinstance(e.get("direction"), str):
            labels[n - 1] = e["direction"].strip().lower()
    if not all(l in ("correct", "partial", "wrong") for l in labels):
        return None, None, None
    real_label = None
    if real and isinstance(out.get("reference_label"), str):
        rl = out["reference_label"].strip().lower()
        real_label = rl if rl in ("correct", "partial", "wrong") else None
    # extras validated strictly; any failure -> None (labels still usable, rejudge_rank backfills the node)
    extras = None
    try:
        groups = [[int(x) for x in g] for g in out["groups"]]
        ranking = [int(x) for x in out["ranking"]]
        ok = (sorted(x for g in groups for x in g) == list(range(1, k + 1))
              and sorted(ranking) == list(range(len(groups))))
        tiers = None
        if real:
            tiers = [str(t).strip().lower() for t in out["tiers"]]
            ok = ok and len(tiers) == len(groups) and all(t in ("below", "par", "above") for t in tiers)
        if ok:
            extras = {"groups": groups, "ranking": ranking}
            if tiers is not None:
                extras["tiers"] = tiers
    except (KeyError, ValueError, TypeError):
        extras = None
    return labels, real_label, extras


def derive_pairs(node: dict, cands: list[str], labels: list[str], rng: random.Random,
                 cap: int = 4) -> list[dict]:
    """correct×wrong siblings -> pairs. chosen must additionally be deterministically grounded in the prompt
    (never reward citing unseen specifics); wrong candidates pair regardless of groundedness (tagged)."""
    prompt = node["prompt"]
    good = [c for c, l in zip(cands, labels) if l == "correct" and _grounded(c, prompt)]
    bad = [c for c, l in zip(cands, labels) if l == "wrong"]
    combos = [(g, b) for g in good for b in bad if _norm_line(g).casefold() != _norm_line(b).casefold()]
    rng.shuffle(combos)
    return [{"type": "preference", "pair_type": "resample", "instance_id": node["instance_id"],
             "protocol": node["protocol"], "strength": "judged", "rejected_source": "judge_wrong",
             "chosen_grounded": True, "rejected_grounded": _grounded(b, prompt), "node": node["node_id"],
             "context_mode": node.get("context_mode", "ledger"), "gen_model": node.get("gen_model", ""),
             "post_feedback": node.get("post_feedback", False),
             "prompt": prompt, "chosen": g, "rejected": b} for g, b in combos[:cap]]


def _selftest() -> None:
    """Offline check of extraction, judge parsing and pair derivation — no proxy."""
    assert _extract_hyp("noise\nHYPOTHESIS: the `parse_meta` loop collapses duplicate keys.\n") == \
        "the `parse_meta` loop collapses duplicate keys."
    assert _extract_hyp("bare text answer") == "bare text answer"
    node = {"instance_id": "x", "protocol": "sanitized", "node_id": "x::pro2::0",
            "prompt": "ISSUE: parse_meta collapses duplicate keys in reader_config"}
    cands = ["the parse_meta loop collapses duplicate keys silently",           # correct + grounded
             "the reader_config default masks duplicate keys",                  # wrong + grounded
             "the frobnicate_cache invalidation drops the second key"]          # wrong + ungrounded anchor
    labels = ["correct", "wrong", "wrong"]
    fake_judge = json.dumps({"labels": [{"n": i + 1, "direction": l} for i, l in enumerate(labels)]})
    parsed = _parse_json(fake_judge)["labels"]
    assert len(parsed) == 3
    pairs = derive_pairs(node, cands, labels, random.Random(0), cap=10)
    assert len(pairs) == 2, f"expected 1 good x 2 bad = 2 pairs, got {len(pairs)}"
    assert all(p["chosen"] == cands[0] for p in pairs)
    assert {p["rejected"] for p in pairs} == {cands[1], cands[2]}
    ung = [p for p in pairs if p["rejected"] == cands[2]]
    assert ung and not ung[0]["rejected_grounded"] and pairs[0]["chosen_grounded"]
    bad_judge = _parse_json('{"labels": [{"n": 1, "direction": "meh"}]}')
    assert bad_judge is not None  # parseable JSON, invalid direction -> _judge would return None
    print("[selftest] PASS — extraction, judge parsing, pair derivation, grounding tags all behave")


@app.command()
def main(run_id: str = typer.Option("pro2", "--run-id"),
         dataset: str = typer.Option("pro", "--dataset", help="verified | full | pro (gold lookup)"),
         k: int = typer.Option(6, "--k", help="fresh candidates per node"),
         cap_per_node: int = typer.Option(4, "--cap-per-node"),
         limit: int = typer.Option(0, "--limit", help="stop after N nodes (smoke)"),
         instances_file: str = typer.Option("", "--instances-file", help="restrict to these instance ids (JSON list)"),
         include_holdout: bool = typer.Option(False, "--include-holdout",
                                              help="allow pro_test instances (ONLY for building eval pairs)"),
         tag: str = typer.Option("train", "--tag", help="output suffix: pairs_<run>_<tag>.jsonl"),
         context: str = typer.Option("raw_masked", "--context",
                                     help="raw_masked (default; deployment-faithful raw prefix, feedback masked) "
                                          "| ledger (ablation arm matching the static eval-items world)"),
         raw_chars: int = typer.Option(24000, "--raw-chars",
                                       help="raw context budget (chars ~= tokens*3.5). Raising it past ~26k "
                                            "overflows the 8192-token training cutoff — raise rm_qlora cutoff_len "
                                            "and prep_rm --max-tokens together if you do"),
         from_candidates: str = typer.Option("", "--from-candidates",
                                             help="skip generation: judge+pair an existing bon_candidates_*.jsonl"),
         items_file: str = typer.Option("", "--items-file", help="eval items.jsonl (prompts for --from-candidates)"),
         seed: int = typer.Option(42, "--seed"),
         list_only: bool = typer.Option(False, "--list-only", help="enumerate nodes + cost estimate, NO proxy calls"),
         selftest: bool = typer.Option(False, "--selftest", help="offline parse/pair check, no proxy")) -> None:
    if selftest:
        _selftest()
        return
    # ---- holdout discipline: these mistakes silently train on the held-out set (2026-07-07 review) ----
    if include_holdout and tag == "train":
        raise typer.BadParameter("--include-holdout requires a non-'train' --tag (e.g. --tag eval): "
                                 "prep_rm merges every pairs_*_train.jsonl into RM TRAINING.")
    if from_candidates:
        if not items_file:
            raise typer.BadParameter("--from-candidates requires --items-file (the prompts live there).")
        if tag == "train":
            raise typer.BadParameter("--from-candidates consumes eval items (built from HELD-OUT instances) — "
                                     "use a non-'train' --tag.")
    from datasets import load_dataset
    if not list_only:
        print("!!! proxy is SERIAL — make sure the collection batch is PAUSED before running this. !!!", flush=True)
    ds = {i["instance_id"]: i for i in load_dataset(DS_NAME[dataset], split="test")}
    rng = random.Random(seed)
    rdir = HERE / "resample"
    rdir.mkdir(exist_ok=True)
    cand_p = rdir / f"candidates_{run_id}_{tag}.jsonl"
    lab_p = rdir / f"labels_{run_id}_{tag}.jsonl"
    pair_p = rdir / f"pairs_{run_id}_{tag}.jsonl"
    rank_p = rdir / f"rankings_{run_id}_{tag}.jsonl"  # combined-judge groups/tiers/ranking (rejudge_rank-compatible)
    stats_p = rdir / f"stats_{run_id}_{tag}.json"
    done = {json.loads(l)["node_id"] for l in lab_p.read_text(encoding="utf-8").splitlines()
            if l.strip()} if lab_p.exists() else set()
    stored = {json.loads(l)["node_id"]: json.loads(l) for l in cand_p.read_text(encoding="utf-8").splitlines()
              if l.strip()} if cand_p.exists() else {}

    # ---- enumerate nodes ----
    nodes = []
    if from_candidates:  # judge+pair pre-generated candidates (e.g. GPU policy samples in bon_candidates format)
        items = {json.loads(l)["item_id"]: json.loads(l)
                 for l in Path(items_file).read_text(encoding="utf-8").splitlines() if l.strip()}
        for l in Path(from_candidates).read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            row = json.loads(l)
            it = items.get(row["item_id"])
            if not it:
                continue
            iid = row["item_id"].split("::")[0]
            if iid not in ds:
                continue
            proto = _protocol(RAW / iid / run_id) if (RAW / iid / run_id).exists() else "unknown"
            nodes.append({"node_id": row["item_id"], "instance_id": iid, "prompt": it["input"],
                          "protocol": proto, "context_mode": "policy", "gen_model": "policy",
                          "cands": [_extract_hyp(c) for c in row["candidates"]],
                          "cause": None, "gold_patch": ds[iid].get("patch") or ds[iid].get("gold_patch", "")})
    else:
        if instances_file:
            targets = json.loads(Path(instances_file).read_text(encoding="utf-8"))
        else:
            targets = sorted(p.name for p in RAW.iterdir() if (p / run_id / "ledger.json").exists())
        holdout = set()
        if dataset == "pro" and not include_holdout:
            hp = ROOT / "dataset" / "splits" / f"{dataset}_test.json"
            if not hp.exists():  # a silently-empty wall would mine held-out instances into training pairs
                raise SystemExit(f"[gen_pairs] HOLDOUT WALL MISSING: {hp} not found — refusing to mine training "
                                 "nodes without it (pass --include-holdout ONLY for the eval tag).")
            holdout = set(json.loads(hp.read_text(encoding="utf-8")))
        for iid in targets:
            if iid not in ds or iid in holdout:
                continue
            rd = RAW / iid / run_id
            if _protocol(rd) == "raw_leak":
                continue  # old-protocol trajectory: per the project decision these are not mined (no card-level rescue)
            w = _walk(iid, run_id, ds[iid])
            if w is None:
                continue
            wn, meta = w
            solved = meta["outcome"] in ("self_solved", "self_corrected", "oracle_redirected")
            cause = _norm_line(meta["by_card"].get(len(meta["cards"]), {}).get("hypothesis")
                               or meta["cards"][-1]["hypothesis"]) if (solved and meta["cards"]) else None
            for n in wn:
                prompt, mode, post_fb = n["propose_input"], "ledger", False
                if context == "raw_masked":
                    rr = re.match(r"msg#(\d+)", str(n["card"].get("raw_ref") or ""))
                    got = _raw_masked_context(rd, n["hyp"], raw_chars,
                                              raw_ref=int(rr.group(1)) if rr else None)
                    if got:
                        prompt, post_fb = got
                        mode = "raw_masked"
                    # else: no boundary (no raw_ref AND fuzzy miss) -> honest fallback to the ledger
                    # input (tagged context_mode=ledger, counted in stats) instead of risking answer leakage
                nodes.append({"node_id": f"{iid}::{run_id}::{n['i']}", "instance_id": iid, "prompt": prompt,
                              "protocol": meta["proto"], "context_mode": mode, "post_feedback": post_fb,
                              "gen_model": "", "real_hyp": _norm_line(n["hyp"]), "cause": cause,
                              "gold_patch": ds[iid].get("patch") or ds[iid].get("gold_patch", "")})
    todo = [n for n in nodes if n["node_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[gen_pairs] {len(nodes)} nodes, {len(done)} already judged, {len(todo)} to go "
          f"(k={k}, context={context}, tag={tag})", flush=True)
    if list_only:
        insts = len({n['instance_id'] for n in nodes})
        calls = len(todo) * (k + 1)
        cm = Counter(n.get("context_mode", "?") for n in nodes)
        pf = sum(1 for n in nodes if n.get("post_feedback"))
        print(f"[list-only] {insts} instances -> {len(nodes)} nodes (context {dict(cm)}, post_feedback {pf}); "
              f"remaining cost ≈ {len(todo)}×({k} gen + 1 judge) = {calls} serial proxy calls "
              f"(~{calls * 15 // 60}-{calls * 25 // 60} min at 15-25s/call; raw contexts run slower). No calls made.")
        return

    st = Counter()
    with cand_p.open("a", encoding="utf-8") as cf, lab_p.open("a", encoding="utf-8") as lf, \
            pair_p.open("a", encoding="utf-8") as pf, rank_p.open("a", encoding="utf-8") as rf:
        for idx, node in enumerate(todo):
            nid = node["node_id"]
            try:
                # ---- phase 1: candidates (resume from stored; --from-candidates ships its own) ----
                if "cands" in node:
                    cands = node["cands"]
                    if nid not in stored:  # pick-mode needs a prompt-bearing candidates row for policy tags too
                        cf.write(json.dumps({"node_id": nid, "instance_id": node["instance_id"],
                                             "gen_model": node["gen_model"], "context_mode": node["context_mode"],
                                             "post_feedback": node.get("post_feedback", False),
                                             "prompt": node["prompt"], "candidates": cands},
                                            ensure_ascii=False) + "\n")
                        cf.flush()
                elif nid in stored:
                    row = stored[nid]
                    if "prompt" not in row:  # judging stored candidates against a REBUILT prompt would break
                        st["resume_prompt_missing"] += 1  # information parity (candidates never saw it)
                        print(f"  [{idx + 1}/{len(todo)}] {nid}: stored candidates lack 'prompt' (old format), "
                              "skipped — delete the row to regenerate", flush=True)
                        continue
                    cands = row["candidates"]
                    node["gen_model"] = row.get("gen_model", "")
                    node["prompt"] = row["prompt"]  # the TRUE generation context — never the rebuilt one
                    node["context_mode"] = row.get("context_mode", node["context_mode"])
                    node["post_feedback"] = row.get("post_feedback", node.get("post_feedback", False))
                else:
                    node["gen_model"] = _kw(GEN_EFFORT)["model"]
                    raw = [_llm(_GEN_SYS, node["prompt"] + _PERTURB[j % len(_PERTURB)], GEN_EFFORT)
                           for j in range(k)]
                    cands = [_extract_hyp(t) for t in raw]
                    cf.write(json.dumps({"node_id": nid, "instance_id": node["instance_id"],
                                         "gen_model": node["gen_model"], "context_mode": node["context_mode"],
                                         "post_feedback": node.get("post_feedback", False),
                                         "perturbs": [j % len(_PERTURB) for j in range(k)],
                                         "prompt": node["prompt"], "candidates": cands},
                                        ensure_ascii=False) + "\n")
                    cf.flush()
                uniq = list(dict.fromkeys(c for c in cands if c))  # order-preserving dedupe
                # judge sees the candidates in a RANDOM order (position-bias control — LLM judges favor early
                # list slots); labels are mapped back to generation order for storage, so first-pick semantics
                # and resume alignment are untouched.
                order = list(range(len(uniq)))
                rng.shuffle(order)
                shuffled = [uniq[j] for j in order]
                # combined judge (2026-07-09): absolute labels + groups/tiers/ranking in ONE call — the real
                # hyp is presented as a MARKED reference (tier anchor); real_label is its absolute label
                labels_shuf, real_label, extras = _judge(node["prompt"], shuffled, node["gold_patch"],
                                                         node.get("cause"), real=node.get("real_hyp"))
                if labels_shuf is None:
                    st["judge_unparseable"] += 1
                    print(f"  [{idx + 1}/{len(todo)}] {nid}: judge output unparseable, skipped", flush=True)
                    continue
                lab_c = [None] * len(uniq)
                for pos, j in enumerate(order):
                    lab_c[j] = labels_shuf[pos]
                labels = lab_c + ([real_label] if node.get("real_hyp") else [])
                lf.write(json.dumps({"node_id": nid, "candidates": uniq + ([node["real_hyp"]] if node.get("real_hyp") else []),
                                     "labels": labels, "real_label": real_label},
                                    ensure_ascii=False) + "\n")
                lf.flush()
                if extras:  # map judge-space (shuffled) 1-based indices back to generation order for storage
                    gen_groups = [[order[n - 1] + 1 for n in g] for g in extras["groups"]]
                    rrow = {"node_id": nid, "candidates": uniq, "real_hyp": node.get("real_hyp"),
                            "groups": gen_groups, "ranking": extras["ranking"]}
                    if "tiers" in extras:
                        rrow["tiers"] = extras["tiers"]
                        for t in extras["tiers"]:
                            st[f"tier_{t}"] += 1
                    rf.write(json.dumps(rrow, ensure_ascii=False) + "\n")
                    rf.flush()
                else:
                    st["extras_missing"] += 1  # labels kept; rejudge_rank backfills this node's ranking later
                pairs = derive_pairs(node, uniq, lab_c, rng, cap_per_node)
                for p in pairs:
                    pf.write(json.dumps(p, ensure_ascii=False) + "\n")
                pf.flush()
                st["nodes"] += 1
                st["pairs"] += len(pairs)
                st[f"dist_{'/'.join(sorted(set(lab_c)))}"] += 1
                if not any(l == "correct" for l in lab_c):
                    st["zero_correct_nodes"] += 1  # ledger-thinness gauge: candidates could not reach the cause
                if not any(l == "wrong" for l in lab_c):
                    st["zero_wrong_nodes"] += 1    # too-easy gauge
                if node.get("real_hyp"):
                    st[f"real_{labels[-1]}"] += 1
                print(f"  [{idx + 1}/{len(todo)}] {nid}: {Counter(lab_c)} -> {len(pairs)} pairs", flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                st["errors"] += 1
                print(f"  [{idx + 1}/{len(todo)}] {nid}: ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
    # ---- stats: CUMULATIVE, rebuilt from the durable files — session counters lie after a resume
    # (a re-run that judges the last 5 nodes must not report zero_correct_frac over 5 nodes) ----
    cum = Counter()
    if lab_p.exists():
        for l in lab_p.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            row = json.loads(l)
            labs = row["labels"]
            if row.get("real_label") is not None:
                labs = labs[:-1]
                cum[f"real_{row['real_label']}"] += 1
            cum["nodes"] += 1
            if not any(x == "correct" for x in labs):
                cum["zero_correct_nodes"] += 1
            if not any(x == "wrong" for x in labs):
                cum["zero_wrong_nodes"] += 1
    cum["pairs"] = (sum(1 for l in pair_p.read_text(encoding="utf-8").splitlines() if l.strip())
                    if pair_p.exists() else 0)
    total = cum["nodes"] or 1
    cum["zero_correct_frac"] = round(cum["zero_correct_nodes"] / total, 3)
    if context == "raw_masked" and not from_candidates:
        cum["ledger_fallback_nodes"] = sum(1 for n in nodes if n.get("context_mode") == "ledger")
        cum["post_feedback_nodes"] = sum(1 for n in nodes if n.get("post_feedback"))
    stats_p.write_text(json.dumps({"cumulative": dict(cum), "session": dict(st)}, indent=2), encoding="utf-8")
    print(f"[gen_pairs] cumulative: {cum['nodes']} nodes -> {cum['pairs']} pairs -> {pair_p.name} | "
          f"zero-correct {cum['zero_correct_nodes']}/{total} ({cum['zero_correct_frac']:.0%}) | stats -> {stats_p.name}")
    if cum["zero_correct_frac"] > 0.5:
        print("  WARNING: >50% of nodes produced NO correct candidate — the context may be too thin for "
              "candidates to reach the cause. Inspect labels before training; consider a larger --raw-chars "
              "budget (sync cutoff_len/--max-tokens), NOT a side channel for candidates.")


if __name__ == "__main__":
    app()
