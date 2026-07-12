"""Raw-trajectory SFT baseline arm (the paper's control against structured SFT).

The claim "structured belief-state supervision beats raw-trajectory SFT" needs a raw arm trained on the
SAME collected trajectories with the STRUCTURE removed — otherwise the comparison is only vs base and the
claim can't be made. This builds that arm as standard reject-sampled agent-trajectory imitation:

  * SAME instances / runs / holdout wall / protocol filter as the structured arm (r1 + pro_test excluded,
    raw_leak trajectories dropped) — so the only variable is representation, not data hygiene.
  * SOLVED trajectories only (self_solved / self_corrected / oracle_redirected): you can only imitate a
    whole trajectory that reached a correct patch. This is itself a paper point — structure mines signal
    from FAILED runs too (via audit), raw cannot; report both instance counts.
  * Crutch masking: every harness/oracle-injected observation is replaced with a neutral placeholder (same
    marker set as gen_pairs), so the raw arm isn't quietly trained on leaked test output / oracle directions
    — that would confound "structure" with "leak hygiene". A masked observation leaves the following
    assistant redirect slightly unexplained; that is the honest cost of a leak-clean raw baseline and is
    noted, not hidden.
  * Multi-turn sharegpt (LLaMA-Factory trains on the gpt turns only): system+issue -> assistant -> obs ->
    assistant -> ... Long tool outputs are middle-truncated (a real agent context is bounded too); whole
    conversations over the token budget are dropped, same rule as prep_sft.

Output: dataset/sft_raw/ with the SAME dataset_info name (hl_sft) so the identical qwen_qlora_sft.yaml
trains it by just pointing dataset_dir at this folder — the three arms differ only by which folder is uploaded.

  python src/prep_raw_arm.py --runs pro1,pro2 --dataset pro
"""
import json
import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import _protocol  # noqa: E402  — same protocol classification as the structured arm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
OUT = ROOT / "dataset" / "sft_raw"
app = typer.Typer(add_completion=False)

SOLVED = {"self_solved", "self_corrected", "oracle_redirected"}
# exact injected-observation markers (mirror of gen_pairs._CRUTCH_MARKERS / collect._on_submit) — at
# deployment none of these exist, so a leak-clean raw baseline masks them all.
CRUTCH_MARKERS = ("Sanitized test-feedback direction", "guidance round", "tests FAILED",
                  "UNCHANGED from your last refuted attempt", "Resubmitted an unchanged",
                  "Still failing after several rounds", "All tests pass. Task solved.")
MASK = "(submission feedback withheld — no oracle / hidden-test verdict exists at deployment)"


def _text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(_text(x) for x in c)
    if isinstance(c, dict):
        return str(c.get("text") or c.get("content") or "")
    return "" if c is None else str(c)


def _obs(text: str, cap: int) -> str:
    if any(mk in text for mk in CRUTCH_MARKERS):
        return MASK
    if len(text) <= cap:
        return text
    half = cap // 2
    return text[:half] + "\n... [output truncated] ...\n" + text[-half:]


def build(iid: str, run: str, obs_cap: int) -> list[dict] | None:
    rd = RAW / iid / run
    tp, op = rd / "trajectory.json", rd / "outcome.json"
    if not tp.exists() or not op.exists():
        return None
    if json.loads(op.read_text(encoding="utf-8")).get("outcome") not in SOLVED:
        return None  # can't imitate a trajectory that never reached a correct patch
    if _protocol(rd) == "raw_leak":
        return None  # old protocol: contaminated, re-collect (same rule as structured)
    msgs = json.loads(tp.read_text(encoding="utf-8")).get("messages", [])
    # map roles to sharegpt turns: assistant -> gpt, everything else -> human; merge consecutive same-kind
    turns: list[tuple[str, str]] = []
    for m in msgs:
        who = "gpt" if m.get("role") == "assistant" else "human"
        t = _text(m.get("content"))
        if who == "human":
            t = _obs(t, obs_cap)
        if not t.strip():
            continue
        if turns and turns[-1][0] == who:
            turns[-1] = (who, turns[-1][1] + "\n\n" + t)
        else:
            turns.append((who, t))
    # LLaMA-Factory sharegpt: must start with human, strict alternation, end on gpt (trailing obs dropped)
    while turns and turns[0][0] != "human":
        turns.pop(0)
    while turns and turns[-1][0] != "gpt":
        turns.pop()
    if len(turns) < 2:
        return None
    return turns  # list[(who, text)]; budget-fitting happens in main() where max_tokens is known


def _fit(turns: list[tuple[str, str]], budget_chars: int, floor: int = 300) -> list[dict] | None:
    """Shrink OBSERVATION (human) turns — never the agent's reasoning (gpt) — until the whole conversation
    fits the training window. Keeping reasoning intact and squeezing tool output is exactly what a bounded
    agent context does; this lets the raw arm train at the SAME cutoff as structured instead of losing 99%
    of trajectories to length. The issue (turn 0) is preserved. Returns None only if reasoning alone overflows."""
    def total():
        return sum(len(t) for _w, t in turns)
    if total() <= budget_chars:
        return [{"from": w, "value": t} for w, t in turns]
    # repeatedly halve the largest non-issue observation until we fit or all obs are at the floor
    for _ in range(2000):
        obs = [(i, len(t)) for i, (w, t) in enumerate(turns) if w == "human" and i != 0 and len(t) > floor]
        if not obs:
            break
        i = max(obs, key=lambda x: x[1])[0]
        t = turns[i][1]
        keep = max(floor, len(t) // 2)
        turns[i] = ("human", t[:keep // 2] + "\n...[truncated]...\n" + t[-keep // 2:])
        if total() <= budget_chars:
            return [{"from": w, "value": t} for w, t in turns]
    return [{"from": w, "value": t} for w, t in turns] if total() <= budget_chars else None


@app.command()
def main(runs: str = typer.Option("pro1,pro2", "--runs"),
         dataset: str = typer.Option("pro", "--dataset", help="pro | verified (holdout wall source)"),
         obs_cap: int = typer.Option(3000, "--obs-cap", help="middle-truncate a single tool observation to N chars"),
         max_tokens: int = typer.Option(24576, "--max-tokens",
                                        help="drop whole conversations over this (== raw yaml cutoff_len). "
                                             "Measured on real data: 8192 keeps 1/18 oracle trajectories, "
                                             "16384 keeps 10/18, 24576 keeps 17/18 — small windows silently "
                                             "reduce the baseline to easy-solve imitation."),
         chars_per_token: float = typer.Option(3.5, "--chars-per-token")) -> None:
    holdout: set[str] = set()
    hp = ROOT / "dataset" / "splits" / f"{dataset}_test.json"
    if dataset == "pro":
        if not hp.exists():
            raise SystemExit(f"[prep_raw_arm] holdout wall missing: {hp}")
        holdout = set(json.loads(hp.read_text(encoding="utf-8")))
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    rows, n_inst, n_drop_len, n_held = [], 0, 0, 0
    seen_iids: set[str] = set()
    for run in run_ids:
        for p in sorted(RAW.iterdir()):
            iid = p.name
            if not (p / run).exists():
                continue
            if iid in holdout:
                n_held += 1
                continue
            if iid in seen_iids:
                continue  # one arm sample per instance; first run that has a solved trajectory wins
            turns = build(iid, run, obs_cap)
            if not turns:
                continue
            conv = _fit(turns, int(max_tokens * chars_per_token)) if max_tokens > 0 else \
                [{"from": w, "value": t} for w, t in turns]
            if not conv:  # agent reasoning alone overflows the window — genuinely can't fit
                n_drop_len += 1
                continue
            seen_iids.add(iid)
            n_inst += 1
            rows.append({"conversations": conv})
    (OUT / "train_sharegpt.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (OUT / "dataset_info.json").write_text(json.dumps({
        "hl_sft": {"file_name": "train_sharegpt.jsonl", "formatting": "sharegpt",
                   "columns": {"messages": "conversations"},
                   "tags": {"role_tag": "from", "content_tag": "value",
                            "user_tag": "human", "assistant_tag": "gpt"}}
    }, indent=2), encoding="utf-8")
    print(f"[prep_raw_arm] {len(rows)} raw-imitation conversations from {n_inst} SOLVED instances "
          f"(dropped over-{max_tokens}tok {n_drop_len}, held-out excluded {n_held}) -> {OUT}\n"
          f"  NOTE: raw arm uses SOLVED instances only; the structured arm additionally mines FAILED runs "
          f"via audit — report both instance counts in the paper (this gap is a structured-arm advantage).")


if __name__ == "__main__":
    app()
