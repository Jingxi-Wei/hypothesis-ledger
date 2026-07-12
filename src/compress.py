"""Layer 0 raw trajectory -> Layer 1 Hypothesis Ledger (deterministic, NO LLM / proxy).

One card per stated HYPOTHESIS (= phase). Under each card: KEY actions — source EDITS and TEST runs
(with results) — kept in detail; low-signal actions (grep / read / inspect) compressed to a count.
Each card also records its TRIGGER (explore / self_rescue after test feedback / oracle round) so the
training-view exporter can later cut the input BEFORE the test/oracle crutch.

  python src/compress.py --run-id r1                 # compress every collected trajectory
  python src/compress.py --run-id r1 -i <instance>   # one instance
"""
import json
import re
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
app = typer.Typer(add_completion=False)

HYP = re.compile(r"HYPOTHESIS:\s*(.+)")
# SWE regexes: UNCHANGED, byte-for-byte — a naive shared widening flipped 124 commands across 30 already-
# collected Pro trajectories (35 with exported ledgers), incl. a grep whose QUOTED pattern contained the word
# "make" (2026-07-07 integration review). Substrate-specific additions live in EDIT_TB / TEST_LCB below and
# apply ONLY to tb__/lcb__ trajectories, so recompressing SWE/Pro ledgers stays idempotent.
EDIT = re.compile(r"\bsed -i\b|\bgit apply\b|\bpatch -p|\bapply_patch\b|cat\s*>|tee\s+\S+|>\s*\S+\.(?:py|cfg|txt)|open\([^)]*['\"][wa]|write_text\(")
TEST = re.compile(r"\bpytest\b|python -m pytest|\bunittest\b|python\s+\S*test\S*\.py|python\s+repro|assert ")
# TB: builds/installs ARE the state changes on this substrate — but HEAD-ANCHORED (start of the command or
# after && / ; / |) so the word "make" inside a quoted grep pattern can never misfire; plus redirects into
# config/source files beyond SWE's .py/.cfg/.txt.
_HEAD = r"(?:^|&&|\|\||;)\s*"
EDIT_TB = re.compile(_HEAD + r"(?:make\b|pip3?\s+install|apt(?:-get)?\s+(?:-y\s+)?install|npm\s+(?:install|ci)\b"
                             r"|cargo\s+(?:build|install)|cmake\b|\./configure\b)"
                     r"|>\s*\S+\.(?:sh|json|ya?ml|toml|ini|conf|c|h|cpp|js|ts|go|rs|java)")
# LCB: running the candidate program against an example IS the test on the algorithmic substrate.
TEST_LCB = re.compile(r"python3?\s+\S*solution\S*\.py|python3?\s+_run\.py")
READ = re.compile(r"^\s*(?:cat|grep|rg|ls|find|head|tail|nl|wc|awk|tree|which|sed -n|git diff|git log|git status|git show)\b|print\(open")
# content-shaped reads (the observation IS file content, unlike grep/ls whose output is hits/names):
# these give the exporter real source snapshots so a correction target is derivable from code the agent saw
CONTENT_READ = re.compile(r"^\s*(?:cat(?!\s*>)|sed -n|head|tail|nl)\b")
PLACEHOLDER = "<one sentence"  # the prompt's own example HYPOTHESIS line
_OUT = re.compile(r"<output>\n?(.*?)\n?</output>", re.S)
_HIT = re.compile(r"^\S+\.[A-Za-z0-9_]+:\d+:")          # grep-style file.ext:line: hit (self-locating signal)
_FILE = re.compile(r"[\w./-]+\.(?:py|pyx|rst|txt|cfg|ini|toml|md|ya?ml|json|c|h|cpp|js|html)\b")


def _grep_hits(obs: str) -> list[str]:
    """grep/search hits (file.ext:line:text) in a read observation — the concrete 'the relevant code is HERE' signal."""
    mo = _OUT.search(obs)
    body = mo.group(1) if mo else obs
    return [ln.strip()[:160] for ln in body.splitlines() if _HIT.match(ln.strip())]


def _text(m: dict) -> str:
    c = m.get("content", "")
    if isinstance(c, list):
        return " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c or ""


def _classify(cmd: str, substrate: str = "swe") -> str:
    c = cmd.strip()
    if "COMPLETE_TASK_AND_SUBMIT" in c:
        return "submit"
    # strip env-var and `cd ... &&` prefixes so the real command is classified
    c = re.sub(r"^(?:[A-Z_]+=\S+\s+)+", "", c)
    c = re.sub(r"^cd\s+\S+\s*&&\s*", "", c)
    if EDIT.search(c) or (substrate == "tb" and EDIT_TB.search(c)):
        return "edit"
    if TEST.search(c) or (substrate == "lcb" and TEST_LCB.search(c)):
        return "test"
    return "read"  # default: low-signal exploration / inspection (the noise we compress)


def _substrate(instance_id: str) -> str:
    return "tb" if instance_id.startswith("tb__") else ("lcb" if instance_id.startswith("lcb__") else "swe")


def _trigger(obs: str) -> str | None:
    if "guidance round" in obs or "WHERE your reasoning is wrong" in obs:
        return "oracle"
    if "tests FAILED" in obs and "Reconsider your current HYPOTHESIS" in obs:
        return "self_rescue"
    return None


# Harness-injected observations that are NOT the agent's own evidence: submit feedback (carries the HIDDEN
# graded tests' stderr/names), oracle text, terminal notices. Their content must never be harvested into
# found/results — the learner judges "where the code falls short" WITHOUT the graded-test verdict.
_INJECTED = re.compile(r"All tests pass\. Task solved\.|UNCHANGED from your last refuted attempt"
                       r"|Resubmitted an unchanged|Still failing after several rounds")


def compress_one(traj: dict, substrate: str = "swe") -> tuple[list[dict], list[str], dict[str, str]]:
    cards: list[dict] = []
    cur: dict | None = None
    preamble: list[str] = []  # grep hits found BEFORE the first HYPOTHESIS — kept SEPARATE from card 1's own
    # hits (which are post-statement): the exporter needs the temporal cut "what was known when H1 was stated".
    pending = "explore"  # trigger for the next stated hypothesis
    preamble_views: dict[str, str] = {}  # content read BEFORE the first hypothesis (grounds H2+ inputs too)
    view_q: list[str | None] = []  # positional pairing: action k of an assistant msg -> k-th following tool msg
    for i, m in enumerate(traj["messages"]):
        t = _text(m)
        if m.get("role") == "assistant":
            hm = HYP.search(t)
            if hm and PLACEHOLDER not in hm.group(1):
                new_hyp = hm.group(1).strip()[:400]
                # dedup: some agents reprint the SAME hypothesis line every message — that is a restatement,
                # not a new conjecture; opening a card per line inflated one instance to 68 cards for 8 real
                # hypotheses and broke card<->audit pairing. Merge into the current card UNLESS feedback
                # intervened (pending self_rescue/oracle): restating after a refutation is real spin — keep
                # it as its own card so the audit can call it out.
                if cur is not None and pending == "explore" and new_hyp == cur["hypothesis"]:
                    pass  # same card continues; its actions keep accruing
                else:
                    cur = {"hypothesis_id": f"H{len(cards) + 1}", "trigger": pending,
                           "hypothesis": new_hyp, "raw_ref": f"msg#{i}",
                           "edits": [], "tests": [], "_readcmds": [], "_hits": [], "_views": {}}
                    cards.append(cur)
                    pending = "explore"
            view_q = []
            for a in m.get("extra", {}).get("actions", []):
                cmd = a.get("command", "") if isinstance(a, dict) else str(a)
                kind = _classify(cmd, substrate)
                slot: str | None = None
                if kind == "read":
                    c = re.sub(r"^cd\s+\S+\s*&&\s*", "", cmd.strip())
                    if cur is not None:
                        cur["_readcmds"].append(c[:80])
                    if CONTENT_READ.match(c):  # observation of this action IS a file's content
                        fs = [f for f in _FILE.findall(c) if "." in f]
                        slot = fs[0] if len(fs) == 1 else None
                elif cur is not None:
                    if kind == "edit":
                        cur["edits"].append(cmd[:800])  # carry enough of the actual change for the audit to judge the CODE
                    elif kind == "test":
                        cur["tests"].append({"cmd": cmd[:200]})
                view_q.append(slot)
        else:  # observation — mini-swe emits ONE tool message per action, in order (verified), so pop positionally
            slot = view_q.pop(0) if view_q else None
            trig = _trigger(t)
            if trig:
                pending = trig
            if trig or _INJECTED.search(t):
                continue  # harness-injected feedback is a crutch, never evidence (hidden-test lines match _HIT!)
            # capture search hits from the result (buffer pre-first-hypothesis ones into preamble); no command pairing
            (cur["_hits"] if cur is not None else preamble).extend(_grep_hits(t))
            if slot:
                mo = _OUT.search(t)
                body = (mo.group(1) if mo else t).strip()
                if len(body) > 40:  # a real content snapshot, not an error/empty read
                    (cur["_views"] if cur is not None else preamble_views)[slot] = body[:2500]  # latest read wins
            if cur and cur["tests"] and "result" not in cur["tests"][-1] and re.search(r"returncode|passed|failed|error", t, re.I):
                cur["tests"][-1]["result"] = t.strip()[:400]
    # finalize: keep the grep RESULTS (deduped file:line hits, ones in edited files first) + the distinct
    # files inspected; collapse the rest of the read steps into a count (the "distractor" middle reads).
    global_edited = " ".join(e for c in cards for e in c["edits"]).lower()
    out = []
    for c in cards:
        hits, readcmds, views = c.pop("_hits"), c.pop("_readcmds"), c.pop("_views")
        found, seen = [], set()
        for h in sorted(hits, key=lambda x: x.split(":", 1)[0].lower() not in global_edited):  # edited-file hits first
            if h not in seen:
                seen.add(h)
                found.append(h)
        inspected: list[str] = []
        for rc in readcmds:
            for f in _FILE.findall(rc):
                if "." in f and f not in inspected:
                    inspected.append(f)
        c["found"] = found[:8]                  # search results: where the relevant code is
        c["inspected"] = inspected[:8]          # files the agent read (the code content itself lives in the edits)
        c["reads_compressed"] = len(readcmds)   # how many read/grep steps this card collapsed
        # source snapshots of files that some card EDITS: the exporter shows these (temporally cut) so a
        # correction hypothesis about "where the last attempt's code falls short" is derivable from code the
        # agent actually saw — not from the graded-test verdict. Non-edited files stay compressed (noise).
        c["read_snippets"] = {f: v for f, v in views.items() if f.lower() in global_edited}
        out.append(c)
    pf, pseen = [], set()                       # preamble hits, same dedup/edited-first/cap treatment
    for h in sorted(preamble, key=lambda x: x.split(":", 1)[0].lower() not in global_edited):
        if h not in pseen:
            pseen.add(h)
            pf.append(h)
    pv = {f: v for f, v in preamble_views.items() if f.lower() in global_edited}
    return out, pf[:8], pv


@app.command()
def main(run_id: str = typer.Option("r1", "--run-id"), instance: str = typer.Option("", "-i", "--instance")) -> None:
    targets = [instance] if instance else sorted(p.name for p in RAW.iterdir() if (p / run_id / "trajectory.json").exists())
    n = 0
    for iid in targets:
        rd = RAW / iid / run_id
        traj = json.loads((rd / "trajectory.json").read_text(encoding="utf-8"))
        cards, preamble_found, preamble_views = compress_one(traj, substrate=_substrate(iid))
        (rd / "ledger.json").write_text(json.dumps({"instance_id": iid, "run_id": run_id,
                                                    "preamble_found": preamble_found,
                                                    "preamble_views": preamble_views, "cards": cards},
                                                   indent=2, ensure_ascii=False), encoding="utf-8")
        n += 1
        print(f"[compress] {iid} -> {len(cards)} cards | triggers={[c['trigger'] for c in cards]}")
    print(f"[compress done] {n} ledgers written")


if __name__ == "__main__":
    app()
