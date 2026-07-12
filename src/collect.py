"""3-stage no-early-oracle collection driver (v1) — ONE continuous agent.

The same agent runs throughout: explore -> submit. On submit we eval; if it failed,
the test failure (then, if self-rescue also fails, an oracle DIRECTION) is fed back AS
THE OBSERVATION of that submit, so the agent keeps its full history and self-corrects
in one continuous trajectory. Oracle is reached only after the agent's own self-rescue
also fails.

Honesty: the agent only ever sees problem_statement (+ its own prior work + the test
failure + an oracle DIRECTION). The gold patch and gold tests are NEVER shown to it;
gold is read only by the oracle step and to run the hidden tests for labeling.
"""
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import litellm
import typer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_wincompat"))  # resource stub for swebench on Windows
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS, TestStatus  # noqa: E402
from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: E402
from swebench.harness.test_spec.python import get_test_directives  # noqa: E402
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER  # noqa: E402

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.exceptions import FormatError, LimitsExceeded, Submitted, TimeExceeded
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import DATASET_MAPPING, get_sb_environment

sys.path.insert(0, str(Path(__file__).resolve().parent))  # src on path
import pro  # noqa: E402  — SWE-bench Pro mode (multi-language env + per-instance run_script/parser eval)
import tb   # noqa: E402  — Terminal-Bench mode (local task dirs, stateful container, solve.sh oracle)
import lcb  # noqa: E402  — LiveCodeBench mode (algorithmic sandbox, public/private tests, no gold)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = str(ROOT / "src" / "configs" / "swebench_hypo.yaml")


def _ds() -> str:  # read at RUNTIME so run_batch can set os.environ["DATASET"] after import
    return os.environ.get("DATASET", "verified")


def _is_pro() -> bool:
    return _ds() == "pro"


def _is_tb() -> bool:
    return _ds() == "tb"


def _is_lcb() -> bool:
    return _ds() == "lcb"


GPT_MODEL_PREFIX = os.environ.get("GPT_MODEL_PREFIX", "openai/gpt-5.5")
GPT_SPEED = os.environ.get("GPT_SPEED", "fast")
# Strategy: get the HARDEST problems actually SOLVED (good trajectories teach the policy), not chaotic
# medium failures. The agent runs at HIGH (run_batch escalates chaotic instances to xhigh). The agent
# NEVER sees gold. The oracle DIRECTION and the posthoc AUDIT are a SEPARATE gold-reading pass; AUDIT
# runs at xhigh (best judgment, one call/instance). Override any of these via env vars.
AGENT_EFFORT = os.environ.get("AGENT_EFFORT", "xhigh")  # teacher must EXCEED student (gpt-medium ~= qwen3.6); xhigh + room for oracle to solve
ORACLE_EFFORT = os.environ.get("ORACLE_EFFORT", "high")
TEST_FEEDBACK_EFFORT = os.environ.get("TEST_FEEDBACK_EFFORT", ORACLE_EFFORT)
AUDIT_EFFORT = os.environ.get("AUDIT_EFFORT", "xhigh")
MAX_ORACLE_ROUNDS = int(os.environ.get("MAX_ORACLE_ROUNDS", "7"))  # relaxed: longer refutation chains on hard problems (no looping — each round forces a NEW hypothesis, refuted ones are 100%-eliminated)

# A context-window overflow is a POISON PILL, not a transient error: the prompt no longer fits the model, so
# every retry re-sends the same over-long context and 502s again — burning tokens each loop round for nothing.
# We detect it by signature (not by trajectory SIZE: a 23MB oracle_redirected trajectory resolved legitimately,
# so size cannot separate poison from productive — only the model's own "won't fit" refusal can).
_OVERFLOW_MARKERS = ("context window", "exceeds the context", "maximum context", "context_length_exceeded",
                     "too many tokens", "reduce the length", "prompt is too long", "input is too long")


def is_context_overflow(err) -> bool:
    s = (err if isinstance(err, str) else repr(err)).lower()
    return any(m in s for m in _OVERFLOW_MARKERS)


class LauncherBroken(RuntimeError):
    """The LOCAL process can no longer spawn children — observed 2026-07-09 as NTSTATUS 0xC0000142
    (rc=3221225794) on every `docker exec` after ~11h of subprocess churn: two TB instances burned 300-420
    steps each of pure launch failures (each still costing an LLM call) before hitting step_limit.
    Detection key: an IN-CONTAINER command can only exit 0-255, so a streak of returncodes > 255 means the
    local launcher itself is failing, not the agent's command. Only cure = restart the worker process
    (fresh process table); run_batch exits on this so the outer loop respawns clean."""


def _guard_launcher(env, threshold: int = 8):
    """Wrap env.execute: raise LauncherBroken after `threshold` CONSECUTIVE >255 returncodes."""
    orig = env.execute
    state = {"streak": 0}

    def guarded(*a, **kw):
        r = orig(*a, **kw)
        rc = r.get("returncode") if isinstance(r, dict) else None
        if isinstance(rc, int) and rc > 255:
            state["streak"] += 1
            if state["streak"] >= threshold:
                raise LauncherBroken(f"{state['streak']} consecutive local-launcher failures (rc={rc})")
        else:
            state["streak"] = 0
        return r

    env.execute = guarded
    return env


app = typer.Typer(add_completion=False)


def _model_for_effort(effort: str) -> str:
    suffix = f"-{GPT_SPEED}" if GPT_SPEED else ""
    return f"{GPT_MODEL_PREFIX}-{effort}{suffix}"


def _kw(effort: str) -> dict:
    return dict(model=_model_for_effort(effort), api_base="http://127.0.0.1:8080/v1", api_key="pwd")


def _jl(x):
    return json.loads(x) if isinstance(x, str) else x


def _sh(env, cmd, timeout=120):
    return env.execute({"command": cmd}, timeout=timeout)


def _hidden_test_tokens(instance) -> list[str]:
    """Leaf identifiers of the hidden F2P/P2P tests. A DIRECTION shown to the agent must NEVER contain these:
    seeing a hidden-test name lets the agent reward-hack toward it instead of reasoning (the very failure this
    project's redirect design exists to avoid). Privileged reviewers may READ the tests; this scrubs their OUTPUT
    before it re-enters the agent's context — a defense the downstream scrub_f2p CANNOT give, because once the agent
    has conditioned on a leaked token the trajectory is already contaminated."""
    toks: set[str] = set()
    # BOTH casings: Verified uses FAIL_TO_PASS, SWE-bench_Pro uses fail_to_pass — reading only the uppercase
    # keys made this a silent no-op on every Pro instance (the 2026-07-07 review's top finding).
    for key in ("FAIL_TO_PASS", "fail_to_pass", "PASS_TO_PASS", "pass_to_pass"):
        val = instance.get(key)
        if val is None:
            continue
        try:
            names = _jl(val)
        except Exception:
            try:
                import ast
                names = ast.literal_eval(val)
            except Exception:
                names = []
        for t in names if isinstance(names, (list, tuple)) else []:
            t = str(t)
            # same token model as scrub_f2p.f2p_tokens: py test_*, Go Test*, JS/TS sentence titles —
            # the old lowercase 'test' prefix rule missed Go and JS names entirely.
            toks.update(re.findall(r"\btest_[A-Za-z0-9_]{3,}\b", t))
            toks.update(re.findall(r"\bTest[A-Z][A-Za-z0-9_]{2,}\b", t))
            s = t.strip()
            if " " in s and len(s) >= 15:
                toks.add(s)
            for seg in re.split(r"[^\w]+", t):
                if seg.startswith("test") and len(seg) > 5:  # test-name identifier, not the bare word "test"
                    toks.add(seg)
    return sorted(toks, key=len, reverse=True)  # longest first so subset tokens don't clip supersets


def _redact_hidden(direction: str, instance) -> str:
    """Best-effort generation-time scrub: replace any leaked hidden-test identifier with a neutral phrase.
    Boundary anchors only where the token edge is alphanumeric (JS sentence titles end in punctuation —
    a blanket \\b there would never match).

    Also consumes the adapter-supplied `_gold_tokens` (TB: distinctive solve.sh identifiers not in the
    instruction; LCB: hidden expected-output values not already public) — the mechanical backstop behind the
    reviewers' direction-only prompts, so a single prompt slip cannot condition the agent on gold content."""
    toks = list(_hidden_test_tokens(instance)) + list(instance.get("_gold_tokens") or [])
    for tk in sorted(set(toks), key=len, reverse=True):  # longest first so subset tokens don't clip supersets
        pat = (r"\b" if tk[:1].isalnum() else "") + re.escape(tk) + (r"\b" if tk[-1:].isalnum() else "")
        direction = re.sub(pat, "the affected behavior", direction)
    return direction


def eval_in_container(env, instance) -> dict:
    """Apply gold test_patch, run the repo's OWN test directives, parse per-test status with swebench's
    repo-specific parser, and compare to F2P / P2P. Robust to unrelated test noise (loader errors,
    other tests in the same module) because it checks only the named F2P/P2P tests' status."""
    if _is_tb():  # Terminal-Bench: run the task's own verifier against the LIVE container state (stateful, no diff)
        # state fingerprint BEFORE the verifier runs (the agent's submit-time state): computing it after let the
        # verifier's own droppings drift the fingerprint on no-op resubmits, defeating the anti-loop guard
        # (2026-07-07 review). stat -c works on GNU *and* busybox (GNU find's -printf does not); %Y (mtime)
        # catches same-size in-place edits that a size-only fingerprint would miss.
        state = _sh(env, r"find /app -type f -not -path '*/.git/*' -exec stat -c '%s %Y %n' {} + 2>/dev/null | sort | md5sum")["output"].strip()
        ev = tb.eval_tb(env, instance, _sh)
        ev["state"] = state
        ev["submission"] = _sh(env, "cd /app && (git diff 2>/dev/null | head -c 4000) || true")["output"]
        return ev
    if _is_lcb():  # LiveCodeBench: grade the agent's solution.py against public then private cases
        ev = lcb.eval_lcb(env, instance, "", _sh)
        ev["state"] = _sh(env, f"md5sum {lcb.SOLUTION_PATH} 2>/dev/null || echo none")["output"].strip()
        ev["submission"] = _sh(env, f"cat {lcb.SOLUTION_PATH} 2>/dev/null || echo ''")["output"]
        return ev
    if _is_pro():  # multi-language Pro eval: capture the agent's source diff, run the instance's own run_script.sh + parser.py
        diff = _sh(env, "cd /app && git add -A && git diff --cached HEAD")["output"]
        return pro.eval_pro(env, instance, diff, _sh)
    f2p, p2p = _jl(instance["FAIL_TO_PASS"]), _jl(instance["PASS_TO_PASS"])
    spec = make_test_spec(instance)
    tc = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"]
    test_cmd = tc[-1] if isinstance(tc, list) else tc
    directives = get_test_directives(instance)
    b64 = base64.b64encode(instance["test_patch"].encode()).decode()
    _sh(env, f"printf %s '{b64}' | base64 -d > /tmp/tp.patch")
    ap = _sh(env, "cd /testbed && (git apply -v /tmp/tp.patch || patch -p1 --fuzz=5 < /tmp/tp.patch)")
    applied = ap["returncode"] == 0
    out = _sh(env, f"cd /testbed && {test_cmd} {' '.join(directives)}", timeout=1800)
    _sh(env, "cd /testbed && (git apply -R /tmp/tp.patch 2>/dev/null || true); rm -f /tmp/tp.patch; true")
    try:
        status = MAP_REPO_TO_PARSER[instance["repo"]](out["output"], spec)
    except Exception:
        status = {}
    ok = TestStatus.PASSED.value
    f2p_pass = bool(f2p) and all(status.get(t) == ok for t in f2p)
    p2p_pass = all(status.get(t) == ok for t in p2p)
    failed = [t for t in f2p if status.get(t) != ok]
    header = f"FAIL_TO_PASS still failing ({len(failed)}/{len(f2p)}): {failed[:6]}\n\n" if failed else ""
    feedback = header + out["output"][-2500:]  # keep the failing-test header (privileged readers need it) + recent output tail
    return {"resolved": applied and f2p_pass and p2p_pass, "applied": applied,
            "f2p_pass": f2p_pass, "p2p_pass": p2p_pass, "feedback": feedback}


def oracle_redirect(problem: str, agent_patch: str, failure: str, gold_patch: str,
                    prior: list[str] | None = None, refuted: list[str] | None = None) -> str:
    sys_p = (
        "You are an expert reviewer. The agent's attempt failed the hidden tests. You can see a GOLD REFERENCE (the "
        "true fix, or — when there is no reference solution — the hidden test cases), but you MUST NOT reveal it. "
        "Give (1) a short NEGATION — what in the agent's CURRENT approach is wrong — and (2) the SMALLEST POSITIVE "
        "direction of what to investigate next. Your negations are AUTHORITATIVE (you verified them against the "
        "reference): steer the agent AWAY from every already-refuted hypothesis toward a NEW region of the search "
        "space — never let it circle back to a refuted one. NEVER give the fix, a file:line, any identifier/symbol "
        "name taken from the reference, any exact expected output value from a hidden test, or any code. Two or three "
        "sentences. If the direction alone is enough for a competent agent to solve it, that is ideal."
    )
    prior_txt = ""
    if prior:
        prior_txt = ("\n\n# Your EARLIER guidance (the agent tried these and still failed — give a DIFFERENT, sharper "
                     "next step; do NOT repeat them):\n" + "\n".join(f"- {p}" for p in prior))
    refuted_txt = ""
    if refuted:
        refuted_txt = ("\n\n# Hypotheses the agent has tried that are DEFINITIVELY WRONG (you verified each against the "
                       "gold — 100% eliminated). Push it toward a NEW conjecture that does not overlap any of these:\n"
                       + "\n".join(f"- {h}" for h in refuted))
    user_p = (
        f"# Issue\n{problem}\n\n# Agent's latest failed attempt\n{agent_patch}\n\n# Test failure (tail)\n{failure}"
        f"{prior_txt}{refuted_txt}\n\n# GOLD reference (FOR YOUR EYES ONLY - the true fix or the hidden test cases; "
        f"never reveal it, any name from it, or any exact expected value)\n{gold_patch}\n\n"
        "Give the negation + minimal positive direction now."
    )
    r = litellm.completion(messages=[{"role": "system", "content": sys_p},
                                     {"role": "user", "content": user_p}], **_kw(ORACLE_EFFORT))
    return r.choices[0].message.content or ""


def test_feedback_redirect(problem: str, agent_patch: str, failure: str, current_hypothesis: str) -> str:
    sys_p = (
        "You are an expert reviewer. The agent's patch failed hidden evaluation. You can see the RAW hidden "
        "test feedback, but you MUST NOT reveal it. Give (1) a short NEGATION of what the failure suggests is "
        "wrong or incomplete in the agent's current approach, and (2) the SMALLEST POSITIVE direction of what "
        "to inspect or check next. You do NOT see the gold patch, so your feedback is evidence-limited: never "
        "claim certainty about the true fix, and if the feedback only proves failure without isolating a cause, "
        "say that the current evidence is insufficient and give a probe direction. NEVER reveal hidden test "
        "names, test file paths, inputs, expected outputs, actual outputs, assertion diffs, stack-trace lines, "
        "parameter values, code, or identifiers/symbol names that appear only in the hidden tests. Use only "
        "concepts already visible in the issue, the agent's patch, or the agent's current hypothesis. Two or "
        "three sentences."
    )
    user_p = (
        f"# Issue\n{problem}\n\n# Agent's current hypothesis\n{current_hypothesis or '(not explicitly stated)'}\n\n"
        f"# Agent's latest failed patch\n{agent_patch}\n\n"
        f"# RAW hidden test feedback (FOR YOUR EYES ONLY - do not quote, name, or paraphrase hidden specifics)\n"
        f"{failure}\n\nGive the evidence-limited negation + minimal positive direction now."
    )
    r = litellm.completion(messages=[{"role": "system", "content": sys_p},
                                     {"role": "user", "content": user_p}],
                           **_kw(TEST_FEEDBACK_EFFORT))
    return r.choices[0].message.content or ""


def _transcript(messages: list[dict], max_obs: int = 800) -> str:
    """Readable transcript of the agent's full trajectory: reasoning (HYPOTHESIS lines), commands, observations."""
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(x.get("text", "") for x in content if isinstance(x, dict))
        if role == "assistant":
            out.append(f"[AGENT] {content}")
            cmds = [a.get("command", "") for a in m.get("extra", {}).get("actions", [])]
            if cmds:
                out.append("  $ " + " ; ".join(c[:200] for c in cmds))
        elif role in ("tool", "user"):
            out.append(f"[OBS] {(content or '')[:max_obs]}")
    return "\n".join(out)


def _card_digest(cards: list[dict]) -> str:
    """Numbered ledger cards + their recorded evidence — the canonical hypothesis list the audit must follow,
    and the ONLY facts why_proposed may cite (the learner sees exactly these)."""
    lines = []
    for i, c in enumerate(cards, 1):
        ev = []
        for t in c.get("tests", []):
            r = t.get("result", "")
            ev.append(f"ran `{t.get('cmd', '')[:80]}`" + (f" -> {r[:100]}" if r else ""))
        for e in c.get("edits", [])[:4]:
            ev.append(f"edit: {e[:200]}")
        for h in c.get("found", [])[:6]:
            ev.append(f"found: {h}")
        if c.get("inspected"):
            ev.append("inspected: " + ", ".join(c["inspected"][:6]))
        body = "\n".join(f"    - {x}" for x in ev) if ev else "    - (exploration/reading only)"
        lines.append(f"CARD {i} [{c.get('trigger')}]: {c['hypothesis']}\n{body}")
    return "\n".join(lines)


def audit_all(messages: list[dict], problem: str, gold_patch: str, outcome: str,
              cards: list[dict] | None = None) -> str:
    """Posthoc audit (xhigh) of EVERY hypothesis across all stages: per-stage what was good / wrong / insufficient.
    May read gold, but the audit itself stays DIRECTION-LEVEL (no gold quote / file:line / identifier / code).
    When `cards` (the compressed ledger) is given, the audit is CARD-ALIGNED: exactly one per_hypothesis entry
    per numbered card, each tagged with its card number — this is what makes the exporter's pairing exact."""
    sys_p = (
        "You are an expert reviewer producing a posthoc AUDIT of a coding agent's full trajectory (stages: explore, "
        "self-rescue after a test failure, possibly after an oracle direction). A HYPOTHESIS is the agent's conjecture "
        "about the bug's CAUSE (not a fix description or a verification step). "
        "Judge EACH hypothesis on its QUALITY AS A CONJECTURE GIVEN THE INFORMATION THE AGENT HAD AT THAT POINT — NOT "
        "merely whether it matched the final answer. A hypothesis can be GOOD (well-reasoned and justified by the "
        "evidence gathered so far, specific and falsifiable) even if later proven wrong, and BAD (a leap, ignoring "
        "evidence the agent had already seen, vague, or mis-framed) even if it happened to be right. Use the gold ONLY "
        "to understand the true cause; judge the conjecture on whether it was well-formed and justified at the time. "
        "For EACH hypothesis, ALSO reconstruct WHY the agent proposed it — the reasoning from the evidence it had at "
        "that point to this conjecture — so a learner can reproduce HOW to GENERATE such a hypothesis, not merely judge it. "
        "Write `why_proposed` as a SELF-CONTAINED evidence->conjecture path the learner can follow: ground it ONLY in the "
        "concrete evidence and in the agent's OWN earlier (now-eliminated) hypotheses. For a hypothesis stated AFTER an "
        "earlier attempt failed, why_proposed MUST start from the specific deficiency of the failed attempt — what it "
        "missed or mis-framed — and derive the new conjecture as the repair of that deficiency ('X failed because it "
        "ignored Y; Y points to Z'), not merely note that X failed. NEVER mention an oracle, hidden "
        "guidance, a correction, a redirect, or a hint — the learner sees none of those, so a why_proposed that cites "
        "them is unusable; phrase eliminations as 'the earlier hypothesis X was tried and its patch failed', not as an external instruction. "
        "THE SAME BAN APPLIES TO THE HIDDEN/GRADED TESTS: at inference the learner has NO access to the project's tests, "
        "their names, or their pass/fail results — the whole point is to learn WHERE the bug is BEFORE submitting, not to "
        "react to a test verdict. So NO prose field (why_proposed, flaw_given_info_at_the_time, should_have_turned, overall, "
        "and the restated hypothesis) may use the hidden tests as the SOURCE of a diagnosis: never write 'the hidden tests', "
        "'the project's tests', 'the failing test(s)', a specific test name/identifier, or 'the tests showed/failed X'. You MAY "
        "state at the OUTCOME level that a prior attempt was tried and did not resolve the issue, but the diagnosis of WHAT it "
        "missed and the derivation of the next hypothesis MUST be reconstructed from the issue text and the code evidence the "
        "agent inspected — written so a learner who never saw any test result could follow the same path. (The agent's own "
        "reproduction that it wrote and ran is legitimate evidence; the graded-test verdict is not.) "
        "You can see the GOLD patch but your audit MUST stay DIRECTION-LEVEL: never quote the gold, a file:line, any "
        "identifier from the gold patch, or code. "
        "CALIBRATION IS GRADED IN BOTH DIRECTIONS: if a conjecture was sound given the information available at that "
        "point, SAY SO — write exactly 'No material flaw.' as flaw_given_info_at_the_time (optionally followed by AT "
        "MOST one minor limitation). Do NOT manufacture deficiencies to appear rigorous — a fabricated flaw is itself "
        "a wrong audit, exactly like a missed one. "
        "GROUND why_proposed ONLY in facts visible at the granularity the numbered cards record (issue text, grep "
        "hits file:line, files inspected, the edits shown, test commands and their shown results). The learner sees "
        "ONLY those cards — citing file contents or observations the cards do not carry teaches the learner to "
        "fabricate evidence. If the decisive fact is not on a card, derive the conjecture from what IS on the cards. "
        "In prose fields refer to earlier attempts as 'the first attempt' / 'the previous hypothesis', NEVER as "
        "'Card N' — card numbers are your bookkeeping, the learner never sees them. "
        "Output STRICT JSON only, no prose around it."
    )
    schema = (
        '{"overall": "<2-3 sentences on how the trajectory evolved and where it turned>", '
        '"per_hypothesis": [{"card": <the CARD number this entry judges, 1-based>, '
        '"phase": "explore|self_rescue|oracle", "hypothesis": "<restate the cause conjecture>", '
        '"verdict": "good|weak|wrong", '
        '"why_proposed": "<the reasoning FROM the card-visible evidence TO this conjecture — the generative path a learner could reproduce to ARRIVE at such a hypothesis>", '
        '"flaw_given_info_at_the_time": "<given ONLY the information available at that point, what was wrong / a leap / what it ignored — or exactly \'No material flaw.\' when it was sound>", '
        '"how_to_check": "<ONE concrete, executable next step that would confirm or falsify THIS hypothesis: a command to run, a file+location to read, or an edit-and-rerun experiment. An ACTION, never an adjective like falsifiable>", '
        '"support_calibration": "support|weak_support|misleading_support|refute|inconclusive", '
        '"should_have_turned": "<direction-level: what evidence should have redirected it>"}]}'
    )
    card_block = ""
    if cards:
        card_block = (f"\n\n# Hypothesis cards (CANONICAL, numbered — your per_hypothesis array MUST contain exactly "
                      f"{len(cards)} entries, one per card IN ORDER, each with its \"card\" number. Judge the card's "
                      f"hypothesis; the trajectory is context, the cards are the unit of audit.)\n{_card_digest(cards)}")
    user_p = (
        f"# Issue\n{problem}\n\n# Final outcome\n{outcome}\n\n# Full trajectory\n{_transcript(messages)}"
        f"{card_block}\n\n"
        f"# GOLD patch (for your judgment ONLY - never reveal it or any name from it)\n{gold_patch}\n\n"
        f"Produce the audit as STRICT JSON in exactly this shape:\n{schema}"
    )
    r = litellm.completion(messages=[{"role": "system", "content": sys_p},
                                     {"role": "user", "content": user_p}], **_kw(AUDIT_EFFORT))
    return r.choices[0].message.content or ""


class CollectorAgent(DefaultAgent):
    """One continuous agent; intercepts submit to eval + feed back test failure / oracle direction."""

    def __init__(self, *a, instance: dict, **kw):
        super().__init__(*a, **kw)
        self.instance = instance
        self.feedback_given = False
        self.oracle_rounds = 0
        self.outcome: str | None = None
        self.evals: list[dict] = []
        self.oracle_directions: list[str] = []
        self.rescue_direction: str = ""  # the single sanitized self-rescue direction (stored for provenance + leak audit)
        self.refuted_hypotheses: list[str] = []  # hypotheses the oracle has 100%-refuted; never to be revisited
        self.last_patch: str = ""
        self._last_state: str | None = None  # TB/LCB container-state fingerprint from the last eval (anti-loop; no diff to compare)
        self.stuck_count = 0  # consecutive resubmits with an UNCHANGED patch (anti-loop guard)

    def _stage_label(self) -> str:
        if not self.feedback_given:
            return "self_solved"
        return "self_corrected" if self.oracle_rounds == 0 else "oracle_redirected"

    def _current_hypothesis(self) -> str:
        """The agent's latest stated HYPOTHESIS (its current cause conjecture) — what a failed patch embodied."""
        for m in reversed(self.messages):
            if m.get("role") != "assistant":
                continue
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            found = re.findall(r"HYPOTHESIS:\s*(.+)", c or "")
            if found:
                return found[-1].strip()[:300]
        return ""

    def _on_submit(self, patch: str) -> dict:
        """Eval the submission; return the observation to feed back (or set terminal outcome)."""
        ev = eval_in_container(self.env, self.instance)
        # the agent's attempt as the reviewers see it: Pro/Verified = the submitted diff; TB/LCB = the eval-captured
        # container state (solution.py / container diff), since neither of those submits a patch.
        prev = self.last_patch
        self.last_patch = ev["submission"] if ev.get("submission") is not None else (patch or self.last_patch)
        # change detection for the anti-loop: TB/LCB compare a container-state fingerprint (no diff exists to compare),
        # Pro/Verified compare the submitted diff.
        if ev.get("state") is not None:
            unchanged = ev["state"] == self._last_state
            self._last_state = ev["state"]
        else:
            unchanged = patch == prev or not (patch or "").strip()  # identical OR empty submission = no progress
        stage = "explore" if not self.feedback_given else ("self_rescue" if self.oracle_rounds == 0 else f"oracle_{self.oracle_rounds}")
        self.evals.append({"stage": stage, "resolved": ev["resolved"], "f2p_pass": ev["f2p_pass"], "p2p_pass": ev["p2p_pass"]})
        if ev["resolved"]:
            self.outcome = self._stage_label()
            return {"output": "All tests pass. Task solved.", "returncode": 0, "exception_info": ""}
        if not self.feedback_given:
            self.feedback_given = True
            direction = test_feedback_redirect(self.instance["problem_statement"], self.last_patch, ev["feedback"],
                                               self._current_hypothesis())
            direction = _redact_hidden(direction, self.instance)  # never let a hidden-test name reach the agent
            self.rescue_direction = direction
            return {"output": "Your submitted patch is applied, but the project's tests FAILED:\n"
                              "Sanitized test-feedback direction (hidden test details withheld):\n"
                              f"{direction}\n\nReconsider your current HYPOTHESIS in light of this failure and keep "
                              "fixing the code. Do not submit again until you believe it passes.",
                    "returncode": 1, "exception_info": ""}
        if unchanged:  # anti-loop: an unchanged patch is NOT progress and must not consume a guidance round
            self.stuck_count += 1
            if self.stuck_count >= 3:
                self.outcome = "chaotic_failed"
                return {"output": "Resubmitted an unchanged, already-refuted patch repeatedly. Stopping.",
                        "returncode": 1, "exception_info": ""}
            return {"output": "Your patch is UNCHANGED from your last refuted attempt — that approach is already proven "
                              "wrong and cannot be retried. Change to a genuinely NEW HYPOTHESIS before submitting again.",
                    "returncode": 1, "exception_info": ""}
        self.stuck_count = 0
        if self.oracle_rounds < MAX_ORACLE_ROUNDS:
            self.oracle_rounds += 1
            hyp = self._current_hypothesis()  # the conjecture this failed patch embodied — now authoritatively refuted
            if hyp and hyp not in self.refuted_hypotheses:
                self.refuted_hypotheses.append(hyp)
            direction = oracle_redirect(self.instance["problem_statement"], self.last_patch, ev["feedback"],
                                        ev.get("oracle_gold") or (self.instance.get("patch") or ""),
                                        prior=self.oracle_directions, refuted=self.refuted_hypotheses)
            direction = _redact_hidden(direction, self.instance)  # scrub any leaked hidden-test name before feed-back
            self.oracle_directions.append(direction)
            refuted_txt = ("\n\nDEFINITIVELY REFUTED (verified WRONG against the true solution — 100% eliminated, never "
                           "return to any):\n" + "\n".join(f"- {h}" for h in self.refuted_hypotheses)) if self.refuted_hypotheses else ""
            return {"output": f"Your patch still fails (guidance round {self.oracle_rounds}).{refuted_txt}\n\n"
                              "Negation + the direction to investigate next (direction only — work out the fix "
                              f"yourself):\n{direction}\n\nState a genuinely NEW HYPOTHESIS — distinct from every refuted "
                              "one above, do not loop back — and fix accordingly.",
                    "returncode": 1, "exception_info": ""}
        self.outcome = "chaotic_failed"
        return {"output": "Still failing after several rounds of guidance. Stopping.", "returncode": 1, "exception_info": ""}

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            try:
                outputs.append(self.env.execute(action))
            except Submitted as e:
                outputs.append(self._on_submit(e.messages[-1].get("extra", {}).get("submission", "")))
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def collect(self, task: str) -> str:
        self.extra_template_vars |= {"task": task}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while self.outcome is None:
            try:
                self.step()
                self.n_consecutive_format_errors = 0
            except (LimitsExceeded, TimeExceeded) as e:
                self.add_messages(*e.messages)
                final = eval_in_container(self.env, self.instance)  # maybe fixed without submitting
                self.outcome = self._stage_label() if final["resolved"] else "chaotic_failed"
            except FormatError as e:
                self.n_consecutive_format_errors += 1
                self.add_messages(*e.messages)
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.outcome = "chaotic_failed"
            except Exception as e:
                if is_context_overflow(e):
                    # terminal poison, NOT a transient failure: the prompt overflowed the model's window.
                    # Mark a distinct outcome (don't re-raise -> no orphan) so run_batch skip-lists this instance
                    # and never re-attempts it (each retry would re-overflow and re-burn tokens). 2026-07-08.
                    self.outcome = "context_overflow"
                else:
                    self.handle_uncaught_exception(e)
                    raise
            finally:
                self.save(self.config.output_path)
        return self.outcome


def collect_one(instance: dict, run_id: str = "r1", agent_effort: str | None = None) -> dict:
    config = get_config_from_spec(CONFIG)
    ec = config.setdefault("environment", {})
    ag = config.setdefault("agent", {})
    if _is_pro():  # Pro: jefzda image (ENTRYPOINT=/bin/bash must be reset), /app cwd, longer timeout, /app in prompt
        instance = pro.normalize_pro(instance)
        ec["cwd"], ec["timeout"], ec["run_args"], ec["pull_timeout"] = "/app", 1800, ["--rm", "--entrypoint", ""], 7200
        for k in ("system_template", "instance_template"):
            if isinstance(ag.get(k), str):
                ag[k] = ag[k].replace("/testbed", "/app")
    elif _is_tb():  # Terminal-Bench: the task's own dockerhub image, WORKDIR /app (like Pro), TB task framing
        instance = tb.normalize_tb(instance)
        ec["cwd"], ec["timeout"], ec["run_args"], ec["pull_timeout"] = "/app", 1800, ["--rm", "--entrypoint", ""], 7200
        ag["instance_template"] = tb.INSTANCE_TEMPLATE
    elif _is_lcb():  # LiveCodeBench: generic python sandbox at /workspace, algorithmic framing, short per-cmd timeout
        instance = lcb.normalize_lcb(instance)
        ec["cwd"], ec["timeout"], ec["run_args"], ec["pull_timeout"] = "/workspace", 120, ["--rm"], 3600
        ag["instance_template"] = lcb.INSTANCE_TEMPLATE
        config.setdefault("run", {})["env_startup_command"] = "mkdir -p /workspace"  # cwd must exist before commands
    out_dir = ROOT / "dataset" / "raw" / instance["instance_id"] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_cfg = {k: v for k, v in config.get("agent", {}).items() if k != "agent_class"}
    agent_cfg["output_path"] = str(out_dir / "trajectory.json")
    mc = config.get("model", {})
    mc["model_name"] = _model_for_effort(agent_effort or AGENT_EFFORT)
    extra_body = mc.setdefault("model_kwargs", {}).setdefault("extra_body", {})
    extra_body.pop("reasoning_effort", None)
    extra_body.pop("speed", None)
    model = get_model(config=mc)
    env = _guard_launcher(get_sb_environment(config, instance))  # fail fast if the local process can't spawn
    try:
        agent = CollectorAgent(model, env, instance=instance, **agent_cfg)
        outcome = agent.collect(instance["problem_statement"])
        # Posthoc Layer-2 audit. DEFAULT OFF since the card-aligned audit contract (2026-07-03): the audit
        # judges the NUMBERED ledger cards, which exist only after compress — so the standard flow is
        # collect (trajectories only) -> compress -> reaudit/audit_run. Set AUDIT_INLINE=1 to audit inline
        # anyway (cards are computed on the fly from the fresh trajectory).
        if os.environ.get("AUDIT_INLINE", "0") == "1":
            from compress import compress_one  # local import: only needed on this opt-in path
            from compress import _substrate
            inline_cards, _pf, _pv = compress_one({"messages": agent.messages},
                                                  substrate=_substrate(instance["instance_id"]))
            audit_raw = audit_all(agent.messages, instance["problem_statement"], instance.get("patch") or "", outcome,
                                  cards=inline_cards)
            (out_dir / "audit.json").write_text(audit_raw, encoding="utf-8")
        summary = {"instance_id": instance["instance_id"], "outcome": outcome, "evals": agent.evals,
                   "rescue_direction": agent.rescue_direction, "oracle_directions": agent.oracle_directions,
                   "refuted_hypotheses": agent.refuted_hypotheses, "n_messages": len(agent.messages)}
        (out_dir / "outcome.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[collect] {instance['instance_id']} -> {outcome} | evals={[e['stage']+':'+('ok' if e['resolved'] else 'fail') for e in agent.evals]}")
        return summary
    finally:
        # Remove THIS instance's container directly by id: concurrency-safe (targets one container, not a
        # global sweep) AND Windows-safe (mini-swe's env.cleanup() uses Unix shell syntax that no-ops on cmd.exe,
        # which leaks the container + its image layers until `--rm`/`sleep` expires).
        try:
            cid = getattr(env, "container_id", None)
            if cid:
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)
            else:
                env.cleanup()
        except Exception:
            pass


def load_instances(dataset: str, split: str = "test", lcb_filtered: bool = True) -> dict:
    """{instance_id: instance} for any benchmark — HF for verified/full/pro, local adapters for tb/lcb.
    Shared by collect.main, run_batch and audit_run so entry points can never disagree on how a source loads.
    lcb_filtered=False ignores LCB_AFTER_DATE — posthoc consumers (audit) must see EVERY collected instance,
    even ones a narrower env var at audit time would filter out (silent-skip footgun, 2026-07-07 review)."""
    if dataset == "tb":
        return {i["instance_id"]: i for i in tb.load_instances()}
    if dataset == "lcb":
        after = os.environ.get("LCB_AFTER_DATE") if lcb_filtered else None
        return {i["instance_id"]: i for i in lcb.load_instances(after_date=after)}
    ds_path = {"verified": "princeton-nlp/SWE-bench_Verified", "full": "princeton-nlp/SWE-bench",
               "pro": "ScaleAI/SWE-bench_Pro"}.get(dataset, dataset)
    return {i["instance_id"]: i for i in load_dataset(ds_path, split=split)}


@app.command()
def main(
    instance_spec: str = typer.Option(..., "-i", "--instance"),
    dataset: str = typer.Option("verified", "--dataset", help="verified | full | pro | tb | lcb (or an HF path)"),
    split: str = typer.Option("test", "--split"),
    run_id: str = typer.Option("r1", "--run-id"),
) -> None:
    if dataset in ("tb", "lcb", "pro", "verified", "full"):
        os.environ["DATASET"] = dataset  # so _is_tb/_is_lcb/_is_pro dispatch in collect_one + eval_in_container
    instances = load_instances(dataset, split)
    if instance_spec.isnumeric():
        instance_spec = sorted(instances)[int(instance_spec)]
    collect_one(instances[instance_spec], run_id=run_id)


if __name__ == "__main__":
    app()
