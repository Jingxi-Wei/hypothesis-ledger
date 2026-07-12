"""LiveCodeBench adaptation. Algorithmic/competitive-programming problems — a DIFFERENT substrate from
repo bug-fixing (whole-program synthesis, not localization). Loaded from a version JSONL
(``dataset/lcb/test6.jsonl`` = the 2025-01..04 window; filter by ``contest_date`` > the student's training
cutoff for contamination control). No repo and no per-problem image: the agent writes ``/workspace/solution.py``
in a generic python sandbox and runs it.

The public/private split maps directly onto our student/oracle privilege boundary:
  public_test_cases  (plain json)                     -> the agent sees these (its own self-check)
  private_test_cases (b64 -> zlib -> pickle -> json)  -> HIDDEN. On submit we grade against BOTH.

No gold solution exists in this dataset, so the privileged reference is the tests themselves:
  * eval returns ``oracle_gold`` = a rendering of the first FAILING PRIVATE case (input -> expected). collect.py
    feeds that to oracle_redirect (via ev['oracle_gold']) — a strong hint (the target output) but NOT an
    algorithm, so the oracle still gives DIRECTION only.
  * self_rescue feedback names only PUBLIC failures + a hidden-fail COUNT (never private values) — the
    sanitizing reviewer then strips even that.
  * fix samples are skipped (no gold patch); LCB contributes the reasoning chain (audit / propose / probe).

Two test formats: 'stdin' (pipe stdin, compare stdout) and 'functional' (starter_code defines a function,
metadata.func_name; a wrapper imports the solution and calls it, printing the return).
"""
import base64
import json
import math
import os
import pickle
import subprocess
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LCB_FILE = Path(os.environ.get("LCB_FILE", ROOT / "dataset" / "lcb" / "test6.jsonl"))
LCB_IMAGE = os.environ.get("LCB_IMAGE", "python:3.11-slim")
SOLUTION_PATH = "/workspace/solution.py"
_LCB_PREFIX = "lcb__"
_MAX_CASES = int(os.environ.get("LCB_MAX_PRIVATE", "40"))  # cap graded private cases (some problems ship 100+)


def _decode_tests(raw) -> list[dict]:
    """public = plain json; private = base64 -> zlib decompress -> pickle -> json. Try plain first."""
    if not isinstance(raw, str):
        return raw or []
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def _prob_id(r: dict) -> str:
    return _LCB_PREFIX + str(r.get("question_id") or r.get("question_title", "")).replace("/", "_").replace(" ", "_")


def load_instances(after_date: str | None = None, difficulties: set[str] | None = None,
                   path: Path | None = None) -> list[dict]:
    """JSONL records -> instance dicts collect.py consumes. `after_date` (YYYY-MM-DD) keeps only problems
    released strictly after it (contamination window); `difficulties` filters easy/medium/hard."""
    out = []
    for line in (path or LCB_FILE).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if difficulties and r.get("difficulty") not in difficulties:
            continue
        if after_date and (r.get("contest_date", "")[:10] <= after_date):
            continue
        meta = r.get("metadata")
        meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
        starter = r.get("starter_code") or ""
        stmt = r["question_content"] + (f"\n\n# Starter code (use this signature):\n{starter}" if starter.strip() else "")
        public = _decode_tests(r["public_test_cases"])
        private = _decode_tests(r["private_test_cases"])
        if len(private) > _MAX_CASES:  # LOUD + head/tail sample: a blind head-slice preferentially drops the
            # stress/edge cases judges put LAST — silently passing a subtly-wrong solution (2026-07-07 review)
            print(f"[lcb] {_prob_id(r)}: {len(private)} private cases > cap {_MAX_CASES} — keeping head+tail sample")
            private = private[:_MAX_CASES // 2] + private[-(_MAX_CASES - _MAX_CASES // 2):]
        # hidden expected values that are NOT already public (not in the problem text / public examples):
        # a mechanical redaction backstop behind the oracle's "never reveal an exact expected value" prompt
        # (collect._redact_hidden consumes these). len>=3 keeps single digits out of the redactor.
        pub_txt = stmt + json.dumps(public)
        gold_tokens = sorted({v for c in private for v in [(c.get("output") or "").strip()]
                              if len(v) >= 3 and v not in pub_txt}, key=len, reverse=True)
        out.append({"instance_id": _prob_id(r),
                    "problem_statement": stmt,
                    "patch": None,                       # no gold solution in LCB
                    "image_name": LCB_IMAGE,
                    "difficulty": r.get("difficulty", "?"),
                    "repo_language": "python",
                    "_public": public,
                    "_private": private,
                    "_func_name": meta.get("func_name"),
                    "_starter": starter,
                    "_gold_tokens": gold_tokens})
    return out


def lcb_image(instance: dict) -> str:
    return instance.get("image_name", LCB_IMAGE)


def normalize_lcb(inst: dict) -> dict:
    inst = dict(inst)
    inst.setdefault("image_name", LCB_IMAGE)
    return inst


_FUNC_WRAPPER = """
import json, sys
sys.setrecursionlimit(100000)
{solution}

_args = [json.loads(_l) for _l in sys.stdin.read().splitlines() if _l.strip()]
_r = Solution().{func}(*_args) if {has_class} else {func}(*_args)
print(json.dumps(_r) if not isinstance(_r, str) else _r)
"""


def _norm_out(s: str) -> str:
    """Judge-style whitespace tolerance: strip the ends, rstrip每line (real expected outputs in test6 carry
    incidental per-line trailing spaces — byte-exact comparison would false-fail a correct solution)."""
    return "\n".join(l.rstrip() for l in (s or "").strip().splitlines())


# In-container batch runner (official-harness shape): reads cases.json, runs every case as a SUBPROCESS of
# the candidate program with the full input piped through a file — the input NEVER rides a command line.
_BATCH_RUNNER = r"""
import base64, json, subprocess, sys
spec = json.load(open("/tmp/hl_eval/cases.json"))
res = []
for prog, inp in zip(spec["progs"], spec["inputs"]):
    open("/tmp/hl_eval/in.txt", "w").write(inp)
    try:
        with open("/tmp/hl_eval/in.txt") as fin:
            r = subprocess.run([sys.executable, "/tmp/hl_eval/%s.py" % prog], stdin=fin,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=spec["timeout"])
        out = r.stdout[:10_000_000]
    except subprocess.TimeoutExpired:
        out = b"__HL_TIMEOUT__"
    res.append(base64.b64encode(out).decode())
print("__HL_RESULTS__" + json.dumps(res))
"""


def _push_eval_dir(env, sh, files: dict[str, str]) -> bool:
    """Ship the eval payload (runner + candidate program(s) + case INPUTS) into the container via `docker cp`.
    This is the whole point of the batch design: the previous per-case `printf '<base64>'` put the payload ON
    the docker-exec command line, which the Windows host caps at 32,767 chars — a >24k-char case silently
    failed to write, /workspace/_in.txt kept the PREVIOUS case's stdin, and the program 'answered the wrong
    question': even a perfect solution could never pass, and the oracle saw phantom tiny outputs (lcb2 smoke,
    2026-07-11). Expected outputs NEVER enter the container — only inputs and programs do; the agent's shell
    could read anything we place here."""
    cid = getattr(env, "container_id", None)
    if not cid:
        return False
    sh(env, "rm -rf /tmp/hl_eval")
    with tempfile.TemporaryDirectory() as td:
        for name, text in files.items():
            Path(td, name).write_text(text, encoding="utf-8", newline="")
        r = subprocess.run([os.environ.get("MSWEA_DOCKER_EXECUTABLE", "docker"), "cp", td, f"{cid}:/tmp/hl_eval"],
                           capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"[lcb] eval payload docker cp FAILED: {r.stderr[:300]}", flush=True)
    return r.returncode == 0


def _json_close(a, b, tol: float = 1e-6) -> bool:
    """Official-judge-style value compare for functional returns: floats by closeness, containers recursively,
    everything else exact. bool checked before number (bool is an int subclass — True would isclose(1))."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_close(x, y, tol) for x, y in zip(a, b))
    return a == b


def _case_ok(actual: str, expected: str) -> bool:
    if _norm_out(actual) == _norm_out(expected):
        return True
    try:  # float-format mismatches ("2.0" vs "2.00000"): compare as JSON values with tolerance
        return _json_close(json.loads(actual), json.loads(expected))
    except (ValueError, TypeError):
        return False


def _run_cases(env, instance: dict, sol_code: str, cases: list[dict], sh, timeout: int = 15) -> list[tuple[bool, str]]:
    """Run the candidate on ALL cases in one shot: one `docker cp` in, one exec of the in-container batch
    runner, all outputs back on stdout (b64 per case). Returns [(passed, actual), ...] aligned with `cases`.
    The class-vs-function choice is COMPUTED FROM THE CANDIDATE'S OWN SOURCE and baked in as a boolean —
    checking the assembled wrapper file always found the literal in its own detection line (self-referential
    always-true; a bare-function candidate then crashed with NameError — 2026-07-07 review, reproduced).
    The runner is invoked WITHOUT cd: sh() already runs in /workspace, so candidate subprocesses keep the
    same cwd the old per-case design gave them."""
    files = {"runner.py": _BATCH_RUNNER, "stdin.py": sol_code,
             "cases.json": json.dumps({
                 "timeout": timeout,
                 "progs": ["func" if (c.get("testtype") == "functional" and instance.get("_func_name")) else "stdin"
                           for c in cases],
                 "inputs": [c.get("input") or "" for c in cases]})}
    if instance.get("_func_name"):
        files["func.py"] = _FUNC_WRAPPER.format(solution=sol_code, func=instance["_func_name"],
                                                has_class=("class Solution" in sol_code))
    if not _push_eval_dir(env, sh, files):
        return [(False, "__HL_TRANSFER_FAILED__ (harness fault, not the solution)")] * len(cases)
    r = sh(env, "python /tmp/hl_eval/runner.py; rc=$?; rm -rf /tmp/hl_eval; exit $rc",
           timeout=timeout * max(1, len(cases)) + 60)
    out = r.get("output") or ""
    line = next((l for l in out.splitlines() if l.startswith("__HL_RESULTS__")), None)
    if line is None:
        return [(False, f"__HL_RUNNER_CRASHED__ (harness fault, not the solution) {out[-300:]!r}")] * len(cases)
    outs = [base64.b64decode(x).decode("utf-8", errors="replace") for x in json.loads(line[len("__HL_RESULTS__"):])]
    return [(_case_ok(a, c.get("output") or ""), a) for c, a in zip(cases, outs)]


def _elide(s: str, keep: int = 3000, total: int = 20000) -> str:
    """Structure-preserving elision for the oracle_gold rendering: NEVER drop a line (one line = one JSON arg
    for functional problems — every argument must stay visible), only middle-elide lines longer than `keep`
    with an explicit marker. A blind 800-char head-cut here once amputated the second argument (k) of a
    long-array case; the oracle then chased phantom argument-binding bugs for ~5 redirect rounds (lcb2 smoke,
    2026-07-11). `total` bounds pathological many-line output the same marked way."""
    out = []
    for l in (s or "").splitlines():
        if len(l) > keep:
            h = keep // 2
            l = f"{l[:h]} ...[{len(l) - 2 * h:,} chars elided]... {l[-h:]}"
        out.append(l)
    j = "\n".join(out)
    if len(j) > total:
        j = f"{j[:total // 2]}\n...[{len(j) - total // 2 - total // 4:,} chars elided]...\n{j[-total // 4:]}"
    return j


def eval_lcb(env, instance: dict, submission: str, sh) -> dict:
    """Grade the agent's /workspace/solution.py against public then private cases. Returns the collect.py
    eval shape PLUS `oracle_gold` (the SMALLEST failing private case rendered for the gold-reading oracle —
    smallest, not first: median inputs are ~24 chars, so the oracle usually sees a complete case with zero
    elision; when only the huge stress case fails, its size IS the diagnostic signal)."""
    sol = sh(env, f"cat {SOLUTION_PATH} 2>/dev/null || echo ''")["output"]
    if not sol.strip():
        return {"resolved": False, "applied": False, "f2p_pass": False, "p2p_pass": False,
                "feedback": f"No solution found. Write your program to {SOLUTION_PATH}, then submit."}
    pub, priv = instance.get("_public", []), instance.get("_private", [])
    results = _run_cases(env, instance, sol, pub + priv, sh)
    pub_fail = [(i, c, actual) for i, (c, (ok, actual)) in enumerate(zip(pub, results[:len(pub)])) if not ok]
    priv_fail, min_priv_fail = 0, None
    for c, (ok, actual) in zip(priv, results[len(pub):]):
        if not ok:
            priv_fail += 1
            if min_priv_fail is None or len(c.get("input") or "") < len(min_priv_fail[0].get("input") or ""):
                min_priv_fail = (c, actual)
    resolved = not pub_fail and priv_fail == 0
    # sanitized feedback for the self_rescue reviewer: PUBLIC failures may show values (the agent can see them
    # already); PRIVATE failures are a COUNT only.
    fb = []
    for i, c, actual in pub_fail[:3]:
        fb.append(f"Public example {i} failed: input {c['input'][:120]!r} -> expected {c['output'][:120]!r}, got {actual[:120]!r}")
    if priv_fail:
        fb.append(f"{priv_fail} hidden test case(s) failed.")
    feedback = ("\n".join(fb) or "All tests pass.")[-2500:]
    # oracle_gold = the privileged view (a hidden case's input->expected), rendered for oracle_redirect
    oracle_gold = None
    if min_priv_fail is not None:
        c, actual = min_priv_fail
        inp, exp = c.get("input") or "", c.get("output") or ""
        oracle_gold = (f"The SMALLEST failing HIDDEN test case (of {priv_fail} failing), which the agent cannot see. "
                       f"For functional problems each line below is one argument; oversized values are middle-elided "
                       f"with an explicit [... chars elided] marker — every line/argument is present:\n"
                       f"  input ({len(inp):,} chars):\n{_elide(inp)}\n"
                       f"  expected output ({len(exp):,} chars):\n{_elide(exp)}\n"
                       f"  the agent's program produced ({len(actual):,} chars):\n{_elide(actual)}")
    return {"resolved": resolved, "applied": True, "f2p_pass": resolved, "p2p_pass": True,
            "feedback": feedback, "oracle_gold": oracle_gold}


# NOTE: no fix samples for LCB (export.py gates them off): there is no gold patch, and the "produce the fix"
# framing doesn't fit whole-program synthesis. LCB contributes the reasoning chain (audit / propose / probe).


# instance_template for LCB. Same reasoning protocol (THOUGHT + one HYPOTHESIS per phase + submit marker);
# reframed for whole-program synthesis on an algorithmic substrate — the hypothesis is about the APPROACH,
# and a correction round is triggered when a hidden case refutes the current algorithm.
INSTANCE_TEMPLATE = """<problem>
{{task}}
</problem>

<instructions>
# Task Instructions

Solve this competitive-programming problem by writing a complete Python program to /workspace/solution.py.
Unless a function signature is given, the program reads its input from STDIN and writes the answer to STDOUT.
A few example cases are shown in the problem; your solution is also graded on HIDDEN cases you cannot see, so
reason about edge cases (large inputs, boundaries, empty/degenerate cases), not just the examples.

For each response: (1) a THOUGHT section with your reasoning, (2) at least one bash tool call (write the file
with a heredoc/base64, run it against an example, etc.).

## Hypothesis discipline (one per phase)

Before each phase, state your current hypothesis on its own line, exactly:

HYPOTHESIS: <one sentence — your current conjecture about the correct APPROACH, or about why your current solution is wrong>

A HYPOTHESIS is a checkable conjecture about the algorithm (the method that solves it, or the specific reason
the current attempt fails a case) — NOT a description of the code you will type. State it concrete enough to be
wrong. State a NEW `HYPOTHESIS:` line only when your conjecture CHANGES; keep working under the current one
(implement / run on examples) without restating it.

## Command execution
- You issue >=1 bash command; it runs in a fresh non-persistent subshell; you see the result.
- Directory/env changes do NOT persist between commands; write to files and re-read them.
- Write your program to /workspace/solution.py. Run it against the examples to check yourself before submitting.

## Submission
When you believe /workspace/solution.py is correct (it must exist), submit with this EXACT command:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

You CANNOT continue after submitting. If hidden tests fail, you will be told and may keep fixing solution.py.
</instructions>
"""
