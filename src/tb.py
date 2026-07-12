"""Terminal-Bench 2.0 adaptation. Tasks are LOCAL dirs (the terminal-bench-2 repo, vendored at
``terminal-bench-2/`` or ``$TB_TASKS_DIR``). Each task dir holds:

  task.toml              metadata incl. [environment].docker_image  (a pre-built dockerhub image)
  instruction.md         the natural-language task  -> problem_statement
  solution/solve.sh      a runnable reference solution  -> the ORACLE gold (+ the fix-sample target)
  tests/test.sh          the verifier: runs pytest, writes /logs/verifier/reward.txt (1 pass / 0 fail)
  tests/test_outputs.py  the graded tests (+ occasional helper files)

Substrate contrast with SWE/Pro: the reward is a BINARY end-state check (all-or-nothing), and the task is
STATEFUL — the agent mutates the container in place (build/configure/fix an environment), there is no git
patch. So eval runs the verifier against the LIVE container state, not against an applied diff. WORKDIR is
``/app`` (same as Pro), so collect.py's Pro env setup is reused unchanged.

Reuses collect.py's agent / oracle_redirect / audit. The oracle privileged info is ``instance['patch']`` =
solve.sh (a reference *procedure*, not a diff — the oracle still gives direction only, never the script).
"""
import base64
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = Path(os.environ.get("TB_TASKS_DIR", ROOT / "terminal-bench-2"))
_TB_PREFIX = "tb__"  # instance_id namespace so TB ids never collide with Pro/Verified ids in dataset/raw


def _b64_write(sh, env, path: str, content) -> None:
    """Write `content` (str or bytes) to `path` inside the container via chunked base64 — one big command blows
    past the Windows docker-exec command-line limit and silently truncates (same failure pro._b64_write guards).
    Bytes are written verbatim so binary test fixtures survive (str would corrupt them)."""
    b64 = base64.b64encode(content if isinstance(content, bytes) else content.encode()).decode()
    sh(env, f": > {path}.b64")
    for i in range(0, len(b64), 4000):
        sh(env, f"printf %s '{b64[i:i + 4000]}' >> {path}.b64")
    sh(env, f"base64 -d {path}.b64 > {path} && rm -f {path}.b64")


def _task_dir(instance: dict) -> Path:
    return TASKS_DIR / instance["instance_id"][len(_TB_PREFIX):]


def tb_image(instance: dict) -> str:
    """The task's pre-built dockerhub image (task.toml [environment].docker_image)."""
    return instance["docker_image"]


_SH_COMMON = {"export", "install", "python", "python3", "source", "chmod", "mkdir", "target", "return",
              "import", "update", "upgrade", "config", "configure", "system", "string", "version", "package"}


def _gold_tokens(solve: str, instruction: str) -> list[str]:
    """Distinctive identifiers/paths that appear in solve.sh but NOT in the task instruction = information the
    agent does not have. A mechanical backstop behind the oracle's direction-only prompt (collect._redact_hidden
    consumes these): if the reviewer LLM ever echoes one, it is scrubbed before reaching the agent. Conservative
    over-redaction (a name the agent already dug up itself gets masked too) is the accepted cost."""
    toks = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_.-]{5,}\b", solve)) | set(re.findall(r"/[\w./-]{6,}", solve))
    return sorted((t for t in toks if t.lower() not in _SH_COMMON and t not in instruction),
                  key=len, reverse=True)


def load_task(task_dir: Path) -> dict | None:
    """One TB task dir -> an instance dict shaped like the SWE instances collect.py consumes."""
    toml_p, instr_p, solve_p = task_dir / "task.toml", task_dir / "instruction.md", task_dir / "solution" / "solve.sh"
    tests_p = task_dir / "tests" / "test.sh"
    if not (toml_p.exists() and instr_p.exists() and solve_p.exists() and tests_p.exists()):
        return None
    meta = tomllib.loads(toml_p.read_text(encoding="utf-8"))
    image = (meta.get("environment", {}) or {}).get("docker_image")
    if not image:
        return None
    instruction = instr_p.read_text(encoding="utf-8")
    solve = solve_p.read_text(encoding="utf-8")
    return {"instance_id": _TB_PREFIX + task_dir.name,
            "problem_statement": instruction,
            "patch": solve,                                  # gold = the reference solve.sh (oracle + fix target)
            "docker_image": image,
            "difficulty": (meta.get("metadata", {}) or {}).get("difficulty", "?"),
            "repo_language": "shell",
            "_gold_tokens": _gold_tokens(solve, instruction)}


def load_instances(tasks_dir: Path | None = None) -> list[dict]:
    d = Path(tasks_dir) if tasks_dir else TASKS_DIR
    out = []
    for td in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")):
        inst = load_task(td)
        if inst:
            out.append(inst)
    return out


def normalize_tb(inst: dict) -> dict:
    """Inject the docker image so get_swebench_docker_image_name picks it (it reads 'docker_image')."""
    inst = dict(inst)
    inst.setdefault("docker_image", tb_image(inst))
    return inst


def _copy_tests(env, task_dir: Path, sh) -> None:
    """Copy the task's whole tests/ tree into the container at /tests (test.sh hardcodes /tests paths).
    Files are read as BYTES: 6 vendored tasks ship binary fixtures (reference .png/.jpg/.pt/.mp4) that
    read_text(errors='ignore') would silently mangle into false 'unresolved' labels (2026-07-07 review)."""
    sh(env, "rm -rf /tests && mkdir -p /tests /logs/verifier")
    tdir = task_dir / "tests"
    for f in sorted(tdir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(tdir).as_posix()
            if "/" in rel:
                sh(env, f"mkdir -p /tests/{rel.rsplit('/', 1)[0]}")
            data = f.read_bytes()
            if b"\0" not in data[:8192]:
                # TEXT file from a Windows git checkout carries CRLF; `bash test.sh` chokes on \r (the classic
                # "CRLF => reward 0" gotcha — read_text used to normalize this silently, read_bytes must do it
                # explicitly). Binary files (null byte in the head) are written verbatim.
                data = data.replace(b"\r\n", b"\n")
            _b64_write(sh, env, f"/tests/{rel}", data)


def eval_tb(env, instance: dict, sh) -> dict:
    """Run the task's own verifier against the LIVE container state (no patch — TB is stateful).
    Returns the {resolved, applied, f2p_pass, p2p_pass, feedback} shape collect.py expects; the binary
    reward maps to f2p_pass (p2p_pass is vacuously True — there is no separate regression set).

    Restore step (pro.restore_agent_state's analog, 2026-07-07 review): several tasks' test.sh drop files
    INTO /app (`cp /tests/test.py /app/`, `uv venv .tb` from $PWD) — without cleanup the agent's next turn
    sees verifier artifacts it never created and the anti-loop state fingerprint drifts on no-op resubmits.
    TB has no `git reset` analog, so we diff-restore: snapshot the /app path list before test.sh, delete
    anything NEW under /app after. (Files test.sh MODIFIES in place, or global pip installs outside /app,
    are not undone — accepted residual, recorded in the adaptation notes.)"""
    task_dir = _task_dir(instance)
    if not (task_dir / "tests" / "test.sh").exists():
        return {"resolved": False, "applied": False, "f2p_pass": False, "p2p_pass": True,
                "feedback": f"no tests for {instance['instance_id']}"}
    _copy_tests(env, task_dir, sh)
    sh(env, "find /app -not -path '*/.git/*' 2>/dev/null | sort > /tmp/tb_pre.txt")
    sh(env, "bash /tests/test.sh > /tmp/tb_verify.log 2>&1 || true", timeout=1800)
    reward = sh(env, "cat /logs/verifier/reward.txt 2>/dev/null || echo 0")["output"].strip()
    resolved = reward == "1"  # every vendored test.sh writes exactly 1 or 0; exact match beats startswith('1')
    tail = sh(env, "tail -c 2000 /tmp/tb_verify.log 2>/dev/null || true")["output"]
    # delete verifier droppings under /app (deepest paths first so files vanish before their dirs)
    sh(env, "find /app -not -path '*/.git/*' 2>/dev/null | sort > /tmp/tb_post.txt && "
            "comm -13 /tmp/tb_pre.txt /tmp/tb_post.txt | awk '{ print length, $0 }' | sort -rn | cut -d' ' -f2- "
            "| while IFS= read -r p; do rm -rf \"$p\"; done")
    sh(env, "rm -rf /tests /logs/verifier /tmp/tb_verify.log /tmp/tb_pre.txt /tmp/tb_post.txt")
    feedback = (("" if resolved else "The task's verification tests FAILED.\n") + tail)[-2500:]
    return {"resolved": resolved, "applied": True, "f2p_pass": resolved, "p2p_pass": True, "feedback": feedback}


def run_oracle_solution(env, instance: dict, sh) -> dict:
    """gold-sanity helper (tests only): execute solve.sh in the container, then eval — the reference
    procedure must make the verifier pass. Never used in collection (the agent never sees solve.sh)."""
    _b64_write(sh, env, "/tmp/solve.sh", instance["patch"])
    sh(env, "cd /app && bash /tmp/solve.sh > /tmp/tb_solve.log 2>&1 || true", timeout=1800)
    return eval_tb(env, instance, sh)


# instance_template for TB (overrides the SWE one in collect_one). Keeps the shared reasoning protocol
# (THOUGHT + one HYPOTHESIS per phase + the exact submit marker) but reframes: the task is to reach an
# end-state in a STATEFUL environment (no repo/patch — the container changes ARE the solution).
INSTANCE_TEMPLATE = """<task>
{{task}}
</task>

<instructions>
# Task Instructions

You are working in a Linux environment at /app (your working directory for all commands). The environment is
incomplete or broken; your job is to reach the end state the task describes by modifying the environment IN
PLACE — installing, building, configuring, editing, or fixing whatever is needed. There is no repository and no
patch: the state of the container when you submit IS your solution.

For each response: (1) a THOUGHT section with your reasoning, (2) at least one bash tool call.

## Hypothesis discipline (one per phase)

Before each phase of work, state your current hypothesis on its own line, exactly:

HYPOTHESIS: <one sentence — your current conjecture about WHY the environment does not yet meet the goal, or what it needs>

A HYPOTHESIS is a checkable conjecture about the current obstacle (a missing dependency, a wrong version, a
misconfiguration, a build error's root cause) — NOT a description of the command you will run. State it concrete
enough to be wrong. State a NEW `HYPOTHESIS:` line only when your conjecture CHANGES; keep working under the
current one (inspect / edit / build / test) without restating it.

## Command execution
- You issue >=1 bash command; the system runs it in a fresh non-persistent subshell; you see the result.
- Directory/env changes do NOT persist between commands; prefix with `cd /app && ...` or write to files.
- Use non-interactive flags (-y, -f). Avoid vi/nano. You may install tools.

## Submission
When the task's end state is reached, submit with this EXACT command (nothing else on the line):

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

You CANNOT continue working after submitting. If the verifier fails, you will be told and may keep working.
</instructions>
"""
