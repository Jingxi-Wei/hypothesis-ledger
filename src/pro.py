"""SWE-bench Pro adaptation. Pro repos are multi-language (js/go/python); each instance ships its OWN
run_script.sh (test command) + parser.py (output parser) under swe_pro/run_scripts/<instance_id>/, so there is
NO per-language logic here — we just orchestrate the official eval. Reuses collect.py's agent/oracle/audit.

eval replicates the official entryscript (scaleapi/SWE-bench_Pro-os swe_bench_pro_eval.create_entryscript):
  cd /app; git reset --hard <base>; git checkout <base>; git apply <patch>; <before_repo_set_cmd last line>;
  bash run_script.sh <selected_test_files>; python parser.py stdout stderr output.json
Then resolved = every fail_to_pass AND every pass_to_pass is PASSED in output.json. ENV vars are baked into the
jefzda image, so (unlike the official script) we do not re-export them.
"""
import ast
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPTS = ROOT / "swe_pro" / "run_scripts"


def _jl(x):
    """Pro's list fields are Python-literal strings (the official harness parses them with eval()), not strict
    JSON — test descriptions contain quotes that break json.loads. Try JSON, fall back to ast.literal_eval."""
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except Exception:
        return ast.literal_eval(x)


def pro_image_uri(instance: dict) -> str:
    """jefzda/sweap-images:{tag} — mirrors helper_code/image_uri.get_dockerhub_image_uri."""
    uid, repo = instance["instance_id"], instance["repo"]
    repo_base, repo_name_only = repo.lower().split("/")
    hsh = uid.replace("instance_", "")
    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo.lower() and "element-web" in repo.lower():
        repo_name_only = "element"
        hsh = hsh[:-5] if hsh.endswith("-vnan") else hsh
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]
    tag = f"{repo_base}.{repo_name_only}-{hsh}"[:128]
    return f"jefzda/sweap-images:{tag}"


def normalize_pro(inst: dict) -> dict:
    """Inject the Pro docker image so get_sb_environment uses it; collect.py reads problem_statement/patch as-is."""
    inst = dict(inst)
    inst["image_name"] = pro_image_uri(inst)
    return inst


def _b64_write(sh, env, path: str, content: str) -> None:
    """Write `content` to `path` inside the container via base64. Large diffs (some Pro gold patches are 100KB+)
    blow past the Windows docker-exec command-line limit if sent as one command, silently truncating the file, so
    we append the base64 in small chunks and decode at the end."""
    b64 = base64.b64encode(content.encode()).decode()
    sh(env, f": > {path}.b64")
    for i in range(0, len(b64), 4000):
        sh(env, f"printf %s '{b64[i:i + 4000]}' >> {path}.b64")
    sh(env, f"base64 -d {path}.b64 > {path} && rm -f {path}.b64")


def eval_pro(env, instance: dict, patch: str, sh) -> dict:
    """Apply `patch` (the agent's source diff, or gold for a sanity check) on top of base and run the instance's
    own test script + parser. Returns {resolved, applied, f2p_pass, p2p_pass, feedback}."""
    iid = instance["instance_id"]
    sd = RUN_SCRIPTS / iid
    if not (sd / "run_script.sh").exists():
        return {"resolved": False, "applied": False, "f2p_pass": False, "p2p_pass": False,
                "feedback": f"no run_script for {iid}"}
    run_script = (sd / "run_script.sh").read_text(encoding="utf-8")
    parser_py = (sd / "parser.py").read_text(encoding="utf-8")
    f2p, p2p = _jl(instance["fail_to_pass"]), _jl(instance["pass_to_pass"])
    test_files = ",".join(_jl(instance["selected_test_files_to_run"]))
    base = instance["base_commit"]
    # The official entryscript runs the WHOLE before_repo_set_cmd *after* `git apply patch` — but for the many
    # instances whose setup starts with `git reset --hard <base>`, that silently discards the candidate patch
    # (gold included -> the instance looks unsolvable). We split before_repo_set_cmd instead: the setup/reset lines
    # run BEFORE applying the patch, and the golden-test installs (`git checkout <fix> -- <testfiles>`, i.e. lines
    # containing ' -- ') run AFTER — so the patch survives yet the graded test files are always the benchmark's own
    # (an agent cannot override them). This mirrors SWE-bench's canonical model-patch-then-test-patch ordering.
    before_lines = [l for l in instance["before_repo_set_cmd"].strip().split("\n") if l.strip()]
    setup_pre = "\n".join(l for l in before_lines if " -- " not in l)
    test_install = "\n".join(l for l in before_lines if " -- " in l)
    apply_cmd = ("cd /app && (git apply -v /workspace/patch.diff || git apply --3way /workspace/patch.diff "
                 "|| patch -p1 --fuzz=5 < /workspace/patch.diff)")

    def restore_agent_state() -> None:
        # The hidden test install mutates /app. Restore the agent-visible tree before self-rescue continues.
        sh(env, f"cd /app && git reset --hard {base} && git checkout {base}", timeout=300)
        sh(env, "cd /app && bash /workspace/setup_pre.sh || true", timeout=300)
        if (patch or "").strip():
            sh(env, apply_cmd, timeout=120)
        sh(env, "rm -f /workspace/run_script.sh /workspace/parser.py /workspace/patch.diff "
                "/workspace/setup_pre.sh /workspace/test_install.sh /workspace/stdout.log "
                "/workspace/stderr.log /workspace/output.json", timeout=120)

    sh(env, "mkdir -p /workspace")
    _b64_write(sh, env, "/workspace/run_script.sh", run_script)
    _b64_write(sh, env, "/workspace/parser.py", parser_py)
    _b64_write(sh, env, "/workspace/patch.diff", patch or "")
    _b64_write(sh, env, "/workspace/setup_pre.sh", setup_pre)
    _b64_write(sh, env, "/workspace/test_install.sh", test_install)
    sh(env, f"cd /app && git reset --hard {base} && git checkout {base}", timeout=300)
    sh(env, "cd /app && bash /workspace/setup_pre.sh || true", timeout=300)          # reset/clean to base
    ap = sh(env, apply_cmd, timeout=120)       # candidate/source patch
    applied = ap["returncode"] == 0 or not (patch or "").strip()
    sh(env, "cd /app && bash /workspace/test_install.sh || true", timeout=300)        # golden tests override patch
    sh(env, f"cd /app && bash /workspace/run_script.sh {test_files} > /workspace/stdout.log 2> /workspace/stderr.log; "
            "python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json 2>/dev/null || true",
       timeout=1800)
    raw = sh(env, "cat /workspace/output.json 2>/dev/null || echo '{}'")["output"]
    mo = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        tests = {t["name"]: t["status"] for t in json.loads(mo.group(0)).get("tests", [])} if mo else {}
    except Exception:
        tests = {}
    f2p_pass = bool(f2p) and all(tests.get(t) == "PASSED" for t in f2p)
    p2p_pass = all(tests.get(t) == "PASSED" for t in p2p)
    failed = [t for t in f2p if tests.get(t) != "PASSED"]
    tail = sh(env, "tail -c 1500 /workspace/stderr.log 2>/dev/null || true")["output"]
    feedback = (f"FAIL_TO_PASS still failing ({len(failed)}/{len(f2p)}): {failed[:5]}\n"
                f"(parsed {len(tests)} tests)\n{tail}")[-2500:]
    restore_agent_state()
    return {"resolved": applied and f2p_pass and p2p_pass, "applied": applied,
            "f2p_pass": f2p_pass, "p2p_pass": p2p_pass, "feedback": feedback}
