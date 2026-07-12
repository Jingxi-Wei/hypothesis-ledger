"""Unit tests for the Terminal-Bench / LiveCodeBench adapters + their pipeline integration (no docker, no proxy).

The docker-dependent gold-sanity (EMPTY fails / GOLD resolves) lives in src/_sanity_tb.py and src/_sanity_lcb.py.
"""
import base64
import json
import pickle
import zlib

import pytest

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parent / "_wincompat"))  # swebench's `import resource` stub (compress/export import chain)

import tb  # noqa: E402
import lcb  # noqa: E402
import compress as C  # noqa: E402


# ---------------- Terminal-Bench ----------------

def test_tb_loads_all_tasks_with_required_fields():
    insts = tb.load_instances()
    assert len(insts) >= 80, "expected the ~89 vendored TB2 tasks"
    for i in insts:
        assert i["instance_id"].startswith("tb__")          # namespaced so it never collides with pro/verified ids
        assert i["problem_statement"].strip()               # instruction.md
        assert i["patch"].strip()                           # solve.sh = the gold/oracle
        assert i["docker_image"] and ":" in i["docker_image"]


def test_tb_task_dir_roundtrips_from_instance_id():
    inst = tb.load_instances()[0]
    assert tb._task_dir(inst).name == inst["instance_id"][len("tb__"):]
    assert (tb._task_dir(inst) / "task.toml").exists()


def test_tb_image_reads_docker_image_field():
    inst = tb.load_instances()[0]
    assert tb.tb_image(inst) == inst["docker_image"]
    assert tb.normalize_tb({"instance_id": "tb__x", "docker_image": "img:1"})["docker_image"] == "img:1"


def test_tb_template_keeps_the_shared_protocol():
    t = tb.INSTANCE_TEMPLATE
    assert "{{task}}" in t                                   # jinja task slot
    assert "HYPOTHESIS:" in t                                # the reasoning discipline is preserved
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in t      # the exact submit marker the agent/compress rely on


# ---------------- LiveCodeBench ----------------

def _encode_private(cases: list[dict]) -> str:
    """Mirror LCB's private-test encoding so the decoder is tested against the real pipeline, not a guess."""
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()


def test_lcb_decode_private_roundtrip_and_plain_public():
    cases = [{"input": "26", "output": "2025\n", "testtype": "stdin"}]
    assert C and lcb._decode_tests(_encode_private(cases)) == cases      # b64 -> zlib -> pickle -> json
    assert lcb._decode_tests(json.dumps(cases)) == cases                 # public path: plain json


def test_lcb_loads_problems_with_split_tests():
    insts = lcb.load_instances()
    assert len(insts) >= 150
    i0 = insts[0]
    assert i0["instance_id"].startswith("lcb__")
    assert i0["patch"] is None                               # LCB has no gold solution
    assert i0["problem_statement"].strip()
    assert isinstance(i0["_public"], list) and isinstance(i0["_private"], list)
    assert i0["_private"], "private (hidden) cases must decode"


def test_lcb_date_and_difficulty_filters():
    all_n = len(lcb.load_instances())
    hard = lcb.load_instances(difficulties={"hard"})
    assert 0 < len(hard) < all_n and all(i["difficulty"] == "hard" for i in hard)
    # a date beyond the file's window (2025-04) keeps nothing; the epoch keeps everything
    assert lcb.load_instances(after_date="2099-01-01") == []
    assert len(lcb.load_instances(after_date="2000-01-01")) == all_n


def test_lcb_private_cases_capped():
    # some problems ship 100+ hidden cases; the cap keeps eval bounded
    assert all(len(i["_private"]) <= lcb._MAX_CASES for i in lcb.load_instances())


def test_lcb_func_wrapper_is_valid_python():
    wrapped = lcb._FUNC_WRAPPER.format(solution="class Solution:\n    def f(self, x):\n        return x+1",
                                       func="f", has_class=True)
    compile(wrapped, "<wrapper>", "exec")                    # must be syntactically valid to run in the sandbox


def test_lcb_template_keeps_the_shared_protocol():
    t = lcb.INSTANCE_TEMPLATE
    assert "{{task}}" in t and "HYPOTHESIS:" in t and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in t
    assert "solution.py" in t                                # the agent must know where to write


# ---------------- compress classification (substrate-conditional; SWE byte-identical) ----------------

@pytest.mark.parametrize(("cmd", "substrate", "kind"), [
    ("cd /app && make -j4", "tb", "edit"),                   # TB build = state change, kept as evidence
    ("pip install numpy", "tb", "edit"),
    ("apt-get install -y gcc", "tb", "edit"),
    ("cd /workspace && python solution.py < _in.txt", "lcb", "test"),  # LCB: running the candidate IS the test
    ("cat > /workspace/solution.py", "lcb", "edit"),
    ("sed -i 's/a/b/' f.py", "swe", "edit"),                 # SWE edit unchanged
    ("pytest -q", "swe", "test"),
    ("ls -la /app", "tb", "read"),                           # genuine noise stays noise
    ("grep -rn foo src", "swe", "read"),
    # ---- REGRESSION guards (2026-07-07 review: 124 flips across 30 real Pro trajectories) ----
    ("pip install numpy", "swe", "read"),                    # SWE keeps the ORIGINAL classification
    ("cd /app && make -j4", "swe", "read"),
    ("cat <<'EOF' > tmp_probe.go", "swe", "read"),           # heredoc Go probe scripts must stay reads on SWE
    ('grep -R "Please make sure namespace" -n test lib', "swe", "read"),   # quoted word 'make'
    ('grep -R "Please make sure namespace" -n test lib', "tb", "read"),    # head-anchor: quoted 'make' safe on TB too
    ("python -m pip install -q --target /tmp/x pkg", "swe", "read"),       # introspection install stays read on SWE
])
def test_compress_classifies_multi_substrate_commands(cmd, substrate, kind):
    assert C._classify(cmd, substrate) == kind


def test_compress_substrate_from_instance_id():
    assert C._substrate("tb__build-cython-ext") == "tb"
    assert C._substrate("lcb__abc387_b") == "lcb"
    assert C._substrate("instance_NodeBB__NodeBB-abc") == "swe"
    assert C._substrate("astropy__astropy-12907") == "swe"


# ---------------- gold-token redaction backstops ----------------

def test_tb_gold_tokens_exclude_instruction_words():
    toks = tb._gold_tokens("git clone https://github.com/SPOCKnots/pyknotid.git /app/pyknotid\nsed -i 's/x/y/' secretmod.py",
                           "Help me build pyknotid from source.")
    assert "secretmod.py" in toks                            # solve-only identifier -> redactable
    assert not any("pyknotid" == t for t in toks)            # named in the instruction -> public, not redacted


def test_lcb_gold_tokens_hidden_values_only():
    insts = lcb.load_instances()
    for i in insts[:20]:
        pub_txt = i["problem_statement"] + json.dumps(i["_public"])
        for t in i["_gold_tokens"]:
            assert len(t) >= 3                               # single digits never enter the redactor
            assert t not in pub_txt                          # already-public values are not "hidden"


def test_lcb_norm_out_judge_tolerance():
    assert lcb._norm_out("R 3 2 \nB 2 2\n") == lcb._norm_out("R 3 2\nB 2 2")   # per-line trailing space tolerated
    assert lcb._norm_out("1\n2") != lcb._norm_out("1\n3")                       # real differences still fail


def test_lcb_func_wrapper_bakes_class_choice_from_candidate():
    w_cls = lcb._FUNC_WRAPPER.format(solution="class Solution:\n    def f(self, x):\n        return x+1",
                                     func="f", has_class=True)
    w_fn = lcb._FUNC_WRAPPER.format(solution="def f(x):\n    return x+1", func="f", has_class=False)
    compile(w_cls, "<w>", "exec")
    compile(w_fn, "<w>", "exec")
    assert "if True else" in w_cls and "if False else" in w_fn   # decision baked from the CANDIDATE, not self-referential


# ---------------- collect.load_instances dispatch ----------------

def test_collect_loader_dispatches_to_adapters():
    import collect
    tbd = collect.load_instances("tb")
    lcd = collect.load_instances("lcb")
    assert len(tbd) >= 80 and next(iter(tbd)).startswith("tb__")
    assert len(lcd) >= 150 and next(iter(lcd)).startswith("lcb__")
    # gold shape the exporter/audit rely on
    assert tbd[next(iter(tbd))]["patch"].strip()             # TB gold present
    assert lcd[next(iter(lcd))]["patch"] is None             # LCB gold absent


# ---------------- poison-pill context-overflow detection (auto-skip safety) ----------------

@pytest.mark.parametrize(("text", "want"), [
    # REAL poison: the model refused because the prompt overflowed its window -> auto-skip is correct
    ("BadGatewayError: OpenAIException - Codex API error (502): Your input exceeds the context window of this model.", True),
    ("openai.BadRequestError: context_length_exceeded - maximum context length is 400000 tokens", True),
    ("This model's maximum context length is 200000 tokens. Please reduce the length of the messages.", True),
    # NOT poison: transient/operational — auto-skipping these would WRONGLY blacklist good instances.
    # The 2026-07-08 auth outage produced 8 such orphans; they MUST stay eligible for retry.
    ("litellm.BadRequestError: OpenAIException - Not authenticated. Please login first at /", False),
    ("litellm.APIConnectionError: Connection refused", False),
    ("docker: Error response from daemon: container not found", False),
    ("litellm.RateLimitError: rate limit exceeded", False),
])
def test_context_overflow_detection_separates_poison_from_transient(text, want):
    import collect
    assert collect.is_context_overflow(text) is want
    assert collect.is_context_overflow(RuntimeError(text)) is want   # works on exception objects too
