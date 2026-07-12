"""Minimal wiring smoke: gpt5.5 via codex-proxy at medium effort.

Uses a reasoning-inducing prompt and compares no-effort vs medium on BOTH
latency and reasoning_tokens, so we can tell whether reasoning_effort actually
reaches the model (the ARC effort-floor bug was effort silently never sent).
extra_body is used so litellm cannot drop reasoning_effort.
"""
import json
import os
import time

os.environ["MSWEA_SILENT_STARTUP"] = "1"
import litellm  # noqa: E402

BASE = "http://127.0.0.1:8080/v1"
KEY = "pwd"
MODEL = "openai/gpt5.5"
PROMPT = (
    "A snail is at the bottom of a 10 meter well. Each day it climbs 3 meters, each night it "
    "slips back 2 meters. On which day does it first reach the top? Reason step by step, then "
    "give the day number."
)


def call(effort: str | None):
    kw = dict(model=MODEL, messages=[{"role": "user", "content": PROMPT}], api_base=BASE, api_key=KEY)
    if effort:
        kw["extra_body"] = {"reasoning_effort": effort}
    t = time.time()
    r = litellm.completion(**kw)
    dt = time.time() - t
    usage = r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage)
    return dt, usage, (r.choices[0].message.content or "")


for eff in (None, "medium"):
    try:
        dt, usage, content = call(eff)
        ctd = usage.get("completion_tokens_details") or {}
        rt = ctd.get("reasoning_tokens") if isinstance(ctd, dict) else None
        print(f"=== effort={eff} === {dt:.1f}s completion_tokens={usage.get('completion_tokens')} reasoning_tokens={rt}")
        if eff == "medium":
            print("   full usage:", json.dumps(usage, default=str))
        print("   answer tail:", content[-140:].replace(chr(10), " "))
    except Exception as e:
        print(f"=== effort={eff} === ERROR: {repr(e)[:300]}")
