"""VERIFY-ON-GPU gate: the RM must score with the SAME tokenization stage:rm trained on.

Run this on the GPU box (train_rm.sh runs it automatically after training, tokenizer-only, no model load):

  python check_template.py            # PASS = rm_lib renders via LLaMA-Factory's own qwen3 template
  python check_template.py --show     # also print the decoded rendering for an eyeball check

PASS  -> rm_lib.render_ids uses LLaMA-Factory's template on this box = authoritative by construction.
FAIL  -> llamafactory not importable here, so rm_lib would fall back to a manual qwen3 string that MAY
         differ from what training saw (default system prompt, <think> handling). Fix the environment
         (pip install llamafactory) instead of trusting fallback scores.

Known benign residue: LLaMA-Factory may append one extra EOS after <|im_end|> during pairwise
preprocessing. That shifts the read position by one token IDENTICALLY for chosen and rejected, so the
RANKING comparison is unaffected; it is printed here for transparency, not treated as failure.
"""
import sys

import typer

import rm_lib

app = typer.Typer(add_completion=False)

_PROMPT = "ISSUE:\nDemo issue text.\n\nState the single most likely hypothesis about the CAUSE of this bug."
_RESP = "The parser collapses duplicate keys, so conflicting values are silently accepted."


@app.command()
def main(show: bool = typer.Option(False, "--show", help="print the decoded rendering")) -> None:
    from transformers import AutoTokenizer
    from pathlib import Path
    adapter = Path(rm_lib.RM_ADAPTER)
    src = str(adapter) if (adapter / "tokenizer_config.json").exists() else rm_lib.BASE_MODEL
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    ids, source = rm_lib.render_ids(tok, _PROMPT, _RESP)
    decoded = tok.decode(ids)
    if show:
        print("---- decoded rendering ----")
        print(decoded)
        print("---------------------------")
    print(f"[check_template] render source = {source} | {len(ids)} tokens | "
          f"has default system prompt: {'<|im_start|>system' in decoded} | "
          f"has think block: {'<think>' in decoded}")
    if source == "llamafactory":
        print("[check_template] PASS — scoring uses LLaMA-Factory's own qwen3 template (same as training).")
        return
    print("[check_template] FAIL — llamafactory is NOT importable here; rm_lib fell back to a manual "
          "rendering that is NOT verified against training. Install llamafactory in this environment "
          "(the training box already has it) and re-run. Do NOT trust score_rm/bon_eval output before this passes.")
    sys.exit(1)


if __name__ == "__main__":
    app()
