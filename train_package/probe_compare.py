"""Side-by-side reasoning probe: BASE Qwen vs the SFT LoRA adapter, on out-of-domain hypothesis-discipline
stories (+1 pure logic riddle). Qualitative smoke test for the eyeball — NOT the held-out eval.

Run on the GPU box AFTER train.sh. In China / offline boxes HuggingFace is unreachable ([Errno 99]), so point
--model at the LOCAL ModelScope cache the training already downloaded (NOT the HF id):
  python probe_compare.py --model /root/autodl-tmp/ms/Qwen/Qwen3-8B --adapter out/qwen3-8b-hlsft            # final adapter
  python probe_compare.py --model /root/autodl-tmp/ms/Qwen/Qwen3-8B --adapter out/qwen3-8b-hlsft/checkpoint-200  # mid-run
(--model also accepts a HF id if the box can reach HF or a mirror. Results saved to probe_results.txt.)
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # if a HF id IS used, go through the CN mirror
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def generate(model, tok, prompt: str, thinking: bool, max_new: int) -> str:
    msgs = [{"role": "user", "content": prompt}]
    try:  # Qwen3 chat template supports enable_thinking; targets were trained WITHOUT <think>, keep it off for both arms
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default="out/qwen3-8b-hlsft")
    ap.add_argument("--probes", default="probes.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--thinking", action="store_true", help="enable Qwen thinking mode for BOTH arms (default off)")
    args = ap.parse_args()

    if not Path(args.adapter).exists():
        raise SystemExit(f"adapter not found: {args.adapter} — run `bash train.sh` first")
    probes = [json.loads(l) for l in Path(args.probes).read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"loading {args.model} (4-bit) + adapter {args.adapter} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto",
                                                trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter)  # one load; disable_adapter() gives the BASE arm
    model.eval()

    lines: list[str] = []
    for i, p in enumerate(probes, 1):
        hdr = f"\n{'=' * 90}\nPROBE {i}/{len(probes)}: {p['title']}  [{p['kind']}]\n{'=' * 90}"
        print(hdr)
        with model.disable_adapter():
            base_out = generate(model, tok, p["prompt"], args.thinking, args.max_new_tokens)
        sft_out = generate(model, tok, p["prompt"], args.thinking, args.max_new_tokens)
        block = (f"{hdr}\n\n--- BASE ({args.model}) ---\n{base_out}\n\n--- SFT (+{args.adapter}) ---\n{sft_out}\n\n"
                 f"--- 看点 (what good looks like) ---\n{p['expect']}\n")
        print(block)
        lines.append(block)

    Path("probe_results.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] probe_results.txt — {len(probes)} probes, base-vs-SFT side by side")


if __name__ == "__main__":
    main()
