"""GPU-side eval inference: run BASE and SFT arms over the held-out items in ONE model load.

Ship this + eval_items.jsonl to the GPU box (it's in train_package/), run AFTER training:

  python eval_infer.py --model /root/autodl-tmp/ms/Qwen/Qwen3-8B --adapter out/qwen3-8b-hlsft        # flow test
  python eval_infer.py --model <local Qwen3.6-27B path> --adapter out/<27b-adapter>                  # real run

Writes eval_outputs.jsonl: {"item_id", "task", "output_base", "output_sft"} — copy it back to the local
machine and grade with eval_grade.py (the grader never sees which arm is which). Greedy decoding, thinking
off, identical params for both arms — the arms differ ONLY by the adapter.
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ADAPTER_KEY_MAPPING = {
    r"^model\.language_model\.": "model.",
    r"^language_model\.": "",
}


def generate(model, tok, prompt: str, max_new: int) -> str:
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="LOCAL base-model path (HF id needs network)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir from train.sh")
    ap.add_argument("--items", default="eval_items.jsonl")
    ap.add_argument("--out", default="eval_outputs.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke)")
    args = ap.parse_args()

    items = [json.loads(l) for l in Path(args.items).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]
    print(f"loading {args.model} (4-bit) + adapter {args.adapter} ... ({len(items)} items x 2 arms)")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto",
                                                trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter, key_mapping=ADAPTER_KEY_MAPPING)
    model.eval()

    outp = Path(args.out)
    done = set()
    if outp.exists():  # resumable: long runs survive SSH hiccups
        done = {json.loads(l)["item_id"] for l in outp.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"resuming: {len(done)} items already done")
    with outp.open("a", encoding="utf-8") as f:
        for k, it in enumerate(items, 1):
            if it["item_id"] in done:
                continue
            with model.disable_adapter():
                ob = generate(model, tok, it["input"], args.max_new_tokens)
            os_ = generate(model, tok, it["input"], args.max_new_tokens)
            f.write(json.dumps({"item_id": it["item_id"], "task": it["task"],
                                "output_base": ob, "output_sft": os_}, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(items):
                print(f"  {k}/{len(items)}", flush=True)
    print(f"[eval_infer] done -> {outp}  (copy back and run: python src/eval_grade.py)")


if __name__ == "__main__":
    main()
