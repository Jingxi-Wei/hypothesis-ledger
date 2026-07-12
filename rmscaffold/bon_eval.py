"""Best-of-N bridge experiment (GPU side): does RM-guided selection beat random/first selection?

This is the cheapest demonstration that the audit ability can steer ACTION selection without RL:
  policy proposes N candidate hypotheses -> RM scores them -> compare RM-pick vs random-pick vs first-pick
on the same fixed blind rubric the main eval uses (src/eval_grade.py --llm).

Self-preference guard (README): DEFAULT policy is the BASE model, scored by the SFT-lineage RM —
a cross combination. Run --policy sft as the second arm and report both.

Distribution alignment with RM training (do not undo these):
  * generation runs with enable_thinking=False (same as eval_infer) — a <think> preamble would be
    format the RM never saw AND would break eval_grade's parsing.
  * the RM scores the EXTRACTED one-line hypothesis (HYPOTHESIS: <line>) — the same response
    normalization its training pairs used. Candidates with no extractable line are scored on full
    text and counted (extract_fallback in bon_meta.json). The OUTPUT files carry the full candidate
    text so the blind grader sees exactly what the policy produced.
  * probe-expected items are EXCLUDED: "pick the best guess" is ill-posed where guessing itself is
    premature (the right output there is a probe, not a hypothesis).

Two sequential phases in one run (each model freed before the next loads):
  1) generate: N samples per propose item        -> bon_candidates_<policy>.jsonl   (resumable by item_id)
  2) score+select: RM scores every candidate     -> bon_scores_<policy>.jsonl
     then writes eval_grade-compatible pair files:
       bon_rm_vs_random_<policy>.jsonl   output_base = RANDOM-pick, output_sft = RM-pick
       bon_rm_vs_first_<policy>.jsonl    output_base = FIRST-pick,  output_sft = RM-pick
     (column names abuse eval_grade's base/sft slots so the existing blind grader runs unchanged;
      the grader never sees which strategy is which.)

Grade locally afterwards:
  python src/eval_grade.py --llm --outputs rmscaffold/bon_rm_vs_random_base.jsonl
"""
import gc
import json
import random
import re
from pathlib import Path

import torch
import typer

import rm_lib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SFT_ADAPTER = str(HERE.parent / "train_package" / "out" / "qwen3.6-27b-hlsft")  # override with --sft-adapter
app = typer.Typer(add_completion=False)

_HYP_LINE = re.compile(r"HYPOTHESIS:\s*(.+)", re.IGNORECASE)


def _extract_hyp(text: str) -> tuple[str, bool]:
    """One-line hypothesis for RM scoring (same normalization as the training pairs). (text, extracted?)"""
    m = _HYP_LINE.search(text or "")
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip()), True
    return re.sub(r"\s+", " ", (text or "").strip())[:400], False


# same remap eval_infer.py uses for the SFT adapter — WITHOUT it PEFT silently drops every mismatched
# LoRA weight and the "sft policy" is actually the base model (the exact base==sft bug this project
# already hit once: autodl_results .../eval_outputs_bad_same66.jsonl)
ADAPTER_KEY_MAPPING = {
    r"^model\.language_model\.": "model.",
    r"^language_model\.": "",
}


def _load_policy(policy: str, sft_adapter: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(rm_lib.BASE_MODEL, trust_remote_code=True)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(rm_lib.BASE_MODEL, quantization_config=qc, device_map="auto",
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True)
    if policy == "sft":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, sft_adapter, key_mapping=ADAPTER_KEY_MAPPING)
    model.eval()
    return model, tok


N_PROMPT_TRUNC = 0  # over-length prompts spliced head+tail (recorded in bon_meta — silent truncation lies)


@torch.no_grad()
def _sample(model, tok, prompt: str, n: int, temperature: float, max_new: int) -> list[str]:
    global N_PROMPT_TRUNC
    msgs = [{"role": "user", "content": prompt}]
    try:  # Qwen3 thinking OFF — same as eval_infer; a <think> preamble is OOD for the RM and the grader
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # over-length: keep template head AND tail — plain right-truncation would sever the trailing
    # <|im_start|>assistant generation header and the policy would just continue the user text
    ids_t = tok(text, return_tensors="pt")["input_ids"][0]
    if ids_t.shape[0] > 8192:
        N_PROMPT_TRUNC += 1
        keep = 64
        ids_t = torch.cat([ids_t[:keep], ids_t[-(8192 - keep):]])
    ids = {"input_ids": ids_t.unsqueeze(0).to(model.device),
           "attention_mask": torch.ones(1, ids_t.shape[0], dtype=torch.long, device=model.device)}
    outs = model.generate(**ids, do_sample=True, temperature=temperature, top_p=0.95,
                          num_return_sequences=n, max_new_tokens=max_new,
                          pad_token_id=tok.eos_token_id)
    return [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in outs]


@app.command()
def main(n: int = typer.Option(8, "--n"),
         policy: str = typer.Option("base", "--policy", help="base (default; cross with SFT-lineage RM) | sft"),
         sft_adapter: str = typer.Option(SFT_ADAPTER, "--sft-adapter"),
         items_file: str = typer.Option("", "--items",
                                        help="eval items.jsonl (default: items.jsonl beside this script — the "
                                             "documented GPU upload location — else dataset/eval/items.jsonl)"),
         temperature: float = typer.Option(0.8, "--temperature"),
         max_new: int = typer.Option(700, "--max-new-tokens", help="same default as eval_infer"),
         limit: int = typer.Option(0, "--limit"),
         seed: int = typer.Option(42, "--seed")) -> None:
    ip = Path(items_file) if items_file else (
        HERE / "items.jsonl" if (HERE / "items.jsonl").exists() else ROOT / "dataset" / "eval" / "items.jsonl")
    if not ip.exists():
        raise SystemExit(f"[bon] items file not found: {ip} — copy dataset/eval/items.jsonl into rmscaffold/ "
                         "(the GPU box has no dataset/ tree) or pass --items explicitly.")
    items = [json.loads(l) for l in ip.read_text(encoding="utf-8").splitlines() if l.strip()]
    # propose items only — BoN selects among candidate HYPOTHESES. Skip schema-transfer twins (renamed
    # labels) and probe-expected items (there the honest output is a probe, not a best guess).
    items = [it for it in items if it["task"] == "propose" and not it.get("schema_transfer")
             and it.get("expected") != "probe"]
    if limit:
        items = items[:limit]
    cand_path = HERE / f"bon_candidates_{policy}.jsonl"
    done = set()
    if cand_path.exists():
        done = {json.loads(l)["item_id"] for l in cand_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [it for it in items if it["item_id"] not in done]
    print(f"[bon] {len(items)} propose items, {len(done)} already generated, {len(todo)} to go "
          f"(policy={policy}, n={n}, max_new={max_new})")

    if todo:  # ---- phase 1: generate candidates (policy model) ----
        model, tok = _load_policy(policy, sft_adapter)
        with cand_path.open("a", encoding="utf-8") as f:
            for i, it in enumerate(todo):
                cands = _sample(model, tok, it["input"], n, temperature, max_new)
                f.write(json.dumps({"item_id": it["item_id"], "candidates": cands}, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 5 == 0:
                    print(f"  gen {i + 1}/{len(todo)}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- phase 2: RM scores every candidate (on the EXTRACTED one-liner), then select ----
    cands = {json.loads(l)["item_id"]: json.loads(l)["candidates"]
             for l in cand_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    rm, rtok = rm_lib.load_rm()
    rng = random.Random(seed)
    rows, f_rand, f_first = [], [], []
    n_scored = n_fallback = 0
    for i, it in enumerate(items):
        cs = cands.get(it["item_id"])
        if not cs:
            continue
        extracted = [_extract_hyp(c) for c in cs]
        n_scored += len(cs)
        n_fallback += sum(1 for _t, ok in extracted if not ok)
        scores = [rm_lib.score(rm, rtok, it["input"], t) for t, _ok in extracted]
        pick_rm = cs[max(range(len(cs)), key=lambda j: scores[j])]
        pick_rand = rng.choice(cs)
        pick_first = cs[0]
        rows.append({"item_id": it["item_id"], "scores": scores,
                     "extracted": [ok for _t, ok in extracted]})
        f_rand.append({"item_id": it["item_id"], "task": "propose", "output_base": pick_rand, "output_sft": pick_rm})
        f_first.append({"item_id": it["item_id"], "task": "propose", "output_base": pick_first, "output_sft": pick_rm})
        if (i + 1) % 10 == 0:
            print(f"  score {i + 1}/{len(items)}", flush=True)
    (HERE / f"bon_scores_{policy}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (HERE / f"bon_rm_vs_random_{policy}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in f_rand), encoding="utf-8")
    (HERE / f"bon_rm_vs_first_{policy}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in f_first), encoding="utf-8")
    meta = {"policy": policy, "n": n, "temperature": temperature, "max_new": max_new,
            "items": len(rows), "candidates_scored": n_scored, "extract_fallback": n_fallback,
            "extract_fallback_frac": round(n_fallback / n_scored, 3) if n_scored else None,
            "prompt_truncated": N_PROMPT_TRUNC, "rm_score_overflows": rm_lib.OVERFLOWS,
            "sft_adapter": sft_adapter if policy == "sft" else None, "seed": seed}
    (HERE / f"bon_meta_{policy}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[bon] wrote bon_rm_vs_random_{policy}.jsonl + bon_rm_vs_first_{policy}.jsonl | "
          f"extract fallback {n_fallback}/{n_scored} -> bon_meta_{policy}.json\n"
          f"  grade locally (proxy, batch paused):\n"
          f"  python src/eval_grade.py --llm --outputs rmscaffold/bon_rm_vs_random_{policy}.jsonl\n"
          "  (output_base = control pick, output_sft = RM pick; the blind grader never knows which is which)")


if __name__ == "__main__":
    app()
