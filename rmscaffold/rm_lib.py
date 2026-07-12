"""Shared GPU-side RM loading/scoring for score_rm.py / bon_eval.py. NO src/ imports (self-contained on GPU).

Two silent-failure landmines, both closed here:
  1. TEMPLATE MISMATCH: LLaMA-Factory stage:rm encoded training pairs with ITS qwen3 template (its own
     system-prompt default and <think> handling). Scoring with a different rendering = the value head sees
     an OOD format and the scores are quiet garbage. render_ids() therefore uses LLaMA-Factory's OWN
     template when importable (authoritative — same box that trained), and only falls back to a manual
     qwen3 chat string otherwise. Run check_template.py BEFORE trusting any score.
  2. VALUE-HEAD NO-OP LOAD: strict=False load_state_dict silently ignores mismatched keys, leaving the
     head at random init (≈constant scores). load_rm() now hard-asserts the v_head weights in the model
     EQUAL the file after loading.
"""
import os
from pathlib import Path

import torch

# Env-overridable (2026-07-12 lesson): rm_lib loads via transformers DIRECTLY, so ModelScope env vars
# (a LLaMA-Factory feature) do NOT apply — a bare hub id re-downloads 55GB and hits hf-mirror's xet 401
# on AutoDL. Set HL_BASE_MODEL to the local weights dir (e.g. /root/autodl-tmp/ms/Qwen/Qwen3.6-27B).
BASE_MODEL = os.environ.get("HL_BASE_MODEL", "Qwen/Qwen3.6-27B")
RM_ADAPTER = os.environ.get("HL_RM_ADAPTER", str(Path(__file__).resolve().parent / "out" / "qwen3.6-27b-hlrm"))
OVERFLOWS = 0  # count of scored sequences that exceeded max_len (callers print it — silent truncation lies)


def render_ids(tok, prompt: str, response: str) -> tuple[list[int], str]:
    """Token ids for one (prompt, response) exactly as stage:rm training encoded it.
    Returns (ids, source) where source is 'llamafactory' (authoritative) or 'manual' (verify first)."""
    try:
        from llamafactory.data.template import TEMPLATES
        tpl = TEMPLATES["qwen3"]
        enc = tpl.encode_oneturn(tok, [{"role": "user", "content": prompt},
                                       {"role": "assistant", "content": response}])
        if isinstance(enc, tuple):
            prompt_ids, resp_ids = enc[0], enc[1]
            return list(prompt_ids) + list(resp_ids), "llamafactory"
        return list(enc), "llamafactory"
    except Exception:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
        return tok(text, add_special_tokens=False)["input_ids"], "manual"


def load_rm(base_model: str = BASE_MODEL, adapter: str = RM_ADAPTER, load_4bit: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from trl import AutoModelForCausalLMWithValueHead
    from safetensors.torch import load_file

    tok = AutoTokenizer.from_pretrained(adapter if (Path(adapter) / "tokenizer_config.json").exists() else base_model,
                                        trust_remote_code=True)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4") if load_4bit else None
    base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=qc, device_map="auto",
                                                torch_dtype=torch.bfloat16, trust_remote_code=True)
    peft = PeftModel.from_pretrained(base, adapter)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(peft)
    vh_path = Path(adapter) / "value_head.safetensors"
    if not vh_path.exists():
        raise FileNotFoundError(f"value_head.safetensors not found in {adapter} — did stage:rm training finish?")
    sd = load_file(str(vh_path))
    vh_keys = [k for k in sd if k.startswith("v_head")]
    if not vh_keys:
        raise RuntimeError(f"no v_head.* keys in {vh_path}; found {list(sd)} — LLaMA-Factory layout changed?")
    model.load_state_dict(sd, strict=False)
    msd = model.state_dict()
    for k in vh_keys:  # HARD assertion: the loaded head must EQUAL the file (strict=False hides misses)
        if k not in msd:
            raise RuntimeError(f"model has no parameter {k} — trl value-head wrapper layout mismatch")
        if not torch.allclose(msd[k].float().cpu(), sd[k].float().cpu(), atol=1e-6):
            raise RuntimeError(f"{k} did NOT load (model != file) — value head would be random init")
    print(f"[rm_lib] value head loaded + verified equal to file: {vh_keys}")
    model.eval()
    return model, tok


@torch.no_grad()
def score(model, tok, prompt: str, response: str, max_len: int = 8192) -> float:
    """Scalar reward: value-head output at the final token of the training-identical rendering.
    Over-length handling: keep the TEMPLATE HEAD (first 64 tokens — a sequence starting mid-prompt without
    <|im_start|> structure is a rendering training never produced) AND the response end (where the reward is
    read); the middle of the prompt is dropped. Overflows are counted in rm_lib.OVERFLOWS — callers report
    them, because upstream length gates (prep_rm/gen_pairs budgets) should make this rare."""
    global OVERFLOWS
    ids, _src = render_ids(tok, prompt, response)
    if len(ids) > max_len:
        OVERFLOWS += 1
        keep_head = 64
        ids = ids[:keep_head] + ids[-(max_len - keep_head):]
    dev = model.pretrained_model.device
    t = torch.tensor([ids], device=dev)
    _, _, values = model(input_ids=t, attention_mask=torch.ones_like(t))  # trl forward -> (logits, loss, values)
    return float(values[0, -1].item())
