# Hypothesis-Ledger SFT — one-click training package

Self-contained QLoRA SFT. Upload this whole folder to a rented GPU, then run:

```bash
bash train.sh
```

## Contents
- `train_sharegpt.jsonl` — SFT data (sharegpt; audit / propose / fix samples)
- `dataset_info.json` — LLaMA-Factory dataset registration (points at the jsonl, dataset name `hl_sft`)
- `qwen_qlora.yaml` — QLoRA config
- `train.sh` — installs LLaMA-Factory + runs training

## Flow test (now): `Qwen/Qwen3-8B` on a small GPU
Qwen3.6 has no <27B dense, so the flow test uses Qwen3-8B (same family → template/flow transfers).
Goal = validate the flow: data loads, QLoRA trains 3 epochs, LoRA adapter saves to `out/`. **Not** the final result.
Sizing: a single 24G GPU (RTX 4090 / A10) is enough.

## Real run (later): `Qwen/Qwen3.6-27B`
Edit `qwen_qlora.yaml` → `model_name_or_path: Qwen/Qwen3.6-27B` (needs ~1× A100/H100-80G).
Keep thinking OFF (`enable_thinking=false`) — our targets are direct, no `<think>` blocks.

## Refresh the data before the real run
The jsonl here is a snapshot (flow-test data). For the latest/complete data: in the main project run
`python src/process_all.py --run-id r1`, then copy `dataset/sft/train_sharegpt.jsonl` + `dataset_info.json` here.

## Notes
- `train.sh` sets `HF_ENDPOINT=https://hf-mirror.com` (fast model download in China / AutoDL). Override if you have your own.
- If the LLaMA-Factory build lacks a `qwen3` template, change `template: qwen3` → `qwen` in the yaml.
