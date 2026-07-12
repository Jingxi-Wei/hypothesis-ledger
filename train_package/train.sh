#!/usr/bin/env bash
# One-click QLoRA SFT on a rented GPU. Upload this whole folder, then: bash train.sh
set -euo pipefail
cd "$(dirname "$0")"

# China mirror for fast HF model download (AutoDL etc.); override with your own HF_ENDPOINT if needed.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export DISABLE_VERSION_CHECK=1
# reclaim fragmented CUDA memory (the ~3.6G "reserved but unallocated" that triggers OOM on long samples)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== [1/3] install LLaMA-Factory + bitsandbytes (each checked separately) ==="
# NOTE: many GPU images (AutoDL etc.) ship LLaMA-Factory but NOT bitsandbytes. Checking only llamafactory
# would skip the bitsandbytes install and QLoRA (4-bit) then fails with ModuleNotFoundError. Check each.
python -c "import llamafactory" 2>/dev/null || pip install -q "llamafactory[torch,metrics]"
python -c "import bitsandbytes" 2>/dev/null || pip install -q -U bitsandbytes  # required for quantization_bit: 4

echo "=== [2/3] sanity: dataset rows ==="
wc -l train_sharegpt.jsonl

echo "=== [3/3] train (model auto-downloads on first run) ==="
llamafactory-cli train qwen_qlora.yaml

echo ""
echo "=== DONE — LoRA adapter saved to ./out/qwen3-8b-hlsft ==="
echo "Flow test passed if: model downloaded, training ran 3 epochs, loss curve in out/, adapter files present."
