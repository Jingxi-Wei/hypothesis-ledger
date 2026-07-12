#!/usr/bin/env bash
# One-click RM (QLoRA + value head) on a rented GPU. Upload the rmscaffold/ folder, then: bash train_rm.sh
# Prereq: prep_rm.py was run locally so rm_pairs.jsonl + dataset_info.json are in this folder.
set -euo pipefail
cd "$(dirname "$0")"

# China mirror for fast HF model download (AutoDL etc.); override with your own HF_ENDPOINT if needed.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# AutoDL 防呆（2026-07-12 实踩）：手动新终端重跑时没带 autopilot 的环境变量 -> LF 去 HF 重下 27B 撞 xet 401，
# 而模型早在 ModelScope 缓存里。检测到缓存目录就自动对齐 SFT 同款环境。
if [ -d /root/autodl-tmp/ms ]; then
  export USE_MODELSCOPE_HUB="${USE_MODELSCOPE_HUB:-1}"
  export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/autodl-tmp/ms}"
  export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
fi
export DISABLE_VERSION_CHECK=1
# 重定向到文件时 stdout 是块缓冲——loss 字典会被憋到缓冲攒满才落盘（进度条走 stderr 看得见,
# loss 看不见, 2026-07-12 实踩）。关掉缓冲让日志实时可 tail。
export PYTHONUNBUFFERED=1
# reclaim fragmented CUDA memory (the "reserved but unallocated" that triggers OOM on long samples)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== [1/3] install LLaMA-Factory + bitsandbytes (each checked separately) ==="
# NOTE: many GPU images ship LLaMA-Factory but NOT bitsandbytes — check each (train_package lesson).
python -c "import llamafactory" 2>/dev/null || pip install -q "llamafactory[torch,metrics]"
python -c "import bitsandbytes" 2>/dev/null || pip install -q -U bitsandbytes  # required for quantization_bit: 4

echo "=== [2/4] gate: prep_rm's >=300-clean-pairs verdict is ENFORCED here (RM_ALLOW_SMALL=1 overrides) ==="
wc -l rm_pairs.jsonl
python - <<'EOF'
import json, os, sys
kept = json.load(open("rm_stats.json")).get("kept", 0)
if kept < 300 and os.environ.get("RM_ALLOW_SMALL") != "1":
    print(f"GATE FAIL: only {kept} clean pairs (<300) — prep_rm said don't train. Set RM_ALLOW_SMALL=1 to override.")
    sys.exit(1)
print(f"GATE OK: {kept} clean pairs")
EOF

echo "=== [3/4] train RM (model auto-downloads on first run) ==="
llamafactory-cli train rm_qlora.yaml

echo "=== [4/4] template gate: scoring must render EXACTLY like training (tokenizer-only, fast) ==="
python check_template.py

echo ""
echo "=== DONE — RM adapter + value_head saved to ./out/qwen3.6-27b-hlrm ==="
echo "Next: python score_rm.py --smoke   (value-head sanity: scores must NOT be constant),"
echo "      python score_rm.py           (held-out chosen>rejected accuracy + slices — the RM headline),"
echo "      python bon_eval.py --n 8     (Best-of-N bridge; then --policy sft as the second arm)."
