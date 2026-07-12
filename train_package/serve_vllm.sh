#!/usr/bin/env bash
# vLLM OpenAI-compatible server for the rollout eval — serves BASE and the SFT/raw LoRA adapters at once,
# so eval_rollout.py just points --model / --api-base at it. Run ON THE GPU BOX after training.
#
#   bash serve_vllm.sh                        # base + whatever adapters exist under out/
#   MODEL=Qwen/Qwen3.6-27B bash serve_vllm.sh
#
# Then from the LOCAL machine (or same box) run the rollout eval, e.g.:
#   EVAL_MODEL=openai/base     EVAL_API_BASE=http://<gpu-ip>:8000/v1 python src/eval_rollout.py ... (base arm)
#   EVAL_MODEL=openai/sft      EVAL_API_BASE=http://<gpu-ip>:8000/v1 python src/eval_rollout.py ... (structured arm)
#   EVAL_MODEL=openai/raw      EVAL_API_BASE=http://<gpu-ip>:8000/v1 python src/eval_rollout.py ... (raw arm)
#
# The served model NAMES are: base / sft / raw  (whichever adapters are present). LoRA lets all three share
# ONE base weight load — no reloading the 27B per arm.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-16384}"        # rollout prompts (full agent context) are long; give headroom over the 8192 train cutoff
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-True}"

echo "=== [1/4] ensure vllm installed ==="
python -c "import vllm" 2>/dev/null || pip install -q vllm

echo "=== [2/4] discover LoRA adapters under ./out ==="
# an adapter dir = one containing adapter_config.json. Name it by its folder unless it's an hlsft/hlraw dir.
LORA_ARGS=()
NAMES=()
for d in out/*/ ; do
  [ -f "${d}adapter_config.json" ] || continue
  base="$(basename "$d")"
  case "$base" in
    *hlsft*) nm="sft" ;;
    *hlraw*) nm="raw" ;;
    *)       nm="$base" ;;
  esac
  LORA_ARGS+=( "${nm}=$(pwd)/${d%/}" )
  NAMES+=( "$nm" )
  echo "  adapter: $nm -> ${d%/}"
done
if [ ${#LORA_ARGS[@]} -eq 0 ]; then
  echo "  (no adapters found under out/ — serving BASE only; that's fine for the base arm)"
fi

echo "=== [3/4] launch vLLM (base name = 'base'; adapters: ${NAMES[*]:-none}) ==="
EXTRA=()
if [ ${#LORA_ARGS[@]} -gt 0 ]; then
  EXTRA=( --enable-lora --max-lora-rank 16 --lora-modules "${LORA_ARGS[@]}" )
fi
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name base \
  --port "$PORT" --max-model-len "$MAXLEN" \
  --dtype bfloat16 --quantization bitsandbytes \
  "${EXTRA[@]}" > vllm_server.log 2>&1 &
echo "  server pid $! -> vllm_server.log"

echo "=== [4/4] wait for health (up to ~10 min while the 27B loads) ==="
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo ""
    echo "=== READY. served models: ==="
    curl -s "http://127.0.0.1:${PORT}/v1/models" | python -c "import sys,json; print('  ' + ', '.join(m['id'] for m in json.load(sys.stdin)['data']))"
    echo ""
    echo "Smoke test one arm:"
    echo "  curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\":\"base\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"max_tokens\":5}'"
    echo ""
    echo "Then run the rollout eval per arm (see the header of this file). Stop the server: kill $!"
    exit 0
  fi
  sleep 5
done
echo "SERVER DID NOT COME UP in ~10 min — check vllm_server.log (OOM? wrong model id? adapter rank mismatch?)"
exit 1
