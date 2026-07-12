#!/bin/bash
# 一键全链 v2（单卡即可，2026-07-11 用户拍板：raw 臂本轮不跑；失败不硬停，能跑的都跑完）：
#   SFT -> 选最优 checkpoint -> 拷 rmscaffold/sft_adapter -> RM -> score_rm 三连 -> bon_eval 双臂 -> eval_infer
# 失败策略：
#   * SFT 失败 = 致命（后面全依赖它）→ 停，留 FAILED。
#   * RM 失败 → 跳过 RM 系评测（score_rm/bon_eval），但 eval_infer 照跑。
#   * 任何单个评测失败 → 记入 FAILURES，继续下一个。
# 收尾标记：ALL_DONE（零失败）或 DONE_WITH_ERRORS（看 FAILURES 明细）。
# 用法: 两个 zip 解压到 /root/autodl-tmp 后:  cd train_package && nohup bash autopilot.sh > autopilot.out 2>&1 &
set -u
cd "$(dirname "$0")"
ROOT=/root/autodl-tmp
mark() { echo "[autopilot $(date '+%H:%M:%S')] $*" | tee -a autopilot.log; }
soft() { # soft "$名字" 命令... —— 失败记账不停车
  local name=$1; shift
  mark "$name start"
  if "$@"; then mark "$name OK"; else mark "$name FAILED（继续后面的）"; echo "$name" >> FAILURES; fi
}
export USE_MODELSCOPE_HUB=1 HF_HOME=$ROOT/hf MODELSCOPE_CACHE=$ROOT/ms
rm -f FAILED ALL_DONE DONE_WITH_ERRORS FAILURES

# ---- 1. SFT（致命段）----
mark "SFT start (GPU0)"
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train qwen27b_qlora.yaml > train_27b.log 2>&1 \
  || { mark "SFT 训练退出非零 — 看 train_27b.log 尾部"; echo "SFT" > FAILED; exit 1; }
BEST=$(python - <<'PY'
import json, glob, os
cands = sorted(glob.glob('out/qwen3.6-27b-hlsft/checkpoint-*'), key=lambda p: int(p.rsplit('-',1)[1]))
best, best_loss = (cands[-1] if cands else ''), float('inf')
for c in cands:
    ts = os.path.join(c, 'trainer_state.json')
    if os.path.exists(ts):
        losses = [e['eval_loss'] for e in json.load(open(ts)).get('log_history', []) if 'eval_loss' in e]
        if losses and losses[-1] < best_loss:
            best_loss, best = losses[-1], c
print(best)
PY
)
[ -n "$BEST" ] || { echo "SFT-ckpt" > FAILED; mark "找不到 SFT checkpoint"; exit 1; }
mark "SFT done — best ckpt: $BEST"

# ---- 2. RM（失败则跳过 RM 系评测）----
RM_OK=0
rm -rf ../rmscaffold/sft_adapter && cp -r "$BEST" ../rmscaffold/sft_adapter \
  && ( cd ../rmscaffold && CUDA_VISIBLE_DEVICES=0 bash train_rm.sh > train_rm.log 2>&1 ) \
  && RM_OK=1 || { mark "RM 训练/门禁失败 — 看 rmscaffold/train_rm.log；跳过 RM 系评测"; echo "RM" >> FAILURES; }

# ---- 3. RM 系评测（RM 成了才有意义）----
if [ "$RM_OK" = "1" ]; then
  cd ../rmscaffold
  soft "score_rm--smoke"  python score_rm.py --smoke  > score_smoke.log 2>&1
  soft "score_rm-full"    python score_rm.py          > score_full.log  2>&1
  soft "score_rm-pickBoN" python score_rm.py --pick-candidates resample/candidates_pro2_eval.jsonl --pick-labels resample/labels_pro2_eval.jsonl > score_pick.log 2>&1
  soft "bon_eval-base"    python bon_eval.py --policy base --items ../train_package/eval_items.jsonl > bon_base.log 2>&1
  soft "bon_eval-sft"     python bon_eval.py --policy sft  --items ../train_package/eval_items.jsonl > bon_sft.log  2>&1
  cd ../train_package
fi

# ---- 4. 静态 prefix 评测（只依赖 SFT）----
soft "eval_infer-sft" python eval_infer.py --model $ROOT/ms/Qwen/Qwen3.6-27B \
  --adapter out/qwen3.6-27b-hlsft --out eval_outputs_sft.jsonl > eval_sft.log 2>&1

# ---- 收尾 ----
if [ -s FAILURES ]; then
  mark "DONE WITH ERRORS — 失败清单: $(tr '\n' ' ' < FAILURES)"
  touch DONE_WITH_ERRORS
else
  mark "ALL DONE"
  touch ALL_DONE
fi
mark "拷回清单: train_package/out/ rmscaffold/out/ rmscaffold/sft_adapter/ rmscaffold/rm_eval_scores.json rmscaffold/bon_*.log rmscaffold/score_*.log eval_outputs_*.jsonl *.log FAILURES(若有)"
