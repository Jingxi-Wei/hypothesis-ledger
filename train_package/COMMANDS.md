# 全部命令清单（环境已存成镜像版，2026-07-04）

> 前提：昨天的环境（PyTorch + LLaMA-Factory 0.9.5 + bitsandbytes + torchaudio 修复）已保存为 **AutoDL 自定义镜像**。
> ⚠️ **镜像只保存系统盘（pip/conda 环境）；`/root/autodl-tmp` 数据盘不跟镜像走** —— 新实例上：环境即用，但 **zip 要重传、模型要重下**。

按顺序：A 本机刷数据 → B 租卡+上传 → C 训练(27B) → D 评测推理(同一张卡) → E 拷回+本机出 Δ → 排错。

---

## A. 本机刷新数据 + 打包（正式跑前做；pilot 直接用现成 zip 可跳过）
> 采集 batch 还在后台跑。正式数字版必须先刷全量；**顺序一步不能少**（scrub 必须在 compress 后 export 前）。
```powershell
cd <PROJECT_ROOT>
$env:PYTHONUTF8='1'
# 若新采实例还没 audit：先停 batch，跑 .venv\Scripts\python.exe src\reaudit.py --all （2-worker，~4min/题）
.venv\Scripts\python.exe src\compress.py --run-id r1
.venv\Scripts\python.exe src\compress.py --run-id pro1
.venv\Scripts\python.exe src\compress.py --run-id pro2
.venv\Scripts\python.exe src\compress.py --run-id lcb2
# ⚠ lcb__abc392_f 修正链标签是反的（LCB 私测越轨，rmscaffold/HANDOFF §0.5）——export 前把
#   dataset\raw\lcb__abc392_f\lcb2 整个移到 dataset\archive\ 下隔离
.venv\Scripts\python.exe src\rewrite_oracle.py --run-id r1     # 旧 prompt 时代的存量清洗；pro2/lcb2 是新 prompt，不用跑
.venv\Scripts\python.exe src\rewrite_oracle.py --run-id pro1
.venv\Scripts\python.exe src\scrub_f2p.py
.venv\Scripts\python.exe src\fetch_code.py --run-id r1   --dataset verified   # fix 样本的代码上下文（幂等，只补缺的；要 Docker）
.venv\Scripts\python.exe src\fetch_code.py --run-id pro1 --dataset pro
.venv\Scripts\python.exe src\fetch_code.py --run-id pro2 --dataset pro
.venv\Scripts\python.exe src\export.py --run-id r1   --dataset verified
.venv\Scripts\python.exe src\export.py --run-id pro1 --dataset pro
.venv\Scripts\python.exe src\export.py --run-id pro2 --dataset pro
.venv\Scripts\python.exe src\export.py --run-id lcb2 --dataset lcb            # LCB 无 repo：不用 fetch_code，无 fix 样本
.venv\Scripts\python.exe src\prep_sft.py                       # 自带 --max-tokens 8192 闸：超长样本丢弃，零截断
.venv\Scripts\python.exe src\eval_build.py                     # 刷新 held-out eval items（随采集增长）
Copy-Item dataset\sft\train_sharegpt.jsonl train_package\train_sharegpt.jsonl -Force
Copy-Item dataset\eval\items.jsonl         train_package\eval_items.jsonl     -Force
Compress-Archive -Path train_package\* -DestinationPath train_package.zip -Force
```
> held-out（pro_test 80 题）由 export.py 自动排除训练、eval_build 只用它们 —— 墙是死的，重跑不会污染。

## B. 租卡 + 上传
1. AutoDL 新建实例：卡选 **A100-80G 或 H100-80G**；镜像选 **「我的镜像」→ 昨天保存的那个**（环境零配置）。
2. JupyterLab 传 `train_package.zip` 到 `/root/autodl-tmp`，Terminal 里：
```bash
cd /root/autodl-tmp && unzip -o train_package.zip -d train_package && cd train_package
source /etc/network_turbo                       # 学术加速，每开新 Terminal 跑一次
export USE_MODELSCOPE_HUB=1 HF_HOME=/root/autodl-tmp/hf MODELSCOPE_CACHE=/root/autodl-tmp/ms
llamafactory-cli version                        # 镜像自带，应直接出版本号
# ★A800 主机是 CUDA 13.0 驱动——驱动向后兼容，镜像的 cu12x torch 照跑。开跑前 10 秒自检：
python -c "import torch, bitsandbytes; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
# 期望：打印 torch 版本 + avail True + 'A800'。若 avail=False（老 torch 撞太新驱动，少见）见排错重装 torch。
```

## C. 训练 27B（模型首次自动下载 ~55GB，走 ModelScope）
> **双卡布局（2026-07-11）**：GPU0 = 本节 SFT → 完了拷 adapter 训 RM（rmscaffold/COMMANDS.md §C）；
> GPU1 = raw 基线臂（`CUDA_VISIBLE_DEVICES=1 llamafactory-cli train qwen27b_qlora_raw.yaml`）。
> RM 不能和 SFT 真并行——RM 从 SFT adapter 初始化（拍板设计），依赖顺序死的。
>
> **★零看管一键链 v2（推荐；2026-07-11 用户拍板：raw 本轮不跑=单卡够；失败不硬停）**：
> 两个 zip 解压后一条命令串完 SFT → 选优 ckpt → RM → score 三连 → bon_eval 双臂 → eval_infer。
> SFT 失败才致命；RM 失败只跳过 RM 系评测；单个评测失败记入 FAILURES 继续。
> 收尾看标记：ALL_DONE（全成）/ DONE_WITH_ERRORS（看 FAILURES）/ FAILED（SFT 死了）。
> ```bash
> cd /root/autodl-tmp/train_package && nohup bash autopilot.sh > autopilot.out 2>&1 &
> tail -f autopilot.log
> ```
> raw 臂 yaml 保留在包里，想补跑随时 `CUDA_VISIBLE_DEVICES=0 llamafactory-cli train qwen27b_qlora_raw.yaml`。
> jlens（机理案例研究）也是 GPU 活但不进自动链——要人工挑例子，主链跑完卡还在时手动做。
```bash
# 27B 专用配置已就绪（qwen27b_qlora.yaml：cutoff 8192 / save per-epoch / out/qwen3.6-27b-hlsft），零编辑：
nohup env USE_MODELSCOPE_HUB=1 HF_HOME=/root/autodl-tmp/hf MODELSCOPE_CACHE=/root/autodl-tmp/ms \
  llamafactory-cli train qwen27b_qlora.yaml > train_27b.log 2>&1 &
tail -f train_27b.log
```
**通过标志**：loss 从 ~2 往下走、跑满 3 epoch、`out/qwen3.6-27b-hlsft/` 下有 **checkpoint-185 / -370 / -555**（每 epoch 一个，eval 选最好的）。预估 ~25-35s/it × 552 步 ≈ **4-5.5h**；峰值显存 ~40-60G。

## D. 评测推理（训练完、同一张卡上，~1-2.5h）
```bash
# 双臂（base vs SFT）一次加载、同参数生成；可断点续跑
python eval_infer.py \
  --model /root/autodl-tmp/ms/Qwen/Qwen3.6-27B \
  --adapter out/qwen3.6-27b-hlsft

# 想按 epoch 选点：实际 checkpoint 是 185 / 370 / 555；下面都是全量 eval
python eval_infer.py \
  --model /root/autodl-tmp/ms/Qwen/Qwen3.6-27B \
  --adapter out/qwen3.6-27b-hlsft/checkpoint-185 \
  --out eval_outputs_ep1_185.jsonl
python eval_infer.py \
  --model /root/autodl-tmp/ms/Qwen/Qwen3.6-27B \
  --adapter out/qwen3.6-27b-hlsft/checkpoint-370 \
  --out eval_outputs_ep2_370.jsonl
python eval_infer.py \
  --model /root/autodl-tmp/ms/Qwen/Qwen3.6-27B \
  --adapter out/qwen3.6-27b-hlsft/checkpoint-555 \
  --out eval_outputs_ep3_555.jsonl
```
产出 `eval_outputs.jsonl` —— 从 JupyterLab 下载回本机。**跑完就可以关实例了（先确认文件下载完）。**

## E. 本机出 Δ（base vs SFT）
```powershell
# eval_outputs.jsonl 放到 dataset\eval\ 下
.venv\Scripts\python.exe src\eval_grade.py                 # 机械指标：免费、即刻
.venv\Scripts\python.exe src\eval_grade.py --llm           # + 盲评 rubric（gpt5.5；要 proxy 空闲 => 先停采集 batch）
```
指标：verdict 一致率 / **premature_refute**（好假设硬挑刺）/ **premature_guess**（证据不足硬猜）/ CHECK 操作性 / grounding + rubric 均分。
**claim 红线：只写 held-out 过程指标 Δ，不 claim 解题率。**

---

## 排错
```bash
# ★probe_compare / eval_infer 连不上 HF（Errno 99）→ --model 用本地缓存目录，别用 HF id
python probe_compare.py --model /root/autodl-tmp/ms/Qwen/Qwen3.6-27B --adapter out/qwen3.6-27b-hlsft

# OOM（不该发生：80G@8192 余量足）→ 确认 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（train.sh 已设，
# 手跑 llamafactory-cli 时要自己 export）；仍炸 → cutoff 降 6144 再试
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# template 'qwen3' 不存在 → sed -i 's#template: qwen3#template: qwen#' qwen27b_qlora.yaml
# 模型下载卡住 → source /etc/network_turbo && export USE_MODELSCOPE_HUB=1
# 磁盘满 → 确认缓存在数据盘：echo $HF_HOME $MODELSCOPE_CACHE; df -h
# 数据格式自检 → head -1 train_sharegpt.jsonl | python -m json.tool

# torch.cuda.is_available()=False（A800 CUDA13 驱动 + 镜像 torch 太老，少见）→ 装匹配 host 的 torch
#   （AutoDL 学术加速下）pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
#   然后重装匹配的 torchaudio（见下）或直接 pip uninstall -y torchaudio（文本 SFT 用不到）

# —— 以下只有「不用保存的镜像、从裸 PyTorch 镜像起」才需要 ——
pip install "llamafactory[torch,metrics]" && pip install -U bitsandbytes
# torchaudio libcudart 不匹配 → pip uninstall -y torchaudio（文本 SFT 用不到）
```

> 包内清单：`train_sharegpt.jsonl`(训练数据) `dataset_info.json` `qwen_qlora.yaml`(8B 流程测试) `qwen27b_qlora.yaml`(正式)
> `train.sh`(8B 一键) `eval_items.jsonl` `eval_infer.py` `probes.jsonl` `probe_compare.py` `GPU_SETUP.md` `COMMANDS.md`
