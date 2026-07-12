# 租 GPU 跑 SFT —— 一步步指南

## 0. 结论先行
- **租 Linux,不要 Windows**(QLoRA 的 bitsandbytes / flash-attn 在 Windows 上巨坑;所有 GPU 云都是 Linux)。
- **流程测试**:单卡 **24G(RTX 4090)** + `Qwen/Qwen3-8B`,验证整条训练跑通。AutoDL ≈¥1.5-2/hr,1-2 小时够。
- **正式跑**:单卡 **A100-80G 或 H100-80G** + `Qwen/Qwen3.6-27B` QLoRA。AutoDL A100-80G ≈¥6.68/hr。
- **`train_package/` 整个文件夹传上去,`bash train.sh` 就开跑**,代码基本不用改(下面列出唯一可能要改的两处)。

---

## 1. 租实例(以 AutoDL 为例)
1. 新建实例 → 选 **RTX 4090(24G)** 先做流程测试。
2. 镜像选 **PyTorch 2.x / CUDA 12.x / Python 3.10+**(基础镜像自带 conda + torch)。
3. 开机进 JupyterLab 或 SSH。

## 2. 传数据
把本机 `train_package/` 整个文件夹传上去(AutoDL 网页有上传,或 `scp -r train_package root@<ip>:<port>`)。
里面有:`train_sharegpt.jsonl`(数据)、`dataset_info.json`、`qwen_qlora.yaml`、`train.sh`。

## 3. 开跑(流程测试)
```bash
cd train_package
# ★中国区强烈建议用 ModelScope 下模型(比 HF 镜像快且稳),二选一:
export USE_MODELSCOPE_HUB=1          # LLaMA-Factory 走 ModelScope 下 Qwen(推荐)
# 或者用 HF 镜像(train.sh 已默认设了 HF_ENDPOINT=https://hf-mirror.com)

bash train.sh
```
`train.sh` 会:①装 LLaMA-Factory ②打印数据行数 ③`llamafactory-cli train qwen_qlora.yaml`(模型首次自动下载 ~16GB)。
**流程测试通过标志**:模型下载完 → 训练跑满 3 epoch → LoRA adapter 落在 `out/qwen3-8b-hlsft/`,`out/` 里有 loss 曲线。

## 3b. 常见环境坑(AutoDL 镜像自带的包不一定匹配)
- **`ModuleNotFoundError: No module named 'bitsandbytes'`**:镜像自带 llamafactory 但没 bitsandbytes(QLoRA 4-bit 必需)。`pip install -U bitsandbytes`。(train.sh 已分开检测,新版会自动装。)
- **`OSError: libcudart.so.13: cannot open shared object file`**(import torchaudio 时):镜像的 torchaudio 按 CUDA 13 编译,torch 却是 CUDA 12=不匹配;文本 SFT 用不到 torchaudio,但 llamafactory 0.9.5 会 import 它。修=按 torch 重装匹配版:
  ```bash
  TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  CU_TAG=cu$(python -c "import torch; print(torch.version.cuda.replace('.',''))")
  pip install --force-reinstall --no-deps "torchaudio==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$CU_TAG"
  ```
  装不到就 `pip uninstall -y torchaudio`(用不上,多数能直接过)。
- **`torch.OutOfMemoryError: CUDA out of memory` 训练中途(不是一开始)**:24G 卡上 `cutoff_len: 8192` 会在遇到长样本时爆(实测 step ~13 崩)。**24G 流程测试必须 `cutoff_len: 4096`**(yaml 已默认 4096;只有 80G 正式跑才 raise 到 8192)。另设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(train.sh 已设)回收碎片显存。**模型崩在训练阶段=已下载缓存,重跑不重下**。
- **`Some modules are dispatched on the CPU or the disk` / bnb 显存不足(跑 probe_compare.py 时)**:训练还在占 GPU,单卡塞不下"训练中的模型"+"probe 再加载一份 8B"。**probe/eval 必须在训练结束后跑**(GPU 腾出来);训练中途只看 `tail train.log` 的 loss,别同时 probe。
- **`[Errno 99] Cannot assign requested address` / HF 连不上(跑 probe_compare.py 时)**:脚本默认按 HF id 找 base 模型,但中国区/离线机连不上 huggingface.co。模型训练时走 ModelScope 已下到本地 → **`--model` 指本地缓存目录**(训练日志里的加载路径,通常 `/root/autodl-tmp/ms/Qwen/Qwen3-8B`),别用 HF id:`python probe_compare.py --model /root/autodl-tmp/ms/Qwen/Qwen3-8B --adapter out/qwen3-8b-hlsft`。半程可指 `--adapter out/qwen3-8b-hlsft/checkpoint-200`。

## 4. 唯一可能要改的两处代码
1. **template**:若 LLaMA-Factory 版本没有 `qwen3` 模板,把 `qwen_qlora.yaml` 里 `template: qwen3` 改成 `template: qwen`。
2. **正式跑换大模型**:编辑 `qwen_qlora.yaml`:
   - `model_name_or_path: Qwen/Qwen3.6-27B`(先去 HF / ModelScope **确认这个 id 存在、拼写对**)
   - 换 A100/H100-80G 卡
   - `cutoff_len` 已默认 **8192**(覆盖 ~99%;correction/fix 的 input 现在带代码=EDITS+源码快照,4096 会截 ~7%)。剩 ~0.3% 是 go 巨型 patch 的 fix 样本,任何 cutoff 都超,正式跑前在 export 里 cap;80G 卡放得下 8192。
   - Qwen3 思考模式默认要关(我们 target 是直接输出、无 `<think>`);`qwen3` 模板配 QLoRA SFT 默认不加 thinking,若产出带 `<think>` 再显式关。

## 5. 数据刷新(正式跑前)
`train_package/train_sharegpt.jsonl` 是当前快照(**1444 条** = audit 737 + propose 488 + fix 219,含 Verified r1 + Pro pro1,会随采集增长)。
⚠️ 这是**流程测试快照**:78 条仍带 test-leak 措辞(受影响实例的 audit 还没重审),正式跑前必须先跑完 re-audit 再重导。
正式跑前在本机重新生成最新合并数据再覆盖上传(**四步一个都不能少,漏 rewrite_oracle 会把机制引用漏进去**):
```bash
# 本机项目根(采集在后台持续产数据)
python src/compress.py      --run-id r1
python src/compress.py      --run-id pro1
python src/rewrite_oracle.py --run-id r1                    # 中和 audit prose 里的 oracle/reviewer/test 引用
python src/rewrite_oracle.py --run-id pro1
python src/export.py        --run-id r1   --dataset verified
python src/export.py        --run-id pro1 --dataset pro     # Pro test split 自动排除,不会污染
python src/prep_sft.py                                      # 合并所有 run -> dataset/sft/train_sharegpt.jsonl
cp dataset/sft/train_sharegpt.jsonl train_package/          # 覆盖后重新上传
```

---

## 6. 评测(eval)——同一次租卡里跑完
**这个作品的真正交付 = held-out 过程指标 Δ(base vs SFT,同一套 items、同一个盲评 grader)。**
eval harness 已建成(2026-07-04),分三步,GPU 上只跑第 2 步:
```bash
# [GPU] 训练完、显存空出来之后(eval_items.jsonl 已在本包里):
python eval_infer.py --model /root/autodl-tmp/ms/<base模型目录> --adapter out/<adapter目录>
# 产出 eval_outputs.jsonl(base/SFT 双臂,同一次加载、同一解码参数;可断点续跑)
# 把 eval_outputs.jsonl 拷回本机 dataset/eval/,然后本机:
#   python src/eval_grade.py            # 机械指标(免费):verdict 一致率/premature_guess/premature_refute/CHECK率/grounding
#   python src/eval_grade.py --llm      # + 盲评 rubric(gpt5.5,要 proxy 空闲=采集 batch 停着时跑)
```
诚实红线不变:claim 只写「held-out 过程指标 Δ」,**不 claim 解题率**;items 来自 pro_test(训练自动排除);两臂同 items 同 grader,grader 不知道臂别。eval 集会随采集增长(现在 ~22 实例/120 items → 全量后 ~80 实例),**正式数字前在本机重跑 `python src/eval_build.py` 刷新 items 再上传**。

## 数量参考(要多少数据)
- **流程测试**:现在 1444 条就够,今天就能跑。
- **正式一版**:~2-3k 样本(≈400-500 实例)就是合理的 SFT 集;Pro ~5 样本/实例 → 采 ~500 实例约 2-3 天。**不必凑满 5k**,也不必跑完全部 731。self_solved / self_corrected 同样有用(它们本身就是「test 报错→转向」的核心模式),不是只有 oracle_redirected 才算数。
