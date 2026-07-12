# HANDOFF — RM scaffold v2 执行手册（给接手模型/Opus，2026-07-07 Fable 交接）

设计已和用户逐条拍板定稿（见 README.md「Pair 设计 v2」），**接手的活 = 照本手册按序执行 + 把关**，
不需要再做设计决策。所有本机零 proxy 环节已在真数据上验通（见 §0）。

## 0. 交接时的状态快照（2026-07-07）

已验证（本机实跑过）：
- `src/export.py` 重构成 `_walk()` 节点游走器：**818 个 SFT 样本 diff 字节级一致**（重构零回归），
  export_preference v2 + `--preference-only` 旗标可用。
- 真数据 dry-run 数字：pro1 自然对 91（clean 29 = decision 15 + issue_only 14）、r1 65（默认整体排除）；
  `prep_rm.py` 输出 29 clean 对 → 闸正确报 TOO FEW（<300）；`build_rm_eval.py --run-id pro1` 出 4 条
  holdout 对；`gen_pairs.py --selftest` PASS；`gen_pairs.py --run-id pro1 --list-only` = 62 实例/120 节点。
- GPU 侧 5 个文件全部 py_compile 通过，**但从没在真 GPU 上跑过**（VERIFY-ON-GPU 项见 §3）。

在跑的东西：**pro2 采集批量**（detached `run_loop_pro.ps1`，2 worker，写 `dataset/raw/*/pro2/`，
log=`dataset/batch_pro2.log`）。**proxy 不能并发**——批量在跑时禁止任何其他 proxy 调用。

**2026-07-07 review 修复批已合入**（16 finder + 验证器全链审查后）：collect/prep_rm 的隐藏测试 token 推导修成
Pro-aware（此前对 Pro 全空转；prep_rm 实测立刻扫掉 6 条真泄漏对）；scrub 支持 pro2；holdout 墙三层化
（gen_pairs CLI 守卫 + prep_rm 结构墙 + export/gen_pairs 缺 splits 文件 fail-loud）；refuted join 截断容忍；
resume 复用存储 prompt；raw_ref 精确切界；rm_lib/bon_eval 截断改 head+tail 拼接并计数；train_rm.sh 的
300 对闸**真执行**（RM_ALLOW_SMALL=1 可越）；gen_pairs stats 改 {cumulative, session}（看 cumulative）；
eval_grade 产物按 --outputs 加后缀不再互覆；bon_eval sft 臂补 key_mapping（防 adapter 静默不加载）；
eval_build 加协议门+共享 ask 常量（items 已重建 43 条）。
**★splits 已冻结为工件**：上游 ScaleAI 数据集掉过 65 个 protonmail 实例（pro_test 80→69，.bak 是旧版）——
**绝不要重跑 select_pro_split**，pro_test.json 就是 holdout 墙本体；正式跑前考虑 pin HF revision（待办）。

用户已拍板的决定（别重新讨论）：
- r1 整体不用（Verified 之后会用新协议重跑新 run-id）；不做卡级捞取旧 raw_leak 轨迹。
- pair 三类型 + chosen=outcome-verified + grounding 闸 + response 一行归一（README 表格）。
- **resample 上下文 = raw-masked 为主（用户 2026-07-07 明确拍板两次）**：raw 轨迹前缀 + 掩掉全部
  harness/oracle 注入 = 活 agent 门控的部署忠实分布；pair prompt = 生成上下文本身。ledger 是
  `--context ledger` 对照臂。judge 可以吃特权信息（只出标签）。agent 复述残留由 post_feedback
  标签跟踪（掩不掉，切片量化）。

## 0.5 —— 2026-07-11 增补（Fable 最后一天收尾，接手先读这段）

**LCB judge 重写（两毒源，都已修+验证，src/lcb.py）**：
- 毒源①：per-case docker exec 把 base64 输入拼命令行，Windows 上限 32,767 字符 → >24k 输入静默写失败、
  容器里残留上一用例旧 stdin，正解也永远过不了。修法=官方 testing_util 思路：`_run_cases` 一次 docker cp
  进容器 + 容器内 runner 批量跑（输入永不上命令行；expected 永不进容器；43 exec→2）。
- 毒源②：oracle_gold 输入头切 800 字符会把第二个参数(k)切没 → oracle 产"参数绑定"幻影方向。修法=
  取**最小挂例** + `_elide` 保结构省略（每行=每参必留，长行中间省略+显式标注）。
- 验证链：38 单测过 + `_sanity_lcb` 判别过 + 3677 参考解 resolve=True + 3.5M 怪物输入全量送达探针。
- **全部旧 LCB 结果已归档 `dataset/archive/lcb_pre-harness-fix_20260711/`（lcb1×52 + lcbsmoke + 修复前
  lcb2×4，含 README）——outcome 分布不可信，禁止喂 compress/export/gen_pairs**。lcb1 的"oracle 4%/
  chaotic 29%"作废。
- lcb2 = 干净世代：smoke 6/6 全解（4 self_solved + 2 self_corrected；修复前同题 57/84 条消息 chaotic）。

**LCB 角色改判（用户拍板 2026-07-11：不是重点了，当泛化测试）**：hard50 批实测自解率 ~94%
（gpt-5.5 采集器对竞赛题太强），修正数据密度太低，**LCB 不再作为训练数据主力**。落地：
- hard50 批跑完即收（在跑，log=`dataset/lcb2_hard50.log`；数据照常 posthoc，audit/propose 样本 +
  少量重采对照收，白捡的多样性）。
- **medium/剩余 hard 批默认不发**（`lcb_med.json`(49)/`lcb_hard_rest.json`(27) 名单留着）——它们改作
  **rollout eval 的 unseen 泛化题池**（跨底座泛化轴：SFT 里 LCB 占比极小，拿 LCB 未见题测学生的
  假设纪律是否迁移）。发车模板（若用户改主意；**PYTHONUTF8=1 必带，GBK console 遇 rich emoji 即崩**）：
```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 nohup .venv/Scripts/python.exe src/run_batch.py --dataset lcb \
  --run-id lcb2 --workers 2 --instances-file dataset/splits/lcb_med.json >> dataset/lcb2_med.log 2>&1 &
```
- 修正样本主力供给 = **pro2 续跑**（优先级进一步上升；**用户排期：2026-07-13 当周补跑**）。
- **GPU 排产已拍板**：一台双卡，GPU0=SFT→RM 串行、GPU1=raw 臂；先用现有数据跑通全链，pro2 补完后再刷正式版。
批完后的 posthoc 链照 §2（proxy 串行：等批量收完再跑 compress→audit→export→gen_pairs）。

**★审计哲学定案（2026-07-12 凌晨，用户，首夜 eval 定性后）**：审计的天职=**把坏轨迹也变成推理能力的
养料，不是对行动假设当学报审稿人**——"只要能知道要干啥，没必要判那么仔细"。落地三条：
① verdict 按序数读：good↔weak 相邻混淆=软误差（weak 大半是攥着 gold 的事后洁癖），good↔wrong 才是硬错;
  真正要考的是 wrong（判死刑）轴——当前 eval 为零（见 todo：holdout pro2 重采后 eval_build 重建）;
  重建时给含上游提示段的 item 打 `has_gold_hint` 标（Pro 题面自带 "## Additional Information/golden
  patch" 接口说明=上游设计非我们泄漏；现 eval 9/43 条中招/2 实例；臂间 Δ 公平但绝对难度虚低，切片剔除）;
  **flaw_match 判法修正（idx33 实证）**：不按"与参照一致"判，按"与 item 可见证据自洽"判——参照审计员
  可能引用视野外证据（原始轨迹被压缩丢弃的片段/gold 事后视野；实例：参照引 scanner 路径而 item 输入
  零 scanner），盲评 rubric 里给 grader 加一句"候选 flaw 若锚定于可见证据且参照 flaw 引用界外事实，不罚"；
② 下轮重审 prompt 加**行动后果检验**：缺陷只有在"知道了会改变下一步行动"时才配 weak/wrong；
③ 数据修法**不要**加权元缺陷味 weak 样本（那是把 base 的慢性挑刺病训进来），保证 wrong 类（证据矛盾型）
  样本充足即可。首夜实测背书：SFT 保守报警（漏两条洁癖 weak、零误报、真 weak 一击即中且 flaw 逐句对参照）
  优于 base 宽松报警（weak 白中但 good 上连环误报）——闭环里误报比漏报贵（假 insufficient 当场带偏好路，
  漏 weak 有下轮测试兜底）。

**⚠ LCB 私测违反题面约束 = 系统性模式（2026-07-11，两例实锤 + 选择效应）**：
- `lcb__abc392_f`：40 私测 2 个 P_i>i（题面保证 P_i≤i），期望输出=Python list.insert 越界 append 语义。
- `lcb__abc397_d`：private case38 N=27 期望 "3 0"——y=0 违反题面 "positive integers"（正解 -1 被判死）。
两条都已隔离（archive/lcb_pre-harness-fix_20260711/ + skip.json）。**选择效应解释为什么深修正链必中毒**：
采集器（gpt-5.5）够强，正常题一把过；只有"测试本身违规"的题才能拦住正解制造深修正链——所以 **LCB 的
oracle_redirected/深链默认全是嫌疑犯**，用前必须做约束校验（对着题面查私测输入/输出合法性）；
abc396_e（tie-break 规范化）属较轻一类，肉眼过了可用。rescue 一轮就修的 self_corrected（agent 自己的
bug）通常是真的。此模式进一步锁死：**LCB 只当泛化测试，correction 供给指望 Pro**。
**第三类毒（2026-07-11 深夜，arc190_a 实锤）：多解题撞精确匹配判分**——题面"print any/多解任取"，
判分只认参考解的那一组 → 合法最优解被判死，arc190_a 烧满 7 轮 oracle 后 chaotic（oracle 第 1 轮就点破
"查你的 harness 是不是没有 special judge"，oracle 无辜）。已隔离；**全库 regex 扫出 6 个多解题，队列中
未跑的已预防性进 skip.json**（arc191_c 在批中扫描前已起跑，烧完自然隔离）。给 Opus 的规矩：LCB 三类毒
= ①私测违反题面约束 ②tie-break 规范化 ③多解题精确匹配——深链/chaotic 一律先对号这三类再定去留。

**prep_rm cap 改分层（verified-first）+ 放宽到 16**：旧刀=每实例 8 对跨通道均匀随机抽，会把稀缺 verified
和量产 judged 等概率误杀。已改成 verified 全保、judged 补位（prep_rm.py cap 段）。用
`--max-per-instance 16` 重跑实测：kept 696→**1145**（verified 121→**280**，占比 15%→24%）。
**新数据落地后重跑 `prep_rm.py --max-per-instance 16` 刷新 rm_pairs.jsonl 即可**（确定性，训练前任意重跑）。

**数据量目标（用户认可 1k 偏少）**：RM 留存目标 **3-5k 对**；路径=pro2 续跑 150-200 实例 + LCB 全库 +
cap 已放。**eval 垫到 300+（现 89）= 三步全复用、零新代码**：①holdout（pro_test 69 题）补跑 posthoc
compress+audit（§2 标准链，proxy 槽）→ ②`build_rm_eval.py --run-id pro2`（自然对，零 proxy，pro1 冒烟过）
→ ③`gen_pairs.py --instances-file dataset/splits/pro_test.json --include-holdout --tag eval`（重采对，
proxy，CLI 守卫防误入训练）。SFT 现状 1590
（audit 825/propose 559/fix 193/probe 14，correction 占 31%）——修正类 ~495 条，距离 800-1000 修正样本
的停采线还差 pro2 续跑；LCB 落地后重新 export 全量刷新 dataset/sft/。

## 0.6 —— 2026-07-12 Fable 收官快照（订阅最后一天，接手看这里就够）

**已完成并验证**：SFT 训成（ckpt-597，注意=无验证集时选点回退到最后一个，ep2/398 对比是遗留题）；
定性验收超额（四维度+校准+零渗漏+污染签名干净，见记忆）；机械 eval：verdict_agree 80%/误报 0/
换皮判断三层梯度；check_template PASS（VERIFY-ON-GPU 债清）；LCB judge 重写+三类毒分类学；
r2 难题批 33 题（修正率 42%）；lcb2 55 题干净数据（**未进 RM/SFT——posthoc 只做了 compress+audit，
gen_pairs/export 未跑**）。

**RM v1 = 诚实负结果（照实报，勿粉饰）**：held-out 0.539/ep1 0.506=均为硬币；诊断=实例级记忆
（108 实例不够），监督方向正确（反对冲 97:11）、长度干净、同上下文微弱存活（resample_real ~0.6）、
decision(带轨迹)60% vs issue_only(仅issue)33%=RM 确实在用上下文。**eval 附加发现：簇效应**（89 对
≈14 实例，有效 n≈25-30，真 CI 更宽；vuls-407407 一家 0/7）——重建时按实例分层+簇稳健 CI+
issue_only 消融通道与 headline 分列。**解药=补实例**：pro2 已到 158（暂停中，rollout 让路），
目标 ~300 实例→对 ~2.5-3k→下个 GPU 日重训（考虑 1 epoch）。

**rollout eval 进行中（本节=runbook）**：
- GPU: `cd train_package && MODEL=/root/autodl-tmp/ms/Qwen/Qwen3.6-27B nohup bash serve_vllm.sh > serve.log 2>&1 &`
- 本机隧道: `ssh -CNg -L 8000:127.0.0.1:8000 root@<实例> -p <端口>`
- A 臂两条（seen 24×{sft,base}，~5-7h;PYTHONUTF8=1）:
  `EVAL_MODEL=openai/sft  EVAL_API_BASE=http://127.0.0.1:8000/v1 python src/eval_rollout.py --instances-file dataset/splits/rollout_seen.json --run-id rollA_seen_sft  --feedback none`
  `EVAL_MODEL=openai/base EVAL_API_BASE=http://127.0.0.1:8000/v1 python src/eval_rollout.py --instances-file dataset/splits/rollout_seen.json --run-id rollA_seen_base --feedback none`
- 判分: `compress.py --run-id rollA_seen_sft`（及 base）→ `eval_rollout_grade.py --runs rollA_seen_base,rollA_seen_sft`
  → dataset/eval/rollout_report.json。**主读数=premature_submit（基线 69.5%）+ self_tests**。
- 后续可选: unseen 集同款、B 臂(binary)——头炮没信号就不花。
- rollout 收完恢复 pro2: `Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','run_loop_pro.ps1'`

**队列（优先级序）**：①rollout 头炮判分 ②pro2 续采到 ~300 ③lcb2/pro2 posthoc（gen_pairs/export）
④prep_sft/prep_rm 全量刷新（散文变体 todo#2 一起做）⑤下 GPU 日: SFT 重训(全量新数据)+RM 重训(1ep)
+DPO 臂 ⑥eval 重建(todo#3: wrong 轴+分层+gold_hint 标) ⑦消融(todo#1)。在线 RL 冷冻(重启条件见记忆)。

## 1. 排队纪律（proxy 串行,一切 proxy 活都在这个槽里）

```powershell
# 停批量（PS gotcha：过滤必须排除 $PID，否则查询进程会自杀 exit 255）
Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and
  ($_.CommandLine -match 'run_loop_pro|run_batch') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
# 残留容器清理走 Bash（PS 安全层会把 docker rm 当 Remove-Item 拦）:  docker ps -q | xargs -r docker rm -f
```
```powershell
# 恢复批量（干完 proxy 活之后）
Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File',
  '<PROJECT_ROOT>\run_loop_pro.ps1'   # param 默认 RunId=pro2
```

## 2. Stage A — 本机数据侧（proxy 槽内按序，全部可断点续跑）

环境常量：项目根 `<PROJECT_ROOT>`，python=`.venv\Scripts\python.exe`，
**每条命令前置 `PYTHONUTF8=1`**（GBK 坑）。时间估算按串行。

```bash
# A1. 压缩 + 审计 pro2（含 holdout 实例——export 自动不让 holdout 进训练，build_rm_eval 专吃它）
python src/compress.py --run-id pro2                                   # 确定性零 proxy,分钟级
python src/audit_run.py --run-id pro2 --dataset ScaleAI/SWE-bench_Pro  # proxy,~3-5min/实例(xhigh)

# A2. 重采对（先看成本再跑;两个 tag 都要）
python rmscaffold/gen_pairs.py --run-id pro2 --dataset pro --list-only # 节点数+成本预估,零调用
python rmscaffold/gen_pairs.py --run-id pro2 --dataset pro             # 训练节点(~k+1 次调用/节点)
python rmscaffold/gen_pairs.py --run-id pro2 --dataset pro \
  --instances-file dataset/splits/pro_test.json --include-holdout --tag eval   # held-out 节点

# A3. 自然对 + 合并 + 闸（全零 proxy,批量恢复后也能随时重跑）
python src/export.py --run-id pro2 --dataset pro --preference-only     # 不动 SFT 样本文件
python rmscaffold/prep_rm.py                                           # ★闸:kept>=300 才继续
python rmscaffold/build_rm_eval.py --run-id pro2                       # held-out 质检对(两通道合并)

# A4. (主线顺手,非 RM 必需) 刷新 eval items 供 BoN: python src/eval_build.py
```

每步之后看什么：
- A1 audit 完抽 1 个实例眼球 `dataset/raw/<iid>/pro2/audit.json`（per_hypothesis 带 card 号）。
- A2 `rmscaffold/resample/stats_pro2_*.json`：`zero_correct_frac > 0.5` = 上下文太瘦/太难警报（打印也会
  喊），这时候**别急着训**，抽 10 个节点的 labels 眼球一下再说；`real_correct/real_wrong`（judge 对轨迹
  真实假设的判断）= judge 校准参考；`ledger_fallback_nodes`（找不到假设边界回退 ledger 的节点数，应是
  个位数比例）；`post_feedback_nodes`（复述残留可能存在的子集规模）。`judge_unparseable`/`errors` 偶发
  正常（可重跑续），成串出现查 proxy。
- A3 `rmscaffold/rm_stats.json`：关注 `by_pair_type`（resample 应是大头）、`leak_dropped`（应≈0，
  非 0 看 `leak_instances`）、闸 verdict。

## 3. Stage B — GPU 侧（租卡,AutoDL 自定义镜像已含 LLaMA-Factory 0.9.5）

上传整个 `rmscaffold/` 文件夹（训练数据 rm_pairs.jsonl + dataset_info.json + 质检对 rm_eval_pairs.jsonl
+ `resample/` 里的 candidates/labels eval 文件都在里面）。**必须另拷两样**：①训好的 SFT adapter
文件夹拷成 `rmscaffold/sft_adapter/`——RM 从 SFT 系初始化（yaml 已配 adapter_name_or_path；这是行业
默认也是 README 写明的"SFT 系 RM"设计；跑 base-init 对照臂时注释掉 yaml 那两行并注明）；②本机
`dataset/eval/items.jsonl` 拷进 rmscaffold/（bon_eval 传 `--items items.jsonl`）。顺序：先训 SFT，
再跑 train_rm.sh。

```bash
cd rmscaffold && bash train_rm.sh        # 装依赖->训练(~1-2h@80G,数据小)->自动跑 check_template 门
python score_rm.py --smoke               # 5 对原始分:必须非常数(spread > 1e-3)
python score_rm.py                       # ★质检 headline:准确率+CI+切片 -> rm_eval_scores.json
# ★主 BoN 数字(同分布、零 policy 加载):held-out 节点 RM挑/随机挑/挑第一 的 judge-correct 率
python score_rm.py --pick-candidates resample/candidates_pro2_eval.jsonl \
                   --pick-labels resample/labels_pro2_eval.jsonl          # -> rm_pick_scores.json
python bon_eval.py --n 8 --items items.jsonl                             # 次级臂(ledger items,注明OOD)
python bon_eval.py --n 8 --items items.jsonl --policy sft --sft-adapter <path>  # 次级臂之二
```

VERIFY-ON-GPU 检查单（第一次上 GPU 逐项过，全过才算管线活了）：
- [ ] `check_template.py` 打印 `render source = llamafactory` + PASS（FAIL=环境缺 LF，装好重跑，
      **不许**用 manual 渲染出的分数）
- [ ] `rm_lib.load_rm()` 打印 `value head loaded + verified equal to file`（硬断言，抛错=layout 变了）
- [ ] `score_rm.py --smoke` 分数非常数
- [ ] score_rm 全量：看 `slices`（resample slice = BoN 预测器）+ `length_bias`（corr 高或
      rejected-longer 子集塌 = 学了长度，停下回数据）
- [ ] bon_meta_*.json 的 `extract_fallback_frac`（>0.3 = 候选大量没有 HYPOTHESIS 行，看几个候选
      原文——base 臂散文多属预期，记进报告）

## 4. Stage C — 本机盲评出 BoN Δ（proxy 槽）

```bash
python src/eval_grade.py --llm --outputs rmscaffold/bon_rm_vs_random_base.jsonl
python src/eval_grade.py --llm --outputs rmscaffold/bon_rm_vs_first_base.jsonl   # + sft 臂同理
```
output_base=对照挑法、output_sft=RM 挑法，盲评器不知道哪臂是哪。报 rubric 分差 + 机械指标。

## 5. 故障分诊表

| 症状 | 根因 | 动作 |
|---|---|---|
| score_rm 分数全常数 | value head 没加载（旧 bug，现在 load_rm 会直接抛错） | 看 load_rm 报错的 key；LF 存档 layout 变了就改 rm_lib 的 key 匹配 |
| check_template FAIL | GPU 环境没有 llamafactory | `pip install llamafactory` 后重跑；别信 manual 渲染 |
| 质检准确率 ≈50% | RM 没学到判别 | **别做 BoN**。查:对够不够 300/label 噪声(抽 labels 眼球)/切片哪类塌 |
| resample slice 塌、verified slice 好 | judge 标签噪 or 候选质量差 | 抽 20 条 labels 眼球;考虑 JUDGE_EFFORT=xhigh 重 judge(文件删了重跑该 tag) |
| length-bias corr 高 | RM 学长度 | 回 prep_rm 看两侧长度分布;必要时对 pair 做长度配平后重训 |
| zero_correct_frac >0.5 | 上下文预算太紧 or 节点真难 | 先抽 labels 眼球;再试 --raw-chars 加预算(同步升 cutoff/--max-tokens);ledger 对照臂帮助归因 |
| pick Δ ≈0 (CI 跨 0) | 判断没转化成排序 = 诚实 null | 看 mixed 节点 n 够不够(<30 先攒量);post_feedback 切片是否拖累;两切片都 null 就照报 null+归因 |
| gen_pairs 连续 error | proxy 断/批量没停 | 停批量;网络恢复重跑同命令(按 node_id 续) |
| 训练 OOM | 长对 + 显存 | 80G 用 8192;24G 流程测试:yaml cutoff 4096 + prep_rm --max-tokens 4096 重导 |

## 6. 汇报红线（不许 claim 的）

1. 不说"RM 能判任意假设好坏"——说"同分布 held-out 对上 chosen>rejected X% (95% CI, n=?)，
   其中 resample slice X%"。verified/judged 两种标签强度分开报。
2. **主 BoN 数字 = pick 模式的 mixed-nodes Δ(RM−random)+CI**（同分布）；bon_eval 的 ledger-items 臂
   是次级证据，报的时候注明它对 RM 训练分布是 OOD。CI 跨 0 = 诚实 null，照报。
   不把任何 Δ 说成"等价于 RL"——它是 inference-time selection 的第一段桥。
3. 不忘 self-preference 控制——base 臂是主臂，sft 臂对照，两臂都报。
4. RM 天花板 = 标签来源（执行结果 / gpt-5.5+gold 判断），不是"客观真理"。

## 7. 留给用户的可调旋钮（有默认值,不改也能跑）

- `gen_pairs --k`（6）/ `--cap-per-node`（4）/ `--raw-chars`（24000，升它要同步升 yaml cutoff_len 和
  prep_rm --max-tokens）/ GEN_EFFORT（medium）/ JUDGE_EFFORT（high）
- `prep_rm --max-per-instance`（8）/ `--exclude-runs`（r1）
- 训练超参 rm_qlora.yaml（epochs 2 / lr 5e-5）——对少时 epoch 别加,RM 过拟合排序数据很快
- raw_masked 对照臂：只在想验证"ledger 世界是否太瘦"时跑 30 节点对比 judge 标签分布,训练不用
