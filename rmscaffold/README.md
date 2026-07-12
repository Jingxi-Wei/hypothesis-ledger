# RM scaffold v2 — audit能力产品化成 Pairwise Reward Model（等 pro2 数据到量再训）

> 状态：**scaffold v2（2026-07-07 重设计，全部代码本机验通）**。数据管线/训练配置/评测脚本齐且
> dry-run 过真数据；**现在不训**——干净对 29 条（闸 300），等 pro2 采集+posthoc audit+gen_pairs 到量。
> 执行顺序照 `HANDOFF.md`（给接手模型的 runbook，含精确命令/检查单/分诊表）。

## 为什么有这个（战略定位，2026-07-06 拍板）

- SFT 教「懂」，不保证 agent 行动「会干」（会审不一定会干 = 会，但**想不起来**）。真压进策略要 RL，
  但 RL 明确不做（太贵/太久/别拉长战线）。
- **RM = 对「想不起来」的工程回应**：把触发从模型内部（不可靠的自发回忆）挪到管线外部
  （每个决策点强制过一遍 judge）= 外置触发器。也是 SFT→RM→[RL] 完整路线的可信证明。
- **RM+BoN = 不跑 RL 的「审→干」第一段桥**：held-out 上 policy 采 N 个候选假设，RM 挑 vs
  random/first 挑，Δ 用同一套盲评 rubric 量。

## Pair 设计 v2（核心，2026-07-07 和用户逐条对齐后定稿）

**训练/使用分布必须一致**：RM 只会判它训练时见过的那种（prompt, response）。BoN 打分的地方是
ledger 派生的 propose input（= eval items 的 input），候选是新鲜的一行假设。v1 的"裸 issue + 轨迹
原文"两侧都失配 → v2 重造。三类对，全带 `pair_type/strength/rejected_source/grounded` 标签：

| pair_type | prompt | chosen | rejected | strength | 教什么 |
|---|---|---|---|---|---|
| `decision` | 解题卡的 propose input（与 SFT/eval 同一构造器 `export._walk`） | outcome 验证的解题假设（过 grounding 闸） | RULED OUT 里各条被否/判坏假设 | `verified` | 此局面下别再选被否方向 |
| `resample` | **raw-masked 轨迹前缀**（部署忠实上下文；`--context ledger` 为对照臂） | 同节点新采候选中 judge 判 correct+grounded 的 | 同节点判 wrong 的 | `judged` | **在真实 agent 语境给新鲜候选排序（=部署门控的真实任务）** |
| `issue_only` | 裸 ISSUE | 验证假设（且 grounding 在 issue 文本内） | 同 decision | `verified` | H1 型（读完 issue 提初始假设）排序 |

**信息对等三原则**（回应"好假设比坏假设多信息不公平"的讨论，结论=公平的账算在 prompt 上）：
1. **chosen 资格 = 被结局验证**（patch 真过了测试）。audit verdict 只是软信号——"推得好"的假设可能
   被 oracle 否掉，拿它当 chosen 会教 RM 偏好被否方向（v1 有此隐患，v2 修死）。
2. **grounding 闸**：chosen 引用的代码锚点必须出现在 prompt 里（`export._grounded`，与 eval 的
   hyp_grounded 同一规则）——对永远不会奖励"引用 prompt 外的具体细节"，幻觉换不来分。
3. **resample 对出生时间对称 + prompt=生成上下文本身**：同节点候选同时生成、同信息基础，配对内部
   零时间差；judge（看 gold 的特权判卷人）只出标签、特权信息不进对文本。**上下文默认 raw-masked
   （用户 2026-07-07 拍板）**：RM 真正要站岗的地方是**活 agent 循环的 propose 步门控**，那里的上下文
   就是 raw 对话历史、且部署时本来没有 oracle——所以"raw 前缀 + 掩掉全部 harness/oracle 注入"= 部署
   分布的忠实重建；pair 的 prompt 就是这个生成上下文，候选引用的一切都可在 prompt 内查证（"奖励
   prompt 外细节"的病根不存在）。ledger 版（对应静态 eval-items 世界）保留为 `--context ledger`
   对照臂。**诚实残留**：agent 后续发言可能复述被掩的方向（织在 agent 原话里掩不掉）→ 首次反馈之后
   的节点带 `post_feedback=true` 标签，报数前先切片看该子集是否异常。

**response 归一**：两侧都是一行假设陈述（`_norm_line`）；BoN 打分时从候选完整输出里抽
`HYPOTHESIS:` 行（抽不到 fallback 整段并计数）。杜绝"结构化格式=好"的表面捷径。

## 文件与运行顺序

```
rmscaffold/
  # 本机（零 proxy）
  prep_rm.py         # 自然对+重采对 -> rm_pairs.jsonl / dataset_info.json / rm_stats.json；<300 对闸
  build_rm_eval.py   # held-out 质检对（自然对复用 export_preference + 重采 eval 切片合并）
  # 本机（proxy 串行，batch 暂停时跑；--list-only 先看成本）
  gen_pairs.py       # 重采对：每节点 K 候选（gpt-5.5）+ 1 次 gold 锚定 judge；--selftest 离线验证
  # GPU 侧（自包含，不 import src/）
  rm_qlora.yaml      # LLaMA-Factory stage:rm 配置（Qwen3.6-27B QLoRA, cutoff 8192）
  train_rm.sh        # 一键：装依赖 -> 训练 -> check_template 门
  check_template.py  # ★VERIFY-ON-GPU：打分渲染必须 == LF 训练渲染（不过不许信任何分数）
  rm_lib.py          # value-head 加载（硬断言=权重必须等于文件）+ 打分（LF 模板优先、左截断）
  score_rm.py        # held-out 准确率 + 95% CI + pair_type/strength 切片 + length-bias 诊断
  bon_eval.py        # BoN：候选生成（关 thinking）-> RM 抽行打分 -> rm/random/first 三挑法出文件
```

顺序（细节和精确命令见 HANDOFF.md）：
1. 【proxy 槽】compress+audit_run 覆盖 pro2（含 holdout）→ gen_pairs（train+eval 两个 tag）
2. `python src/export.py --run-id pro2 --dataset pro --preference-only` → `python rmscaffold/prep_rm.py`
   —— 看闸：**≥300 才训**（打印会告诉你）
3. `python rmscaffold/build_rm_eval.py --run-id pro2` —— held-out 质检对
4. 传 `rmscaffold/` 上 GPU → `bash train_rm.sh`（末尾自动跑 check_template 门）
5. `python score_rm.py --smoke` → `python score_rm.py` —— **先看质检**：~50% = RM 没学到，别做 BoN
6. **主 BoN 数字（同分布，零 policy 加载）**：`python score_rm.py --pick-candidates
   resample/candidates_pro2_eval.jsonl --pick-labels resample/labels_pro2_eval.jsonl` —— held-out 节点上
   RM 挑的候选 judge-correct 率 vs 随机挑 vs 挑第一个，mixed 节点 Δ + CI = headline
7. （次级臂）`python bon_eval.py --n 8` （再 `--policy sft`）→ 本机 `python src/eval_grade.py --llm
   --outputs rmscaffold/bon_rm_vs_random_base.jsonl` —— 注意：eval items 是 ledger 世界 = 对 RM 训练
   分布是 OOD，报的时候注明这层失配

## 诚实边界（★写进任何汇报前先看这里）

1. **RM 天花板 = 标签来源的判断力**：verified 对 = 执行结果（铁），judged 对 = gpt-5.5 拿着 gold 的
   方向判断（没执行过）。报数按 strength 分开报，别混。claim 是"held-out 对上 chosen>rejected X%
   (CI)"，分 slice 报，**resample slice 才是 BoN 的预测器**。
2. **BoN 的 self-preference bias**：默认候选由 BASE 生成、SFT 系 RM 挑（交叉）；`--policy sft`
   是第二臂，两臂都报。
3. **BoN Δ 可能是 null**——null 也是干净负结果，照报，别预设必赢。桥是第一段不是全程："BoN 有 Δ"
   ≠ "等价于 RL"。
4. **协议纯度**：raw_leak 对自动丢；**r1 整个排除**（用户拍板，Verified 会新协议重跑）；不做卡级
   捞取（用户拍板）。
5. **gen_pairs 的 zero-correct 诊断**：>50% 节点采不出 correct 候选 = ledger 太瘦的信号——修法是
   加厚 ledger input（训练/eval 共用构造器，一起厚），不是给候选开侧信道。
6. **length-bias 诊断**（score_rm）：score↔长度相关性高、或"rejected 更长"子集准确率塌 = RM 学了
   长度不是内容，BoN 之前拦下。
7. **held-out 质检对依赖 posthoc audit**：pro2 的 holdout 实例审完 build_rm_eval 才有量；resample
   eval 切片要单独 `gen_pairs --tag eval`。

## 数据 schema v2

`export_preference()`（src/export.py，走 `_walk` 同一构造器）+ `gen_pairs.py`，每行：
```json
{"type": "preference", "pair_type": "decision|issue_only|resample", "instance_id": "...",
 "protocol": "sanitized|none|raw_leak", "strength": "verified|judged",
 "rejected_source": "oracle_refuted|verdict_wrong|verdict_weak|judge_wrong",
 "chosen_grounded": true, "rejected_grounded": false, "node": "H4",
 "prompt": "<决策点 input 或裸 issue>", "chosen": "<一行假设>", "rejected": "<一行假设>"}
```
prep_rm 转 LLaMA-Factory sharegpt-ranking（`hl_rm`）：conversations/chosen/rejected。

## GPU 侧已知坑（继承 train_package/GPU_SETUP.md + 本轮新增）

- 镜像可能没 bitsandbytes——train_rm.sh 分开检查各装各的；`template: qwen3`（0.9.5 有）
- **check_template 是门不是建议**：不过（llamafactory 不可 import → rm_lib 落到手写渲染）就别信分数
- rm_lib.score **左截断**（保 response 端；右截断会把打分位置切掉）；cutoff/max_len 8192
- 24G 流程测试卡：yaml cutoff 改 4096 **且** prep_rm --max-tokens 4096 重导（OOM 教训）
- value-head 加载有硬断言（权重必须等于文件），smoke 的非常数 spread 是第二道保险
