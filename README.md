# Hypothesis Ledger

**Manufacturing self-correction training data for LLM agents.**

Pretraining corpora are overwhelmingly "solved-in-one-pass" text. This project manufactures the scarce kind: trajectories where an agent proposes a wrong hypothesis, gets it refuted, changes direction, and fixes the bug — then turns every step of that process into supervision.

## How it works

1. **Collection (minimal intervention).** A strong agent ([mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), extended) solves real repair tasks (SWE-bench Pro / Verified, LiveCodeBench, Terminal-Bench). Tasks it can solve produce natural trajectories. When it fails, it first gets one round of sanitized test feedback; if it still fails, an oracle (which can see the gold patch but never reveals it) states what is wrong with the current hypothesis and a deliberately vague direction to investigate — forcing a genuine re-derivation instead of an answer copy.
2. **Hypothesis ledger.** The agent must explicitly declare one hypothesis per phase. Trajectories are cut into per-hypothesis cards: code edits and test results are kept verbatim, exploratory reads are compressed to summaries, and evidence is stored with temporal fidelity (what was known *before* each hypothesis). Context shrinks by roughly an order of magnitude and samples focus on the claim → evidence → conclusion chain.
3. **Post-hoc audit.** An LLM judge with access to the gold answer audits every hypothesis: was it justified *given the evidence available at the time*, what was wrong with it, and what should have been checked next. Failed trajectories therefore still produce supervision — every bad hypothesis becomes teaching material.
4. **Training views.** Each trajectory is exported into four sample types — **audit** (judge a hypothesis), **propose** (suggest the next hypothesis), **fix** (produce the repair), **decline** (admit insufficient evidence and say what to probe) — plus **preference pairs** for reward modeling (outcome-verified chosen/rejected at decision points, and same-context resampled candidates ranked by a judge with randomized presentation order).
5. **Leakage discipline.** Oracle text never enters training inputs or targets; it is rewritten into diagnoses derivable from the input alone (training distribution = inference distribution). Underivable samples are downgraded to decline samples. Three defense layers (generation-time redaction, post-hoc scanning, structural holdout walls) are enforced in code.

## Repository map

| Path | What it is |
|---|---|
| `src/` | Collection harness, ledger compression, audit, export, eval (rollout / grading / leak scans) |
| `rmscaffold/` | Reward-model side: preference-pair construction, RM training (QLoRA + value head), scoring, Best-of-N |
| `train_package/` | GPU-side training bundle: SFT configs, serving, one-command autopilot |
| `openspec/` | Design specs for the data pipeline |

## Status

Research code, actively evolving. A 27B QLoRA fine-tune on 1.6k manufactured samples already shows the target behaviors on held-out items (calibrated audits with zero false refutations, executable verification steps, redirect-after-refutation instead of hypothesis repetition, no format bleed on unrelated tasks). Dataset release and write-up in progress.

## Acknowledgements

Built on [SWE-bench](https://github.com/princeton-nlp/SWE-bench) / [SWE-bench Pro (MIT)](https://github.com/scaleapi/SWE-bench_Pro-os), [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), [Terminal-Bench](https://github.com/laude-institute/terminal-bench) and [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent).

## License

MIT
