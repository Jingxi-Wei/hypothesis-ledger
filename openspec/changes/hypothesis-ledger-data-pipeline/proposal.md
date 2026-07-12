## Why

Left alone, a coding agent generates bad supervision. The repo's earlier **oracle-free Elenchus experiments** showed that without an oracle the agent (1) proposes **weak hypotheses**, (2) after editing **self-confirms its hypothesis on the test result alone** — not checking whether the prior state was already correct or whether the edit is what fixed it, and (3) lets **band-aid / proxy patches** pass as "solved". These are exactly the failure modes worth training a model to catch — and an oracle is the only thing that can supply the corrective *direction* the agent cannot generate for itself.

This change builds a **simple** data pipeline: a thin mini-SWE-agent that states a hypothesis before acting and explores on its own; on failure it sees the test, and if still failing an oracle (reading gold) tells it **where it is wrong and the direction — never the answer**. The trajectories become a runtime Hypothesis Ledger, a posthoc audit, and prefix-level audit/redirect training samples. A later SFT change trains on them.

Scope is the **data pipeline only** (plan Day 1–3). Model training and eval are follow-up changes. Timeline is a goal, not a deadline.

## What Changes

- A thin **collection driver**: a vanilla mini-SWE-agent + one prompt change (state a hypothesis, then act, explore yourself) + a 3-stage loop — explore → submit → on-fail show test (self-rescue) → on-fail again an **oracle direction** (where-wrong + direction, not the answer). **No Elenchus engine**, no matcher / residual / verifier-diagnostics / challenger.
- A `dataset/` **four-layer** store: Layer 0 raw trace, Layer 1 runtime Hypothesis Ledger (the agent's own live hypotheses), Layer 2 posthoc strong-model audit, Layer 3 training/eval views. Ledger ≠ Audit.
- The oracle redirect is **direction-level** (never patch / file:line / gold), stored as a typed artifact, and **fails loud** rather than silently skipping.
- Training **inputs are oracle-free** (gold is only the oracle-step midwife); targets are a minimal fixed-section text block; pre-submit targets never predict a hidden failure.
- A held-out turning-point eval anchored on **gold / protocol turning points**, scored by a grader that uses a **different model** than the audit labeler.

## Capabilities

### New Capabilities
- `trajectory-collection`: thin mini-SWE-agent, hypothesis-first prompt, 3-stage no-early-oracle loop, Layer 0 persistence.
- `hypothesis-ledger`: Layer 1 runtime ledger = the agent's own live hypotheses + actions + evidence, with traceability and no posthoc / audit fields.
- `audit-redirect-supervision`: posthoc strong-model audit emitting direction-level audit/redirect labels.
- `training-view-export`: Layer 3 prefix samples (oracle-free input, minimal target), instance-id splits, held-out turning-point eval + gold-anchored independent grader.

### Modified Capabilities
- None (greenfield).

## Impact

- **New code / dirs**: `dataset/{raw,ledgers,audits,samples,splits}/`; a thin collection driver; an oracle-direction step; a ledger compressor; a posthoc audit generator; a prefix exporter; a grader.
- **Infra reuse (only)**: `run/swebench_single.py` (single-instance docker run) + `scripts/_wincompat/` (resource stub + CRLF fix). **Not** the Elenchus engine.
- **Out of scope / deferred (reviewer gaps, reference only — see design Non-Goals)**: the Elenchus matcher/residual/verifier-diagnostics/challenger and C-differential; grader human-κ calibration + controls; base tool-use baseline; near-duplicate/diversity controls; per-sample-type floors; formal label-quality audits.
- **Honesty constraints carried into specs**: oracle-free training input; direction-level oracle (where-wrong + direction, not the answer); ledger = live agent statements (no hindsight); trace ⟂ eval; gold-anchored, independent grader; claims limited to held-out turning-point Δ.
