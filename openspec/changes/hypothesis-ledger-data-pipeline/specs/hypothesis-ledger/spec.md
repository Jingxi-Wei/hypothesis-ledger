## ADDED Requirements

### Requirement: Runtime ledger card schema
The Layer 1 ledger SHALL record, per hypothesis, only runtime-visible fields: `task_id`, `run_id`, `hypothesis_id`, `active_hypothesis` (a problem-state abstraction, not a per-tool-call reason), `why_plausible_at_that_time`, `actions_under_hypothesis`, `evidence_observed` (each with the agent's interpretation), `agent_self_verdict` (support | weak_support | refute | inconclusive | ready | unknown), and `next_hypothesis_id`.

#### Scenario: One card per hypothesis
- **WHEN** a raw trace is compressed
- **THEN** each distinct stated hypothesis becomes one card with only runtime fields

### Requirement: Ledger sourced from the agent's live statements
Cards SHALL be built from the agent's own hypotheses and observations stated live during the run. A posthoc model pass over the completed trajectory MUST NOT author any runtime field; if a model assists extraction it SHALL be blind to the outcome and to steps after the card's step.

#### Scenario: Outcome-aware authorship rejected
- **WHEN** a runtime field would be written using the final result or a later step
- **THEN** that card is rejected as hindsight-contaminated

### Requirement: No posthoc / audit fields (allowlist)
A card MUST NOT contain audit fields (`audit_verdict`, `weakness`, `failure_mode`, `support_calibration`, `band_aid_or_proxy_risk`, `should_have_turned_at_step`). The validator SHALL enforce an allowlist: any key outside the runtime set is rejected.

#### Scenario: Extra / audit field rejected
- **WHEN** a card contains a key outside the runtime allowlist
- **THEN** the validator rejects the card

### Requirement: Traceability and validation
Every card claim and evidence item SHALL carry a `raw_ref` resolving to Layer 0; a validator SHALL check schema, allowlist, and reference resolvability over `dataset/ledgers/` and exit non-zero on any invalid ledger.

#### Scenario: Unreferenced claim rejected
- **WHEN** a card has a conclusion with no `raw_ref`
- **THEN** the validator rejects it
