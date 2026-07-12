## ADDED Requirements

### Requirement: Hypothesis-per-phase agent
The collection driver SHALL run a thin mini-SWE-agent with one prompt change: it states **one hypothesis per phase** — a hypothesis (its current bug / repair understanding) governs a phase of verification actions, not every step — and explores on its own. A new hypothesis opens a new phase. No matcher, residual, promotion, or challenger machinery is used.

#### Scenario: One hypothesis governs a phase
- **WHEN** the agent states a hypothesis
- **THEN** the following actions (inspect / patch / test) are recorded under that hypothesis until it states a new one, which opens the next phase

#### Scenario: Phase hypothesis captured with a reference
- **WHEN** a phase opens
- **THEN** its hypothesis is captured with a `raw_ref` to the originating message

### Requirement: Three-stage no-early-oracle loop
The driver SHALL let the agent explore and submit unaided; on failure it SHALL return the test / harness output for a self-rescue attempt; only if it still fails SHALL an oracle (reading gold) provide a direction-level redirect. Oracle content before the agent's first submit is PROHIBITED.

#### Scenario: Agent submits unaided
- **WHEN** a run begins
- **THEN** the agent inspects / patches / tests / submits with no oracle or gold shown to it

#### Scenario: Test feedback before oracle
- **WHEN** the submission fails the harness
- **THEN** the test / harness output is returned to the agent for a self-rescue attempt before any oracle redirect

#### Scenario: Oracle redirect only after self-rescue fails
- **WHEN** the agent still fails after self-rescue
- **THEN** an oracle reading gold provides a where-wrong + direction redirect (never the answer), and the run records the oracle-intervention step

#### Scenario: Early oracle rejected
- **WHEN** oracle / gold content reaches the agent before its first submit
- **THEN** the run is rejected as contaminated

### Requirement: Oracle redirect is minimal-direction, typed, and fails loud
The oracle redirect SHALL give only a where-wrong assessment + a direction (probe / search), never a patch, `file:line`, gold identifier, or hidden-test assertion. It SHALL give the **minimal** direction that lets the agent proceed — ideally just enough that pointing the direction alone enables the agent to solve it — and MUST NOT drift toward specifying the fix. It SHALL pass a direction-level leakage check before being shown to the agent, SHALL be stored as a typed Layer 0 artifact (`oracle_redirect_*.json` with step + content), and SHALL fail loud if it cannot be produced.

#### Scenario: Minimal direction preferred
- **WHEN** the oracle composes a redirect
- **THEN** it states the least direction needed for the agent to make progress, not a fuller specification that approaches the fix

#### Scenario: Leaky redirect blocked
- **WHEN** a redirect contains `file:line`, a gold identifier, or the concrete fix
- **THEN** the leakage check rejects it and it is not shown to the agent

#### Scenario: Redirect failure is loud
- **WHEN** the oracle cannot produce a redirect (error / empty)
- **THEN** the step is marked failed-loud, never silently continued

### Requirement: Layer 0 persistence and outcome labeling
Each run SHALL persist its raw trace, patches, test outputs, and harness report under `dataset/raw/<instance_id>/<run_id>/`, and SHALL be labeled with one outcome in {`self_solved`, `self_corrected`, `oracle_redirected`, `chaotic_failed`}.

#### Scenario: Self-correction labeled
- **WHEN** the agent fixes the task after test feedback but with no oracle redirect
- **THEN** the run is labeled `self_corrected`
