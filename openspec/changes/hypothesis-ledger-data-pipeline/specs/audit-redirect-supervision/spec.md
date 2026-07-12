## ADDED Requirements

### Requirement: Posthoc full-trajectory audit
A strong model SHALL read the full trajectory, ledger, and outcome and emit a per-hypothesis audit — `why_it_looked_plausible`, `why_it_was_weak_or_wrong`, `support_calibration`, `band_aid_or_proxy_risk`, the turning point, and a direction-level next move — to `dataset/audits/<instance_id>_<run_id>.audit.json`.

#### Scenario: Structured audit produced
- **WHEN** a completed run is audited
- **THEN** a per-hypothesis audit with the required fields is written

### Requirement: Direction-level only
The audit MUST NOT contain a patch, `file:line`, hidden-test assertion, or gold mechanism; a leakage check rejects any audit that does.

#### Scenario: Fix expressed as a probe
- **WHEN** the audit says the current patch is insufficient
- **THEN** it names a probe / direction (preservation / anti-case / discriminative), not the code change

### Requirement: Support calibration and band-aid flagging
The audit SHALL mark a visible-test-only pass as `weak_support` (not `support`) and flag a patch that does not match its stated hypothesis as a band-aid / proxy risk.

#### Scenario: Visible pass is weak support
- **WHEN** the only positive evidence is a visible test pass
- **THEN** `support_calibration` is `weak_support` and a stronger probe is required before submit
