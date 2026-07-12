## 1. Scaffolding

- [ ] 1.1 Create `dataset/{raw,ledgers,audits,samples,splits}/` (four layers) + a note on Ledger ≠ Audit and the honesty constraints
- [ ] 1.2 Pick the SWE-style task source; reserve a held-out test slice by `instance_id`

## 2. Collection driver (trajectory-collection)

- [ ] 2.1 Thin mini-SWE-agent + prompt change: state a hypothesis before acting, explore on your own (no engine machinery)
- [ ] 2.2 Capture the stated hypothesis + actions + evidence live, each with a `raw_ref`
- [ ] 2.3 Three-stage loop: explore → submit → test-feedback self-rescue → oracle direction
- [ ] 2.4 Oracle-direction step: strong model reads gold → where-wrong + direction (not the answer); direction-level check; typed Layer 0 artifact; fail-loud
- [ ] 2.5 Run `swebench_single` in docker with `_wincompat` on PYTHONPATH; sanity precheck (gold→resolved=true, bad→fail, codex-proxy actually sent the requested effort) before any bulk collection
- [ ] 2.6 Persist Layer 0 + label outcome; smoke-run a few instances

## 3. Runtime ledger (hypothesis-ledger)

- [ ] 3.1 Define the runtime-only ledger card schema
- [ ] 3.2 Compressor: the agent's live statements → cards (`raw_ref` + agent interpretation), no posthoc authorship
- [ ] 3.3 Validator: schema + allowlist + reference resolvability; exit non-zero on any invalid ledger

## 4. Posthoc audit (audit-redirect-supervision)

- [ ] 4.1 Audit generator: full trajectory → per-hypothesis audit (plausible / weak, `support_calibration`, band-aid, turning point, direction-level next move)
- [ ] 4.2 Direction-level leakage check on audit labels (`file:line` / diff / gold identifier) + a sampled human spot-review

## 5. Training views (training-view-export)

- [ ] 5.1 Prefix cutter (input ≤ `cut_step` + feedback; oracle-free; drop future-leaking)
- [ ] 5.2 Minimal fixed-section target; provenance out-of-band
- [ ] 5.3 Pre / post-submit discipline; pre-submit target regenerated prefix-only
- [ ] 5.4 Quality gate (log every drop reason) + instance-id splits
- [ ] 5.5 Held-out turning-point eval set (test split only) anchored on gold / protocol turns
- [ ] 5.6 Grader: a different model than the labeler, rubric + scale frozen; report the four single-turn metrics; no SWE-resolve claim

## 6. End-to-end

- [ ] 6.1 Run collection → ledger → audit → export over a handful of instances; produce gated samples + the eval set; log realized yield (a goal, not a deadline)
- [ ] 6.2 Hand-verify 3–5 sample cards (old hypo / evidence / why weak / turning point / redirect)
- [ ] 6.3 Dataset README with the honesty constraints (oracle-free input, direction-level oracle, ledger = live, gold-anchored independent grader, claim = held-out turning-point Δ)
