# Parity report: <model name>

The document that says what "correct" means here and shows it holds. Written during Phase 3,
finished before Phase 5 starts, because a performance number on an unverified port is worthless.

```bash
./env/bin/python3 scripts/port_gate.py report docs/yourmodel-parity.md \
    --require-heading "Component parity" --require-heading "Negative controls"
```

## Golden definition

- Reference commit, config, checkpoint hash:
- Capture command:
- Fixture location and size:
- Determinism proof: capture run twice, byte-identical (paste the hashes)

## Component parity

| Component | Metric | Threshold | Achieved | Sizes tested | Test |
|---|---|---|---|---|---|

State why each threshold is what it is. "It passed" is not a justification. A threshold should come
from either the numerical analysis (this is the expected bf16 accumulation error at this depth) or
from the downstream metric (this deviation moves the task metric by less than its own noise).

## End-to-end parity

| Input | Metric | Threshold | Achieved |
|---|---|---|---|

## Task-level metric

The metric a domain expert would ask about, on real inputs, against the reference:

| Set | N | Reference | Device | Delta | Interpretation |
|---|---|---|---|---|---|

## Known deviations

Anything that does not match, with the mechanism and why it is acceptable:

## Negative controls

Evidence each test can fail. For each test, what was broken to make it go red, and that it did.
Keep the verdict under 40 characters, `yes, 0.712` rather than a sentence, and put anything longer
in the Notes column: the gate checks the verdict as a field and does not parse prose. With nothing
to note, write `none needed`; the gate rejects a blank cell and a bare dash.

| Test | Injected fault | Went red? | Notes |
|---|---|---|---|
