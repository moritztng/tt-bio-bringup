# Performance report: <model name>

Every number here must name the hardware, the shape, the dtype, the warm state, the repeat count,
and the exact command. A number without those is not reproducible and does not belong in this file.

```bash
python3 scripts/port_gate.py report docs/yourmodel-perf.md \
    --require-heading "Measured roofs" --require-heading "Op census" --require-heading "Levers"
```

## Setup

- Chip generation, card, host:
- Package versions:
- Command:
- Warm-up policy, repeats, statistic reported (median):

## Measured roofs

Measured on this card with a microbenchmark, not quoted from a datasheet.

| Roof | Method | Value |
|---|---|---|
| Peak matmul throughput | large square matmul, dtype X | |
| DRAM bandwidth | large streaming copy | |
| L1 bandwidth | L1-resident elementwise | |

## Op census

| Op | Calls | Device time | Share | Arithmetic intensity | Bound by | Distance to roof |
|---|---|---|---|---|---|---|

Wall clock: . Sum of device time: . Residual (host + dispatch): . Explain the residual.

## Levers

| Lever | Predicted ceiling | Predicted win | Measured win | Landed? | Re-census done? |
|---|---|---|---|---|---|

A prediction is written before the work starts. Scoring it afterwards is how the cost model gets
better, and a lever whose prediction was wrong by an order of magnitude is a finding worth a note.

## End-to-end

| Configuration | Cold | Warm steady state | Host share |
|---|---|---|---|

## Backlog

Ranked, with each item's ceiling. Re-rank after every landing, because levers dismissed as too small
grow as the ones above them shrink. A killed lever stays in this table with the number that killed
it, so it cannot come back later on a different metric.

| Item | Ceiling | Decision |
|---|---|---|
