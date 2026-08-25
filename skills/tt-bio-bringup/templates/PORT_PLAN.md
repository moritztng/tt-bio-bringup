# Port plan: <model name>

Written in Phase 0, before any device code. Updated when the shape of the work changes, never
deleted. If a section below is empty, Phase 0 is not finished, and the Phase 0 gate says so:

```bash
python3 scripts/port_gate.py plan PORT_PLAN.md
```

## Reference

- Repository and pinned commit:
- Config / checkpoint used, with its hash:
- Command that runs it on CPU, and its runtime:
- Known reference-side nondeterminism, and how it is pinned:

## Target

Fill the hardware lines from `tt-smi -ls` once you have a card. Before then, state your intent and
append "unconfirmed"; the gate wants a stated answer, not a confirmed one, and an unconfirmed value
you can see is better than a blank you cannot.

- Chip generation and card count:
- Supported size range to ship (state numbers, not "large"):
- Input modes to support:
- Outputs to produce, in what formats:

## Module tree

Leaf modules first. One row per module. Every row needs a golden.

| Module | Params | Input shape | Output shape | Golden fixture | Parity threshold | Status |
|---|---|---|---|---|---|---|
| | | | | | | not started |

## Axes

One row per axis your model actually has. Delete the rows that do not apply and add the ones that
do: a structure predictor usually has token, pair, sample, MSA and atom; an embedder has token and
maybe layer; a scalar predictor has token and a pocket or ligand axis. Every remaining row must be
filled, and the tile-multiple column is where the padding and masking decision is recorded.

| Axis | Symbol | Static or dynamic | Range | Tile-multiple handling |
|---|---|---|---|---|
| | | | | |

## Op inventory

| torch op | Count in model | ttnn equivalent | Risk |
|---|---|---|---|

**No equivalent (the risk register):**

## Control flow

Loops, their trip counts, and whether the trip count depends on data:

## Host-side pipelines

Everything the model needs that is not the sequence: MSA construction, templates, ligand
featurization, constraints. Each one is a port of its own and is routinely underestimated.

## Randomness

| Source | How the reference seeds it | How both sides will share draws |
|---|---|---|

## Evaluation set

Phase 3's gate needs one task-level metric on real inputs, which needs inputs and a ground truth.
Neither appears by itself, and for a proprietary model the licence question is real. Settle it here
or Phase 3 stalls three weeks in.

| Item | Decision |
|---|---|
| Inputs, named | |
| Ground truth, and where it lives | |
| The metric, and what a domain expert considers passing | |
| Licensed for use with this model | |
| Who curates it | |

## Risks, ranked

1.
