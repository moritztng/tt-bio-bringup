# Port plan: <model name>

Written in Phase 0, before any device code. Updated when the shape of the work changes, never
deleted. If a section below is empty, Phase 0 is not finished.

## Reference

- Repository and pinned commit:
- Config / checkpoint used, with its hash:
- Command that runs it on CPU, and its runtime:
- Known reference-side nondeterminism, and how it is pinned:

## Target

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

| Axis | Symbol | Static or dynamic | Range | Tile-multiple handling |
|---|---|---|---|---|
| token | | | | |
| pair | | | | |
| sample | | | | |
| msa | | | | |
| atom | | | | |

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

## Risks, ranked

1.
