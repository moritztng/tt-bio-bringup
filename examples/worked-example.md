# A worked example: MiniFold, Phase 0 to Phase 6

One small model taken through every phase, with the artifact each gate reads and the output each
gate prints. It exists because the reference documents describe the method and a filled-in example
shows it, and the second is what tells you whether you have understood the first.

**What is real here and what is not.** Everything in Phase 0 and Phase 1 is real: the commands ran
on a CPU, and the gate output is pasted from the run. From Phase 2 on the commands need a
Tenstorrent card, so the shapes of the commands are real and **every number is illustrative**. They
are labelled where they appear. Do not quote a number from this file as a measurement of anything.

`minifold_capture.py` in this directory is the Phase 1 capture, runnable with torch and no card.

## The model

MiniFold: an embedding, four pre-norm transformer blocks, a per-token linear head. 814,480
parameters. No recycling, no diffusion, no MSA. It is the easiest possible shape on purpose, so
nothing distracts from the workflow, and one detail is not a simplification: `blocks[i]` is called
as `block(x, mask=mask)`, with a keyword, which is what most real bio models do and what breaks a
capture that walks only `*args`.

```python
class MiniFold(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, vocab=22, bins=16):
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.head = nn.Linear(d, bins)

    def forward(self, tokens, mask=None):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, mask=mask)
        return {"logits": self.head(x), "single": x}
```

Your model is larger and has a recycling loop or a diffusion sampler. That changes how long each
phase takes. It does not change the order of the phases or the shape of any gate.

---

## Phase 0: the plan, and a gate you cannot argue with

```bash
mkdir -p notes scripts
cp "$SKILL/templates/PORT_PLAN.md" notes/
cp "$SKILL/gates/port_gate.py" scripts/
python3 scripts/port_gate.py plan notes/PORT_PLAN.md
```

The template starts red, and it names every hole:

```
GATE 1: phase 0 plan notes/PORT_PLAN.md is not finished. 11 problem(s):
  notes/PORT_PLAN.md:1: unfilled placeholder '<model name>'
  notes/PORT_PLAN.md:30: empty cell(s) under 'Module, Params, Input shape, Output shape, Golden fixture, Parity threshold'
  notes/PORT_PLAN.md:41: empty cell(s) under 'Axis, Symbol, Static or dynamic, Range, Tile-multiple handling'
  notes/PORT_PLAN.md:45: table [torch op | Count in model | ttnn equivalent | Risk] has no data rows
  notes/PORT_PLAN.md:61: table [Source | How the reference seeds it | How both sides will share draws] has no data rows
  notes/PORT_PLAN.md:72: empty cell(s) under 'Decision'
  ...
  notes/PORT_PLAN.md: the Target section states no numbers. The supported size range is a range of integers, not an adjective.
```

Four sections carry the weight. The rest is bookkeeping.

**The module tree, leaves first, and no row without a golden.** This is the one thing Phase 0 exists
to force. A module with no named golden is a module you will "verify" at the end by looking at the
whole model, which does not work.

| Module | Params | Input shape | Output shape | Golden fixture | Parity threshold | Status |
|---|---|---|---|---|---|---|
| `embed` | 2,816 | `[1, L]` int64 | `[1, L, 128]` | `embed.pt` | maxdiff 0 (a gather) | not started |
| `blocks.0.norm1` | 256 | `[1, L, 128]` | `[1, L, 128]` | `block0_norm1.pt` | PCC >= 0.9999 | not started |
| `blocks.0.attn` | 66,048 | `[1, L, 128]` + mask | `[1, L, 128]` | `block0_attn.pt` | PCC >= 0.999 | not started |
| `blocks.0.ffn` | 131,712 | `[1, L, 128]` | `[1, L, 128]` | `block0_ffn.pt` | PCC >= 0.9995 | not started |
| `blocks.0` (whole) | 198,272 | `[1, L, 128]` + mask | `[1, L, 128]` | `block0.pt` | PCC >= 0.999 | not started |
| `blocks` (4 blocks) | 793,088 | `[1, L, 128]` + mask | `[1, L, 128]` | `trunk.pt` | PCC >= 0.998 | not started |
| `head` | 18,576 | `[1, L, 128]` | `[1, L, L, 16]` | `head.pt` | PCC >= 0.998 | not started |
| `MiniFold` (whole) | 814,480 | `[1, L]` + mask | dict of 2 | `e2e.pt` | PCC >= 0.998 both | not started |

Those thresholds are a first guess from `03-precision-and-numerics.md`'s plausibility bands. Phase 1
replaces every one of them with the measured bf16 self-envelope for that module. Writing a guess now
is right; shipping the guess as the gate is not.

**The axes, with the tile decision recorded.** The token axis is where the tiling work is. Note that
the pair axis is *derived*: it has no separate decision, it inherits the token axis's padding, and
writing that down stops someone padding it twice.

| Axis | Symbol | Static or dynamic | Range | Tile-multiple handling |
|---|---|---|---|---|
| token | L | dynamic | 16 to 512 | pad to a multiple of 32, mask the tail, slice back before returning |
| pair | L x L | dynamic, derived | 256 to 262,144 | inherits the token axis's padding on both pair axes |
| channel | d = 128 | static | 128 | already 4 tiles, no handling needed |
| bins | 16 | static | 16 | under one tile: pad to 32, slice the first 16 back |
| batch | B | static at 1 | 1 | not batched in this port |

**The risk register: the ops with no clean equivalent.** Two entries, and the first one is the whole
performance story of this model.

1. The head's outer sum materialises `[1, L, L, 128]` before projecting to 16 bins. At L=512 that is
   32 M elements, 64 MB in bf16, and it is the only allocation that scales as L squared. Plan:
   project to bins first where the algebra allows, otherwise chunk over the first L axis.
2. `torch.use_deterministic_algorithms(True)` has no device-side equivalent. Reproducibility on
   device is asserted by running twice and comparing bits, not by setting a flag.

**The evaluation set, named now.** Phase 3's gate needs one task-level metric on real inputs, and
inputs with a ground truth do not appear by themselves. Three sequences at L=64, 117 and 380, the
published contact maps for them, top-L/5 long-range contact precision, licence confirmed. Three
weeks from now this is the thing that is not ready.

With those filled in, the gate flips:

```
GATE 0: phase 0 plan notes/PORT_PLAN.md is complete: every section present, every table row
filled, no placeholders, nothing deferred.
```

Then, before trusting it, make it fail. Blank one golden cell and confirm the gate notices:

```bash
python3 scripts/port_gate.py prove-red \
  --check   'python3 scripts/port_gate.py plan notes/PORT_PLAN.md' \
  --break   "sed -i '0,/| \`embed\`/s/| \`embed.pt\` |/|  |/' notes/PORT_PLAN.md" \
  --restore 'git checkout notes/PORT_PLAN.md'
```

```
GATE 0: green (0) -> fault injected -> red (1) -> restored -> green (0). This check can fail, so
its green means something.
```

Thirty seconds, and the difference between a gate and a decoration.

---

## Phase 1: the golden, on CPU, reproducible

Capture per-module inputs and outputs at L=64, L=117 and L=380. **117 is not a multiple of 32, and
that is the point**: it is the fixture that catches an unmasked tile tail, which is the defect class
that produced 72x the reference error in a shipped model with no error message and no log line.

```bash
python3 examples/minifold_capture.py --len 117 --out scripts/minifold_port/parity_artifacts
```

```
captured 39 modules, 117 entries, 0.005s -> scripts/minifold_port/parity_artifacts
blocks.0 kwargs captured: ['mask']
blocks.0 mask is a tensor: True
```

Those last two lines are the assertion that matters. A capture that walks only `*args` prints an
empty kwargs list, still saves, still loads, still has a key for every module, and has silently lost
the mask. You find out in Phase 3, when a module matches its golden and the model does not match the
reference.

**The gate**: the capture reproduces, byte for byte.

```bash
python3 scripts/port_gate.py determinism \
  --run 'python3 examples/minifold_capture.py --len 117 --out scripts/minifold_port/parity_artifacts' \
  --artifact scripts/minifold_port/parity_artifacts/minifold_117.pt
```

```
--- run 1: python3 examples/minifold_capture.py --len 117 --out .../parity_artifacts
--- run 2: python3 examples/minifold_capture.py --len 117 --out .../parity_artifacts
  same     .../minifold_117.pt  c8b4c6547abf7915  c8b4c6547abf7915
GATE 0: 1 artifact(s) byte-identical across two runs.
```

The gate deletes the artifact before each run, so a capture that exits 0 without writing anything
fails instead of passing on yesterday's file.

**And the negative control, which is not optional.** `--unpinned` skips the seeding and `eval()`,
which is what a rushed capture actually forgets:

```
  DIFFERS  .../minifold_117.pt  2a5b606500b6e582  5e39bdcb5a0674ab
GATE 1: 1 artifact(s) changed between two identical runs. Pin the seeds, the thread count and the
iteration order before going on: a golden you cannot reproduce cannot prove anything later.
```

Note what `--unpinned` had to do to earn that red. Skipping the seed alone was not enough: the toy's
forward pass is deterministic anyway, so the "broken" capture came back byte-identical and the
control proved nothing. It needed live dropout to have any entropy to lose. **A negative control
that does not go red is not a control, and finding that out takes one run.**

The `.meta.json` beside the fixture is deliberately not in the artifact list. It records `runtime_s`,
which is a measurement and differs between runs of a real reference, so hashing it would fail the
gate for a reason that is not a defect.

**Then measure the thresholds.** Run the same graph in torch at bf16 storage with fp32 accumulation
and record, per module, how far bf16 lands from fp32. That envelope is the gate, and it replaces the
guesses in the Phase 0 table. A module whose bf16 self-error is 4e-3 cannot be held to 1e-4 no matter
how good the port is, and holding it to 1e-2 passes a real bug.

---

## Phase 2 onward: illustrative

Everything below needs a card. The commands are the real shapes; **the numbers are made up for the
example** and are marked so.

### Phase 2: skeleton on device

Weights loaded, one forward at L=64, allowed to be slow and allowed to be wrong.

```bash
TT_VISIBLE_DEVICES=0 python3 -m pytest tests/test_minifold_weights.py -q
python3 scripts/port_gate.py determinism \
  --run 'TT_VISIBLE_DEVICES=0 python3 scripts/minifold_port/forward.py --len 64 --out /tmp/fw.npy' \
  --artifact /tmp/fw.npy
```

The weight test asserts a set equality in both directions, not a loop that checks each name it
happens to know about:

```python
assert consumed == set(reference.state_dict()), (
    f"unconsumed: {set(reference.state_dict()) - consumed}, invented: {consumed - set(reference.state_dict())}")
```

Both directions, because a remap that drops `blocks.3.norm2.bias` and a remap that invents
`blocks.3.norm2.beta` are different bugs and a one-directional check catches only one of them. Prove
it red by deleting one key from the remap.

The L-squared allocation gets measured here, before any parity work, because if L=512 cannot fit then
the size range in the plan is wrong and everything downstream was scoped against a promise you cannot
keep. *Illustrative:* it fits at 512 with 61% of L1 free, and OOMs at 900.

### Phase 3: component parity, leaves first

One module at a time, in the order the plan lists them, each against its own golden. Not the whole
model against the whole reference.

*Illustrative* run, and the interesting rows are the ones that failed first:

| Module | Threshold (measured bf16 envelope) | First attempt | After the fix | What it was |
|---|---|---|---|---|
| `embed` | maxdiff 0 | 0 | 0 | passed first time, it is a gather |
| `blocks.0.norm1` | PCC >= 0.99995 | 0.99999 | | |
| `blocks.0.ffn` | PCC >= 0.9997 | 0.9962 | 0.99991 | GELU: reference used exact, ttnn defaulted to tanh |
| `blocks.0.attn` | PCC >= 0.9991 | 0.712 | 0.9996 | mask orientation, transposed |
| `blocks.0` | PCC >= 0.9990 | 0.9994 | | |
| `blocks` | PCC >= 0.9980 | 0.9987 | | |
| `head` | PCC >= 0.9980 | 0.9991 | | |
| `MiniFold` | PCC >= 0.9980 | 0.9984 | | |

Two things in that table are the actual lesson.

**The attention failure was invisible at L=64 and L=128.** The mask was all-ones there, so a
transposed mask is the same mask. It only failed at L=117, where the padded tail makes the mask
asymmetric. This is why the ragged fixture exists, and it is why a size ladder built by truncating
one sequence is one input rather than a sweep.

**The GELU failure looked like precision.** A uniform, small, whole-model deviation that grows
smoothly with depth is exactly what an accumulated bf16 error looks like, and the tempting response
is to widen the threshold until the trunk passes. The plan's risk register named it in Phase 0
(risk 2), which is the only reason it was checked in ten minutes instead of chased for two days.
**A deviation you accept at a leaf compounds coherently and reappears at the end as an
unattributable end-to-end failure.**

Then the task metric, which is a different question from PCC: *illustrative*, top-L/5 long-range
contact precision 0.71 on device against 0.72 for the reference, on the three-target eval set, inside
the reference's own seed-to-seed spread of 0.03.

```bash
python3 scripts/port_gate.py report docs/minifold-parity.md \
  --require-heading "Component parity" --require-heading "Negative controls"
```

The negative-controls table is why that gate exists: it fails on any blank row, so "I broke it and
watched it go red" has to have happened for every test in the suite, not for the two you remember.

### Phase 4: generality

The ladder, across the whole range the plan promised: 16, 17, 31, 32, 33, 64, 117, 128, 255, 256,
380, 511, 512. Every input mode. The OOM boundary, written down.

```bash
TT_VISIBLE_DEVICES=0 python3 -m pytest tests/test_minifold_ladder.py -q
```

*Illustrative* finding, and the reason this phase is not a formality: the chunked head path was
tuned at L=512 and its guard read `if L >= 384`. At L=380 it silently took the unchunked path,
which fit, ran, and was correct, so nothing failed. The ladder only caught it because the test
asserts the chunked path *fired*, not that the answer was right:

```python
assert stats["head_chunks"] > 1, f"chunked head did not fire at L={L}"
```

An assertion on the mechanism, not an inference from the timing. A guard tuned at one size stops
firing at another, every time, and timing is too noisy to notice a path that quietly went away.

### Phase 5: performance

Census first, roofs second, prediction third, build fourth. *Every number illustrative.*

| Op | Calls | Device time | Share | Bound by |
|---|---|---|---|---|
| `ttnn.linear` | 22 | 4.1 ms | 38% | compute, 1.4x off the measured roof |
| broadcast add in the head | 1 | 3.3 ms | 31% | DRAM bandwidth, 1.1x off |
| `ttnn.transformer.sdpa` | 4 | 1.8 ms | 17% | compute, 2.2x off |
| everything else | 61 | 1.5 ms | 14% | mixed |

Wall clock 31.4 ms, summed device time 10.7 ms, **residual 20.7 ms, 66% of the wall**. That residual
is the whole finding. 88 op dispatches at a small size, so the host round-trip dominates and the
biggest number in the census table is not in the census table. A campaign that starts from the op
list optimizes `ttnn.linear` and captures at most 38% of a third of the time.

| Lever | Ceiling (Amdahl) | Predicted | Measured | Decision |
|---|---|---|---|---|
| trace capture, whole forward | 66% (the residual) | 2.4x | 2.1x | landed |
| project to bins before the outer sum | 31% (the broadcast add) | 1.35x | 1.28x | landed |
| fused attention kernel | 17% (sdpa) | 1.09x | not built | killed: 9%, below the effort bar |

The prediction is written before the build, so a miss is informative. The killed lever stays in the
table with the number that killed it, so it cannot come back in six weeks as a fresh proposal on a
different metric.

Then re-census, because every label in the table above expired the moment trace capture removed the
traffic that produced it.

### Phase 6: integration

One row in `_MODEL_RESULTS_PREFIX`, one branch in `_WorkerState.load_model`, one
`_predict_minifold_one`. Shared helpers, not private copies. Then:

```bash
python3 scripts/packaging_smoke.py
TT_VISIBLE_DEVICES=0 python3 scripts/release_gate.py
python3 -m pytest tests/test_perf_model_coverage.py tests/test_repo_root_clean.py -q
```

*Illustrative:* `packaging_smoke.py` failed the first time. The chunked-head kernel's `.cpp` was on
disk, imported fine in the editable install, and was absent from the wheel. Zero signal from every
other check, because an editable install reads the file from the source tree. That is the only guard
that catches this class, and it has caught it repeatedly.

Every bug from Phase 3 becomes a gate arm before Phase 6 closes: one for the GELU approximation, one
for the mask orientation at a ragged length. Each demonstrated failing on the commit before its fix,
or it is theatre.

---

## What this example does not show

Honest list, because the gap between this and your port is mostly here.

- **A recycling or diffusion loop.** A feedback loop amplifies every upstream deviation and has to be
  ported last, after the trunk is exact. Capturing one needs `num_timesteps >= 2`, or the common
  `zip(schedule, schedule[1:])` idiom yields an empty loop and every per-step module silently never
  runs.
- **Host-side pipelines.** MSA construction, templates, ligand featurization, constraints. MiniFold
  has none, and in a real port this is routinely the underestimated half of the work.
- **Multi-card anything.** One card throughout.
- **A real accuracy investigation.** The two failures above were found in minutes because the plan
  predicted them. The ones that cost days are in `13-failure-atlas.md`, indexed by symptom.
- **Custom kernels.** The census killed that lever at 9%, which is the normal outcome.

The order, though, is the same at any size, and so is every gate.
