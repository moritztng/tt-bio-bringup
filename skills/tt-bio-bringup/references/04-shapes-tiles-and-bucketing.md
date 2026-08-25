# Shapes, tiles and bucketing

This document decides one number for you: **pad every variable-length axis to a multiple of 32, mask the pad
everywhere it can reach a reduction, slice back**. It then says which ops stay bit-exact when you do that and which
do not, how to price padded compute against kernel recompilation, why a layout mismatch appears as a 36x slowdown
instead of an error, and what guard keeps all of it from rotting at the next size you run.

Read this when wiring a variable-length input into a ttnn module, when a port is correct at one sequence length and
wrong at another, when a per-op profile is dominated by something that is not a matmul, or when a lever measured as
a win does nothing in a real run.

## 1. The 32x32 tile is the only shape the hardware has

`ttnn.TILE_LAYOUT` stores a tensor as 32x32 tiles. The last two dims are each padded up to a multiple of 32
**physically**, while the **logical** shape stays at the true length. Both are visible:

```python
t.shape          # logical,  e.g. [1, 8, 98, 98]
t.padded_shape   # physical, e.g. [1, 8, 128, 128]
# axis a is RAGGED exactly when t.shape[a] != t.padded_shape[a]
```

**Cost is quantised.** An axis of 33 costs 64. For an O(S^3) op such as triangle multiplication, rounding a token
axis to the next multiple of 64 rather than 32 costs `(128/96)^3 = 2.37x` the work at N in 65..96, every run,
forever. The general penalty is `((k+2)/(k+1))^3` where `ceil32(N) = 32(k+1)`, so it is worst at small N (8x below 32)
and fades as N grows.

**The pad's contents are unspecified.** Some ops zero it, some leave stale buffer data, some leave it unwritten
(`ttnn.scatter` does). Never assume zero, and never assume the consumer ignores it.

**A predicate on the logical shape tests the wrong number.** `fuse_batch` folds leading dims into M *after* tile
padding, so the M a matmul runs is `prod(leading_dims) * ceil32(rows)`. A real case: an L1-residency guard required
`M % 32 == 0` on the flattened M of an `[S, S, c_z]` pair tensor, i.e. it tested `S^2 % 32`: 88804 at S=298
(`% 32 == 4`), 13689 at S=117 (`% 32 == 25`). It refused at every real size and the merged "2.17x" win was dead in
every fold, while its validating A/B ran identical code on both arms and so looked clean. Derive tile counts from
`padded_shape`, never from the logical shape.

## 2. The hard rule: pad the token axis to 32, mask the tail

**Every variable-length token/sequence/atom axis is padded to a multiple of 32, and the padded tail is masked at
every site where it can influence a result**: softmax, mean, sum, variance, layernorm, any `reduce`, argmax/top-k,
any outer product building a pair track, any additive pairwise bias.

This is a correctness rule, not a perf preference. **An unmasked tail does not degrade accuracy slightly.** In
attention, padded key columns enter softmax at score 0 while real scores sit well below 0, so `exp(0)` wins and
the row's attention mass lands on padding. Measured against an fp64 reference: **71-76x the reference error at
every ragged length, ~1.4x at every aligned one.** At fold level, masking the tail moved one target's error
metric from 2.43 to 0.47.

Bucketing is three operations and all three are mandatory:

```python
TILE = 32
def pad_amount(n, mult=TILE): return (-n) % mult

L = tokens.shape[-2]; pad = pad_amount(L)
if pad:
    tokens = F.pad(tokens, (0, 0, 0, pad))                     # 1. PAD
    tmask = torch.zeros(1, L + pad); tmask[:, :L] = 1.0
    pmask = tmask[..., :, None] * tmask[..., None, :]          # outer product, both axes
    attn_bias = attn_bias + (1.0 - pmask) * -1e9               # 2. MASK  (and zero K/V:)
    key = key * tmask[..., None]
out = out[..., :L, :]                                          # 3. SLICE BACK
```

- Use `-1e9`, not `-inf`: in bf16 a masked lane times an `inf` bias gives `0 * inf = NaN` in valid lanes. Zero the
  padded K/V **as well**, so exactness does not depend on how bf16 rounds `exp(-1e9)`.
- Mask the pair track with the **outer product** of the token mask; a 1-D mask on one axis leaves the other alive.
- The same rule covers a fixed-budget validity mask (padded atoms inside a fixed atom count): unmasked invalid
  entries blow up through high-gain real weights and contaminate valid ones via attention.

## 3. Detection recipe

The defect is invisible to any test whose lengths are multiples of 32: screens at 128 and 512 tokens are correct
and blind simultaneously.
1. **Fold at a deliberately ragged length beside an aligned one** (N=98 and N=128), both against the CPU golden; a
   ragged/aligned error ratio above ~2x is the signature.
2. **Use captured real tensors.** Random operands at a ragged length reproduce only ~1.55x of a 72x defect: the
   damage needs the real score distribution as well as the ragged length.
3. **Poison the pad.** Fill the padded region with a large constant instead of zeros; output must be
   **bit-identical**. Independent of the error's size, and the only test that proves the padding is inert.
4. **A/A control.** At an already-aligned length, enabling bucketing must be a `torch.equal` no-op.
5. **Census by counter, not by reading source.** Wrap the reducing primitives, count `ragged` vs `aligned` per call
   site on one real run (`tests/token_axis_probe.py`). The CLI runs the model in a per-card worker **subprocess**,
   so patching in the parent counts nothing: install via a `sitecustomize.py` import hook, one JSON per pid. A
   real-run census inverted which models a previous pass had *guessed* were exposed.

## 4. Ragged tails in fused and custom kernels

**Whether a ragged tail is safe is a property of the primitive, not of the model.** Measured answers differ
inside the same library:

| primitive | ragged tail | why |
|---|---|---|
| `ttnn.softmax(dim=-1)` | **masked for you** | tt-metal sets `mask_padded_data` from the logical shape when the padded last dim is wider |
| `ttnn.transformer.scaled_dot_product_attention` | **not masked** | extends the key axis with a defined-zero bias while the caller's bias covers only the logical length |
| a hand-written fused SDPA transcription | **not masked** | its planner derives `Sq`/`Sk` from `padded_shape`, so at logical 98 over physical 128 it sees Sk=128, finds a dividing chunk, and accepts |
| `ttnn.scatter` | **leaves output pad unwritten** | any downstream reduce over that axis reads garbage |

The question is never "does this model pad", it is "which reduce sites does this axis reach". Record the answer
per model in one file with the counters that produced it. tt-bio uses `tt_bio/token_axis.py`: a table of
`(status, multiple, site, evidence)` with statuses `bucketed / immune / partial / exposed / uncensused`, and
`tests/test_token_axis_bucketing.py` fails if a model appears in a CLI `--model` choice without a row. That test is
what stops the next port inheriting the bug: older ports were immune only as a side effect of bucketing for
compile-time reasons, and the first port that skipped the padding wrapper ran 117 tokens raw.

- **Bucketing the caller hides a ragged-kernel bug, it does not fix it.** Fix the kernel too.
- **Census the call sites, not the model list.** One pass fixed 1 of 4 call sites of a defective primitive and
  declared victory. "Believed to bucket" is not a census.
- **A pad that crosses a length gate changes route selection.** If the fix lifts a model past a
  `_FUSED_MIN_S = 128`-style threshold, the measured delta bundles correctness with a route change; split them.

## 5. Bucketing: the ladder, and what recompilation costs

Each distinct kernel *shape* compiles once, cold. **Measure your own**: it scales with the kernel, and the
figures in this repository span **~0.85 s for a small language model, ~8-10 s for a typical trunk op, and up
to ~60 s for a large fused kernel**. Multiplying a shape count by the wrong end of that range is a 70x error
in a GO/NO-GO. Time one cold compile of your own hottest op and use that. It then persists in the on-disk cache
(`TT_METAL_CACHE`). Bucketing exists so the second input of a different length reuses the first's programs.

**Use 32, one constant for the whole codebase.**

- `ceil(N/32)*32 <= ceil(N/64)*64` for every N, with equality on exactly half of all lengths. On the other half
  64 adds a whole tile. 64 is never faster at any length, much slower at half of them, and uses no less DRAM.
- The ttnn program cache keys on the **logical** shape, so raising N to `ceil(N/32)*32` changes no *padded* shape.
  Those tiles were already computed with a zero tail: a 32-bucket is compute-free by construction.
- 64's only advantage is compiled-variant count (64 lengths per program rather than 32). That saving is
  **one-time and disk-cached**; padded compute is paid on every run forever.

A per-model multiple encodes when the model was written, not what the hardware needs. An exception requires that
model's own measured numbers in an evidence table beside it, plus a test that refuses an undocumented one.

**When bucketing is net-negative:**

- Coarsening the bucket on a superlinear op. On an O(S^3) trunk, 32 -> 64 costs up to 2.37x the triangle work for
  a length just above a boundary. A coarse trunk bucket was implemented on one port and reverted for exactly this.
- A different axis may want a different constant: an MSA-depth axis padded to 1024 is cheap per unit and dominated
  by program count, so a coarse bucket is right there and wrong on the token axis. Do not confuse either with an
  op's internal chunk-size constants (triangle/transition chunk widths), which are L1-budget constants.

## 6. Bit-exactness under bucketing: which class is which

**Exact under padding** when a zeroed/masked lane cannot enter an accumulator that produces a live output:

- Flash-attention SDPA with fully-masked padded key chunks. A fully-masked chunk is an exact no-op; verified
  PCC 1.000000, maxdiff 0.0 on an 80-layer LM trunk.
- Elementwise ops, and any op whose reduction axis is not the axis you padded.

**Not exact under padding** when the pad changes a matmul's **contraction size** K, or when the padded axis *is* the
reduction axis: bf16/fp32 accumulation order changes even with exact zeros.

- Triangle multiplication. Pad-32 vs pad-64 on a trunk: PCC 0.9997 but `max|Δ| = 192`, and a structure RMSD shift
  of ~0.065 Å: within noise, but a real distribution shift. Same for anything normalising by a count you did not
  correct for the pad (`mean`, `var`, layernorm).

**Verify per op; do not assume by class name.** Two further traps:
- Per-chunk bit-exactness does not imply fold-level bit-exactness. A transition op's output was found to depend on
  **allocation sequence**: identical values, identical boundaries, all 52 chunks `torch.equal` in isolation, and the
  fold's output still moved. Allocation order is a hidden input to reduction order, so check the final output.
- `ttnn.slice` is a copy, never a view (`buffer_address()` differs from the parent's) for every real subrange;
  only a whole-tensor slice with unit step short-circuits and returns the input. Any plan of the form
  "write into a slice instead of concatenating" is not an optimisation.

## 7. When a batch or sample dimension defeats bucketing

Padding an axis helps only if that axis is what a program's shape keys on, and often it is not.
- **Windowed attention.** A model that reshapes to `(B, n_windows, WINDOW, -1)` **before** the encoder has folded
  the window index into the batch dim: one dispatched program already covers all windows regardless of how many
  are real. Bucketing that axis up to a fixed 3584 from a real 2560 removed **zero** programs on a
  dispatch-bound model and made every tensor 1.4x bigger for nothing.
- **Per-sample-shaped ops.** With best-of-N batched as B=N, the pair state replicates to `[N,L,L,c]` in both
  sampler and confidence head, so the effective shape varies with N inside a fixed token bucket: the bucket neither
  stabilises the program set nor bounds memory.
- **A submodule built directly, bypassing the padding wrapper**, arrives at the raw length, so thresholds expressed
  relative to the padded length misclassify it: one case counted `too_short: 5440, served: 0` at S=76/117 for a
  fused route on a model whose nominal residue count was well above the gate.

**The instrument is an op census (dispatched program count), not a byte count.** If padding an axis does not change
the number of dispatched programs, bucketing on it buys nothing.

## 8. Layout: tile vs row-major, and the cost of a retilize

Ops declare which layout they accept; when they disagree ttnn inserts a conversion and **you get no error, only
time.** A layout mismatch is a perf cliff, visible in a per-op profile as a large non-GEMM data-movement fraction
and nowhere else.

- **Host tilize on upload.** `ttnn.from_torch(..., layout=ttnn.TILE_LAYOUT)` tilizes on the host, single-threaded.
  On a diffusion sampler at 250 residues / 3359 atoms this was **288 of 596 ms per step (48%)**. Fix: upload
  row-major pre-cast to the target dtype, then `ttnn.to_layout` on device (2.8-8.5x per tensor, legal inside a
  trace-capture region); for persistent trace buffers, pre-cast dtype in torch before tilizing (2.5x). Both
  bit-exact. **Not** universal: on another port's upload mix this measured net slower. Profile the upload path.
- **Permutes are not equal.** The last-two transpose `(0,1,3,2)` hits a fast path, ~368 GB/s. A channel-move
  permute `(0,3,1,2)` is a full cross-tile re-tile (untilize, blocked transpose, retilize) and measured
  **55 GB/s, 12.6% of the measured 435.2 GB/s roof** (not the 512 GB/s datasheet figure). On one
  triangle-multiplication op at N=1024 that single permute was
  **70.4 of 118.1 ms (60%)** while the contraction matmul ran at 48 TFLOP/s, 48% of the measured 100.55 TFLOP/s HiFi4 roof in under 5% of the time. The
  fix is structural: keep the tensor channel-major through the chunk loop, or fold the transpose into the matmul
  operand read (`transpose_b`).
- **`untilize`/`to_layout` has a silent single-core fallback** for particular (tile-row, tile-column) combinations,
  with a per-chip trigger boundary: measured 36070 us at 10.1 GB/s with 1 of 130 cores engaged, versus 998 us at
  364.2 GB/s for a neighbouring shape. `use_multicore=True` does **not** override it. Confirm by forcing
  `use_multicore=False` on a known-fast shape and checking it reproduces the slow number.
- **Non-tile-aligned head dims cost layout work and can scramble data.** At `head_dim=48`, reshape/permute ops cost
  6x a full linear (72 us vs 11.9 us), and `ttnn.experimental.nlp_create_qkv_heads` reads each head at stride
  `ceil32(head_dim)` = 64, so a tensor packed at native stride 48 is scrambled (random-weight tests pass, real
  peaked weights give PCC < 0.1). Pad head dims to 32, keep that layout through attention, scale by the real
  `head_dim ** -0.5`.

## 9. Core-grid geometry and per-element ops

Work is split over a rectangular Tensix grid (13x10, 11x10, 8x9, depending on part and harvesting). Two failure
modes come from the grid, not the tensor.
**Unsatisfiable splits leave holes.** `ttnn.split_work_to_cores(all_cores, units)` raises
`TT_FATAL work_split.cpp: remaining == 0` under an exact rule: **`units > cores` and `units % cores` is a non-zero
multiple of the grid height**. Verified over every unit count 1..4000 on ten grids with zero mismatches; on a
13x10 grid, 358 of the first 4000 unit counts fail. No tensor is involved, it is the work-split arithmetic. Stock
ops do not hit it, but **any custom `ttnn.generic_op` kernel calling the utility inherits the holes**, and they are
grid-height dependent, so a screen on one grid misses most of them. Fix: try the full grid first, on a throw search
rectangular sub-grids with the utility itself as the sole authority on viability, memoise per `(device, grid,
units)` (a throwing call costs ~357 us versus ~0.6 us cached), and fall through to the stock op when nothing splits.

**A shard's core count goes dark by pure arithmetic.** One materialised-softmax path pinned its height shard to a
64-core rectangle requiring `rows * heads * S ≡ 0 (mod 2048)` while the L1 byte budget affords `rows ∝ S^-2`, so
above ~512 tokens the walk silently reaches 0 rows and the tail falls back to DRAM. The guard reported "not firing"
correctly the whole time; that clause being true says nothing about whether the guard is doing its job.

**Per-element ops are cycles-per-element bound, not bandwidth bound.** Measured on a Blackhole p150,
`[1,4,3359,3360]` bf16 (45.1M elements):

| op | time | throughput | vs `add` |
|---|---|---|---|
| `ttnn.add` | 0.649 ms | 417 GB/s, 69.5 G elem/s | 1x |
| `ttnn.scatter` | 4.657 ms | 39 GB/s, 9.7 G elem/s | 7.2x slower |
| `ttnn.scatter_add` | 9.65 ms | 4.7 G elem/s | 14.9x slower |
| `ttnn.gather` | 38.0 ms | 7 GB/s, 1.2 G elem/s | 58x slower |

Proof it is element rate, not bytes: the same dense op in `bfloat8_b`, half the bytes, costs 4.662 ms, 0.1%
different. That is ~14 cycles/element against `add`'s ~2, the signature of a scalar inner loop, and no knob moves it
(index dtype, ROW_MAJOR vs TILE, `sub_core_grids`, `out=` preallocation). **Do not design a port around
scatter/gather as cheap sparse indirection.** The levers are calling it less (one decoder called `scatter` twice
with bit-identical inputs across recycles and cached instead: free, bit-exact, ~7% of the step) or an upstream
kernel fix. `ttnn.scatter` in TILE layout also rejects fp32 (`scatter.cpp:106`) and int32/uint32 rows longer than 256
elements; both limits are gated on the tile layout, not on the dtype alone.

## 10. Shape generality: the size-ladder guard

**A port tuned at one size silently goes dark at another.** No error, no log line, only a number that looks like
"the model got slower here". Three mechanisms found simultaneously above one model's tuning size:

- A fused-kernel precondition declining **1120 of 1120** calls on its own check.
- An L1-headroom predicate answering DRAM instead of L1 above N >= 560.
- An SDPA q-chunk program config overflowing per-core L1 and falling back to the slow path: it asked for
  **2.20 MB per core against the 1.46 MB the allocator reports per bank** at 768 tokens, and 3.39 MB at 1024.
  (The 1.80 and 2.34 MB figures that once appeared here were the *requested* totals under a different
  chunk setting, not budgets; `08-memory-and-residency.md` §1 has the one number to size against.)

Together: **11.1% of fold wall-clock at 768 tokens, 14.2% at 1024**, on a model whose levers were validated at 512.

**Detection is the log-log runtime exponent between consecutive rungs**, not reading configs: a jump from N^2.03
(256 -> 512) to N^3.62 (512 -> 768) inside one interval is the signature. Two neighbouring size-only failures:

- **An OOM naming an allocation size is often an allocation-count problem.** An fp32 softmax held two live
  `[S, heads, S, S]` score tensors, 16 GiB each at 1024 tokens; the fix that mattered was deleting one live copy
  (fold the scale into the bias-add, reduce in place), not blocking the failing allocation. Count co-live tensors
  of the failing shape before building a chunker.
- **One specific shape can hang.** A narrow token window `[500, 507]` wedged on a device readback on a 13x10
  compute grid while 496 and 512 passed clean, and the same repro was clean on 11x10. A coarse sweep never sees it.

**The guard is a size-ladder arm in the release gate:**

- Fold every model at a fixed rung set (`256, 512, 640, 768`), baselined per card type in
  `docs/size_ladder_baseline.json` alongside the grid it was measured on. Take exponent intervals over a **sparse**
  subset (`256, 512, 768`): rungs too close together make the exponent a coin flip against noise, and a tolerance
  wide enough for that noise cannot catch the ~1.6 apparent-exponent cliff above (N^2.03 to N^3.62).
- Record per rung **which levers actually fired**, by effect (`served` / `declined` counters), not by config. Fail
  when the fired/dark lever set changes, not only when runtime moves: a run that merely succeeds proves nothing
  about whether the optimised path ran. Discard the first, compile-cold measurement per rung; keeping it inflated
  sigma to 25% and made the exponent band useless.
- No lever lands default-ON on one sequence length. A win at one size that regresses another is a NO-GO.
- Build the ladder from **different** inputs, not one sequence tiled or truncated to each length: a ladder derived
  from one protein holds every other property of it (MSA depth, chain count, ligand count) constant, and reports a
  defect conditioned on that property as if it were conditioned on size.

A one-off sweep does not survive a week of merges: if nothing in the repo *requires* the ladder, it will not run.

## Minimum guard set for a new port

1. A row in the token-axis table naming the bucket, the pad sites, and counters proving `ragged == 0` on a real run,
   plus a test that fails when a `--model` choice has no row.
2. A ragged-length parity test (N not a multiple of 32) against the CPU golden, beside an aligned one, on real
   captured tensors.
3. A pad-poison test: a large constant in the padding leaves the output bit-identical.
4. An A/A control: at an already-aligned length, enabling bucketing is a `torch.equal` no-op.
5. A size-ladder gate arm over the full supported range asserting both the runtime exponent and the per-rung lever
   fired/dark set against a checked-in baseline.
