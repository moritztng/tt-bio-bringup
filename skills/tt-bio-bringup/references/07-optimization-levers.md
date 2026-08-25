# 07: optimization levers, ranked

This document decides the *order* in which you try things once a bio model is numerically correct on Tenstorrent. Twelve
levers, each with the precondition that makes it pay, the magnitude to expect, and the way it fails. Working the list
top-down on a naive port typically takes a first working fold from "far slower than the CPU reference" to the
3x-6x first-week range this document's own schedule predicts, and 1.2x-1.5x again over the following month. Working it bottom-up (custom kernels first, program-config knobs one op at a time)
is the most expensive mistake in this repo's history.

Read this when the port is correct, `05-perf-method-and-roofline.md` has given you a measured decomposition and a floor, and you need
to pick what to build next.

## The ranking

| # | Lever | Precondition that makes it pay | Typical magnitude |
|---|---|---|---|
| 1 | Device residency across calls/iterations | any host round-trip inside a loop | 1.15x-12x |
| 2 | Trace capture | host-dispatch-bound AND shape-stable loop | 1.0x-2.2x |
| 3 | L1 residency / memory config | fits L1 *and* every consumer stays on-chip | 1.02x-2.2x on the op |
| 4 | Op fusion | absorbing kernel is bandwidth-bound, not compute-bound | 1.06x-1.34x on the fold |
| 5 | Sample/design-dim batching | device underutilized at your sequence length | 1.5x-5x throughput |
| 6 | Weight preprocessing + bf16 load | load is a visible share of wall clock | 2.6x on load |
| 7 | Multi-device fanout | axis is independent samples/designs/targets | 1.8x-27.8x, sublinear |
| 8 | Matmul program config | one named op is far off its measured roof | 1.02x-1.2x on the op |
| 9 | Remove per-element ops | any scatter/gather/index op in a hot path | up to 58x on that op |
| 10 | Bucketing | many distinct shapes, compile-bound | bounds a ~60 s cold cost |
| 11 | Host-side work | host share > 5% of wall clock | up to 1/(1 - host share); measure yours |
| 12 | Custom kernels | everything above exhausted, floor says room remains | 1.06x-1.34x, days of work |

Magnitudes are whole-fold unless stated. **Never multiply two levers' headline numbers.** Levers measured from different
baselines do not compose: individually-measured 5.07x and 4.77x arms landed at 4.37x when integrated. Measure one
integrated arm.

---

## 1. Keep the model resident on device

**What it is.** The forward pass hands ttnn tensors to ttnn ops without calling `ttnn.to_torch` / `ttnn.from_torch`
inside a loop. Weights stay uploaded across predictions; loop state (a trunk's pair tensor, a sampler's coordinates)
stays on device across iterations; loop-invariant conditioning is computed once and kept.

**When it pays.** Any recurrence, recycling, or sampling loop that bounces state to host. A port written
module-by-module against a PyTorch reference almost always does, because the reference's loop stayed in torch and only
the inner modules were swapped. Largest win on a first port, most often left on the table.

**How to predict the win.** Count host round-trips per loop and price each at the transfer's real cost.
`ttnn.from_torch` does host-side tile-layout conversion, so an `[L, L, c_z]` pair tensor is expensive in host CPU, not
just PCIe bytes. Tell: the reference loop does not speed up when warm, because host tilization is not a cacheable
kernel. Warm == cold for a loop means the cost is transfer, and residency removes all of it.

**How to implement.** Two patterns, both in-tree. *Resident recurrence loop*: hoist the loop into a ttnn module
(`TrunkRecycle` / `TrunkModule` style in `tt_bio/tenstorrent.py`), keep pair/single state as device tensors, mark
inference-invariant submodules compute-once. Worth ~8-16% of a folding fold; on a design model whose trunk runs at ~100
residues it compounded to ~31% end-to-end because two pipeline stages share the loop. *Resident sampler context*: split
conditioning into step-invariant `f(z_trunk, relpos)` (computed once, kept resident) and per-step (cheap,
`t`-dependent), exposed as `prepare()` / `step()` / `release_cache()`. One diffusion head: 30.6 s → 2.5 s.

**How it fails.** Residency changes accumulation order, so it is not bit-identical to the host-glue path; validate as a
distribution, not one fold (in one case accuracy *improved*, because the old host-fp32 recycle glue was the deviation).
Caching a resident driver on a module that hot-swaps checkpoints mid-run reuses stale weights, and `self._tt_trunk =
trunk` auto-registers as an `nn.Module` submodule and then breaks `load_state_dict`. Residency raises peak device
memory, so the ceiling size that used to fit may now OOM (lever 5).

**How to guard.** Keep the host loop behind an explicit fallback flag (`use_resident_trunk=False`) so a regression is
isolable in one run. Gate the merge on an n≥8 quality comparison, not n=2.

## 2. Trace capture

**What it is.** `ttnn.begin_trace_capture` / `execute_trace` records the exact device instruction stream and replays it
with zero host dispatch.

**When it pays.** Two gates, both required. *Structural*: the loop body is shape-stable and allocates nothing new.
*Economic*: dispatch is actually the cost, proven by a warm eager-vs-traced A/B. On a small-protein diffusion loop of
many tiny atom-level matmuls, a score model went 38.4 → 26.7 ms/step partially traced, 17.1 ms/step whole-step.

**How to predict the win.** Multiply the loop body's op count by ~17-20 µs of per-program dispatch, compare to measured
wall, then confirm the prediction with a warm eager-vs-traced A/B before writing any capture code. Both halves are
load-bearing. A "91.5% host-dispatch-bound" headline for one folding trunk was a unit slip
(`05-perf-method-and-roofline.md` §5 has the arithmetic); measured directly, that trunk was device-compute-bound and
the trace would have bought zero. A 512-residue diffusion loop measured 31.9 s traced against 31.7 s eager, and was
not a candidate either.

**How to implement.** Reserve `trace_region_size` at `ttnn.open_device` (it cannot be added later),
`enable_program_cache()` before warmup, run one warmup pass to compile every kernel, pre-allocate persistent input
buffers, capture, then feed each iteration by `ttnn.copy_host_to_device_tensor` + `execute_trace`.

**How it fails.**
- **No loop primitive.** N Python iterations record N copies of the body. A 200-step sampler cannot be one trace: at a 1
  GB region it overflows the region assert; at 4 GB `begin_trace_capture` hangs indefinitely. A tt-metal limit, not a
  tuning knob.
- Allocating inside the captured region throws, and the replay owns its output buffer, so return a copy.
- **Tracing on top of an already device-resident loop is a no-gain.** Residency already made input staging free, and
  replay collapses op *enqueue* without touching the per-step sync round trip. On one 32-chip fanout, tracing away 399
  of 400 op launches removed 0.001 of 1.007 host cores per fold. If the remaining cost is `ttnn.to_torch` drains, cut
  `n_sync`, not launches.
- Useless for one-shot graphs and shape-varying loops. Slicing a stacked `[T, ...]` RNG buffer alone produces T distinct
  programs and a compile explosion.

**How to guard.** Keep the eager path behind a flag, assert shape-stability at capture, and A/A the traced arm against
itself for a noise floor before quoting a delta.

## 3. L1 residency and memory config

**What it is.** Where a tensor lives (DRAM interleaved, L1 interleaved, L1 sharded) and whether a matmul writes its
result to DRAM or leaves it in L1 for the next op. Blackhole L1 is 1,572,864 B per core as the device reports it, but budget against the allocator's
1,461,760 B per bank (`08-memory-and-residency.md` §1: sizing on the device number admits configs
that do not fit on an idle chip); measured DRAM bandwidth through
ttnn is ~435 GB/s and machine balance at HiFi4 is ~231 FLOP/byte, which almost nothing in a bio model clears. These
workloads are bandwidth and latency bound, so where the bytes live is the dominant device-side lever.

**When it pays.** The tensor fits L1 at full size *and* every consumer of it also stays on-chip.

**How to predict the win.** Model the op as `compute + write`, not `max(compute, write)`. For a tall/narrow matmul
(small K, large M) the output write can dominate DRAM traffic at modest FLOPs: one dense projection moved 33.95 MB of
which 25.17 MB was the result write, and the additive model predicted its 0.17443 ms wall to 0.1% while the overlapped
model was wrong by 45%. Ablate by removing the DRAM destination and measuring the delta; no device profiler needed.

**How to implement.** Keep the result in L1 via program config under two guards: `in0_block_w` must be the whole of K (a
narrower K block is a different accumulation order and breaks bit-exactness), and both operands plus the result must fit
aggregate L1 with room for the block's other allocations. Measured: projection 0.1733 → 0.0800 ms (2.17x),
projection+heads+attention chain 1.97x, whole trunk block 1.079x, bit-exact by `torch.equal`.

**How it fails.**
- **A guard written against the logical shape checks the wrong number.** TILE_LAYOUT pads the last two dims to 32
  independently and `fuse_batch` folds leading dims into M *after* padding, so real M is `prod(leading) * ceil32(rows)`.
  An `M % 32 == 0` predicate on a `[S, S, c_z]` pair tensor tests `S² % 32`, true only by coincidence. One merged
  "2.17x" fix never fired in a single real fold at any size; its A/B looked bit-exact because both arms ran the same
  code.
- **An L1-resident output nobody consumes on-chip is pure overhead.** The same projection was 2.66x faster isolated and
  1.45x *slower* end-to-end because the attention consumer read q/k/v from DRAM regardless. Check the whole consumer
  chain, not the producer op.
- **The row-blocking pincer.** If the tensor does not fit L1 at full size, residency needs row blocking, and ttnn has no
  row-range operand, so each block materializes its own slice. On a 134 MB pair tensor blocking cost +1.820 ms/call
  while residency returned 1.116 ms: net +0.704 ms/call, bit-exact, 37 noise floors clear. Row blocking is not a cheap
  substitute for fusion.
- **A budget tuned at one size goes dark at another, silently.** One model's L1 stack switched itself off above ~640
  residues on three independent gates with no error and no log line. The tell was the fold's own scaling exponent
  jumping from N^2.03 (256→512) to N^3.62 (512→768): 11.1% of the fold at 768, 14.2% at 1024.
- **A budget that prices only the tiled dims crashes on the first `batch>1` caller.** A helper budgeting `chunk *
  seq_len²` ignored a leading replica dim, widened the chunk 32→128 at 5 diffusion samples, and threw a circular-buffer
  clash in an unrelated op several calls downstream. A grid-scaled budget must also include the matmul's own *static*
  circular buffers on the same core.

**How to guard.** Put a fire counter on every capacity gate and assert in CI that it admits on a real `predict_one` fold
at a small, a middle, and a large size. A lever that fires at your tuned size and is dark elsewhere is the defect, not
the tuning. Standalone microbenchmarks feed tile-aligned N (128/256/384) and never trip a broken `% 32` guard.

## 4. Op fusion

**What it is.** Replacing a sequence of ttnn ops with one kernel pass, either an existing fused ttnn op or a new one.

**When it pays.** *Direct*: the IR shows `matmul` + `add` where `ttnn.linear` exists, or an unweighted `layer_norm`
followed by affine modulation where the weight/bias operands would do. *Theoretical*: the pattern is one kernel in
torch/triton (`matmul → softmax → matmul` = SDPA) but ttnn has no equivalent yet. The second class is a kernel feature
request, not a today-lever.

**How to predict the win. This is the counterintuitive one.** The subtractive screen ("the DRAM traffic this op costs is
the prize if fusion deletes it") is correct only when the *absorbing* kernel is bandwidth-bound and has spare compute to
hide the absorbed math behind. When the absorber is already compute-bound, the absorbed arithmetic gets *un-hidden* and
runs in addition. In one candidate the absorber sat at 50% of its DRAM roof while the op sat at 88% of its: the fused
arm's compute floor was 0.790-1.185 ms/call against a 0.612 ms/call gate and an 0.862 ms/call prize. NO-GO in four
minutes of microbenchmark instead of a week of kernel work. So: measure the absorber's roofline color first, then price
the added arithmetic with a single on/off toggle of one in-DST SFPU pass already in a kernel (one sigmoid pass measured
0.6630 ms/call, one integer-round pass 0.1976) rather than building a prototype.

**How to implement.** Dump TTIR and TTNN IR, align by `loc(#locNNNN)`, enumerate candidate patterns with an instance
count (the count decides whether it is worth fixing). For a new kernel, a shipped ttnn op's program factory transcribes
into a Python `ProgramDescriptor` at 1.002-1.018x of native and bit-exact, so `ttnn.generic_op` ships kernels from the
production wheel with no tt-metal build.

**How it fails.**
- **Fusion into a transaction-bound kernel returns about two thirds of what it looks like it deletes.** One merged
  kernel predicted ~4.876 ms/call recovered and delivered 63-73%; 1.32-1.79 ms/call reappeared because two page reads
  before one barrier expose roughly twice the reader latency. Price a ~1/3 comeback by default.
- **A fusion's addressable size range is computable before you build it.** One fused softmax was bit-exact only where
  two independently derived blockings coincide, which was exactly 50% of engaged key widths. Enumerate that first and
  make the guard decline safely on the rest.
- Copying a compute-kernel config from a neighbouring op is a silent correctness bug. One fold called SDPA with
  `compute_kernel_config=None` so the stock op ran HiFi2/approx; the new kernel that inherited the trunk's HiFi4 config
  compiled, ran at the expected speed, and was wrong across 88% of elements. Only `torch.equal` caught it.
- `ttnn.slice` is **always a copy, never a view** (its `buffer_address()` differs from the parent's), so "write into a
  slice instead of concat" is not an optimization. `ttnn.experimental.slice_write` is ROW_MAJOR only.

**How to guard.** Bit-exactness at the whole-fold level, not per chunk. Allocation *sequence* is a hidden input to
reduction order: one rewrite was `torch.equal` on all 52 chunks including the ragged tail and still moved the fold's
confidence score and output digest.

## 5. Batching along a sample or design dimension

**What it is.** Running N diffusion samples / designs / sequences as one batched pass instead of N serial passes.

**When it pays.** When the device is underutilized at your sequence length, which for a bio model is most of the time
below ~512 residues. Measured: best-of-5 folding 4.06 → 2.44 s warm, with B=4 costing about 1.20x of B=1. On a protein
LM, 16 seq/s → 250 seq/s at batch 8 (~15x), saturating at batch 16-32 (batch 64 adds ~5%).

**How to predict the win.** Place the fold on the roofline (`05-perf-method-and-roofline.md`). Below N≈1024 in matmul terms you are
measuring dispatch, not compute, and batching converts serial dispatch into one program: if the op count per sample is
unchanged, batching removes `(N-1)` dispatch of everything.

**How to implement.** Replicate conditioning to `[N, ...]` and run one pass. Then add the escape hatch: chunk over the
sample axis at a calibrated `B·L²` budget with distinct per-chunk seeds, a shrink-on-OOM net (`chunk = max(1, k//2)`), a
grid-aware default (halve on smaller grids) and an env override. Leave the B=1 path byte-identical.

**How it fails.**
- Replication happens in more places than you think. A confidence head that also replicates pair state to `[N, L, L, c]`
  OOMs at large N·L even when the sampler is fine. Chunk *both* stages.
- An OOM names the allocation that failed to place, not the cause. One capacity wall at 1024 residues was two live fp32
  `[S, H, S, S]` score tensors (16 GiB each); the fix that mattered was deleting one live copy (fold the scale into the
  bias-add, reduce softmax in place), not blocking the failing allocation. Count co-live tensors of the failing shape
  before building a blocking lever.
- Batching is lossless for a deterministic encoder and lossy for a stochastic decoder: batched design slots can drift
  0.5-2.8 Å from their unbatched selves, so each slot is a different design. A batched design counts as throughput only
  if each slot is an independent, production-valid output.
- Best-of-N only helps when confidence is informative. On low-confidence targets the confidence score can be flat or
  anti-correlated with true error, and best-of-N picks a worse structure.

**How to guard.** Prove batching numerically with a **row-independence** test, not a PCC threshold: hold the batch shape
`(B, L_bucketed)` fixed, change the *content* of the other rows, assert the row's output is bit-exact (Δ=0). That
separates shape-driven accumulation reordering (fine) from contamination (a bug). Batching is also the lever that most
often closes a gap against a GPU baseline, and conversely a batch-1-only comparison is not a throughput comparison: on
design models a GPU gains 1.79-5.2x from batching where the TT card gains ~1.04x. Use matched batch on both sides, or
state which batch you used.

## 6. Weight preprocessing

**What it is.** Doing the key remap, transpose, tilize and dtype conversion once at load, caching the device-ready
tensors instead of rebuilding them per call.

**When it pays.** Any batch/folder workload or server, and any model whose load time is visible next to predict time (a
6B-parameter protein LM is minutes of load and ~2% of a fold's compute). Time `load` and `predict` separately and warm;
if load is >10% of a typical request it is a lever.

**How to implement.** Load the checkpoint **straight to bf16 at read time** rather than reading fp32 and converting per
weight: 62.6 → 23.5 s (2.6x), bit-identical at default settings. Bake the transform into the upload (ttnn's
`torch_to_tt` transposes linear weights by default; layer-norm weights and biases upload *without* transpose). Cache
uploaded weights per key so the hierarchy uploads once, and keep weights resident across predictions.

**How it fails.** Gate the bf16 read for quantization-sensitive paths: bf16→bfloat8_b rounding differs from
fp32→bfloat8_b and moves output quality, so a block-fp8 `--fast` mode should keep the fp32 read. Two measured dead ends,
do not retry: parallel `from_torch` with a thread pool (ttnn serializes the device queue, 1.0x) and row-major upload
with on-device tilize (net slower than host TILE_LAYOUT). Per-weight upload time is irreducible today. Remap tables
silently drop keys.

**How to guard.** Assert the loaded-key count and fail on any unconsumed checkpoint key not on an explicit skip list.
Keep a load-time regression number, and a whole-model bit-exactness check between the fp32-read and bf16-read paths at
default settings.

## 7. Multi-device fanout

**What it is.** One model worker per chip, each pinned so its physical chip appears as logical device 0.

**When it pays.** When the axis is embarrassingly parallel: independent targets, designs, sequences. Real sharding of
one model across chips is a different and much harder project, not this lever.

**How to predict the win.** Assume sublinear and measure. A 32-chip fanout on independent folds measured 27.8x (86.8%
efficiency), and the residual was named: ~1.007 host CPU cores of synchronization per fold, independent of host, card
generation, and op-launch count.

**How to implement.** Set `TT_VISIBLE_DEVICES=<chip>` **before** importing ttnn, defer every ttnn import inside
functions so a spawned child can pin before import, and use `multiprocessing.get_context("spawn")` (fork reuses the
parent's already-imported ttnn). Length-sort and stripe work round-robin; gather by id and restore input order.

**How it fails.**
- **Weight-load contention destroys scaling for large models.** A 6B-parameter LM's fanout is flat or worse beyond 2
  chips because N processes each pull multiple GB through the same host at once. Only smaller models at large N scaled
  cleanly (~2x at 4 chips).
- **Per-card throughput can collapse super-linearly on one shared-bus box**: for one design model, 2-chip beat 4-chip.
  Real scaling needs independent hosts, not more chips per box.
- **CPU oversubscription on host-bound stages.** Each per-chip process defaults torch to all physical cores, so N
  workers thrash the CPU N-fold: 4 concurrent CPU-bound shards at `OMP_NUM_THREADS=16` took 102 s vs 2.5 s at 8 (~40x).
  Set `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = len(os.sched_getaffinity(0)) // n_devices` via `env.setdefault` before the
  child imports torch.
- Launching 4+ device-opening processes at once deadlocks on device-init and queue-lock contention (0.7% CPU, `Sl`
  state, silent). Stagger launches.
- `TT_VISIBLE_DEVICES` uses the tool's UMD chip ID (PCI BDF order), not the `/dev/tenstorrent/N` node number. If your
  detection code reads `/dev` directly, pin with both.

**How to guard.** A bit-exactness parity harness running the reference and each shard sequentially in short-lived pinned
subprocesses on one chip, at batch size 1 so sharded == single-shot by construction. Verify the top-level orchestrator
end-to-end with `devices=[N]` (one shard), so the real tempdir/spawn/await/reassemble/cleanup path is covered, not just
its primitives.

## 8. Matmul specifics

**What it is.** The matmul program config: 1D vs 2D multicast, `in0_block_w`, per-core M/N, the compute grid passed per
op, fidelity (LoFi/HiFi2/HiFi4) and `fp32_dest_acc_en`.

**When it pays.** When a *named* matmul sits far off its measured roof and you know why. Measured roofs on one Blackhole
part: 161.14 TFLOP/s bf16 LoFi, 140.35 HiFi2, **100.55 HiFi4**, 435.2 GB/s DRAM. Most bio ops run HiFi4, so 100.55 is
the honest denominator, not the spec sheet.

**How to predict the win.** Compute arithmetic intensity and compare to machine balance (231 FLOP/byte at HiFi4). Below
that you cannot be compute-bound and no block-shape or fidelity knob helps: go back to lever 3 or 4. One ttnn pathology
worth checking: `k_tiles_per_core = div_up(k_tiles, num_cores)` collapses to 1 for any matmul with `k_tiles <
num_cores`, which on a ~110-130-core grid is *every* dense projection in a Pairformer-style trunk at `c_z=256` (8
K-tiles).

**How it fails.**
- **At production sizes the library matmul usually wins, and a hand-written one is a last resort.** Head to head,
  `ttnn.matmul` hit 48.9 TFLOP/s at N=1024 against 3.2-3.9 for a custom kernel lacking L1 reuse and multicast blocking,
  and reached 98.7% of the compute roof once fed from L1. Swapping the custom op into a real fold made it *slower* (105
  s vs 86 s). Its micro-benchmark "15x" was against a whole ttnn module at tiny N, not against `ttnn.matmul` at
  production N.
- Knobs that measured ~nothing on this op class, do not re-litigate: LoFi/HiFi2 on a non-math-bound op (1.04x),
  `fp32_dest_acc_en=False` (1.3% for real numerics risk), flattening 4D→2D (0.3%, ttnn already folds leading dims), 2D
  multicast on a narrow/tall shape.
- Changing the global compute grid to gain 1.02x on one op changes every op. Pass the grid per op instead.
- `from tt_bio.tenstorrent import CORE_GRID_MAIN` captures the value at *import* time, before the device is opened and
  the active grid configured. Probes must read the module attribute after device open.

**How to guard.** Cache block configs keyed off the **weight** shape, not the activation, so the lookup is
size-invariant by construction. Re-measure any config lever at three sizes before it lands default-on: a shipped ratio
can differ by 0.12 absolute between two core-grid geometries with byte-identical output.

## 9. Reduce per-element ops

**What it is.** Removing `scatter`, `gather`, index ops, and Python loops of `slice` + `concat` from hot paths.

**When it pays.** Always, when they are hot. On Blackhole these ops are **per-element rate limited, not bandwidth
limited**. On a 45.1M-element tensor: `ttnn.add` 0.649 ms (69.5 G elem/s), `ttnn.scatter` 4.657 ms (9.7 G elem/s, 7.2x
slower), `ttnn.scatter_add` 9.65 ms (14.9x), `ttnn.gather` 38.0 ms (1.2 G elem/s, **58x**). Proof it is element rate: a
`bfloat8_b` tensor at half the bytes cost 4.662 ms, 0.1% different.

**How to predict the win.** Count elements traversed, not bytes moved, and price at ~10-14 cycles/element. At larger
scales it is worse than linear: one `ttnn.gather` cost the *product* of the gathered dimension and the output count.

**How to implement.** Restructure the indirection away.
- A relative-position one-hot times a weight matrix *is* a row lookup: `onehot(bin) @ Wᵀ = Wᵀ[bin]`, exactly. This
  replaced an `[N, N, 139]` one-hot and a 73 MB upload with small integer-bin uploads.
- A windowed K/V gather written as a Python slice loop (85% of a diffusion step) collapses to one `ttnn.embedding` with
  a uint32 ROW_MAJOR index built once: 113 → 41 ms/step, bit-identical.
- If the op must stay, call it less. One decoder invoked `scatter` twice per step with bit-identical inputs across
  recycles; caching was free, bit-exact, and worth ~7% of the step.

**How it fails.** `ttnn.scatter` rejects fp32 outright and rejects int32/uint32 tensors whose scatter row exceeds 256
elements (route index building through bf16 + typecast). No knob moves the rate: index dtype, ROW_MAJOR vs TILE,
`sub_core_grids` and `out=` preallocation all measured flat. Do not design a port around scatter/gather as the cheap
sparse-indirection primitive, and do not assume a fused kernel rescues it. The op itself is the wall.

## 10. Bucketing and recompilation avoidance

**What it is.** Padding a variable-length axis to a fixed multiple so different inputs reuse compiled kernels. Full
treatment in `04-shapes-tiles-and-bucketing.md` §5; this is the ranking summary.

**When it pays.** When per-shape JIT compile dominates. Measure your own per-shape cost
(`04-shapes-tiles-and-bucketing.md` §5): ~0.85 s per new shape for a small LM, tens of seconds for a
folding trunk. Compiles are disk-cached (`~/.cache/tt-metal-cache`), so it is one-time per shape per machine; an
in-process program cache removes a further ~8-10 s/shape. Bucketing bounds the distinct-shape count so the cache
saturates after one run. Predict by counting distinct lifetime shapes times per-shape compile, against the padded work
you pay on every input forever.

**How it fails.**
- **Over-bucketing an O(L³) op is a permanent loss.** Coarsening a trunk's pad multiple 32→64 costs up to `128³/96³ ≈
  2.4x` of trunk work on every fold forever for a length just above a boundary. One such change was implemented and
  reverted.
- **Bucketing is bit-exact only where the padding is attention-masked.** Fully-masked padded key chunks are exact no-ops
  in SDPA (PCC 1.000000, maxdiff 0.0). Padding a *contraction* size changes bf16 accumulation: pad-32 vs pad-64 on one
  trunk gave PCC 0.9997 with max|Δ| = 192 and a real +0.065 Å shift. Keep contraction axes at 32.
- **The axis may already be folded into the batch dim upstream.** One windowed-attention model reshapes to `(B, NW,
  WINDOW, -1)` before the encoder, so one dispatched program already covers all windows and bucketing that axis removes
  *zero* programs. An op census, not a byte count, reveals this.

**How to guard.** Zero the padded keys/values explicitly so exactness does not depend on bf16 `exp(-inf)`. Never call
`disable_and_clear_program_cache()` per input; enable once and never clear.

## 11. Host-side work

**What it is.** Featurization, MSA processing, structure assembly, rigid alignment, sampling glue: precompute what is
loop-invariant, cache what is recomputed with identical inputs, parallelize what is independent, move to device only
what is heavy. Some things stay on host permanently: exact SVD for Kabsch alignment runs on host fp32 because the device
alternative loses accuracy.

**When it pays.** Once device work has shrunk. A per-card lead over a GPU can be *entirely* host and dispatch: in one
comparison a TT card's 23.240 s beat a GPU's 30.813 s wall, but the GPU's own utilization trace showed 19.78-19.84 s
device-busy, so the GPU was ~1.17x faster on device seconds and the lead was their dispatch bubble plus 2.08 s of host
featurization our port did in 0.069 s. Name the mechanism or the claim reads as a silicon claim it cannot support.

**How it fails. Every cache you add needs a key test.** A memo on a template embedding was correct in its stated
justification (the embedding is constant in one of its inputs) and had **no key**, so the first design's template
silently substituted into every later design in the same process. Output stayed plausible, nothing threw, and an earlier
investigation misattributed the resulting cold/warm divergence to on-device state. Same class: a hoisted precomputation
that dropped a scaling factor present in the inline path produced a ~12.5 Å regression.

**How to guard.** For every cache: run input A, then B, then A again; assert B's output differs from A's *and* A's
second output is bit-identical to its first. Prefer source-tensor-identity keys (`if self._src is not z: recompute`)
over no key. For a hoisted precompute, assert bit-exactness against the inline path on a real input before deleting the
inline code.

## 12. Custom kernels

Last lever, covered in `10-custom-kernels.md`. Preconditions: every lever above measured and either landed or explicitly
NO-GO; a floor derived from measured roofs; the predicted landing written down *before* building. Expect 1.06x-1.34x on
the fold for days of work, and note that the analysis which kills a megakernel often produces a smaller kernel worth 84%
of it in one day. Two structural facts make it cheaper than it sounds: a shipped ttnn op's program factory transcribes
into a Python `ProgramDescriptor` at 1.002-1.018x of native and bit-exact, so `ttnn.generic_op` deploys kernels from the
production wheel with no tt-metal build; and any custom kernel calling `ttnn.split_work_to_cores` inherits its
unsatisfiable-split holes (it throws whenever `units > cores` and `units % cores` is a non-zero multiple of the grid
height, 358 of the first 4000 unit counts on a 13x10 grid), so plan a sub-grid search and a safe decline to the stock
op.

---

## First week of optimization on a fresh port

Order matters because each step changes what the next measurement means. Do not reorder to put a kernel first.

| Day | Action | Expected cumulative |
|---|---|---|
| 0 | Warm-vs-cold baseline: two same-length inputs in one process on one chip, take the **second** one's time. Enable the program cache once. Discard the first-ever run on a fresh machine (a cold cache inflates it ~10x). Measure your card's roofs with the committed micro-benchmark, not the spec sheet. | baseline + named limiter |
| 1 | Stage decomposition with `ttnn.synchronize_device` before every clock stop. Expect a counterintuitive answer: on one folding model the O(L³) trunk was ~67% and the 6B-parameter LM ~2%. | hotspot named |
| 1-2 | **Lever 1: device residency.** Kill every host round-trip in the trunk loop and the sampler loop; split conditioning into invariant and per-step. | 1.15x-2x, up to 12x on a sampler |
| 2 | **Lever 11 (cheap half): recompute hoisting** for anything >25% of per-step cost whose inputs exclude the step variable (one diffusion head: 364 → 113 ms/step). **Lever 9: per-element ops**. slice loops and one-hot matmuls become `ttnn.embedding` lookups (113 → 41 ms/step, bit-identical). | +1.5x-4x on those stages |
| 3 | **Lever 6: bf16 load + weight cache.** **Lever 10: bucketing**, masked-attention axes only, contraction axes stay at 32. | load 2.6x, cold cost bounded |
| 3-4 | **Lever 5: sample-dim batching** with the OOM chunking net, if the product exposes a best-of-N or multi-design axis. | 1.5x-5x throughput |
| 4 | Re-decompose. Levers correctly dismissed as too small now matter: one confidence head went 5% → 15% of a fold once the trunk shrank around it. **"Too small" is not a permanent verdict.** Then **lever 2: trace**, only if a warm eager-vs-traced A/B shows a real delta and the loop is not already resident. | 1.0x-2.2x on that loop |
| 5 | **Lever 3: L1 residency** with the consumer-chain check and a fire counter. **Lever 8: matmul program config** for any op still far off its measured roof. **Lever 7: multi-device fanout** if throughput rather than latency is the product metric. | +1.05x-1.15x, plus 1.8x-3.5x per 4 chips |
| 6 | **Lever 4: fusion**, screened absorber-first. Most candidates die in a four-minute microbenchmark; an evidenced NO-GO is a full pass. | 1.06x-1.34x |
| 6-7 | Multi-size validation: re-run the ladder at a small rung, your tuned size, and a large rung. Any gate that fires at one and is dark at another is a defect, not a tradeoff. | no dark gates |

A realistic first-week outcome on a naive but correct port is 3x-6x, dominated by levers 1, 5, 9 and 11. Levers 3, 4
and 8 start at the end of that week, in the rows above, and most of their value lands in the second month along with
lever 12: roughly 1.2x-1.5x combined on top of a stack that already has the first four. If a
week produces less than 2x, the most likely cause is that the measurement was never warm, never synced, or never
decomposed.
