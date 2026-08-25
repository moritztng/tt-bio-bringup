# Memory model and residency

This document decides where every tensor in your port lives, how you size it before writing code,
and what to do when it does not fit. The default: activations interleaved in DRAM, weights uploaded
once and resident, L1 only where a producer and *every* consumer of a tensor stay on chip. The rest
is the arithmetic and the failure modes that make that default hold or break.

Read this when choosing a `memory_config`, sizing a chunk width, hitting an OOM or a
circular-buffer clash, or deciding whether to keep state on device between calls.

## 1. The hierarchy

| Level | Capacity | Granularity | Managed by |
|---|---|---|---|
| L1 SRAM | ~1.5 MB **per core** (allocator reports 1,461,760 B/bank on a Blackhole p150-class part) | per-core bank | you, explicitly |
| Device DRAM | 32 GB on a p150-class Blackhole part, ~12 GB per Wormhole chip | interleaved over 8 banks (Blackhole) / 12 (Wormhole) | ttnn allocator |
| Host RAM | whatever the box has | pageable | Python/torch |

Aggregate L1 is small but not negligible, and the number depends on which grid you are on. Say
which. The p150 hardware grid is 13x10, so 130 x 1,461,760 B = **190.0 MB**; tt-bio clamps its main
compute grid to 11x10, so 110 x 1,461,760 B = **160.8 MB**. Both use the allocator's per-bank figure
rather than the device's reported total, for the reason in §4. A `[512, 512, 256]` bf16 pair tensor
is 134.22 MB: **70.6% of the full grid's L1, 83.5% of the clamped grid's**. Either way it kills most
naive "keep the pair track resident" plans, and a plan sized on one grid and run on the other is the
failure in `13-failure-atlas.md` §1.

**The mental shift from CUDA.** There is no cache hierarchy. L1 is an explicitly addressed per-core
scratchpad and a compute kernel can address nothing else. Two *independent* allocators share each
bank: the tensor allocator places buffers, and the program's static circular buffers (CBs) are
planned bottom-up at program-creation time. Neither knows about the other, so a live tensor and the
next op's CBs can overlap and you get `Statically allocated circular buffers in program N clash with
L1 buffers on core range [...]` instead of an out-of-memory error. That message is a *shape*
problem, not a capacity problem.

Do not quote datasheet bandwidths; measure them (`05-perf-method-and-roofline.md`). Measured on one
Blackhole part: DRAM read roof 435.2 GB/s, write roof 277.6 GB/s, compute 161.14 TFLOP/s at LoFi and
100.55 at HiFi4 (`05-perf-method-and-roofline.md` §3 measures all three fidelities; most bio ops run
HiFi4, so quote that one) (one
data-movement RISC issuing a writeback reaches only ~59% of the write roof).

`ttnn.get_max_worker_l1_unreserved_size()` is **not** the number to budget against: on Blackhole it
reads 1,532,416 B while the allocator reports 1,461,760 B/bank, so a gate sized on the device number
admits configs that do not fit on a completely idle device. Read the allocator once and cache it;
`get_memory_view` behaves like a pipeline drain (~6 us), never call it per-op in a timed region.

```python
l1_per_bank = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
```

## 2. Memory configs and shard specs

- **Interleaved DRAM** (`ttnn.DRAM_MEMORY_CONFIG`), the default. Pages round-robin across banks; an
  allocation of S bytes needs `S / num_banks` **contiguous** bytes in *every* bank.
- **Interleaved L1** (`ttnn.L1_MEMORY_CONFIG`), same scheme, tiny pool. Easy to hit by accident: an
  intermediate created without an explicit `memory_config` may default to L1. Pin the buffer type on
  every step of a multi-step move, or a DRAM→DRAM permute places its intermediate in L1 and dies.
- **Height / width / block sharded.** Height splits rows (M), width splits the last dim (N), block
  splits both across a 2D core grid.

A shard spec is `(core range set, shard shape, orientation)`. Rules that bite: shard shape is in
**elements** and both dims must be multiples of 32 in TILE_LAYOUT; block sharding needs
`M_tiles % grid_y == 0` and `N_tiles % grid_x == 0`, else ttnn refuses or silently derives a
different grid; `ttnn.create_sharded_memory_config` will not derive a shard shape for a
`CoreRangeSet`, so pass it explicitly when the cores are not a full rectangle; and shard count has
hardware caps (one p150-class part refuses more than 110 shards despite a larger compute grid).

A **prime** tile count is the pathological shard case: it collapses the derived grid to a single
10-core column whose CB region inflates to ~86% of its bank, versus ~45% at a composite tile count.
The worst case is not the biggest input.

Choose: start every tensor in interleaved DRAM. Move to L1 only when you can name every consumer and
show none round-trips through DRAM. One op-level L1-residency win measured 2.66x in isolation and
ran 1.45x *slower* end-to-end, because the downstream attention read its operands from DRAM anyway
and residency just added a copy nobody used.

## 3. Sizing arithmetic, before you write code

A tile is 32x32 = 1024 elements. Bytes per tile: `float32` 4096, `bfloat16` 2048, `bfloat8_b` 1088
(1024 data + 64 shared-exponent), `bfloat4_b` 576. TILE_LAYOUT pads the **last two dims** to 32
independently (`04-shapes-tiles-and-bucketing.md`), then leading dims multiply:

```
tiles = prod(shape[:-2]) * ceil(shape[-2]/32) * ceil(shape[-1]/32)
bytes = tiles * bytes_per_tile[dtype]
```

The trap: with `fuse_batch=True` the M a matmul runs is `prod(leading_dims) * ceil32(rows)`, computed
**after** padding, not the logical flattened product, so a `% 32` predicate on the logical shape
checks the wrong number. A shipped guard computed `M = S²` for an `[S, S, c]` pair tensor and refused
at every real target size (298 aa gives 88804, `% 32 = 4`); the "2.17x" it advertised never fired in
a real fold, and its A/B looked bit-exact because both arms ran identical code.

**Worked example: can a projection's result stay in L1?** `[1, 1, 16384, 256] @ [256, 768]` in bf16
on a 13x10 grid (130 cores): `m_tiles=512`, `k_tiles=8`, `n_tiles=24`, `tile=2048`, `l1=1,461,760`.

```
per_core_M = smallest divisor of m_tiles >= ceil(512/130) = 4        -> 4
output CB + fp32 accumulation CB : per_core_M * n_tiles * (2048+4096) =   589,824
ttnn fixed per-core overhead     : 128 KiB                           =   131,072
the resident result's own share  : ceil(512*24/130) * 2048           =   194,560
per K block (in0 + in1)          : (per_core_M + n_tiles) * 2048     =    57,344
total = 589,824 + 131,072 + 194,560 + 8 * 57,344                     = 1,374,208 B  -> fits (94%)
```

The same op at 224 tokens: `m_tiles = ceil(224*224/32) = 1568`, smallest divisor >= `ceil(1568/130)`
is 14, so the first term alone is `14*24*6144 = 2,064,384 B` > L1. No legal whole-K block exists and
the guard must decline to the DRAM-writing path. Three rules this encodes:

1. Price the **result**, not just the operands. ttnn allocates the output before the program factory
   places a single CB, so it comes off the same budget.
2. Price **every leading dimension**. A helper written and tested at the caller's default `batch=1`
   encodes that default silently. One shipped helper budgeted `chunk * seq_len²` while the tensors
   it guarded were `[batch, chunk, seq, seq]`; the first caller with `diffusion_samples > 1` widened
   the chunk 32→128 and threw a CB clash in an unrelated op several calls downstream.
3. Include fixed overhead. Two figures are in circulation and they are not the same measurement:
   **131,072 B** is the round 128 KiB the worked example above budgets with, and **111,104 B** is
   what a fit to measured allocations returned, exact across two grids and two wheels. Budget with
   131,072 and you have ~20 KB of slack per core; fit your own if you are trying to explain a
   clash rather than avoid one. Do not mix them inside one calculation.

**Guard hygiene.** Any capacity guard needs a counter proving it ADMITS at least once on a real
input, not on a synthetic benchmark at N=128/256/384 (already tile multiples): log served/declined,
assert `served > 0`. A guard that declines 100% of calls is indistinguishable from one that works by
every bit-exactness check you can write.

## 4. OOM is often about the number of allocations, not the total size

```
TT_FATAL @ .../bank_manager.cpp:439 ... Out of Memory: Not enough space to allocate 17179869184 B
  DRAM buffer across 8 banks ... (allocated: ..., free: ..., largest free block: 1107296256 B)

TT_THROW ... Statically allocated circular buffers in program 527 clash with L1 buffers on core
  range [(0,0)-(2,9)]. L1 buffer allocated at X and static circular buffer region ends at Y
```

The first is capacity or fragmentation, the second a shape step. **Capture at least 2000 characters
of exception text, from both ends.** `TT_FATAL` puts the diagnostic parenthetical *last* behind a
fixed ~78-char prefix, so `str(exc)[:200]` records nothing usable; `TT_THROW` puts its payload in the
*middle*, followed by the frames naming the op, so head+tail alone elides diagnosis and op together.

Detection signal for fragmentation: total free is comfortable, largest contiguous block is not.

```python
mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
used = (mv.total_bytes_per_bank - mv.total_bytes_free_per_bank) * mv.num_banks
lcf  = mv.largest_contiguous_bytes_free_per_bank      # min over banks if it returns a list
```

Fingerprint of an allocation-count problem: 2.0-2.4x the requested space is free and the largest free
block is 0.89-0.93x of the request. Shrinking the tensor 10% will not fix that; reducing how many
buffers are live will.

Same lesson, second form: an OOM names the allocation that failed to place, and that allocation is
often the last straw among several co-live tensors of the same shape. One model held **two** live
fp32 `[S, n_heads, S, S]` score tensors across a multiply/add/softmax chain, 16 GiB each at 1024
tokens. The fix that mattered was deleting one live copy (fold the scale into the bias-add's
activation, reduce softmax in place), not blocking the survivor; that made the *unblocked* path fit
at every size up to 768. **Count live tensors of the failing shape before building a chunking lever.**

## 5. Allocation order changes what fits, and sometimes what you compute

Same peak footprint, different order, different outcome:

- Rebinding a name (`z = zb`) does not free the old buffer where you stopped reading it. It still
  held its L1 bank when the next matmul's CBs were planned 17 lines later, producing a clash that
  looked like a capacity cliff across a whole band of sizes. `ttnn.deallocate(z)` at the last read
  fixed it in two different models at zero cost.
- Hoisting intermediates into named locals for readability kept ~3.7 GiB alive past the point the
  original inline expression freed them, and moved the OOM to a different allocation.

Moving an allocation (or a free) *earlier* is therefore a legitimate OOM fix, and usually cheaper
than any chunking scheme. The uncomfortable corollary: allocation sequence is a hidden input to
reduction order on this hardware. A rewrite whose chunks were individually `torch.equal` to the
original (max abs diff exactly 0.0 on all 52 chunks including the ragged tail) still moved the
whole-model output digest. **Verify bit-exactness at whole-output granularity, never only per
intermediate.**

## 6. `ttnn.slice` is not a view

A slice allocates and copies: `slice.buffer_address() != parent.buffer_address()`, always.

- Any plan of the form "write into a slice of a preallocated buffer instead of concatenating" is not
  an optimization; the slice allocation *is* the copy you were trying to delete.
  `ttnn.experimental.slice_write` does not rescue it (ROW_MAJOR only).
- A slice inside a hot loop is a hidden allocation per iteration and a hidden DRAM round trip. A
  row-blocking scheme that materialized each block cost +1.820 ms/call (+805 MB/call of re-read,
  re-write and closing concat) while the L1 residency it enabled returned only 1.116 ms: a net loss,
  bit-exact, 37 A/A floors clear of noise.

Only a kernel that reads and writes at a page offset deletes assembly cost, i.e. real fusion. Row
blocking is not a cheap substitute; do the blocking-cost-versus-residency-return arithmetic
explicitly before crediting a blocking scheme with any time.

## 7. Deallocation discipline

ttnn frees a buffer when its last Python reference drops, which sounds like GC is enough. It is not,
for one reason: CBs are planned at *program creation*, so what matters is whether the buffer is live
at that instant, not whether it will be collected soon after. In any chain longer than about three
ops: call `ttnn.deallocate(t)` at the point of last read, and do not bind hot-path intermediates to
locals unless you deallocate them.

Prefer prevention over recovery. A catch-the-throw-and-memoize-a-fallback wrapper is a valid
backstop, but it still throws internally on every call; a `deallocate` at the right line means the
clash never happens. If you ship a fallback, assert its refusal set is non-empty in the test that
claims to exercise it, otherwise an empty set in a passing arm proves nothing.

## 8. Chunking, in order of preference

1. **Sample / batch axis first.** Samples are independent, so the split is an exact partition. Run
   the largest sub-batch that fits a `B * L²` budget, loop until all N are drawn, halve on OOM:

   ```python
   chunk = max(1, min(n, budget // (L * L)))
   while done < n:
       k = min(chunk, n - done)
       try:
           out.append(run(..., multiplicity=k)); done += k
       except RuntimeError as exc:
           if k == 1 or not _is_oom(exc): raise
           chunk = max(1, k // 2)
   ```
   Classify the exception on `("out of memory", "circular buffer", "clash", "not enough space",
   "allocate")`. Give each chunk a distinct seed. Make the budget grid-aware (halve on a small
   Wormhole grid) and expose an env override.
2. **A row-local spatial axis second.** Legal only where the region's ops are row-local: softmax
   reduces over the last dim only and a bias broadcasts over the leading dim, so blocking the leading
   dim is a partition rather than a reordering. Illegal across triangle ops, attention over the
   chunked axis, or any normalization whose statistics span it.
3. **Return the chunks, not a concatenation**, when every consumer is chunk-wise. A terminal concat
   needs a third full-size buffer (source + parts + destination); that 3x peak is what makes a
   chunked path still OOM.

Key the budget on the allocation's own **byte count**, never on sequence length, and set the
threshold well below the smallest input you support rather than near the measured boundary: that
boundary is usually driven by a different tensor than the one you are gating.

**Proving a chunked path equals the unchunked one.** In one process, run both and compare final host
tensors with `torch.equal` (max abs diff exactly 0.0) at three sizes minimum: chunk divides the axis,
chunk does not (ragged tail), and two chunked axes with different padded lengths. Then compare the
whole-output digest (`02-parity-and-correctness.md`). Two widths agreeing bit-for-bit is **not**
evidence a width-varying scheme is inert: ttnn picks block sizes, core grids and memory configs from
the shapes it is handed, and the chunk dim is one of those shapes, so it takes a third width to
falsify. Check also that a divisibility invariant covers *every* axis it logically applies to: one
enforced on q but not k silently declined a fused kernel on 100% of calls (served 0 of 5184),
invisible at every tile-aligned benchmark size.

## 9. Host RAM and addressable range

The device path can be fine while the host dies. A preparation-side census tensor scaling as
atom-axis² was 14.53 GB at 4576 atoms and ~25.6 GB at 6080, so a 30 GB host with no swap was
OOM-killed ~3% short of the target size while the card had ample free DRAM. Detection: the process
dies with exit 137 / a kernel OOM-killer line and **no ttnn exception at all**. Check host RAM when
picking a machine for a large fixture, not just whether a card is free.

Inside a kernel, a fused region has an addressable size range, and that range is computable before
you build. One fused-softmax region is bit-exact only where two independently derived blockings
coincide (`find_max_divisor(Wt, 4)` versus the consumer matmul's `in0_block_w`), true for exactly 50%
of engaged key widths. Both are one-line functions of the shape. Enumerate the coincidence *before*
building, and ask of every fusion: what fraction of the size axis does this serve, and does its guard
decline safely on the rest?

## 10. Residency across calls

- **Weights resident per device.** Upload once at worker load, fold many. Load straight to bf16 at
  read time rather than fp32-then-convert (2.6x faster load, bit-identical).
- **Resident recurrence loop.** Keep the pair/trunk state on device across recycling iterations and
  make inference-invariant submodules compute-once. A reference loop re-transfers the `L²·c` pair
  tensor host↔device every iteration and cannot be warmed away, because host-side tile-layout
  conversion in `from_torch` is not a cacheable kernel. Measured ~32% off a trunk stage, compounding
  to ~31% end-to-end where several pipeline stages share the loop.
- **Resident sampler context.** Split conditioning into step-invariant (`f(trunk_state, relpos)`,
  computed once, kept resident) versus per-step (t-dependent, tiny), transferring only the noisy
  coordinates per step. Measured 30.6 s → 2.5 s on a diffusion sampler.

**The API surface has to change for this to be possible.** A single `forward(inputs) -> outputs`
cannot express it. The shape that works is `prepare(...) / step(...) / release_cache()`: `prepare`
uploads and computes everything invariant and returns nothing to the host, `step` takes and returns
only the small per-iteration tensors, `release_cache` frees the resident set so the next input size
starts clean. "No OOM at the size ceiling" is largely a question of calling `release_cache` at the
right size-adaptive points.

Residency and trace capture compete rather than compound: a trace's payoff is capped by the host
dispatch it replaces, but replay re-stages every input at a fixed replay address each step, and those
inputs are exactly the ones residency made resident. Measured −12.3% and −74.5% end-to-end on two
regions whose isolated screen said 1.25x.

**The correctness risk is stale device-side state.** Two mechanisms to guard:

- **A cached module driver plus hot-swapped weights.** If the caller can change checkpoints mid-run,
  do not cache the driver across calls: it silently reuses stale weights, and a plain
  `self._trunk = mod` also auto-registers as an `nn.Module` submodule with no state dict and crashes
  `load_state_dict`. Found by testing, not inspection.
- **Shape-dependent masks and buffers.** An invalidation that fires when crossing a threshold in one
  direction is a latent bug in the other. One port invalidated its device mask cache on the way
  *down* through a crop threshold but not on the way back *up*, so every input above the threshold
  died on the **second** prediction in a long-lived process. It stayed invisible through a whole pass
  because every fixture was single-shot. The robust fix sweeps every submodule at the top of
  `forward` rather than tracking which tensors need invalidating.

**Standing rule: any conditional cache needs an invalidation test that would fail if the key were
wrong.** Concretely, the test drives one long-lived object across the threshold at least twice in
both orders (down-then-up, up-then-down), and separately asserts that perturbing the cache key
changes the result. A test that still passes with a deliberately wrong key is testing nothing.

## 11. Diagnostic flow: "it OOMs at size N but works at N-1"

1. **Read the whole exception** (2000 chars, both ends) and classify: DRAM allocator OOM
   (`bank_manager.cpp`, "Not enough space to allocate ... across K banks") or L1 static-CB clash.
2. **CB clash means a shape step, not capacity.** Quote the overlap as `cb_end − addr`, never as
   `held + CB > bank`: the bank constant algebraically cancels, so the overlap is card-independent
   while `held_per_bank` is not. Never carry one card's bank constant to another.
3. **Sweep by tile-count factorization, not round token counts.** A prime tile count collapses the
   derived grid and can need the *largest* CB allocation at the *smallest* tensor in the band, so a
   fix validated at one size routinely leaves neighbours broken.
4. **DRAM OOM: compare `largest_contiguous_bytes_free_per_bank` with `request / num_banks`.** Free
   total much larger than the request with the largest block just under it means fragmentation.
   Reduce the number of live buffers or move an allocation earlier; do not shrink the tensor.
5. **Count live tensors of the failing shape at the failure point** and delete a redundant or
   derivable copy before building a blocking or chunking lever.
6. **Bisect the boundary one token at a time, fresh process each side.** Both edges are typically one
   token wide and are usually *different* mechanisms (a CB-side tile-boundary jump at one end, the L1
   term vanishing to DRAM at the other, which self-heals and is not a fix).
7. **Do not trust a single pass near a boundary.** In a long-lived serving process the clashing
   address depends on that process's own allocator history, so a size that folds today can throw
   tomorrow. Publish the largest size strictly *below* the first observed failure; never interpolate
   across it.
8. **Guard it.** Add the failing size to the release gate's size ladder plus a footprint check
   (`12-testing-and-gates.md`): a footprint regression is invisible to a numerical parity fixture, so
   measure DRAM high-water at the largest supported input. Never A/B performance with the footprint
   probe enabled; it drains the pipeline and roughly doubles fold time.

See `05-perf-method-and-roofline.md` for measuring roofs and A/A floors before crediting any lever
here with time.
