# Precision and numerics on Tenstorrent

This document decides your dtype policy: storage in `bfloat16` everywhere by default, every matmul and
softmax at `fp32_dest_acc_en=True` and `MathFidelity.HiFi4` until you have measured that a lower
setting is safe, the residual/accumulator stream and the structure-head coordinate arithmetic in
`float32`, and exact linear algebra (SVD, reflection correction) on host torch. Everything else is
decided one experiment at a time against a model-level metric, never by taste.

Read this when a ported module is structurally correct but numerically off, when choosing a dtype for
a new submodule, when about to claim a precision flag costs or saves something, or when two runs of
the same input disagree.

---

## 1. Three knobs, not one

| Knob | What it sets | Where it lives | Cost |
|---|---|---|---|
| **Storage dtype** | how a tensor is held in DRAM/L1 and moved | `dtype=` on `from_torch`/op, `typecast` | bytes/elem: bf16 2, fp32 4, bfloat8_b ~1.06 |
| **Math fidelity** | how many mantissa bits the matrix engine multiplies | `math_fidelity=` in the compute kernel config | cycles/tile: LoFi 16, HiFi2 32, HiFi3 48, HiFi4 64 |
| **Accumulator width** | width of the destination register the matmul sums into | `fp32_dest_acc_en=True/False` | usually free; sometimes blocks a fused kernel |

```python
ckc = ttnn.WormholeComputeKernelConfig(      # or BlackholeComputeKernelConfig
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
out = ttnn.matmul(a, b, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
```

**A fidelity setting is not a dtype.** The Tensix matrix engine multiplies in bf16 even when both
operands are stored fp32; fp32 storage buys fp32 *accumulation*, not an fp32 multiply. Conversely a
flag named `fp32_softmax` or `accurate_softmax` usually gates the compute kernel config (HiFi4 +
`fp32_dest_acc_en` vs an op default of HiFi2 + `math_approx_mode=True`), not any storage dtype. Both
directions of the conflation have shipped:

- A port believed `fp32_softmax=True` traded speed for accuracy. Measured against fp64 on identical
  bf16 operands, the *fused* attention kernel at HiFi4 + `fp32_dest_acc_en` was 1.25-2.49x **more
  accurate** than the "fp32" materialised path and 16-20x faster. The op is bandwidth-bound, so
  fidelity was free. 13 configurations spanned 27% on time and 10x on error.
- The same flag was also the *perf* bug: it routed the op past the fused kernel into a path
  materialising `heads x I^3` scores, upcasting the whole tensor for the reduction, downcasting.
  ~10 GB of DRAM traffic per call, 7.3x under its own measured roof
  (`05-perf-method-and-roofline.md`).

**Guard:** before pricing any precision flag, grep it to its definition and record in a comment
whether the branch changes `dtype=`, `compute_kernel_config=`, or which kernel is dispatched.

---

## 2. What bf16 costs

bf16 is fp32 with 16 mantissa bits deleted: same 8 exponent bits (no overflow surprises on downcast),
7 stored mantissa bits, relative rounding up to 2^-8 ≈ 0.4% per operation against 6e-8 for fp32.

Harmless: storage for linears, projections and MLPs; matmuls **provided** `fp32_dest_acc_en=True` so
the K-axis sum accumulates in fp32 (references in this model family do the same, bf16 operands with
fp32 einsum via autocast disable, so you are matching not degrading); LayerNorm and SDPA as shipped,
where the bf16 kernels have measured *more* accurate than their nominally-fp32 counterparts because
they already accumulate in fp32 while the fp32 variants use a different reduction strategy. "Make it
fp32" is not monotone.

Not harmless:
- **Long-axis accumulation done in the storage dtype**, with no fp32 destination.
- **Softmax denominators.** The reference for this family disables autocast specifically around
  attention-score einsums, the softmax, triangle-attention softmax, triangle-multiplication
  contractions and outer-product-mean, then casts back to bf16. Selective-fp32 on the *reductions*
  with bf16 storage, not whole-op fp32. Match that boundary; do not over-shoot it.
- **LayerNorm variance.** torch normalizes in fp32 even from bf16 input; ttnn does not by default. Six
  norms per block over 48 blocks is 288 places to pick up a directional bias.
- **Coordinates in a diffusion or structure head**: large numbers with small meaningful differences
  inside a 200-step feedback loop. bf16 coordinates through rigid alignment measured 1.9-3.1 Å RMSD
  where fp32 gave 1.45-2.29 Å. Same reason: any difference of large similar numbers (centring,
  whitening, finite differences).
- **Residual streams that grow.** A backbone without residual scaling can take its residual std from
  0.1 to 25 (max ~1e3) over 30 layers, and bf16 error rides that growth.

The lever that is nearly always available and free: **bf16 matmuls with an fp32 residual stream.**
Every linear/norm/attention stays bf16; only the running residual-sum tensor threaded across blocks is
fp32. Zero measured wall-clock cost, and it moved an 18-block diffusion trajectory PCC from 0.867 to
0.902 on the hardest arm. It works on a *pure* block stack terminating at an explicit bf16 cast. If an
interleaved op (cross-attention, gather, embedding) has no cast on its input, the fp32 tensor leaks
into a kernel with no fp32 path and the op falls back to host: ~30x slower, CPU pinned, and on some
builds it wedges the card. Trace the cast boundaries before extending the lever.

There is no on-device fp32 matmul kernel: `dtype=float32` on a matmul is a host fallback, not a
precision option, and a host fallback also changes the answer, so verify device residency before
believing any "fp32 helped" number. `ttnn.embedding` requires bf16 weights, so raising a module's dtype
anywhere that gathers will crash unless the table is round-tripped through bf16 for the gather.

---

## 3. bfloat8_b: what "block float" does to outliers

`bfloat8_b` (BFP8_B) is not FP8. Sixteen consecutive datums share one 8-bit exponent; each carries a
sign and 7 mantissa bits. Datum size is a little over half of bf16, and the *range* is wider than
E4M3/E5M2 because the exponent is shared rather than shrunk.

**Precision inside a block is set by the largest element of that block.** One outlier raises the shared
exponent and every small neighbour in the same 16-element group loses mantissa or flushes toward zero;
mantissa overflow on round-up clamps to all-ones rather than re-normalising the block. So bfloat8_b is
safe on flat-magnitude tensors and dangerous on heavy-tailed ones: attention logits without qk-norm,
pair biases, coordinates.

Measured on this model family: a trunk/pairformer in bfloat8_b is fine (component PCC ~0.99 against
the bf16 arm); a diffusion/structure head in bfloat8_b can collapse the structure outright (radius of
gyration 22.2 Å → 4.7 Å in all-bf8), while **bf8 trunk + bf16 diffusion** was 29% faster at 1.0 Å
Cα-RMSD. `bfloat4_b` measured no speedup plus accuracy loss on these shapes.

Ship the selective form: a `--fast` mode that lowers the trunk only, toggled around the trunk call
rather than set globally, because submodules read the dtype at call time. A 4.6 Å fast-vs-bf16 RMSD on
a synthetic or repeat-heavy sequence is a degenerate-landscape artifact, not accuracy loss: validate on
a structured real target.

---

## 4. Deciding what stays wide: one op, one metric, one experiment

1. Get the whole model running at bf16 storage / HiFi4 / `fp32_dest_acc_en=True`. That is the baseline,
   not fp32.
2. Widen **one** op or submodule (fp32 storage, or fp32 reduction, or host torch).
3. Re-measure the **model-level** metric you ship on: Cα-RMSD or lDDT against a known structure, pLDDT,
   a ranking correlation. Not per-op PCC.
4. Keep the widening only if the metric moves further than the run-to-run spread, in the right
   direction. Otherwise revert, and record that it was tried.

Step 3 is load-bearing, because per-op and model-level instruments disagree in both directions:

- Per-block error measured 1.0-1.3x the bf16 envelope, in tolerance by the instrument the port had used
  throughout. The same blocks chained over one real pass read 32x and 82x. Random per-block error
  compounds identically in the device arm and a torch-bf16 arm and holds the ratio near 1.2x end to
  end. It did not, which proves a *coherent* error term the reference does not have. Halving the two
  worst per-op terms then cut isolated per-block error 2.1x and moved the chained 48-block tap score by
  exactly 0.0%.
- Reverse direction: an op-level fp64 screen showed a fused attention path 1.88x more accurate per
  call. At fold level (1088 calls) it *lost* 0.0085 Spearman on the pair track, one-signed across 3
  seeds, CI outside the shipped arm's seed spread. A single-pass online softmax's rounding residual is
  correlated along the reduction axis where a two-pass materialised softmax's is not; one call's fp64
  comparison cannot see whether 1088 residuals reinforce or cancel.

Swapping one op class for a host-torch twin is an **upper bound on that class's contribution, not a
decomposition**: substituting a class removes whatever error cancellation it provided along with its
own error, so results are non-monotone and do not sum. Use it to rank candidates, never to budget.

---

## 5. One-sidedness predicts whether a widening pays

Before spending a fix on a bf16 op inside a deep residual or recycle chain, measure **what fraction of
its disagreements with torch point the same direction**, not how many elements disagree.

```python
d = (tt_out.float() - ref_out.float()).flatten()
nz = d[d != 0]
print("mismatch %", 100 * nz.numel() / d.numel(),
      "one-sided %", 100 * max((nz > 0).float().mean(), (nz < 0).float().mean()).item())
```

Near 100% one-sided is a real bias that compounds linearly with depth, worth widening. Near 50% is
symmetric rounding noise: "fixing" it redraws a random walk and can land worse. Both outcomes measured
on the same model in the same week.

`ttnn.add`/`add_` on two bf16 operands break rounding ties **away from zero** where torch and JAX break
ties **to even**: 11.16% of elements disagree by 1 ulp at equal operand magnitudes, 100% one-sided.
Over a 48-block x 4-recycle residual trunk (432 adds) the per-block error growth rate was 1.074 device
vs 1.036 for a clean torch-bf16 arm, and `(1.0740/1.0359)**47 = 5.5` correctly predicted the observed
5.2x miss at block 47. Routing that add through fp32 (typecast, add, typecast) is bit-identical to
torch and cost 0.42 s over four trunk passes. The identical fix on bf16 `sigmoid` in the same model,
where disagreements were 10.38% of elements but only 46.2% one-sided, was bit-identical to torch
per-op and a **measured end-to-end regression**, reproduced twice.

Blast radius: across five models sharing one op, the same rounding fix improved one 3.4x, improved
another 37%, and made a third 1.85x worse, because that third had been silently benefiting from
cancellation. Score any shared-op default change per model before flipping the default.

---

## 6. Silent slow paths and path-select caps

**(a) A correct-but-slow path selected implicitly.** An "accurate softmax" or "fp32 path" flag can
bypass the one fused kernel the model's performance depends on. Detect by op trace, not by reading the
flag: enable the device profiler and grep the op CSV for the fused kernel's name. Cross-check the
roofline, since an op sitting >3x under a roof measured on *its own shape* is a path detour before it
is a slow kernel. Materialisation is the tell: the slow path's DRAM traffic grows as a higher power of
sequence length than the fused path's.

**(b) A size or byte-count cap that selects host vs device assembly changes the arithmetic**, even when
the blocks assembled are bit-identical bytes. Raising one such cap moved the output structure digest
and pLDDT deterministically on both repeats of both arms: the host path rebuilds the tensor via
`ttnn.from_torch(..., layout=TILE_LAYOUT)` with **no `memory_config`**, and a host round-trip re-zeroes
tile padding that the device-resident path carries through untouched. Either difference can change a
downstream kernel's program-config choice or its padded lanes.

**Guard:** enumerate every constant gating a host-vs-device or chunked-vs-unchunked branch, and test
the same input just below and just above each cap. "Raising a cap only changes which path executes,
not the math" is a claim to disprove, not a premise.

One case where the "same math" claim *is* sound: bf16 → fp32 (host) → bf16 is bit-exact by
construction (the upcast is zero-extension, the downcast of an already-bf16 value is the identity), so
a device-resident fast path whose only change is deleting such a round-trip between two bf16-native
tensors is parity-identical without re-running the accuracy suite. Not so if either side is
fp32-native, or if the sync also does real work.

---

## 7. Reductions, accumulation order, tolerance

Device reductions parallelise over cores and sum in an order torch never uses. Exact equality against
torch is unattainable and is not a target.

- **Compare against a torch-bf16 arm, not only fp32.** Run the same graph in torch at bf16 storage with
  fp32 accumulation and report `device_error / torch_bf16_error`. 1.0-1.3 means you are inside the
  format's own envelope; an absolute PCC cannot tell you that.
- **Print `ratio = ||mine|| / ||ref||` beside every PCC.** PCC is scale-invariant: a 7.9x norm ratio at
  PCC 0.5 is a scaling bug, not float noise.
- Bands: per-op ~0.999, per-block >0.99, assembled trunk 0.98-0.999. Deep stacks degrade legitimately.
- **PCC has an SNR ceiling.** For true-signal std `sig_R` and device `rmse`, the best achievable is
  `PCC_max = sig_R / sqrt(sig_R**2 + rmse**2)`. A near-zero-signal target can be structurally unable to
  clear a flat bar its siblings clear easily, so compute the ceiling before calling a PCC fail a
  regression. Keep a tighter MAE/RMSE bar as the real guard, since those are not SNR-distorted.
- **A row-sum deficit is not a bug to chase to zero.** A softmax whose rows sum to 0.9955 is real error,
  but three independent attempts to close it moved the fold measurably *worse*: the deficit is
  entangled with a compensating accumulation order. Score any candidate against a torch fp32 reference,
  never arm-vs-arm RMSD, which only proves the structure moved.
- **A stochastic head needs shared draws, re-seeded at sampler entry on both paths**, not a matching
  up-front seed: the device and reference trunks consume the global RNG differently in between and
  desync the stream. Protocol in `02-parity-and-correctness.md`.

---

## 8. Determinism comes before parity

Same input, same card, same seed, same process must give byte-identical output. If it does not, stop.
That is a hardware or allocation-order problem, and every parity number on top of it is noise.

```python
import hashlib
h = lambda t: hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
print([h(model(x)) for _ in range(3)])          # must print three identical strings
```

Then run the same three lines as three **separate processes**: allocation history changes what a
freshly allocated buffer's tile padding contains, and that can change the answer.

What non-determinism has actually been:

- **A faulty card.** One card silently miscomputed some matmuls at a low, location-keyed,
  data-independent rate: transfers clean over 4x256 MB, `concat`/`layer_norm` bit-stable, but one
  512→256 matmul differed on 15/15 repeats in fp32 and 2/31 in bf16. There was no size threshold; the
  original "only at the largest size" framing was a single-sample-per-size artifact. Control: run
  unchanged code N>=3 times on the candidate card before attributing any diff to your change. See
  `09-devices-and-hardware-operations.md`.
- **Uninitialised tile padding.** `ttnn.softmax(dim=-1)` reads the tile padding of the axis it reduces
  over, and not every op guarantees that padding is written. A scatter leaving its output padding as
  stale DRAM meant a process that had already folded a larger structure carried `-inf` and 3e38-scale
  bit patterns in 18 pad columns of a 2702-wide reduction, moving ligand coordinates 0.335 Å. Symptom:
  output depends on what ran earlier in the same process. Fix: make the reduction axis a tile multiple
  and pad the extra keys with the mask's own large-negative value, so their post-`exp` weight is
  exactly 0 by construction. Read padding with `Tensor.cpu().to_torch_with_padded_shape()`: a checksum
  over the logical region alone calls two tensors identical when they differ solely in padding, which
  is exactly how such a bug survives a bisection.
- **A model genuinely irreproducible above a size.** One port was bit-exact at 128 tokens and up to
  3.17 Å RMSD between two identical solo runs at 384. Any hash-equality claim there needs a same-size
  solo-vs-solo floor control before any A/B is interpreted.

---

## 9. Precision-debugging procedure

1. **Hook every reference submodule and capture first-call I/O** (`02-parity-and-correctness.md` has
   the harness). Capture the **first** loop iteration only; later ones carry compounded state.
2. **Replay each device component on the captured input**, reporting PCC *and* norm ratio per tap.
3. **Binary-search downward:** full module, block, op. Wrap every ttnn op in the suspect block logging
   args, output checksum and program-cache count, and find the first op whose output diverges from
   inputs verified bit-identical with `torch.equal`.
4. **Classify the first divergence:**
   - *dtype*: reproduces in torch when you cast the same tensor to bf16; scales with depth; check
     one-sidedness (§5).
   - *layout/padding*: logical outputs match but padded outputs differ; output depends on process
     history; the dimension is not a multiple of 32.
   - *masking*: error concentrated on padded rows/columns, or the valid-only slice is clean while the
     all-atom tensor is not. Compute `mine[:, vmask]` vs `ref[:, vmask]` separately and check
     `ref_mask.sum()` against `N` first.
   - *porting bug*: large, structural, present with random weights, or the norm ratio is a clean
     multiple. A ratio of exactly 76x on a bias term is a convention mismatch, not float noise.
5. **Prove the mechanism, not just the fix.** An end-to-end single-variable test proves causation, not
   mechanism. Two plausible wrong mechanisms were each "confirmed" by a working fix before an op-level
   probe found the real one.

Random-weight tests catch structural bugs and hide magnitude-regime bugs: attention logits reach absmax
111 with real weights where random tests produce ~1, and bf16 softmax then flips the argmax.

---

## 10. Named traps

- **Unmasked padding entering a softmax or a mean.** `TILE_LAYOUT` pads physically to 32 while the
  logical shape stays ragged, and an op reducing over the physical extent reads the tail. In a fused
  attention the bias covers only the logical length, so padded key columns enter the softmax at score 0
  while real scores sit well below 0, `exp(0)` wins, and attention mass lands on padding: measured **72x
  the fp64-reference error at every ragged length, ~1.4x at every aligned one**. A precision screen run
  only at 128 and 512 tokens cannot see this class, because both are multiples of 32. Rule and guards in
  `04-shapes-tiles-and-bucketing.md`. Related: a 1-D `[1,S]` mask `unsqueeze`d to `-1` broadcasts along
  the second token axis only, masking the triangle-multiplication variant that contracts that axis and
  leaving the other variant summing padded rows in unmasked.
- **An epsilon fine in fp32 and destructive in bf16.** A normalisation `eps` of 1e-8 is below bf16
  resolution next to a variance of order 1 and vanishes on the downcast; a 1e-4 guard negligible in fp32
  becomes a visible term. Add eps in fp32 and downcast after, and copy the reference's eps value rather
  than the framework default.
- **A serialization step that silently drops a dtype.** A capture harness filtering pytree leaves with
  `leaf.dtype.kind in "biuf"` drops every `ml_dtypes.bfloat16` array, because that dtype reports
  `kind == "V"` (void). Nothing raises: the run completes, an artifact is written, only the leaf count
  is low. Filter by an explicit allowlist of dtype *objects*, or assert the captured leaf count against
  the expected tree structure.
- **A kernel counter or flag name that describes intent, not execution.** A kernel-variant counter tells
  you which kernel ran, never what dtype flowed through it. A published claim that a reference model
  "ships fp32" survived weeks on that inference; hooking every `nn.Module` forward showed bf16 operands
  with fp32 weights and autocast doing the casting. Measure the dtype at the call site.
- **Unfusing an activation rounds its input.** Pulling a `silu` out of a matmul epilogue swaps the exact
  sigmoid for an approximate one *and* rounds the input, because the boundary is now a materialised
  bf16 tensor instead of an fp32 accumulator. Measured split: rounding hit 23.9% of elements at rms
  1.89e-3, the algorithm swap 9.3% at rms 6.23e-4. Quote the two separately.
- **A dtype kwarg on a binary op does not widen the math.** `ttnn.multiply(a, b, dtype=float32)` computes
  in the input dtype's destination register and widens only on pack (measured max_abs 2.7e-4 vs the
  exact chain). A real fp32 intermediate needs an explicit `typecast` first. `ttnn.typecast` with no
  `memory_config` can also send a downstream fused attention into a multi-minute pathological recompile,
  so pin the source `memory_config()`.
- **The fused attention kernel rejects fp32 operands outright** (bf16/bfloat8_b/bfloat4_b only). The
  viable hybrid is fp32 storage for linears/residuals/norms, cast q/k/v/mask to bf16 at the kernel
  boundary, cast the output back.
- **Small fixtures cannot gate a lever whose accuracy floor sits above them.** A fused kernel measured
  12.87x worse than the materialised path at a single-tile sequence (32) and ~1.5x better from 64 tokens
  up; fixtures of 8-53 tokens read the lever as a 3.4-5.3x regression while it was a win at every real
  size. Likewise, test at the input regime the product actually uses: one model's logit PCC was 0.999 on
  representative inputs and 0.88-0.98 on the all-mask input its generation loop always starts from.

---

## 11. Checklist before claiming a precision result

- [ ] Determinism control (§8) passes, in-process and across processes.
- [ ] Baseline is bf16 storage + HiFi4 + `fp32_dest_acc_en=True`, not fp32.
- [ ] The reference's precision boundary was read from its config defaults, not inferred from
      `autocast(` call sites. The boundary is usually reduction-level, not op-level.
- [ ] Both PCC and norm ratio reported; PCC's SNR ceiling computed if the bar is close.
- [ ] Device error divided by a torch-bf16 arm's error, not only by fp32.
- [ ] Metric is model-level, over >=2 seeds, with the seed spread reported alongside.
- [ ] Any widening's one-sidedness measured before it was proposed.
- [ ] Every host/device path-select cap tested on both sides.
- [ ] Device residency verified for any "fp32 helped" result.
