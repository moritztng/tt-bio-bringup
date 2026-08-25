# Failure atlas

This document decides your first three moves when a bring-up misbehaves. Find your symptom in
the reader's own words, read the mechanism, run the check, apply the fix, and land the guard so
the class cannot come back. Every entry here is a mechanism that was found the expensive way on a
real biomolecular port to Tenstorrent; none of them is a thing you would guess from the stack
trace. Where two mechanisms produce the same symptom, both are listed, cheapest check first.

**Read this when** a port is producing wrong numbers, hanging, running slow, or measuring
differently than it did an hour ago, and you do not yet have a hypothesis worth spending a card on.

Sibling docs: `05-perf-method-and-roofline.md` (how to measure before you optimise),
`12-testing-and-gates.md` (the full gate-failure catalogue),
`02-parity-and-correctness.md` and `03-precision-and-numerics.md` (parity protocol and bf16 policy).

---

## 1. "The output is wrong": accuracy off by a lot

### PCC sits at 0.79-0.9 everywhere, not garbage, not NaN
**Mechanism.** A convention mismatch between two AF3-family codebases, not a port bug. The most
common pair: one repo's diffusion module takes raw `x_noisy` and applies the EDM `c_in` scaling
and preconditioning internally; the other takes an already-scaled `x` and returns a bare
`x_update` with EDM applied outside by the sampler. Plausible-but-wrong output is the signature.
**Check.** Print the magnitude of the tensor entering the denoiser: `~sigma` means raw,
`~1` means already `c_in`-scaled. Then check whether the network returns `x_update` or the
preconditioned `denoised`. Also check where the golden capture hook sits: on the bare network or
on the sampler.
**Fix.** Extract a pure `_denoise_net` with no `c_in` and no EDM, and let each sampler apply its
own convention around it. One such fix moved step-0 PCC 0.791 -> 0.99996.
**Guard.** A step-0 denoiser parity test that asserts both the raw and the preconditioned tensor,
not just one of them.

### Error is ~70x the reference at some lengths and ~1.4x at others
**Mechanism.** `ttnn.TILE_LAYOUT` pads physically to 32 on both tile axes while the logical shape
stays ragged. A fused SDPA kernel reduces over the *physical* extent, and the additive bias only
covers the logical length, so padded key columns enter softmax at score 0 while real scores sit
well below 0. `exp(0)` wins and the row's attention mass lands on padding.
**Check.** Screen the op against an fp64 reference at a deliberately ragged token count (117, 298)
using *real captured tensors*. Synthetic operands at a ragged length reproduce only ~1.5x of the
72x, because the damage needs the real score distribution as well as the ragged length.
**Fix.** Bucket every token axis to a multiple of 32 (pad + mask + slice back, all three), and
also mask the tail inside the kernel. Bucketing hides a ragged-input kernel bug, it does not fix it.
**Guard.** A parity sweep whose *first* fixture is a non-tile-aligned length. A sweep made only of
multiples of 32 is structurally blind to this whole class.

### One of two triangle-multiplication variants is dirty, the other is clean
**Mechanism.** A 1-D `[1,S]` pair mask, unsqueezed to `[1,S,1]` and multiplied into a `[1,S,S,C]`
tensor, broadcasts along the second token axis only. The outgoing variant contracts that axis and
is masked correctly; the incoming variant contracts the *first* axis and sums padded rows in
unmasked.
**Check.** Add a `--maskmode` axis switch (`1d` / `i` / `j` / `pair`) and see which variant goes
dirty with which axis. The flip is the proof, not the score.
**Fix.** Build the real 2-D outer-product mask instead of relying on 1-D broadcast.
**Guard.** Assert mask rank at the entry of every op that contracts a token axis; reject rank-1
masks in modules that have two contraction directions.

### Output is off by a constant-looking factor that grows with MSA depth
**Mechanism.** An OuterProductMean-style module divides by the pair norm either *before* the
output projection (bias applied post-divide) or *after* it (bias included at full strength). Two
upstream families disagree. Picking the wrong one over-weights `proj_o.bias` by exactly the MSA
row count.
**Check.** Crop the token axis across a tile-32 boundary and watch the error. Flat inflation at
every crop means a scale convention bug; a step at the boundary means a padding leak.
**Fix.** Set the scale-bias flag per call site, never per module default. The removed error term
should match the closed form `bias * (1 - 1/n_rows)` to several digits; verify it does.
**Guard.** A unit test that asserts the closed-form residual, so the convention cannot silently
flip back.

### An auxiliary head's output is exactly 2x too large
**Mechanism.** The attribute-swap porting pattern (monkey-patch a device module into an unmodified
reference model's attribute slot) puts your wrapper at a call site whose calling convention you
never read. Here the reference's own forward already pre-symmetrised `z + z.transpose(-2,-3)`
before calling the head, and the device head symmetrised again.
**Check.** Read the *reference's* forward at the call site, not your unit test. Then
`max_abs` the head's output against a reference of known magnitude.
**Fix.** Delete the duplicated work in the class that is not the one deciding the convention, and
fix the unit test to feed what the real call site feeds.
**Guard.** For every attribute-swapped component, one test that drives the *reference forward*
with the device component installed, and a numeric anchor on every output-only head. Output-only
heads (distogram, PAE, PDE) never feed back into coordinates, so an affine error in them is
invisible to CIF-hash and pLDDT anchors.

### Every prediction after the first in a process is subtly wrong
**Mechanism.** An unkeyed memo. A cache whose correctness argument is "constant within one unit of
work" leaks across units when one long-lived model instance serves many units. A template-pair
memo justified by "constant in `pair` for a single template" is *not* constant in the template
features, which depend on the input coordinates.
**Check.** Run the same population twice with a different row leading. If the answer depends on
*which* row led (not merely on position), it is a leak, not order-dependence. Then enumerate every
memo and dict cache in the port and write down its key.
**Fix.** Hash every input the correctness argument depends on into the key, including masks.
**Guard.** A two-input same-process test that asserts input B's result is identical whether it ran
first or second.

### Per-op parity is green, chained parity is 30-80x off
**Mechanism.** The device path carries a *coherent* (one-sided) error term the torch reference does
not. Random per-op error compounds identically in both arms and holds the ratio flat; a directional
bias compounds multiplicatively. Measured injector in one case: `ttnn.add` on two bf16 operands
breaks rounding ties away from zero where torch breaks ties to even, at 11% of elements.
**Check.** Fit a per-block growth rate in each arm. `(1.0740/1.0359)^47 = 5.5` predicted an
observed 5.2x miss at block 47. Then substitute one op class at a time for its host-torch twin
*in chain*.
**Fix.** Route the offending elementwise op through fp32 (typecast, op, typecast) if its
disagreements are near-100% one-sided. See §2 for why you must check one-sidedness first.
**Guard.** A chained tap gate at real depth. No per-op tolerance table can see this.

### Residues are silently placed at the origin, conditioning looks empty
**Mechanism.** A parser subclass overrode a pipeline-defining list of filter/fixup steps and
dropped the parent's `remove_water` / `remove_hydrogens`. Hydrogens then reach the CCD name
matcher, which drops a residue when `len(matched) <= len(mismatched)`, so any residue with more H
than heavy atoms is marked fully unresolved.
**Check.** Diff the subclass's overridden method against the parent's, counting what got *dropped*,
not what got added. Confirm both ways with a hydrogen-free control file.
**Fix.** Re-add the parent's steps, or call `super()` and extend.
**Guard.** Assert the resolved-residue count against the input residue count after featurisation.

### The model runs, the weights "load", the output is structurally wrong
**Mechanism.** `load_state_dict(..., strict=False)` exists to tolerate legitimately unused keys,
and also silently drops every shape-mismatched tensor, so a wrong config runs on uninitialised
weights.
**Check.** Run the port's determinism protocol first. If reference-self and device-self are both
1.00000 while device-vs-reference is 0.2-0.4, the problem is structural (wrong config, wrong
checkpoint, wrong shape), not numerical. Then dump the checkpoint's own tensor shapes.
**Fix.** Read the checkpoint's `config.json` and raise on any architecture mismatch before
loading. Keep `strict=False` only for extra or missing *keys*.
**Guard.** A loader assertion that every declared shape matches the checkpoint's actual shape.

### A capture/tap artifact is complete-looking and missing most of its tensors
**Mechanism.** Leaf filtering by `dtype.kind in "biuf"`. `ml_dtypes.bfloat16` reports
`dtype.kind == "V"` (void), so every bf16 array is treated as non-numeric and dropped. Nothing
raises; the capture writes a file and reports success.
**Check.** Count captured leaves against the expected tree structure. A 16-of-20 tap-group capture
looks fine unless you count.
**Fix.** Filter with an explicit allowlist of dtype *objects*, not a kind string.
**Guard.** Assert expected-vs-actual leaf count in the capture harness itself.

### The model computes the right answer and ships the wrong one
**Mechanism.** Computed-then-dropped output. The checkpoint carries a head, the port builds and
runs it every step, and three downstream layers discard it: the post-processor drops the masked
argmax, the sampler reads only coordinates, and the writer names residues from the *input* feature
tensor.
**Check.** Before declaring an upstream capability absent, list the checkpoint's own weight keys.
That settles in minutes what several call-graph reviews miss.
**Fix.** Thread the value through to the writer. Verify coordinates are bit-identical pre/post, so
you know it was pure plumbing.
**Guard.** A test asserting the shipped artifact contains the head's output, not just that the head
ran.

### A quality validator passes on output that is obviously broken
**Mechanism.** The validator sees copied/fixed input and generated output in the same object, and
scores their union. Copied motif residues dilute the defect below the threshold. A failing cell can
be the only honest one in the table.
**Check.** Recompute the metric over the generated subset alone and compare.
**Fix.** Restrict the validator to the generated part explicitly, by index.
**Guard.** A negative control: feed the validator an output where every designed position is
degenerate and assert it fails.

### A featurizer key marked "stochastic, expected to vary" is actually wrong
**Mechanism.** A reference-molecule builder discarded 3D coordinates without calling
`AssignStereochemistryFrom3D`, so every ligand lost its chirality and the conformer generator drew
a random handedness per stereocentre. The port's featurizer diff excluded that key as noisy, so a
34/35 key match hid it.
**Check.** For every key excluded as noisy, ask what should be invariant *under* the noise, and
check that instead. Here: the sign of the signed volume at each stereocentre.
**Fix.** Assign stereochemistry from the coordinates before discarding them.
**Guard.** An invariant assertion per excluded key. An exclusion is a blind spot, not a null result.

### A user-facing toggle is echoed back correctly and ignored
**Mechanism.** A boolean implemented as "add a flag when true, omit otherwise" is silently correct
only if the underlying engine's default already matches the false state. Validation, echo-back and
logging all look consistent because the flag really was omitted as requested.
**Check.** Run the job with the toggle off and grep the engine's own log for the behaviour, not
the API response.
**Fix.** Emit an explicit flag for both states when the engine's default disagrees.
**Guard.** A sweep cell that asserts the *knob was honoured*, not merely that the job succeeded.

### A shared CLI default silently belongs to a different model
**Mechanism.** One flag, several models, one convention. A `--recycling_steps` default of 3 (one
family's convention) applied to a model whose spec is a fixed 10-cycle trunk gives an unconverged
trunk, a bimodal ensemble, and a confidence head that then mis-ranks it. The confidence head looks
like the bug; it is a symptom.
**Check.** Diff every shared CLI default against each model's upstream config, not against the
other models in your repo.
**Fix.** Default to `None` and resolve per model inside `predict()`, honouring explicit overrides.
**Guard.** A test asserting the resolved value per model id.

### A fast head-layout op returns a correctly-shaped, wrong-valued tensor
**Mechanism.** `nlp_concat_heads` and `nlp_create_qkv_heads` assume `head_dim` is a multiple of 32
and do not check. At `head_dim=48` they read 64-wide heads out of a 48-wide tensor. No error at the
wrong-value site.
**Check.** `max_abs` against the two-op `permute` + `reshape` reference at your actual head_dim.
**Fix.** Gate adoption on `head_dim % 32 == 0`; pad head_dim to 64 otherwise. When aligned they are
bit-exact and 6-7x faster.
**Guard.** An assertion on `head_dim % 32` at the call site, with the reference form as fallback.

### A rigid-alignment RMSD is large for a structure that is an exact rigid copy
**Mechanism.** A bespoke Kabsch helper derives the rotation mapping P onto Q, then applies its
transpose to Q instead of to P. Near-identity comparisons still score about right, which is why it
survives spot checks; only a genuinely rigid-but-displaced pair exposes it.
**Check.** `RMSD(P, apply(P, R, t)) == 0` for a rigidly transformed copy of arbitrary P.
**Fix.** Correct the operand/transpose pairing.
**Guard.** That one-line identity assertion next to every bespoke superposition helper. And when a
diagnostic probe contradicts a green gate on the same files, re-score the probe's own output with
the gate's statistic before believing the probe.

### A fold is wrong only when the process has already run a bigger one
**Mechanism.** `ttnn.softmax(dim=-1)` reads the tile padding of the axis it reduces over, and no op
guarantees that padding is written. A scatter into a dense mask leaves whatever the freshly
allocated DRAM buffer held, which in a long-lived process is `-inf` and 3e38-scale bit patterns
from an earlier, larger fold.
**Check.** `Tensor.cpu().to_torch_with_padded_shape()` reads tile padding non-destructively. Wrap
every op in the block, log arg checksums, and find the first divergence from bit-identical inputs.
Clearing the program cache and recompiling returns the same wrong answer, which rules out a
cache mis-hit.
**Fix.** Make the reduced axis a tile multiple, or explicitly zero/`-1e4`-fill the padding.
**Guard.** A same-process ladder that folds a large input before a small one and asserts the small
one matches its solo result.

---

## 2. "The output is slightly wrong": accuracy off by a little

### bf16 error that is invisible per op and fatal end to end
**Mechanism.** One-sidedness, not magnitude, predicts compounding. An op whose disagreements with
torch are ~100% one-signed injects a bias that grows linearly with chain length. An op whose
disagreements are ~50/50 is symmetric noise, and "fixing" it just redraws the random walk.
**Check.** Measure the *fraction of disagreements pointing the same direction*, not the mismatch
rate. One op at 11% mismatch and 100% one-sided was worth a clean win; a sibling at 10% mismatch
and 46% one-sided was a measured regression when widened to fp32.
**Fix.** Widen only the one-sided ops.
**Guard.** Record the one-sidedness number in the port's precision doc next to each widened op, so
the next pass does not re-propose the symmetric one.

### "Make it fp32" makes it worse
**Mechanism.** Which kernel implementation ttnn ships matters more than the nominal dtype. bf16
SDPA and bf16 LayerNorm can beat their fp32 variants because the accuracy-critical accumulation is
already covered by `fp32_dest_acc_en` while the fp32 variant uses a different accumulation
strategy. Separately, on-device ttnn fp32 and CPU-torch fp32 are not the same fp32.
**Check.** A/B the single op swap at the real input magnitude before committing to the class.
**Fix.** Targeted hybrid: bf16 SDPA and LayerNorm kept, fp32 matmuls, fp32 residual stream, fp32
host-side embedding lookup. The fused SDPA kernel rejects fp32 inputs outright (bf16/bf8/bf4 only),
so the viable pattern is fp32 storage with a bf16 cast at the SDPA call boundary.
**Guard.** Never mark a precision lever a dead end from a *host* implementation alone; re-test it
as an on-device dtype change before concluding.

### A "precision" flag turns out to be a compute-config flag
**Mechanism.** A flag named `fp32_softmax` can gate a `compute_kernel_config` (HiFi level,
`fp32_dest_acc`, `math_approx`) rather than a storage dtype. Turning it on can route the op past
the fused kernel into a materialised path, buying nothing and costing an order of magnitude.
**Check.** Read the flag's consumer. Then score both arms against an fp64 reference on identical
bf16 operands. In one measured case the fused kernel at HiFi4 + `fp32_dest_acc` was 1.25-2.49x
*more accurate* than the "fp32" path it was believed to trade against, and 16-20x faster.
**Fix.** Set fidelity explicitly on the fused path. On a bandwidth-bound op, fidelity is free.
**Guard.** A comment at the flag's definition naming which knob it actually sets.

### Unfusing an activation costs more than the activation
**Mechanism.** An activation fused into a matmul epilogue reads its input at full fp32 accumulator
precision. Unfusing it for kernel simplicity forces the input to round to storage dtype at the new
op boundary. Measured split: input rounding hit 23.9% of elements at rms 1.89e-3; the approximate
sigmoid alone hit 9.3% at 6.23e-4. The rounding half dominates.
**Check.** Score algorithm-only (accumulator preserved) separately from input-rounded.
**Fix.** Keep the op fused, or read the accumulator directly.
**Guard.** Do not accept "a sibling kernel already ships this error class" without checking whether
that sibling also rounds its input.

### A widen folded into a binary op changes nothing
**Mechanism.** Passing `dtype=float32` to `ttnn.multiply` / `ttnn.add` widens only when *packing
the output*; the intermediate math still runs in the input dtype's destination register. Measured
max_abs 2.7e-4 against the exact chain.
**Check.** Compare against an explicit typecast-then-op chain.
**Fix.** Insert an explicit typed op before the binary.
**Guard.** Treat any `dtype=` kwarg on a ttnn binary as a packing hint, never a precision fix.

### A softmax row sum that is not exactly 1.0
**Mechanism.** The deficit looks like an obvious bug and the instrument is cheap and legible, so it
gets chased to zero. It is often entangled with a compensating accumulation order that tracks the
true softmax *better* than the corrected one. Three independent instances in one op family closed
the same way.
**Check.** `sum_i(exp~_i / sum_j exp~_j) = 1` exactly for any exp approximation, so a non-1.0 row
sum refutes "it is just the approximation" rather than confirming it. Use it to pick which
mechanism is broken, never as an elimination argument.
**Fix.** Score any candidate against an fp32 reference, never arm-vs-arm RMSD. Arm-vs-arm proves
the fix *moves* the structure, never that it moves it closer to truth.
**Guard.** At least two seeds; a one-seed accuracy comparison has an empty spread and cannot pass
or fail by construction.

### Device-vs-golden RMSD is multiple Angstroms on a stochastic model
**Mechanism.** The golden was produced by an RNG backend you cannot reproduce (CUDA Philox vs CPU
MT19937 vs ttnn, which has no `randn`). Two independent noise realisations of the same model
diverge by the full run-to-run spread, which swamps any real port error.
**Check.** One case: device-vs-golden 7.93 A, device-vs-CPU-reference sharing an identical seeded
noise realisation 1.23 A at PCC 0.99997.
**Fix.** Generate the full draw sequence once, in the exact order the sampler consumes it, save it,
and feed the identical draws to both samplers.
**Guard.** Before trusting any same-seed conclusion, confirm bit-reproducibility: same seed twice
must give an identical output hash. See §4 for why it often does not.

### A bf16 host round-trip that costs nothing
**Mechanism.** bf16 -> fp32 is zero-extension, and fp32 -> bf16 of an already-bf16 value is the
identity. A `--fast` path whose only change is deleting a bf16 host sync between two bf16-native
tensors is parity-by-construction.
**Check.** Confirm both sides are bf16-native and the sync does no other work (no cast, no
reduction).
**Fix.** Claim parity by construction with a bit-exact spot check rather than re-running the full
suite.
**Guard.** The claim is void the moment either side is fp32-native.

---

## 3. "It is wrong only at some sizes"

### Scaling exponent jumps from N^2 to N^3.6 inside one interval
**Mechanism.** Capacity gates switching off silently. In one measured interval, three at once: a
fused-kernel precondition declining 1120/1120 calls, an L1 headroom check answering DRAM above a
threshold, and an SDPA q-chunk program config overflowing its per-core L1 budget and falling back.
None throws, none logs.
**Check.** Fit the log-log exponent between consecutive size rungs. There is no other signal.
**Fix.** Extend the gate's validity range or key it on the resource it consumes. Re-fitting the
constant at a second point just moves the cliff.
**Guard.** A lever census that counts which levers fire *by effect* at every rung, not only at the
size the campaign tuned at. A one-off sweep does not survive a week of merges.

### A guard that reports firing correctly and is dead at every real size
**Mechanism.** A `% 32` predicate written against the *logical* flattened M. TILE_LAYOUT pads the
last two dims independently and `fuse_batch` folds leading dims into M *after* that padding, so the
real M is `prod(leading) * ceil32(rows)`. For a `[S,S,C]` pair tensor the naive M is `S^2`, a
multiple of 32 only by coincidence.
**Check.** Evaluate the predicate against a real fold's padded tile count, not a synthetic
benchmark N. Every standalone harness feeds 128/256/320/384, all already tile multiples.
**Fix.** Derive tiles from the padded shape; clamp the last slice instead of requiring an even
divisor.
**Guard.** A guard's validation harness must prove the guard *admits* on a real input. A dead guard
on both arms of an A/B is trivially bit-exact and reads as a clean pass.

### A shipped kernel that has never executed in production
**Mechanism.** An `eligible_shape` predicate rejecting `batch != 1`, validated against a synthetic
default while production runs batch 2 (or `--diffusion_samples > 1`, which replicates the pair
state to one copy per sample before the trunk).
**Check.** Grep every `eligible_*` / `_fits` / guard predicate for the real batch, size and dtype
the shipping call site uses. Do not trust the merge's own reported speedup.
**Fix.** Loop the kernel per item inside the batch instead of rejecting the batch.
**Guard.** A `served > 0` assertion on the lever arm of any A/B over a gated kernel. Without it you
get an A/A wearing an A/B's clothes: byte-identical arms and a confident zero.

### A refusal cache that retires a shape class forever
**Mechanism.** "Try L1, fall back to DRAM on RuntimeError" implemented as a `set` of refused shape
keys. The first refusal at a given key retires it for the rest of the process. In one case a single
refusal sent 434 of the remaining 435 calls per recycle down four full DRAM passes over a tensor
cubic in tokens.
**Check.** `grep` for `.add(` on a refusal-tracking set. That is the signature.
**Fix.** Replace the set with a per-key cap dict that backs off by one row on refusal. The first
accepted size is also the fastest.
**Guard.** Audit sibling latches in the same module; they cluster.

### An L1 budget that prices some dimensions and not others
**Mechanism.** A chunk-size helper budgeting `chunk * seq_len^2` against a measured L1 ceiling,
while every tensor the chunk loop holds is `[batch, chunk, seq, seq]`. The helper was never given
`batch` and encodes the caller's default of 1. The first sign is a circular-buffer clash in an
unrelated op several calls downstream.
**Check.** Confirm the formula prices *every leading dimension* of every tensor the guarded region
holds in L1, not just the dims that vary in the helper's own sweep.
**Fix.** Pass the missing dims. A fix that only narrows a chunk is free of any parity decision;
narrowing a channel-loop chunk is a finer partition of an independent sum, bit-exact at every width.
**Guard.** A crash-boundary sweep on more than one core-grid shape, since a CB sizing safe on a
130-core grid can overflow a 110-core grid at the identical logical tensor size.

### A capacity constant that gates host-vs-device assembly changes the math
**Mechanism.** Raising a byte cap that selects between a host-assembled and a device-resident
concat is *not* bit-exact, even though the concat itself is. The host path rebuilds via
`ttnn.from_torch(..., layout=TILE_LAYOUT)` with no `memory_config`, and the round trip re-zeroes
tile padding the device path carried through untouched.
**Check.** Fold at a real size on both arms and diff the output hash and pLDDT. In one case both
moved deterministically on both repeats.
**Fix.** Treat any host-vs-device path-select constant as a numerics lever until proven otherwise
at a real fold, not at the op in isolation.
**Guard.** A/B one call site at a time when several share the constant.

### A fused kernel breaks every length that is not a multiple of 32
**Mechanism.** A kernel that allocates its own output took logical dims from `xa.padded_shape`
instead of `xa.shape`, so the output came back logical-320 where the residual add expected
logical-298 and the broadcast threw. Invisible at 512, where logical equals padded.
**Check.** Run the kernel's parity sweep at 20, 62, 76, 96, 100, 298, 300, 512.
**Fix.** Build the output `ttnn.Shape` from `tensor.shape`. Buffer size and grid are unchanged, so
the math is untouched.
**Guard.** Lead every kernel parity sweep with a non-tile-aligned size.

### A model's own internal crop makes your benchmark size a fiction
**Mechanism.** A pocket-crop or template-crop stage pins the trunk at N <= 256 after the first
recycle regardless of input length, so most trunk passes run at the crop size whatever you fed in.
Screens then get written at the crop size and the headline quotes the easiest rung.
**Check.** Log the actual N entering the trunk per recycle, not the input length.
**Fix.** Screen at the easiest rung if you like (a FAIL there kills the lever everywhere), but a
PASS at the easiest rung is not a result. Quote the fleet comparison size first.
**Guard.** Per-model note of "benchmark size vs internal crop size" in the perf doc.

### A shape-dependent cache that only invalidates in one direction
**Mechanism.** Invalidation fires when a later prediction's padded N shrinks toward a threshold but
not when a subsequent one grows past it again. Every input above the threshold dies on the *second*
prediction in the process with a broadcast `TT_FATAL`.
**Check.** Drive one long-lived object across the threshold twice, down-then-up and up-then-down.
Single-shot fixtures each run once and structurally cannot see this.
**Fix.** Sweep every submodule's masks at the top of `forward` rather than tracking which tensors
need invalidating.
**Guard.** A same-process multi-size sequence in the determinism test, not a battery of single-shot
fixtures at different sizes.

---

## 4. "It is nondeterministic"

### The same code, same input, four different outputs
**Mechanism.** A faulty card. In one case the fault was matmul-only and location-keyed, not
data-keyed: concat and layer_norm were bit-stable at every size and precision while a 512->256
matmul differed on 15/15 repeats at fp32 and 2/31 at bf16, with the same victim row clusters
reproduced by a synthetic-weight probe.
**Check.** Run the *unchanged* code N>=3 times to measure the noise floor before attributing any
diff to a change under test. Then run the same probe on a second card.
**Fix.** Move bit-exact gating off the suspect card entirely. A clean run proves nothing about the
next one, and there is no size threshold to hide behind.
**Guard.** Never let dispatch place a hash-equality check on an unvetted card. A per-boot smoke
fold per card, at a large size, costs two minutes and catches this before a campaign burns on it.

### Same seed, two different structures
**Mechanism.** `mp.get_context("spawn")` workers do not inherit the parent's RNG state, so a
controller's `torch.manual_seed` never reaches them. It hides on legs with wide self-floors and
surfaces on a leg with a tight one, where it looks exactly like a real host-vs-reference
implementation difference.
**Check.** Two device runs with the same `--seed` disagreeing by more than the reference's own
floor proves an unseeded RNG. No need to diff schedule code.
**Fix.** Seed `random`, `numpy`, `torch` once before the first stochastic forward *in the spawned
worker*, and confirm the seed is actually threaded into the worker config dict, not merely read
inside it.
**Guard.** After the fix, watch the *device self-consistency* number tighten toward the reference
floor. Device-vs-reference on a wide floor is too noisy to show the fix landed either way.

### Fold 1 is right, fold 2 with a different target is wrong
**Mechanism.** A captured trace cached on `(batch, N_padded)` only. Per-step inputs get re-staged
each step; the per-fold conditioning is baked into device buffer addresses at first capture and
goes stale for a second, different target that happens to pad to the same shape. Nothing on the
per-job path resets the static cache between targets.
**Check.** Fold target A, then target B, then target A again in one process and compare A's two
outputs.
**Fix.** Key the trace on the conditioning identity, or reset the static cache per job.
**Guard.** A two-target same-process test in the trace leg of the gate.

### A partial trace inside a sampler loop drifts across steps
**Mechanism.** Trace capture assumes the captured region's buffers are stable across replays. An
eager op running between replays that touches the same intermediate pool breaks that invariant
silently. Signature: step-0 output correlates ~1.0 with the untraced reference, step 3 is ~-0.04.
The same violation can instead present as a device *hang*, when the interleaved eager op triggers
an allocator wait rather than aliasing a buffer.
**Check.** Score PCC per step, not step 0.
**Fix.** Trace the full step as one unit, staging host work in via `copy_host_to_device_tensor`.
This is viable exactly when the loop has no host-only numerical op (an exact SVD-based rigid
alignment is the usual blocker).
**Guard.** A two-gate rule: isolated PCC, then trajectory PCC. The first alone never catches it.

### Allocation order changes the answer
**Mechanism.** Changing *when* a tensor's row chunks are allocated (loop vs chunk op, DRAM vs L1),
with identical values and identical boundaries and every chunk `torch.equal` in isolation, still
moved a fold's pLDDT and output hash. Allocation sequence is a hidden input to reduction order on
this hardware. Relatedly, `ttnn.slice` is *always* a copy, never a view.
**Check.** Verify bit-exactness at the whole-fold level, not per intermediate.
**Fix.** None generic. Treat "bit-exact by construction" as a claim requiring a fold-level check
before any multi-day fusion commit.
**Guard.** Fold-level `torch.equal` in the acceptance criteria of every kernel-rewrite branch.

### A lazily built cache kills concurrent workers
**Mechanism.** A chemical-component cache built lazily on first use inside whichever worker asks
first. N concurrent workers against an uncached target all race to build and write the same path
and die at ~70s with rc=1. Looks like an MSA mismatch or a timeout.
**Check.** Watch the cache build progress live while the failures happen.
**Fix.** Pre-build once in a single process, archive it, and ship it via an env var.
**Guard.** A cold-cache concurrency test, or a build lock.

### A determinism check on a freshly restarted service pool always passes
**Mechanism.** After a restart every worker's RNG starts at draw 0, so identical inputs agree to
six digits whether or not the seeding bug is present. The divergence only appears once workers have
processed enough traffic to desynchronise.
**Check.** Run one throwaway round, then measure on the warmed pool. Pre-fix, the same protocol
split 10 identical inputs 2-vs-8; post-fix all agreed.
**Fix.** Seed per request, not per worker.
**Guard.** Any post-restart determinism check must state that the pool was warmed.

---

## 5. "It hangs or crashes"

### The whole host looks idle and nothing can open a device
**Mechanism.** Device bring-up serialised behind one global file lock held across the whole open
call, with no timeout by design (the assumption is that every caller is a pool supervisor with its
own watchdog). One process stalling mid-bring-up queues every other device job on the box
regardless of which card it leases. Presentation: `tt-smi` exit 0, cards at idle power, load near
zero, which reads exactly like a hardware fault.
**Check.** `ls -l /proc/<pid>/fd | grep tenstorrent` across candidate pids, and look for who is
blocked acquiring the lock versus who holds all N device fds.
**Fix.** Kill *only your own* stalled opener by explicit pid. That releases the lock without
touching anyone else's card fds.
**Guard.** Never run an unpinned bare `pytest` on a host doing real measurement; it enumerates and
opens every card. A `cpu`-only dispatch does not prevent this: it governs leases, not what the
process touches.

### An import opens hardware
**Mechanism.** A nominally CPU-only task that imports the model package with no `TT_VISIBLE_DEVICES`
pin probes every enumerated device, including cards another process is actively holding. The
kernel-driver signature is a power-state error in the last log lines before the box stops
responding.
**Check.** `cat /proc/<pid>/environ | tr '\0' '\n' | grep TT_VISIBLE` on any running pytest. Check
the previous boot's kernel log after an unexplained host hang.
**Fix.** Pin `TT_VISIBLE_DEVICES` before the import, always, including in scripts you believe are
host-only.
**Guard.** Scope test invocations to specific non-hardware paths rather than running the suite.

### `ARC core failed to start` at `ttnn.open_device`
**Mechanism.** Not necessarily a bad board. Many back-to-back isolated open/close cycles
(calibration sweeps, per-op micro-benchmarks) plus one `kill -9` that skips the device-close
cleanup path will wedge a healthy card.
**Check.** Reset the chip, then verify with a *fast* canary script. A slow script cannot
distinguish "recovered" from "wedged again" in reasonable time.
**Fix.** `tt-smi -r <chip>`.
**Guard.** Prefer SIGTERM over SIGKILL for a process stuck host-side, but use `timeout -s INT`
(SIGINT) for any bounded run of a process that might hang *inside* a device call: SIGTERM on a
device-hung process has itself wedged cards.

### OOM naming an allocation that is not really the problem
**Mechanism.** The error names the allocation that failed to place, which is often the last straw
of several co-live tensors of the same shape. One case held two fp32 `[S, n_heads, S, S]` score
tensors live across three ops: 16 GiB each at 1024 tokens.
**Check.** Count how many tensors of the failing shape are simultaneously live at the failure
point, before reaching for a chunking lever.
**Fix.** Delete a redundant live copy (fold the scale into the bias-add's activation, reduce in
place). That halved the peak and made the *unblocked* path fit again at every size below the wall.
**Guard.** A live-tensor-count assertion in the op, or a comment stating the peak.

### A hang inside a narrow token window, clean on either side
**Mechanism.** A compute-grid-shape-selected defect. One reproducible case hung for token counts in
a 7-wide window, cleanly folded at the sizes immediately above and below, and disappeared entirely
when the main compute grid was clamped from 13x10 to 11x10. It reproduced only in the full fold's
state, never in a standalone replay of the same op chain.
**Check.** Confirm the window with a synthetic single-chain input at the exact token count, then
A/B the grid clamp on one freshly reset card in one session.
**Fix.** Env-gated grid clamp, verified against a control that hangs and survives a long timeout.
**Guard.** Sweep the crash boundary on every distinct grid shape you ship to, not just the one the
lever was discovered on.

### `TT_FATAL work_split.cpp: remaining == 0`
**Mechanism.** `ttnn.split_work_to_cores(all_cores, units)` throws under an exact rule: `units >
cores` and `units % cores` is a non-zero multiple of the grid height. No tensors are involved; it
is the work-split arithmetic itself. Only hand-written `ttnn.generic_op` kernels that call the
utility directly inherit these holes, and they are grid-height dependent.
**Check.** Evaluate the rule for your grid across the unit counts you will see. On one 13x10 grid,
358 of the first 4000 unit counts fail.
**Fix.** Try the full grid first, and on a throw search rectangular sub-grids, using the rule only
to order candidates and the utility itself as the sole authority on whether a candidate splits.
Memoise per `(device, grid, units)`; a throwing call costs ~357us versus ~0.6us for a cached hit.
**Guard.** Fall through to the stock op when no sub-grid works, never a crash.

### A residue-count ceiling that moves between runs
**Mechanism.** An L1 static circular-buffer ceiling whose clashing address depends on the serving
process's own allocator history. A long-lived worker serving many folds hits it at a different size
than a fresh process. One measured ladder folded at 512 and 544, threw at 576, folded at 608, and
threw at 640 one day and folded the next.
**Check.** Walk a ladder and re-verify near a boundary. One passing fold at a size never establishes
that size as a ceiling.
**Fix.** Publish the largest size *below* the first observed failure, never an interpolated point
past it. The real fix is a caught-throw fallback at the crash site.
**Guard.** A cap-setting rule that forbids interpolation across a crash.

### Host OOM presenting as a mystery hang
**Mechanism.** Device lease checks answer a device question, not a host-memory question. Loading a
checkpoint an order of magnitude larger than its neighbours, alongside other live jobs on the same
host, can take the whole box down while every device lease looks clean. From the outside it is
indistinguishable from a hardware hang.
**Check.** Before counting a hang as unexplained, check what was mid-load at the timestamp.
**Fix.** Check free host RAM against checkpoint size and enumerate co-resident jobs before loading.
**Guard.** Do not let a mundane OOM inflate the count of "recurring unexplained hangs"; that count
drives hardware decisions.

---

## 6. "It is slow"

### 48% of a step is `ttnn.from_torch`
**Mechanism.** Uploading with `layout=TILE_LAYOUT` tilizes on the host, single-threaded. It does
not appear as device compute in a naive wall-clock breakdown, so it masquerades as an unavoidable
per-op cost.
**Check.** Profile the upload path specifically. In one case 288 of 596 ms/step at 3359 atoms.
**Fix.** Upload row-major pre-cast to the target dtype and `ttnn.to_layout` on device (2.8-8.5x per
tensor, legal inside a trace-capture region). For persistent trace buffers, pre-cast dtype in torch
before tilizing (2.5x). Both bit-exact.
**Guard.** Before accepting any "hardware ceiling" verdict, profile upload and tilize separately
from kernel time.

### A tall/narrow matmul at 27% of the compute roof and 33% of the bandwidth roof
**Mechanism.** The op computes, then writes its result to DRAM in series rather than overlapped.
`time = compute + write` predicted the measured wall to 0.1%; `max(compute, write)` was wrong by
45%. For a matmul with small K and large M the output write can be three quarters of DRAM traffic.
The general trigger is `k_tiles < num_cores`, which collapses `k_tiles_per_core` to 1 and holds for
essentially every dense projection in a pair trunk at typical channel widths.
**Check.** Ablate the DRAM destination and measure the delta. That proves the mechanism without a
device profiler.
**Fix.** Keep the result in L1, under two hard guards: `in0_block_w` must come out as the *whole*
of K (a narrower K block is a different accumulation order and is not bit-exact), and both operands
plus the result must fit L1 with room for the block's other allocations. Measured 2.17x on the op,
1.079x on the block, bit-exact.
**Guard.** An L1-residency win is only real if every consumer of the resident tensor also stays
on-chip. If any consumer round-trips through DRAM anyway, the residency is pure overhead: one such
lever was 2.66x in isolation and 1.45x *slower* end to end.

### 60% of a trimul op is one `ttnn.permute`
**Mechanism.** A channel-moving permute `(0,3,1,2)` takes the row-major untilize / blocked-transpose
/ retilize path at ~55 GB/s, versus ~368 GB/s for a last-two-dims transpose and ~383 GB/s for a
clone. It is implementation-bound, not bandwidth-bound.
**Check.** Bandwidth-probe each permutation on the same byte count. A 7x gap against a clone on the
same tensor is the tell.
**Fix.** Keep the tensor channel-major through the chunk loop so only the cheap transpose is needed,
or fold the transpose into the matmul operand read. Failing that, block the permute over a spatial
dim so it runs L1 to L1.
**Guard.** Do not try to speed a permute at the op level; all decompositions measured slower.

### Sparse indirection is the wall, not the kernel around it
**Mechanism.** `ttnn.scatter` and `ttnn.gather` are limited by per-element traversal rate, not
bandwidth. Measured on one Blackhole card at 45.1M elements: `add` 69.5 G elem/s, `scatter` 9.7,
`scatter_add` 4.7, `gather` 1.2. A bfloat8 dense (half the bytes) costs within 0.1% of bf16, which
proves it is element-rate. No knob moves it.
**Check.** Price the indirection before designing a port around it.
**Fix.** Call the op less often. One decoder called scatter twice with bit-identical inputs across
recycles and cached instead: free, bit-exact, ~7% of the step.
**Guard.** Any new indirection-heavy op (volumetric gather/scatter) must price this floor before a
fused kernel is proposed as the answer.

### `to_layout` / `untilize` collapses to a single core
**Mechanism.** For shapes whose trigger is a joint function of both row and column tile counts,
`ttnn.untilize` silently falls back to a single-core kernel, and the boundary differs per chip.
`use_multicore=True` does not override it and there is no signal.
**Check.** Force `use_multicore=False` on a known-fast shape of the same size. If that reproduces
the pathology exactly (one measured case: 36070 us at 10.1 GB/s versus 999 us at 364 GB/s), you
have confirmed it.
**Fix.** Row-block the op for the affected shape island only. A blind global swap costs 3x elsewhere.
**Guard.** A shape-conditional gate, and a note of the bad island's boundary per chip.

### Deleting an op does not move the wall
**Mechanism.** The stage is host-dispatch-bound, so the deleted op's latency was already overlapped
with device compute and never on the critical path. A per-call cost times a call census is not a
fold-level gain.
**Check.** Measure host issue time versus device time per block directly. And check the denominator
of any inherited "N% dispatch-bound" claim against the actual loop structure (recycles x samples x
batch, not just block count). One inherited headline was a 10x units slip that reversed the verdict.
**Fix.** If dispatch-bound, reach for trace capture or fusion. If device-bound, do not.
**Guard.** Re-derive any inherited bound/unbound classification with a fresh profile before it
justifies a multi-day route.

### An "X-bound" label that expired
**Mechanism.** Every "DRAM-bound" or "compute-bound" classification is a claim about a measured
state, not a property of the code. The moment a landed lever changes the traffic/compute balance on
that chain, the label is wrong, and a subtract-the-bytes screen built on the old label never even
asks the compute-side question. One pass missed a 1.3 s/fold bit-exact win this way.
**Check.** After landing any lever on a chain, re-split that chain's op costs against *both*
measured roofs.
**Fix.** Rank the next lever against the new split.
**Guard.** Timestamp roofline classifications in the perf ledger.

### A fusion that should be free is not
**Mechanism.** A subtractive prize screen ("the DRAM traffic this op costs is the prize for
fusing it away") is correct only when the *absorbing* kernel is DRAM-bound and has spare compute to
hide the absorbed math behind. When the absorber is already at 88% of its DRAM roof, the absorbed
op's arithmetic runs in addition, not underneath.
**Check.** Price one added SFPU pass with a toggle already in the kernel and no prototype. Measured:
one sigmoid pass 0.663 ms/call, one round pass 0.198 ms/call.
**Fix.** Two-step screen: roofline colour of the absorber, then toggle-priced added compute.
**Guard.** Record the absorber's roofline percentage next to every fusion GO/NO-GO.

### A shape-specialised JIT that never reaches steady state
**Mechanism.** A design loop that draws a new sequence length per trajectory recompiles every
trajectory, and that compile cost does not amortise. One measured workload spent 55.4% of total
wall clock on non-amortising compile, and quoting a stage's "% of wall" with compile left in
overstated a port prize by 3.4x.
**Check.** Ask whether the input shape is stable across the loop before pricing anything.
**Fix.** Bucket the varying axis, or measure and exclude compile explicitly.
**Guard.** A negative "unattributed time" residual is the signature of nested double-counting (stage
B invoked from inside stage A), not a good sign. Build the accounting from interval containment,
never from a sum of named durations.

### Isolated op timings that do not add up to the chain
**Mechanism.** Timing an op alone with a sync on both sides charges a full device drain to that op.
One set of isolated per-op numbers summed to 17.93 ms against a measured chain cost of 14.657 ms; a
single leg read 0.0433 ms/chunk isolated and 0.0222 ms/chunk when issued 52-at-a-time the way the
fold actually issues it.
**Check.** Re-time candidates back-to-back without per-op syncs.
**Fix.** Treat isolated-timing-derived gates as provisional. A gate near its threshold flips.
**Guard.** The bias runs *both* directions: isolation inflates simple op costs via sync tax, and
*underprices* residency levers by hiding the DRAM-queue relief they give every concurrent op. One
residency lever screened at -3.87 ms/step and delivered -6.974 ms/step. Re-measure in the real fold
either way, and record which direction it moved.

### A first run that is 10x the steady-state number
**Mechanism.** Kernel compilation is cached per machine and environment, not shipped with the repo.
A fresh card or fresh venv measures compile plus run. One first-ever run measured ~101 s against a
steady state of ~8.7 s.
**Check.** Run twice, discard the first.
**Fix.** Seed baselines from steady state only.
**Guard.** A bad seed silently poisons every future comparison on that host, and makes real
regressions look like improvements.

---

## 7. "The measurement does not reproduce"

### A GPU-vs-Tenstorrent ratio that is an artifact of batch
**Mechanism.** One side's cell measured as throughput (N designs from one forward, divided by N)
and the other as latency (one at a time). One published 9.52x factored exactly into 2.119x of
batch amortisation on the reference side times 4.492x of real matched-batch gap. The amortisation
is not a collectible win: it is the reference filling idle silicon.
**Check.** Confirm both cells were measured at the same batch size and config, not each side's
shipped default.
**Fix.** Pin one enforced protocol per row of any comparison table.
**Guard.** A FLOOR verdict inherits its target's denominator. Re-derive the target from the
published protocol before accepting any "unreachable" conclusion, and before shelving levers as
"too small": lever worth is absolute seconds, only the deficit moves.

### A lever credited with three months of other people's work
**Mechanism.** Scoring `published_cell - new_measurement`. The published cell records the number at
the time it was written, and unrelated levers land afterwards without anyone republishing it. One
lever measured 1.318 s against a stale cell and 0.526 s against a same-run control.
**Check.** Run the A/B against current HEAD without the flag, on the same card, tree, process and
tick.
**Fix.** Treat the published cell as the number to *update*, never the number to diff against.
**Guard.** If several cells on the page have drifted, refresh the page before scoring anything.

### A regression alarm that is the baseline's fault
**Mechanism.** Legs that take one cold timed call with no warm-median loop carry 20-30% single-shot
noise, so a 15% threshold sits inside the noise band and alarms forever. And host contention is
*one-sided*: it can only slow a rep, never speed one up, so a median of three still reports a
contended value when two of three reps are contended.
**Check.** Run an endpoint A/B first (seed commit versus HEAD, interleaved reps, same host). If the
endpoints match, the seed was the outlier. Do not bisect commits first.
**Fix.** Median of three at minimum; for one-sided contention noise the robust statistic is the
*minimum* rep, which is the closest the run got to an uncontended measurement.
**Guard.** A tree that produces one rep faster than baseline did not regress.

### A perf A/B where the arms differ by card warmth
**Mechanism.** Running all of one arm then all of the other. Two runs of identical code on the same
fixture differed ~10%, the same order as the effect under test. Sequential measurement overstated a
win at +13.3%/+15.7%; interleaved over three paired rounds the honest number was +5.2%/+15.4%.
**Check.** Confirm base and fix were interleaved per fixture and repeated.
**Fix.** Interleave.
**Guard.** Also confirm no co-tenant holds the host. Under contention, two arms running *identical
code* moved a block wall by 1.3% at one size and 10.4% at another. Do not relaunch expecting a
cleaner number; the floor is a property of host state, not sample size.

### An isolated probe number that does not survive the fold
**Mechanism.** Isolated probe and in-fold attribution for the identical site measured 13x apart,
and the gap was not contention: the probe stayed flat to within 5% under 72 MB of deliberate L1
ballast. Separately, a byte model that prices an L1 destination write as *free* manufactures fake
mysteries; measured, L1 is only ~1.9-2.1x faster than DRAM for bulk out-projection moves, nowhere
near the ~20x a "L1 is basically free" assumption implies.
**Check.** Re-derive the byte model with the op's own measured L1 write rate before treating a
measured-vs-model gap as a real inefficiency.
**Fix.** Run a fold-level census before any ranking decision.
**Guard.** Never quote a micro-probe number as the in-fold cost of the same site.

### The gate is measuring a runtime nobody ships
**Mechanism.** A gate script launched with the system `python3` picks up whatever ttnn is on that
interpreter's path, not the version pinned in `pyproject.toml`. Under the wrong wheel two accuracy
legs read high enough to fail a release on a difference that was never code.
**Check.** `python -c "import ttnn; print(ttnn.__version__)"` under the exact interpreter the gate
runs, and compare against the pin.
**Fix.** Launch via the release venv's `sys.executable` with `PYTHONPATH` pointed at the tag tree,
so `tt_bio` stays the code under test while `ttnn` comes from the pinned venv.
**Guard.** A bare `python3` in a release runbook is a red flag.

### Every fold model fails the perf gate by 20-35% on one machine
**Mechanism.** A baseline seeded on a different card generation with no card-type detection in the
gate. The gap tracks silicon, not code, and is confirmed by re-running *unchanged* code.
**Check.** Re-run the unchanged tree on the same day and compare against the same-day warm number.
**Fix.** Detect card type at runtime and compare only against the matching baseline. Fail loudly
with NO BASELINE on an unseen card type.
**Guard.** Never "fix" a cross-card false-fail by loosening the threshold; that masks the next real
regression.

---

## 8. "The gate passed but the thing is broken"

Three worst offenders; the full catalogue is in `12-testing-and-gates.md`.

### The gate scored the installed package, not your checkout
**Mechanism.** A gate script that does `import tt_bio` in-process gets `sys.path[0] = scripts/`, so
the import falls through to whatever is `pip install -e`'d. If that tree is *older* than yours, the
gate passes while measuring the wrong code, with no signal at all.
**Check.** Print `tt_bio.__file__` from inside the gate leg.
**Fix.** Run legs as subprocesses with `PYTHONPATH=<repo root>` set explicitly.
**Guard.** Any new leg that imports the model package in-process is suspect by construction, because
every worktree differs from the main checkout.

### A PCC gate that passes without the op it is named after
**Mechanism.** On a residual stack the skip connection carries the correlation regardless. One
measured LayerNorm gate scored 0.99899 for the correct implementation, 0.99999 for applying the norm
*twice*, and 0.99503 for scoring the pre-norm activation with the norm skipped entirely. A `>0.98`
bar passes all three.
**Check.** Score the deliberately broken controls (op skipped, op applied twice, wrong tensor) and
confirm the gate distinguishes them.
**Fix.** Move the tap, tighten the bar, or score a residual-free quantity.
**Guard.** A gate that cannot fail on a control that obviously lacks the op is not testing its name.

### A gate that recomputes everything from one captured artifact
**Mechanism.** Every arm re-derives features from the same capture, so a defect baked into the
capture reproduces identically in all arms and the gate agrees with itself forever. The sibling
shape is a fixture check that guards on the file's *existence* while depending on its *contents*.
**Check.** Grep what the test actually reads out of the fixture and assert on those keys or a
content hash.
**Fix.** Add one arm that scores the capture itself against an independent source (a second
capture, an upstream run, hand-inspected ground truth).
**Guard.** Otherwise port and upstream can go wrong together and every arm still passes.

---

## 9. "It worked yesterday"

### A dependency bump that silently drags a second one
**Mechanism.** A framework's own dependency spec ties its major version to a hub/client library
major floor, so bumping the one package a security alert names forces a coupled bump the alert
never mentions.
**Check.** Run a real resolver check in an isolated venv before assuming single-package scope.
**Fix.** Apply both, then verify model output bit-exactness across the bump with `torch.equal`.
**Guard.** Keep the blast radius written down: which modules actually import the bumped package.

### A runtime upgrade that is faster on device and fatal in the sampler
**Mechanism.** A newer ttnn was 4-5% faster on the device side once traced, 3.65x *slower* on one
eager encoder, and broke diffusion numerics outright (multi-Angstrom moves against a committed
fixture) while encoder PCC went *up*. The breakage was encoder-invisible and sampler-fatal.
**Check.** Gate any runtime bump on a diffusion-sampler fixture, not an encoder PCC.
**Fix.** Stay pinned; reach for tracing on the current version instead.
**Guard.** Re-scout a *later* release rather than assuming the verdict carries forward.

### A revert on main that never reaches the branch
**Mechanism.** `git revert` adds an inverse commit on one ref. It does not rewrite history and does
not propagate to refs that already contain the original. A branch that merged main as its assembly
base *before* the revert still contains the rejected commit as an ancestor, and merging that branch
re-lands it with every check green.
**Check.** `git merge-base --is-ancestor <rejected-sha> <branch>` for every live branch.
**Fix.** Carry the revert through to each hit.
**Guard.** For a vendored downstream copy, the acceptance test is byte identity of the vendored
files against the source repo's main (`diff -r`, `sha256sum`), not a green import or a passing suite.

### A merge that duplicates a fix instead of conflicting on it
**Mechanism.** Two branches fixing the same bug independently, in structurally similar ways at
different textual positions in one file. Git merges both additions cleanly. Result: a duplicated
tuple entry (harmless) sitting next to a fully redundant global whose companion mechanism is the one
actually read post-merge (dead scaffolding).
**Check.** Whenever a conflict resolution reveals "the other side already fixed this", grep the
whole diff for that mechanism's other touch points.
**Fix.** Delete the now-dead half.
**Guard.** A conflict on one line is often the visible tip of a duplicated fix whose other parts
merged silently.

### A clean auto-merge that breaks imports
**Mechanism.** One branch edits a flat module while mainline moved that code into a package. Git's
rename detection applies the edits to the new path with no conflict markers, and `from . import x`
relative imports keep their old meaning at the new nesting depth.
**Check.** After any merge where `git diff --stat` shows a file-into-package rename on either side,
grep the merged result for `from \.` / `from \.\.` on the renamed paths.
**Fix.** Re-point the imports.
**Guard.** Do this even when git reports zero conflicts. Also `git show --stat` any commit whose
subject describes a small change before trusting it: one 3-line-sounding subject was a 90-file,
6117-deletion whole-tree rollback that a later merge reported as clean because the revert is in
history.

### A deploy script that replaced itself mid-deploy
**Mechanism.** The script was executed *from the checkout it was about to replace*, so a deploy that
changed the deploy script ran the old one, skipping the pin checkout, the stale-tree removal and
both installs, then failed on a console script that did not exist yet.
**Check.** Rehearse the change end to end on a staging pair of checkouts, not by diffing code.
**Fix.** Make the deploy script re-exec the *target* tree's copy of itself.
**Guard.** Never edit a running shell script in place. Bash re-reads from a saved byte offset after
each command, so growing the file mid-run makes a live interpreter resume inside code it already
executed. Write a temp file and `mv` it over: the rename gives new content a new inode and running
interpreters finish on consistent text.

### A stale shadow of the file you edited is what actually runs
**Mechanism.** Four independent variants of one shape. (1) A leftover package directory with no
`__init__.py` still shadows a properly installed package on `sys.path`, and stale `.pyc` files
survive the source's deletion. (2) A venv's console-script shim next to the interpreter is preferred
over `python -m`, so subprocesses run the shared checkout even when `PYTHONPATH` was set for the
parent. (3) A service unit's `VIRTUAL_ENV` path can name a stale clone while the venv installs the
package editable against a *different* directory. (4) An undeclared production dependency can hide
behind a package someone installed by hand on that box years ago.
**Check.** Resolve `tt_bio.__file__` from inside the running process. That is the only authority.
**Fix.** Invoke with an interpreter that has no sibling console script, with an explicit
`PYTHONPATH` ahead of site-packages.
**Guard.** Verify a genuinely clean install from the manifest before trusting the manifest.

### A kernel merge that passes every test and crashes on a clean install
**Mechanism.** Custom kernel sources are loaded at runtime *by file path*, not imported as Python,
so a new `tt_bio/kernels/<name>/` directory that packaging does not name silently drops out of the
wheel with zero signal until the first eligible on-device call. Hit three times before the glob was
made recursive.
**Check.** A packaging smoke leg that installs the *built wheel* and calls every kernel path. A
pytest run against the source tree never catches this, since the files are on disk in dev mode.
**Fix.** `recursive-include tt_bio/kernels *.cpp *.hpp` plus `kernels/**/*.cpp` package-data.
**Guard.** The inverse failure exists too: an asset path derived by walking up from `ttnn.__file__`
is correct for the pip-wheel layout and one path segment wrong for a source build, which fails every
`generic_op` JIT build across every model with no defect in the kernels. Test both layouts.

### A shared default flipped for one model regressed another
**Mechanism.** A global env-var or module-level default read inside a shared class's `__init__`.
A model that composes that class verbatim inherits the flip with zero code touching it. One
fp32-diffusion default flipped for an accuracy fix made a sibling model's atom-level fold 60x
slower. A related shape: a unification pass adds a *second consumer* of an existing flag, and a
model that opted in for one site silently gets it at the new sites too, across a 48-layer trunk.
**Check.** `git grep` every caller before flipping any shared default, and count consumers of any
flag a unification pass touches.
**Fix.** An explicit constructor kwarg defaulting to the env var, so existing callers are preserved
and every other caller can pin its own value. Per-site opt-in for flags with multiple consumers.
**Guard.** Score a shared-default change per model before flipping it, and note that a size-tuning
regression can be a *victim* of a flag leak rather than an independent cause: in one case the two
suspected size-retune commits were innocent and all three sizes improved together with the one fix.
