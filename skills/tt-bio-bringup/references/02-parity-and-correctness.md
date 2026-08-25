# Parity and correctness: proving the port matches the reference

This document decides your correctness protocol. The unmodified PyTorch reference, run on CPU in fp32, is the
only golden. You port one submodule at a time and gate each against its own captured golden before moving on.
Your accuracy threshold is not a number you pick, it is a number you measure: the device may differ from the
fp32 reference by no more than a bf16 recomputation of that reference differs from itself. Every escaped bug
becomes a permanent test arm.

Read this when you are about to port a module, when a module diverges, when you are choosing a PCC threshold,
or before writing the words "parity verified". Precision levers are `03-precision-and-numerics.md`, tile and
bucketing mechanics `04-shapes-tiles-and-bucketing.md`, gate wiring `12-testing-and-gates.md`. Performance work
is `05-perf-method-and-roofline.md`: do not start it until §10 passes.

## 1. The golden-reference discipline

### 1.1 What is golden

The **unmodified reference implementation, on CPU, in torch, fp32**, with your code out of the path. Not a
device run. Not a "reference" produced by a script that imports your ttnn module. Not a vendor-supplied output
file. Not a cloud GPU run you cannot reproduce.

No GPU is needed. A CPU reference fold of a 100-300 residue target is minutes to a few hours, paid once per
fixture. For a **deterministic** reference, CPU fixtures represent what a GPU user sees: same code, same
weights, same inputs, and the CPU-vs-GPU difference is float-association noise. For a **stochastic**
one they do not, and the reason is in §8: `torch.randn(device='cuda')` draws from Philox and CPU from
MT19937, so the two are independent noise realizations, not a small numerical difference. Seed and
compare on one device, or share the draws explicitly.

### 1.2 Capture protocol

Complete, because a half-written capture is the single most expensive mistake available in Phase 1:
it saves, it loads, it has every key you expect, and it is missing an input.

```python
import dataclasses, random
import numpy as np, torch

def _to_cpu(x):
    """Recurse into every container the reference might return. Leaves stay leaves."""
    if torch.is_tensor(x):
        return x.detach().to(torch.float32).cpu()
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*(_to_cpu(v) for v in x))   # a namedtuple takes positional args, not an iterable
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(v) for v in x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _to_cpu(getattr(x, f.name)) for f in dataclasses.fields(x)}
    return x                                     # int, float, str, None, a config object

def capture(ref, seed, **real_input):
    """`ref` is the unmodified reference model, already holding its checkpoint.
    `real_input` is one real target's features, the kwargs you would pass to `ref(...)`."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.use_deterministic_algorithms(True)
    ref.eval()
    cap = {}

    def hook(name, mod):
        fwd = mod.forward
        def h(*a, **k):
            out = fwd(*a, **k)
            if name + "/out" not in cap:                  # FIRST call only
                cap[name + "/args"] = _to_cpu(a)
                cap[name + "/kwargs"] = _to_cpu(k)        # keyword inputs too, see below
                cap[name + "/out"] = _to_cpu(out)
            return out
        mod.forward = h

    for name, mod in ref.named_modules():
        hook(name or "<root>", mod)
    with torch.no_grad():
        ref(**real_input)
    return cap
```

Save it under `scripts/<model>_port/parity_artifacts/`, gitignored, fetched from a release asset by a
script (§1.4). Multi-MB captures do not go in `tests/fixtures/`; that directory is for the kilobyte
inputs a card-free unit test needs.

**Capture `**kwargs`, not just `*args`.** A submodule called as `blk(s, z, mask=m)` is the dominant
convention in bio models, and a capture that walks only the positional tuple loses the mask. The
fixture still saves, still loads, and still has a key for every module, so nothing complains: you
find out weeks later when the module matches its golden and the model does not match the reference.

**Check the contents, not the keys.** After capturing, assert that the tensors you expect are tensors
and the shapes are what the module tree says. `_to_cpu` above returns non-tensors untouched on
purpose, so a `None` where a mask should be survives into the fixture and reads exactly like a
correctly captured optional argument.

- **First call, not last.** Later recycles and diffusion steps carry compounded state and cannot isolate a module.
- **Inputs AND outputs.** An output-only golden cannot be replayed into your module.
- **Save fp32 on CPU**, whatever the reference computed in. Casting at capture time destroys your ability to
  separate your error from the reference's own bf16 error.
- **fp64 is for op-level screens only.** To rank two kernels, score both against `torch.float64` on identical
  bf16 operands. It is not an end-to-end golden, and an isolated fp64 screen is blind to whether an op's error
  is coherent in the chain (§6).
- **Diffusion capture needs `num_timesteps >= 2`.** The common `zip(schedule, schedule[1:])` sampler idiom yields
  an *empty* loop at 1, so every per-step module silently never runs and the golden holds only the
  step-invariant parts.

### 1.2b What the capture writes, and how a test names it

`capture()` above returns a dict and stops there. Settle the rest once, because the plan's Golden
column has to name something a test can load, and three different conventions in one port is three
different bugs.

**One file per input length**, not one per module. The capture walks every module in a single
forward, so splitting the result into fifty files buys nothing and loses the guarantee that they all
came from the same pass.

```
scripts/<model>_port/parity_artifacts/<model>_<len>.pt        # the golden
scripts/<model>_port/parity_artifacts/<model>_<len>.meta.json # provenance, see §1.4
```

**Keys are `<module path>/args`, `<module path>/kwargs`, `<module path>/out`**, with the module path
exactly as `named_modules()` spells it and `<root>` for the top-level model. The example's capture at
`--len 117` writes 41 modules as 123 entries under those keys.

**So a plan's Golden cell names the file and the key**, for example
`minifold_117.pt : blocks.0.attn/out`. That is loadable:

```python
golden = torch.load("scripts/minifold_port/parity_artifacts/minifold_117.pt", weights_only=False)
expected = golden["blocks.0.attn/out"]
args, kwargs = golden["blocks.0.attn/args"], golden["blocks.0.attn/kwargs"]
```

The `.meta.json` sits beside the fixture and is deliberately **not** part of any byte-identity gate:
it records a runtime, which is a measurement, so hashing it fails the gate for a reason that is not
a defect.

### 1.3 Fixture size

| Fixture | Size | Purpose |
|---|---|---|
| Unit / per-module | 1 block, seq 32-128, real captured tensors | fast red-green loop |
| Ragged control | a length **not** a multiple of 32 (e.g. 117) | tile-tail bugs (§4) |
| End-to-end | one real 100-300 residue target, MSA on and off | integration envelope |
| Task metric | 1-3 targets with experimental structures | lDDT / TM / RMSD |

**A tiny fixture cannot gate a size-dependent kernel.** A fused attention kernel measured 12.87x *worse* than the
materialized path at sequence length 32 and ~1.5x *better* from 64 tokens up; a port whose parity fixtures were
8-53 tokens read the correct lever as a 3.4-5.3x regression. Check where a kernel's accuracy floor sits relative
to fixture size before trusting a verdict.

**A ladder built by tiling or truncating one sequence is one input, not a sweep.** It holds MSA depth, chain
count and composition constant across all rungs, so a defect conditioned on one of those is reported as
conditioned on size. One non-tiled input at a "failing" size falsifies it.

### 1.4 Provenance: a fixture without it is worthless

```json
{"model": "<name>", "target": "<id>", "settings_tag": "msa-on-default", "reference_impl": "<upstream package>",
 "reference_version": "2.0.0", "reference_commit": "<upstream sha>", "runtime_s": 1376.4, "seeds": [0,1,2],
 "command": "python -m <ref> predict ... --seed 0", "settings": {"recycles": 3, "samples": 1, "dtype": "fp32"},
 "msa": "<a3m sha256>", "provenance": "CPU torch reference, cold run", "date": "2026-01-01",
 "invalidation_rule": "regenerate if reference_commit or settings change"}
```

**Bug signature:** a *bare stub* meta.json holding only `reference_impl`/`reference_version`/`reference_commit`/
`seeds`, where `reference_impl` names **your** package and `reference_commit` holds **your** repo SHA. That means
a regen script ran a **device fold** and saved its output as the reference, turning every later parity check into
a device-vs-device tautology that passes cleanly and means nothing. Second tell: `runtime_s`. A CPU reference
fold is hundreds to thousands of seconds; a "reference regen" that finished in 17 seconds ran the device path.

Storage: small JSON scores and provenance in git; multi-MB binaries (CIFs, MSAs, `.pt` captures) as tagged
release assets fetched by a script, with the binary directory gitignored. Fixture sets are append-only, so git
LFS quotas bite and release assets do not.

**The metadata is not part of the byte-identity check.** `runtime_s` is a wall-clock measurement and
`date` is a date, so two correct captures produce two different meta files. Hash the golden; check the
metadata for required keys and for the tautology tells above. A determinism gate that hashes the meta
file alarms on a loaded host and teaches the reader to ignore it.

## 2. Component-by-component porting

Never port the whole model then debug end to end. Order: **leaf ops** the shared library lacks (RoPE, a fused
norm, a gather pattern) → **blocks** (attention, triangle multiplication, triangle attention, transition, DiT
block) → **trunk/Pairformer stack including the recycle loop** → **heads** (distogram, confidence, affinity) →
**sampling/diffusion loop last**, because it is a feedback loop that amplifies everything upstream.

**Pass A, random weights, structural.** One identical random state dict into both the torch reference and the
ttnn module (`load_state_dict(..., strict=False)`). Gate at PCC > 0.99. No checkpoint needed, so correctness work
starts before weights exist. Catches transposes, wrong axis, wrong mask orientation, wrong head split.

**Pass B, real weights on real captured input.** Catches magnitude-regime bugs random weights cannot express:
random inputs give attention logits of order 1, real weights reach absmax 111 where bf16 softmax flips the
argmax; real adaLN scales reach absmax ~7 against ~0.2 from a random init. **Pass A green with Pass B red is a
precision-regime bug, not a structural one.**

### 2.1 In-situ injection

Isolated parity is necessary, not sufficient. Run the unmodified reference forward with one submodule swapped for
your ttnn wrapper, and score at the **model** output:

```python
ref.trunk.blocks[7] = TTAdapter(tt_block)      # host tensors in, host tensors out
d(ref(**real_input), out_pure_reference)       # gate at the model output, not the block output
```

- **Orchestration bugs.** Every component passing isolated while the pipeline diverges means loop order,
  additive-vs-overwrite accumulation, or mask threading. Injection localizes it to the block whose insertion
  first moves the model output.
- **Amplification.** A downstream module can be a non-monotone amplifier on one input, its error tracking which
  arm produced input A and ignoring input B entirely. When a consumer's error does not move with a clean
  per-input metric, stop tuning producers and score the consumer in isolation at a fixed captured input.

### 2.2 Capture-and-bisect

Run the device module on the captured reference input; report **PCC and the norm ratio** `|mine|/|ref|` together.
Binary-search downward (module → blocks → ops) replaying the same tensors.

| Isolated | Assembled | Diagnosis |
|---|---|---|
| pass | fail | orchestration: loop order, accumulation, mask threading |
| fail with real weights only | fail | precision regime: peaked logits, large adaLN scale, padding |
| fail with random weights | fail | structural: transpose, axis, head split |
| pass | pass, e2e still off | error accumulation (§6) or the sampler/RNG (§8) |

## 3. Metrics: which one, where

| Metric | Use for | Blind to |
|---|---|---|
| PCC | per-tensor module parity, embeddings | scale, localized error, whether the op ran |
| norm ratio | always print beside PCC | shape/structure errors |
| max-abs error + index | localized catastrophe, tile-tail bugs | overall drift |
| median relative error | heavy-tailed tensors where PCC saturates | sparse large errors |
| `maxdiff == 0` | bit-exactness claims only (§4) | nothing, where it applies |
| Kabsch RMSD | structural end-to-end parity | needs §5 discipline |
| lDDT / TM-score | task accuracy vs experimental truth | port-vs-reference identity |
| ranking correlation | confidence head / selector fidelity | absolute coordinate error |
| pLDDT, any confidence scalar | **nothing in a parity decision** | almost everything |

### 3.1 PCC can pass while the op under test never runs

On a residual or near-identity stack the surrounding structure carries the correlation. Measured on a hoisted
LayerNorm across 24 DiT blocks: correct implementation **0.99899**; norm applied **twice** (wrong) **0.99999**;
norm **skipped entirely**, scoring the pre-norm activation, **0.99503**. A `>0.98` gate passes all three.

**Detection:** score the deliberately broken controls (op removed, op doubled, wrong tensor scored) before
trusting any gate named after an op. If the gate does not separate them, it is not testing what its name says.
Keep the control arms in the test file as permanent `xfail`s.

### 3.2 A single scalar hides a localized catastrophe

PCC over 10^7 elements cannot see 18 wrong columns. A tile-padding read poisoning 18 of 2702 columns in a softmax
reduction moved ligand coordinates 0.335 Å with every aggregate metric clean. Always report alongside the scalar:
`max_abs_err` **and its unravelled index**; the metric recomputed on the **valid/unmasked slice only**; and for
padded models `ref_mask.sum()` vs the physical extent. Padding cuts both ways: blow-up in an invalid region drags
all-atom PCC from 1.0 to 0.5 while every valid atom is perfect.

### 3.3 PCC has an SNR ceiling

`PCC_max ≈ sigma_R / sqrt(sigma_R**2 + RMSE**2)`. For a target whose true signal is small, a bf16 port cannot
clear a flat bar shared with higher-signal siblings. One cell measured 0.8926 against a flat 0.9 bar with a
predicted ceiling of 0.908: 1.6% from its own floor, unreachable by any port. Before calling a PCC failure a
regression, check the reference is deterministic across reruns and compute the ceiling. If measured sits at it,
re-baseline that cell with the derivation inline and keep an MAE bar as the real guard, since MAE is not
SNR-distorted.

### 3.4 Choosing the threshold

**There is no universal PCC minimum.** Accepted values by class, across model families and not only
biomolecular ones, so read the bio-specific plausibility bands in `03-precision-and-numerics.md` §7
alongside these: per-op ~0.99 here against ~0.999 there, and the tighter one is the right expectation
for a single bio op. Full dense decoder stacks
0.94-0.999, degrading with depth (32-layer models accepted at 0.60); MoE 0.86-0.96; encoder-decoder 0.90-0.97;
vision components 0.75-0.79. For a folding model, per-module 0.98-0.99 on real captured inputs is the bar.

For end-to-end structure output, do not pick a number, **measure the envelope**. A diffusion model is a
deterministic function of its input noise, so run three closed-loop folds on byte-identical noise:

```
device_bf16     the port on Tenstorrent
reference_fp32  the same code path on CPU torch, fp32           <- ground truth
reference_bf16  the same code path on CPU torch, bf16 autocast

numerator = d(device_bf16, reference_fp32)
envelope  = d(reference_bf16, reference_fp32)
PASS iff  numerator <= envelope * (1 + margin) + abs_floor      for EVERY metric
```

The device may drift from fp32 by no more than a bf16 recompute of the reference drifts from itself, plus a
residual for TT-bf16 vs torch-bf16 accumulation absorbed by `margin`. In tt-bio, `margin = 0.50` with per-metric
`abs_floor` (kabsch_rmsd 0.05 Å, coord_pcc 0.001, log-scale scalar 0.01) in `scripts/integration_envelope.py`.
`abs_floor` exists only so a degenerate zero envelope (fp32 and bf16 agreeing bit-for-bit on a scalar) does not
fail every nonzero residual by construction. Justify the margin from the measured device-vs-envelope ratio across
your clean legs *before* committing the value, and record that derivation.

**This is a design constraint on the port.** The envelope exists only if your model class stays a real
`nn.Module` runnable on CPU torch with a backend toggle. Replace it with a device-only class whose
`load_from_checkpoint` opens a device unconditionally and you have destroyed your own reference, permanently.

**Fake-envelope signature:** numerator and envelope both round to ~0 with a clean PASS (`num=5.49e-15
env=5.49e-15 ratio=0.00`). That is not excellent parity, it means both sides came from the same code and nothing
external was compared. Treat it as a bug report on the gate.

## 4. Bit-exactness

**Legitimately exact, claimable by construction:**

- `bf16 -> fp32 -> bf16` round-trips. bf16→fp32 is zero-extension; fp32→bf16 of an already-bf16 value is the
  identity. Removing a host sync between two bf16-native tensors gives maxdiff 0, PCC 1.0. Void if either side is
  a different dtype, or if the "sync" also does real work (a cast, a reduction).
- **Bucketing on a masked axis**, as all three of pad + mask + slice-back: pad the ids, add a `-1e9` additive
  mask, **and zero the padded keys and values** so exactness does not rest on bf16 `exp(-inf)`. Verify PCC == 1.0
  and maxdiff == 0 against the unpadded run.
- Host-side code you did not touch.

**Never exact:** any change to a matmul's **contraction size** (padding a reduction dim in bf16 is not exact
across sizes and moves structure output); any change to a **shape**, including batch or sample-chunk width,
because ttnn picks block sizes, core grids and memory configs from the shapes it is handed; anything across ttnn
versions, card revisions or grids.

Two chunk widths agreeing bit-for-bit is a lucky pair, not a law: a third width falsified exactly that claim on a
shipped port. **A shape-varying scheme claimed inert needs at least three values, and output artifacts that can
actually see per-sample differences** (a CIF rounded to 3 dp and a results.json sorted by rank cannot).

**The claim itself:** `assert torch.equal(a, b)` or `assert (a-b).abs().max().item() == 0.0`, never a tolerance,
and demonstrate the test **failing** on the pre-fix commit or a deliberately perturbed arm. A test never observed
red is not evidence.

**The ragged-tile trap** is why bucketing is a correctness rule, not a perf preference (mechanics in
`04-shapes-tiles-and-bucketing.md`). `ttnn.TILE_LAYOUT` pads physically to 32 on both tile axes while the logical
shape keeps the true length, so any op reducing over the physical extent reads that tail. An unmasked tail in a
fused attention put attention mass on padding at **72x the fp64-reference error at every ragged length and ~1.4x
at every aligned one**; uninitialized tile padding left by a scatter fed `-inf`/3e38 patterns from an earlier
fold into a softmax, moving coordinates 0.335 Å.

Consequences for parity: bucket every token axis to a multiple of 32, reusing the shared constants in
`tt_bio/token_axis.py`, and grep that module when auditing whether a port buckets. **Bucketing hides a
ragged-input kernel bug, it does not fix it**: fix the kernel too and keep a ragged fixture so it stays visible.
A per-op screen run only at 128 and 512 tokens is correct and blind simultaneously.

## 5. Structural and geometry metric traps

**Kabsch is not transitive.** Any ensemble-agreement score defined by superposing N samples onto one shared frame
is reference-*dependent*: rigid fits do not compose across a non-rigid molecule. Re-deriving with a different
arbitrary frame moved a pairwise agreement score by up to 0.23 Å and dropped per-cell rank agreement to 0.0036,
two "reference-free" definitions of one quantity that barely correlate. Use a closed-form pairwise definition,
validated against explicit per-pair Kabsch (agreement to ~1e-13 is achievable).

**An inverted rotation convention produces plausible wrong numbers.** A helper deriving `R` that maps `P` onto `Q`
and then applying `R.T` to `Q` returns **15.18 Å for an exact rigid copy of a structure** while scoring
nearly-aligned pairs about right, which is why spot checks miss it. It manufactured a phantom model defect that
reached a production investigation. Guard every bespoke superposition helper:

```python
def test_kabsch_identity():
    P = torch.randn(64, 3, dtype=torch.float64)
    R, t = random_rotation(), torch.randn(3, dtype=torch.float64)
    Q = P @ R.T + t
    assert kabsch_rmsd(P, Q) < 1e-9 and kabsch_rmsd(Q, P) < 1e-9
```

**Grep the scorer, not the label.** A metric labelled "Cα-RMSD" whose `load_atoms` has no CA filter is an all-atom
RMSD, and it stayed mislabelled across a model family's whole published history with every numeric gate green,
because no gate validates a *label* against a *computation*. Worse for TM-score, whose `d0` scales with the number
of items it is given: a CA-intended normalization applied to an all-atom set changes the number's meaning.

**A chimeric or multi-domain fixture saturates.** A fixture fusing a domain to a truncated copy of itself has an
unconstrained hinge as its softest degree of freedom. Four independent unrelated non-bit-exact changes all landed
in the same 7.7-9.0 Å RMSD band; per-domain superposition showed each domain held at 0.73-1.45 Å with only a
45-72 degree hinge rotation. **A number invariant to which change produced it is not measuring the change.** Pair
such a fixture with a monomeric control of similar size: there the same changes moved 0.28-0.33 Å, 20x smaller.
Bit-exact arms are immune, which is how this survives whole campaigns unseen.

**Multi-sample models: compare identity, not rank.** If output files are named by confidence rank and the top-two
confidences tie to 3e-4 while the arithmetic under test is orders of magnitude larger, rank 0 is a coin flip and
the gate measures the distance between two correctly-reproduced samples. **Tell:** the reported error equals the
run's own sample-to-sample spread. **Detection:** build the NxN Kabsch cross-matrix between device and reference
samples; a permutation (device rank 1 matches reference rank 0 at 0.139 Å) means ranking artifact. Fix by
anchoring on the reference's top structure and finding its counterpart across all device samples, requiring the
match to be 3x closer than the next-best candidate, else falling back to strict rank-0.

**Design models break folding invariants**, so a validator written with folding habits rejects good output:
sibling designs legitimately differ in atom count, a binder appended to the target's chain is one chain not two,
and mmCIF column order is header-defined, so a fixed-index slice can read a residue-name field as a coordinate.
Resolve CIF columns **by header tag**, never by index.

## 6. Error accumulation: a per-op error bar is not a model error bar

Measured on a 48-block trunk: per-block error 1.0-1.3x the bf16 envelope, inside tolerance by that instrument.
Chained over one real pass the same blocks read **32x** on one track and **82x** on the other, missing the
task-level bar by 45x.

**Mechanism.** Random per-op error compounds identically in the device arm and a torch-bf16 reference arm, so the
ratio stays near 1.2x end to end. It did not, so the device path carries a **coherent** (systematically signed)
term the reference does not. Coherent error grows as a product of per-block factors; random error grows like a
square root and largely cancels.

**A concrete injector.** `ttnn.add` on two bf16 operands breaks rounding ties **away from zero** where torch
breaks ties **to even**: 11.16% of elements disagree by 1 ulp at equal operand magnitudes. Invisible per op. Over
a 48-block, 4-recycle residual trunk (432 adds) measured per-block growth was 1.074 device vs 1.036
clean-bf16-torch, and `(1.0740/1.0359)**47 = 5.5` correctly predicted the observed 5.2x miss at block 47.

**Instrument 1, per-block drift trace.** Run reference and device in lockstep on the same input:

```python
trace.append(((dev - ref).norm() / ref.norm()).item())        # after every block
growth = (trace[-1] / trace[0]) ** (1 / (len(trace) - 1))     # per-block factor
```

Fit the growth factor and extrapolate to full depth **before** spending a week on per-op precision configs. At or
below the reference arm's own factor means the residual is random and bounded. Above it means a coherent term,
and the extrapolation predicts the end-to-end miss.

**Instrument 2, one-sidedness of the residual.** For any op you consider widening to fp32:

```python
d = (tt_out - torch_out).flatten(); nz = d[d != 0]
one_sided = max((nz > 0).float().mean(), (nz < 0).float().mean()).item()
```

Near 1.0 is a compounding bias worth fixing. Near 0.5 is symmetric rounding noise, and "fixing" it redraws a
random walk: a bf16 `sigmoid` at 46.2% one-sidedness was made bit-identical to torch per op and was a measured
**end-to-end regression**, reproduced twice. Mismatch *rate* does not predict payoff; one-sidedness does.

**Instrument 3, in-chain substitution, and its limit.** Swapping one op class for its host-torch twin in the live
chain **bounds** attribution, it does not **decompose** it: substitution also removes whatever error cancellation
that class contributed, so results are non-monotone (two substituted classes each made the chained residual
*worse*). Rank candidates with it; never build a per-class budget that should sum.

**Corollary:** a per-op win that does not move the chained metric is information, not failure. Halving the two
worst per-op error terms in one port cut isolated per-block error 2.1x and moved the chained gate score by
exactly 0.0%, proving the fixed term was not the coherent one.

The same argument runs along the diffusion axis. 200 sampler steps is a feedback loop, and the Tensix matmul
engine multiplies in bf16 even for fp32 tensors (fp32 buys fp32 *accumulation*, `fp32_dest_acc_en`), so per-step
drift invisible at step 1 becomes trajectory divergence by step 200.

## 7. Fixture and gate blind spots

Each of these produced a green gate over a broken or unmeasured thing.

| Blind spot | Detection recipe |
|---|---|
| **Chimeric/hinge fixture** cannot express the failure mode; every change lands in one RMSD band | Superpose each domain separately. Small per-domain RMSD with only the inter-domain angle moving means the fixture is the problem. Add a monomeric control. |
| **Existence checked, contents depended on**. a partial capture reads as a code regression | Grep what the test actually reads from the fixture and assert on **those keys** or a content hash. Better: wrap the golden in a dict whose `__missing__` calls `pytest.skip("golden has no <key>, regenerate with scripts/<x>_golden.py")`, wrapping nested dicts too, since captures are partial at every depth. |
| **Gate scores the installed package, not the checkout**. in-process `import <pkg>` inside `scripts/foo.py` resolves via `sys.path[0]=scripts/` to the editable install; invisible when install == checkout | Run gate legs as **subprocesses with `PYTHONPATH=<repo root>`**. To detect, run `python -c "import <pkg>; print(<pkg>.__file__)"` from the gate's own working directory and diff against the tree you meant to score. |
| **Gate reproduces a committed verdict, not the value**. a "reproduces committed" tag confirms the PASS/FAIL bucket, so a number drifts under a held verdict and still reads clean | Re-run the number and diff it against the figure quoted in the doc, not against the verdict category. |
| **Reference regenerated from device output and relabelled** (§1.4) | Check the meta.json shape (stub vs rich) and `runtime_s` order of magnitude. |
| **Gate passes with zero legs scored**. an all-blocked run still prints GATE PASS because `all_pass` starts `True` and is ANDed only for legs that reach scoring | Read the tally line, not the banner: it must show scored verdicts. Fix the gate (`all_pass = False if scored == 0`). Same class: a `--seeds` arg taking a comma list silently parsing a bare `5` as `[5]`, matching no fixture and blocking every leg. |
| **Stale exemption**. a model still in a `*_EXEMPT`/`*_SKIP` list with a reason that stopped being true | Before merging a model past its "not covered yet" stage, grep its slug in every exempt list and rewrite or delete the entry in the same merge. |
| **Untracked per-host golden**. the same filename holds different keys on different machines, and copying one over another silently re-baselines the tests that pass | Never `dict[key]` an untracked pickle. Skip with a reason naming the regeneration script. Which capture is canonical is the capture-script owner's call, not a decision a test may make. |
| **Fixture and doc drift apart after a regen** | Re-derive from current main before any regen, and check whether branch and mainline both touched the fixture since the fork point. |

## 8. Making a stochastic model comparable

A diffusion model is a deterministic function of its input noise. Everything here follows.

**Generate the draws once, feed both sides.** Produce the whole random sequence up front in the exact order the
sampler consumes it (initial coordinates, per-step epsilon, augmentation rotations and translations, corrector
noise), save it, and inject it into both the device sampler and the CPU reference sampler. On one port,
device-vs-supplied-golden read 7.93 Å and looked like total failure; device-vs-CPU-reference on identical shared
draws read **1.23 Å RMSD, PCC 0.99997**. Same code, same weights, different noise.

**Why a supplied golden is usually unusable.** `torch.randn(device='cuda')` uses Philox, CPU uses MT19937, ttnn
has no `randn` at all. Two independent noise realizations of one stochastic model diverge by the full run-to-run
spread, multiple Å for a protein, swamping any real port error. Only accept a supplied golden whose exact RNG
stream is reproducible in your environment.

**Verify the sharing, do not assume it.** (1) **Bit-reproducibility:** run the device path twice at the same seed
and `torch.equal` the dumped noise arrays. (2) **Same backend:** confirm both arms draw from the same generator by
dumping and comparing, not by reading code. Two traps that break it:

- **The global-RNG confound.** The reference often draws some state (initial pair state, template sampling, MSA
  subsampling) from the *global* torch RNG while its `seed` argument seeds only the sampler, so sequential runs
  drift even at `seed=0`. Call `torch.manual_seed(s); np.random.seed(s); random.seed(s)` before each prediction,
  matching upstream's once-per-run `seed_everything` flow. Do not re-seed before a later stage in the same
  process: that diverges from the reference's single continuous stream.
- **Spawned workers do not inherit RNG state.** `mp.get_context("spawn")` starts with a fresh global RNG, so a
  controller's `manual_seed` never reaches it, and a `seed` not threaded into the worker's config dict is dead
  code reading `None`. **Diagnostic signature:** two device runs at the same `--seed` disagree by more than the
  reference's own floor. That delta alone proves an unseeded RNG, with no need to diff schedule code. It hides
  behind wide structure floors and surfaces first on a tight scalar metric.

**MSA subsampling** is the same problem: pin the MSA by sha256 in the fixture meta, record its seed and depth cap.

**When the envelope legitimately fails on a chaotic target.** If a leg fails and standard precision levers do not
move it, re-run the identical triple at one or two more seeds and look at the **reference's own** bf16-vs-fp32
envelope. On one no-MSA target it swung 3.45 / 2.16 / 0.81 Å across seeds 0/1/2 with zero device involvement, and
the leg passed cleanly at seeds 1 and 2. That is a chaotic reverse-diffusion trajectory landing on a bad draw,
not a port defect. Record it as evidenced-and-explained; do **not** loosen the margin and do **not** cherry-pick
the seed for one leg.

**Never adjudicate with a confidence scalar.** A precision bug moved a structure 3.849 Å against a 0.000 Å
self-RMSD floor while pLDDT moved -0.0259. pLDDT is the model's self-assessment of local confidence, not a
distance from a reference; the two are loosely coupled and point in opposite directions in exactly the
silent-precision-loss cases that matter. The deciding number is always structural or bit-exact.

## 9. The suite the port ships with

Under `tests/`, running from the checkout, not an installed wheel.

1. **Unit parity per component**, both passes, parametrized over at least one ragged length. Skip cleanly at
   module level when the reference package or checkpoint is absent so collection stays green on a bare machine.
2. **Deliberately-broken control arms** for every gate named after a specific op (§3.1), kept as permanent `xfail`s.
3. **In-situ injection test**: reference graph, one device module swapped, scored at the model output.
4. **End-to-end integration envelope** on a real target, MSA on and off, via the three-run triple.
5. **Task-metric test** against an experimental structure, with the scorer's atom filter asserted in the test,
   not merely documented.
6. **Bit-exactness tests** for every property claimed exact (§4), asserting `maxdiff == 0`.
7. **Geometry helper identity tests** (§5).
8. **A regression arm for every bug you fix.** The same change that fixes an escaped bug adds a permanent arm to
   the **default** arm set, and **that arm must be demonstrated failing on the pre-fix commit** or it is theatre.
9. **A knob with no test at its non-default value is broken at its non-default value.** Every env flag, `--fast`
   path, chunk width and bucket size gets a test exercising it end to end. A host-stub test proves control flow,
   not the device fix: one shipped commit passed green on a host stub while never running its device leg.

## 10. Checklist before claiming "parity verified"

- [ ] Golden is an **unmodified reference, CPU, torch, fp32**; meta.json records `reference_impl`/
      `reference_version`/`reference_commit`/`command`/`settings`/`seeds`/`runtime_s`, none naming this repo, and
      `runtime_s` is in the reference's range, not device-fold speed.
- [ ] Every ported component has a unit parity test passing with **real captured input**, not only random weights.
- [ ] Every absolute-threshold gate named after an op has been shown to **fail** on op-removed and op-doubled controls.
- [ ] PCC is reported with the **norm ratio**, the **max-abs error and its index**, and the **valid-mask-only** variant.
- [ ] The threshold is derived, not chosen: an envelope ratio with a recorded margin, a documented SNR ceiling,
      or an explicitly cited peer-class band.
- [ ] The end-to-end score is a **shared-draws** comparison, with the noise arrays dumped and `torch.equal`-checked.
- [ ] Two same-seed device runs were compared to each other; any disagreement above the reference floor was
      root-caused before anything else was believed.
- [ ] At least one fixture is **ragged** (non-multiple-of-32), and every token axis buckets to a multiple of 32
      through pad + mask + slice.
- [ ] A **per-block drift trace** exists at full depth and its growth factor does not exceed the reference arm's.
- [ ] Structural metrics come from a helper with a passing rigid-copy identity assertion, and the scorer's actual
      atom filter matches the metric's label.
- [ ] For multi-sample output, comparison is by **matched sample identity** verified with an NxN cross-matrix.
- [ ] Every bit-exact claim has an `== 0` test that has been observed **red** on a perturbed arm.
- [ ] The gate ran the **working checkout** (subprocess, `PYTHONPATH` set) and its tally line shows a nonzero
      number of **scored** legs.
- [ ] No decision in the chain rests on pLDDT or another confidence scalar.
