# Performance method: census, roofline, predict-then-build

This document decides the *order* of a performance campaign for you: measure the whole model, build
an op census, measure the two roofs on the actual card, place each hot op on the roofline, compute
the Amdahl-bounded ceiling of the lever you are considering, write that prediction down, and only
then build. It also lists the screening traps that make a prediction wrong, and the reporting rules
that make a win believable. Skipping straight to "tune this op" is the failure mode this document
exists to prevent: a knob-sweeping campaign across ~100 agent passes on this codebase returned about
2 % end to end; the method below returned 1.228x byte-exact on the same model in a day.

**Read this when** the port is numerically correct and you are about to make it fast, or when you are
reviewing a performance claim someone else made.

## 1. The method, in order

**The effort bar, first, because steps 6 and 9 need it.** It is a number you set before the campaign
and write in `notes/PORT_STATE.md`, not a feeling you have per lever. A workable default: build a
lever if its Amdahl ceiling is at least 5% of end-to-end time, or at least 2% and under a day of
work; kill anything below that in writing, with the number. Set your own two figures and keep them
fixed for the campaign, because a bar that moves per lever is not a bar, it is a preference. Raise
the percentage if your card time is scarce, lower it if the model is already close to its floor and
the remaining wins are all small.

Do not reorder the steps. The order is what makes the predictions land.

1. **Measure the real wall.** Retire every number in the lineage that was never a measurement.
2. **Op census.** Device time per op type, call count, share of total, at the shape you care about.
3. **Measure the roofs** on the actual card with a microbenchmark (§3). Never quote a datasheet.
4. **Place each hot op** on the roofline by arithmetic intensity (§4). Name the binding resource.
5. **Derive a floor**: minimum bytes that must cross DRAM, divided by the measured DRAM roof. Now
   "optimal" has a reference and the residual is accountable.
6. **Bound the lever by Amdahl** (§5). Write the ceiling as a number.
7. **Design a screen that predicts THE ACTUAL CHANGE**, not a proxy (§6). State up front what result
   makes you stop.
8. **Write the predicted landing before building.** Non-negotiable. A prediction can be wrong
   informatively; a post-hoc explanation cannot.
9. **GO or NO-GO explicitly. An evidenced NO-GO is a complete, successful pass.**
10. **Build.**
11. **One whole-model A/B**, matched protocol, interleaved arms, with an A/A control (§7).
12. **A named mechanism for every millisecond** between the measured result and the floor.

**Never combine two levers arithmetically.** Two levers measured from different baselines cannot be
multiplied or added. Measure one integrated arm. In one campaign two levers screened at 5.07x and
4.77x against near-identical baselines; the integrated arm was 4.37x.

Done properly the predictions are tight: a fused triangle-attention kernel was predicted at
21.2 ms/block before a line of kernel code existed and measured 21.148, 0.3 % off, per-kernel
decomposition included. The same analysis phase killed a planned megakernel on its predicted landing
and, in killing it, produced a cheap kernel worth 84 % of the megakernel's gain in one day.

## 2. The op census

The census is the ranked list you re-derive after every landed lever. Build it with the device
profiler, not host wall-clock, and read `DEVICE KERNEL DURATION [ns]` per op:

```sh
python -m tracy -r -o OUT --op-support-count 20000 -- /abs/path/to/your_script.py
```

Per op type, record: **total device time, call count, mean per call, share of the whole model.**

Rules that make a census trustworthy:

- **Profile the production call site, not a generic stand-in.** One census under-reported a
  contraction 2.28x by timing `ttnn.matmul` with no program config while production ran with one
  (17.83 vs 39.5-40.1 TFLOP/s). Whatever `program_config`, `memory_config` or fidelity the model
  passes, the census passes too.
- **Segment one-time from per-step cost before quoting any percentage.** Run at `n1` and `n2` steps:
  `per_step = (t2 - t1) / (n2 - n1)`, `one_time = t1 - n1 * per_step`. Getting this wrong was ~2x off
  and inverted an op ranking.
- **The profiler perturbs what it measures** (+20.7 % at 100 ops, +36.2 % at 5000). Attribute with
  the profiler, time A/B legs bare, keep the two in separate tables.
- **An arithmetic floor is not a wall.** `FLOPs / a measured rate` is a lower bound on time. One
  lineage carried "this module costs 11.36 s" for days; it was a floor and the real wall was 30.9 s.
  Dividing real bytes by the fake time implied 668 GB/s on a ~400 GB/s card, which was the tell.
- **Census by effect, not by config.** A merged, default-ON lever delivers nothing if its guard
  declines every real call. Give each lever a `[served, declined]` counter next to its guard plus a
  decline reason, then count what fired (`scripts/lever_census.py`). Run it as a subprocess: folds
  happen in spawned workers, so counters read in the launcher are always zero and every lever reads
  as dark.

## 3. Measure the roofs. Do not assert them.

A roofline needs two numbers: peak FLOP/s and peak byte/s. **Both must be measured on the card in
front of you, through the same software stack the model uses.** A datasheet number misleads you about
how much headroom is left. A "% of peak" against an unmeasured roof produced three wrong conclusions
in one thread: (1) *wrong roof entirely*, an op at "27 % of compute peak" had intensity 190 FLOP/byte
against a machine balance of 231, so it can never reach compute peak, and the tell was already there
(lowering fidelity bought 1.04x where math-bound would buy ~4x); (2) *achieved mistaken for
achievable*, the 112-141 GB/s those ops reached was read as the roof when the real roof was ~3x
higher, hiding a 3x gap neither roof explains; (3) *never measured at all*, a number appearing once
in planning text gets quoted as physics for months.

### The microbenchmark

Three shapes, one script, run on the target card:

| Roof | Shape | FLOPs / bytes | Sanity predicate |
|---|---|---|---|
| **DRAM bandwidth** | large streaming eltwise on DRAM-interleaved bf16 tensors: `ttnn.add` at 4096², 8192x4096, 8192² | `3 * N * 2` bytes for add, `2 * N * 2` for `ttnn.clone` | GB/s must be **flat across sizes**. A bandwidth number still climbing with size is measuring fixed overhead. |
| **Compute** | large square matmul, N = 4096 / 8192 / 12288, bf16 | `2 * N³` FLOPs | Must saturate by N≈4096 and reproduce across runs to <1 %. |
| **L1** | the same eltwise with `memory_config` in L1, sized to fit | same | Should be several times the DRAM figure; if it is not, the tensor is not actually resident. |

The timing harness is the whole trick:

```python
def timed(fn, device, iters, warmup):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)          # drain the warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)          # drain the work you are timing
    return (time.perf_counter() - t0) / iters
```

A ttnn op call is an asynchronous enqueue. Without the trailing `synchronize_device` the host clock
stops while the device is still working. Measured on a 2048³ matmul: 0.019 ms/iter unsynced vs
0.145 ms/iter honest, a **7.63x under-report**. The factor is not a constant, it is however much work
fits in the command queue, so it grows with the loop you time. tt-bio ships this as
`scripts/profiling/roofline_bh.py`; run it once per card model and commit the numbers.

### Measured roofs, Blackhole p150a, bf16, through ttnn

| Quantity | Value | Note |
|---|---|---|
| Peak DRAM bandwidth | **435.2 GB/s** | `ttnn.add`, 8192² bf16, 402.7 MB/call. Flat: 428.4 / 434.9 / 435.2 across the three shapes. Datasheet says 512 GB/s, so ~85 %. `ttnn.clone` lands at 414-415, a 2-stream copy does not beat a 3-stream add. |
| Peak bf16 matmul | **161.1 / 140.4 / 100.6 TFLOP/s** at LoFi / HiFi2 / HiFi4 | N=8192, 1099.5 GFLOP in 6.823 ms at LoFi. Most bio-model ops run HiFi4, so 100.6 is the honest ceiling for them; 161.1 is the hardware ceiling. |
| Machine balance, LoFi | **370 FLOP/byte** | 161.1e12 ÷ 435.2e9 |
| Machine balance, HiFi4 | **231 FLOP/byte** | 100.6e12 ÷ 435.2e9. Use this unless you know the op runs LoFi. |

Below N≈1024 you are not measuring compute at all: at N=512 the fidelity ordering *inverts*
(HiFi4 14.52 > LoFi 11.11 TFLOP/s) because 18-24 µs of dispatch dominates the math. Do not derive a
compute roof from small shapes.

## 4. Placing an op: arithmetic intensity

```
AI = FLOPs_of_the_op / bytes_moved_by_the_op        # bytes = (operands + output) * dtype_size
attainable = min(compute_roof, AI * bandwidth_roof)
```

`AI < machine_balance` means bandwidth-bound and it can never reach compute peak, whatever the FPU
counter says. `AI > machine_balance` means compute-bound.

Reference intensities: with the FLOP and byte counts above (`2N³` FLOPs against `6N²` bytes), a square matmul is
`N/3` FLOP/byte, so it only crosses 231 above **N≈693**. Below that a matmul cannot be compute-bound however you
block it, which is most projections in a bio trunk.
An eltwise add is fixed at **0.17 FLOP/byte**, three orders of magnitude short. Almost nothing in a
biomolecular model clears 231, which is why these workloads are bandwidth- and latency-bound in
practice, and why "add more FLOPs cheaply" (fusion, precision) usually beats "do fewer FLOPs".

Two free cross-checks, worth running before you trust a placement. **A fidelity A/B is a bound
check**: if LoFi vs HiFi4 moves the wall by nothing, the op is not math-bound whatever percentage of
compute peak you computed. **Per-RISC durations**: in the same ops report compare
`DEVICE TRISC1 KERNEL DURATION` (math) against `BRISC`/`NCRISC` (dataflow). Caveat, TRISC1 being
*long* proves nothing: for `ttnn.add` TRISC1 runs 226 of the kernel's 227 µs, resident the whole time
and idle inside it. Two independent signals agreeing is what makes the call trustworthy.

**Never quote a bare "% of peak"** without saying which roof: compute, bandwidth, or
`min(compute, AI * bandwidth)`. And check the denominator of every derived rate. One "the write runs
at 102 % of the write roof" had a correctly measured roof and a bad denominator: bytes written
divided by the *gap between two arm timings* (93.7 µs) when the writer kernel was active 153.5 µs.
Corrected, 59 % of roof.

## 5. Amdahl discipline: bound the lever before you build it

The ceiling of any lever is the share of time it touches. Write it down as a number, in the units of
the whole-model metric, before starting:

```
best_case_speedup = 1 / (1 - share_touched)          # if the lever makes its region free
best_case_saving  = share_touched * total_time * fraction_of_region_removed
```

Worked: a triangle-multiplication module's phase breakdown at N=128 was LN 0.08, projection+gate
0.74, contraction 0.85, output 0.22 ms, so 1.89 ms in total. A perfect contraction kernel leaves
1.04 ms and therefore caps the *module* at **1.89 / 1.04 = 1.82x**, and after it lands the projection
at 0.74 of 1.04 ms is the new bottleneck. Do the division; "it is the biggest phase so the win is
big" is not a bound. That bound was worth more than the
kernel.

Rules:

- **Kill a lever whose best case is below the effort bar.** One fused-kernel proposal died on
  "predicted landing 12.92 ms/call against an existing kernel's 13.33" before any code was written.
- **Scale per-call rates by the model's own repeat count before comparing to headroom.** A trunk
  runs its blocks once per recycling cycle, so a per-recycle cost has to be multiplied by the trip
  count before it meets a fold-level margin. A 0.54 %-per-recycle cost against a 0.33 % fold margin
  was called "room"; over 10 recycles it is 5.4 % against 0.33 %, a 16x miss.
- **A ratio is only meaningful if numerator and denominator are the same unit and protocol.** One
  headline ("91.5 % host-dispatch-bound, 16x host/device gap") divided fold time by 48 blocks instead
  of 48 blocks x 10 recycles. Corrected and measured directly, the trunk was device-compute-bound and
  the natural fix for the claimed problem would have bought zero. That 10x slip nearly funded a
  multi-day trace-capture effort against a target that did not exist.

## 6. Screening traps

A screen predicts what a lever will do. Each of these makes the prediction wrong in a specific,
reproducible direction.

| Trap | Mechanism | Correction |
|---|---|---|
| **Isolated single-op timing over-reads cost** | The per-op `synchronize_device` forces a full pipeline drain that a real chain never pays between consecutive ops. The drain is charged to the op. | Re-time the candidate ops **batched, back to back, the way the model issues them**, with one sync around the group. |
| **Isolated screening under-prices contextual levers** | Syncing both sides of one op hides the DRAM-queue and L1-pressure relief that residency gives every *other* concurrent op. | Any residency/locality lever must be measured in the real chain. Expect the built number to beat its own screen. |
| **A high-precision isolated screen is blind to the sign end-to-end** | A per-call fp64 comparison scores one call's error magnitude. It says nothing about whether N sequential calls' rounding residuals reinforce or cancel. | Score accuracy end-to-end against the task's own reference, with a pre-registered floor rule and seed spread. |
| **cost-per-op x census count** | Both biases above act at once and the mix depends on shape. | Treat it as a hypothesis. Verify against a whole-model measurement before acting. |
| **Subtractive-bytes fusion screen into a compute-bound absorber** | "The DRAM traffic this op costs = the prize if I fuse it away" is only right if the absorbing kernel is DRAM-bound and has spare compute to hide the absorbed math behind. | Measure the absorber's roofline colour first. Then price the absorbed op's own arithmetic with a one-line SFPU on/off toggle already in the kernel. |

Magnitudes, so you know how far these move a number:

- **Oversync inflation, 1.2x on a chain and ~2x on a single op.** Isolated per-op timings for one
  FFN's pieces summed to 17.93 ms against a measured chain cost of 14.657 ms, 1.22x. A single `ttnn.slice` read 0.0433 ms/chunk isolated vs
  **0.0222 ms/chunk when issued 52-at-a-time** the way the model issues them, a 95 % over-read on one
  leg, enough to flip a pre-committed kill gate.
- **Residency under-pricing, 1.8x.** An L1-residency lever screened at -3.87 ms/step delivered
  -6.974 ms/step in the real fold.
- **Census prediction misses in both directions.** The same lever's isolated cost x block count
  predicted **1.5x LOW at 512 residues and 2.3x HIGH at 1024**. There is no way to know which without
  measuring.
- **Microbenchmarks overstate shipped cost generally** (measured 2.2x): the op has the whole chip and
  no queue contention. A custom matmul that looked 15x better in a microbenchmark made the fold
  *slower* end to end (105 s vs 86 s), because the benchmark compared against the framework module at
  tiny N, not against `ttnn.matmul` at production N where the library op already hits 48.9 TFLOP/s.
  **`ttnn.matmul` is often already near peak; a naive custom-matmul swap is a known trap.**

## 7. The whole-model A/B: one protocol, interleaved, with an A/A floor

Every arm of a comparison must match on: **batch size, input size, warm state, host, card, process,
tree, seed, and fidelity flags.** Unmatched batch is the single most common way a large fake speedup
or regression gets reported.

A published cross-platform ratio of 9.52x on one design model factored exactly into
`2.119 (batch amortisation on the other side) * 4.492 (the real matched-batch gap)`: one cell was 8
designs from one forward divided by 8 (throughput) while every other cell was one-at-a-time
(latency). It recurred three times in one lineage, the third time denominating a whole campaign's
"unreachable floor" verdict, which shelved four levers that together cleared the correctly-derived
target twice over. Cause: the page inherited *each side's shipped default batch* instead of pinning
one protocol per row.

Protocol:

- **Interleave arms A,B,A,B and compare medians.** All-A-then-all-B read +13.3 % where the honest
  interleaved answer was +5.2 %: the card heated up.
- **Toggle exactly one flag.** A five-flag OFF arm bounds the *sum* and can never isolate which flag
  did the work; one such run attributed +691 ms/fold to a site later proven to be exactly zero at
  that shape.
- **Run an A/A control** (same config twice) and quote its spread next to every claimed delta.
  Compute the GO/NO-GO **in code as a function of the A/A spread**, do not print it next to a verdict
  for a human to notice. A harness that logs contamination but does not act on it will manufacture
  the hoped-for result silently across every future run that reuses it.
- **Warm every distinct arm shape before timing any of them.** ttnn keys its program cache by tensor
  shape, so a harness that times N reps of shape A then N reps of shape B compiles B inside B's first
  *measured* rep. One such harness priced ~77 s of compile as ~19 s/design of "batching penalty" and
  flipped a real win into a NO-GO.
- **Know your resolution floor.** A single-shot A/B cannot resolve an effect below roughly **1.5 % of
  the wall it measures**, and on a co-tenanted host the floor is **1-10 %** even on an identical
  executed code path. It does not average out, because it scales with what else the host is doing.
  Get an exclusive card, or switch to the device profiler.

## 8. Warm vs cold: report both, never mix them

Three cache layers, each with a different lifetime:

| Layer | Where | Lifetime | Cost it removes |
|---|---|---|---|
| Disk kernel cache | `~/.cache/tt-metal-cache` | machine | the ~60-84 s JIT compile of a first fold |
| In-process program cache | `enable_program_cache()` | process | ~8-10 s per new shape |
| Device DRAM | resident tensors | process | weight upload |

- **Define the headline as warm steady-state per call**: model loaded once in-process, one warmup
  call absorbs the first-kernel compile, then N timed calls give a median. Model load and first
  compile excluded, and say so.
- **Measure the cold cost separately and report it**, because a user pays it. Cold-vs-warm spread is
  roughly **10-13x** (one first-ever run measured ~101 s against an ~8.7-8.9 s steady state).
- **Discard the first-ever run on a fresh host or fresh card.** Seeding a committed baseline from a
  cold run inflates it ~10x permanently and makes every later run on that host look like a huge
  improvement, defeating the gate.
- **Never `disable_and_clear_program_cache()` per input.** Enable once, never clear: kernels are
  then reused across shapes, and a run of `[512, 117, 384, 512]` lands the last one at the warm floor.
- Warm both sides of a cross-platform comparison with the same convention. A reference run timed at
  `n_batches=1` pays a one-time autotune/compile cost that batch 8 amortises, which systematically
  **overstates** the other platform's batching advantage.

## 9. Host cost is part of the number

A port can be device-fast and end-to-end slow. Feature construction, MSA assembly, tokenization,
ligand/CCD parsing and per-call Python overhead are not free and often are not cached. Always publish
**end-to-end wall alongside device time**, and split host vs device on *both* sides of a
cross-platform comparison.

Mechanism to watch for: **a large platform-independent additive cost pulls every ratio toward 1.** In
one measured fold, host featurisation ran uncached on every timed fold of both arms. The whole-fold
ratio came out 3.69x and the device-only ratio ~9.5x: the whole-fold number was correct for its
protocol and understated the accelerator gap by 2.5x, flattering whichever side is proportionally
slower.

Recover the shared cost rather than assuming it. With reference total `T`, whole-fold ratio `r` and
device-only ratio `d`, the additive term is

```
H = T * (1/r - 1/d) / (1 - 1/d)
```

At `T` = 21.995 s, `r` = 3.69 and `d` = 9.5 that is **4.07 s**. Compute it for your own pair before
quoting either ratio: if `H` comes out larger than the whole-fold time divided by `r`, your two
ratios are not describing the same run.

One corollary. Host prep that is **invariant across calls** is a lever, not a constant: one template tensor
re-uploaded every recycling cycle instead of once per fold cost ~3.6 s/fold, worth 1.06-1.10x for a
few lines. And when host prep swings 17 % run to run while device phases hold at 2.8 % spread, the
end-to-end median needs more reps than the device median does.

## 10. Attribution hygiene across a campaign

Every one of these is a *stale-number* defect. They recur because nothing in a repo forces a
re-measurement.

- **A perf label expires the moment a lever removes the traffic that earned it.** A chain correctly
  labelled DRAM-bound kept that label after its own predecessor levers removed the traffic. The
  shipped chain then saturated neither roof (20.8 % of compute, 0.64 ms of unavoidable traffic
  against 14.14 ms of matmul) and the next real win was a compute-side fix hiding behind a DRAM-side
  label. **Re-split the chain against both measured roofs after every landed lever.**
- **"Too small to chase" is a ratio, and the denominator moves.** One confidence-head reduction went
  from 5 % of a fold (correctly dismissed) to 15 % after an unrelated fusion shrank everything around
  it, becoming the pass's biggest win (26x on that op). **Re-run the op-share breakdown across ALL
  components after any lever that meaningfully shrinks the total, not just the component you are
  working on.** Keep a ranked backlog and re-rank it.
- **A killed lever comes back on a different metric.** A change killed on a measured 4.133 Å
  all-atom RMSD against a 0.114 Å bound was re-proposed later as a 2.5 % speed win framed purely on a
  confidence score, never citing the RMSD. **Record every kill with the deciding metric and its
  number.** When a proposal leads with a different metric than the one that decided it last time,
  grep the lineage first. The one legitimate re-open: the bound stands, the target was wrong.
- **One-size tuning is a standing defect class.** A lever screened at one sequence length is a
  calibration point applied outside its validity range. A model tuned at 512 residues scaled N^2.03
  from 256 to 512 and **N^3.62 from 512 to 768**, because three capacity gates switched off with no
  error and no log line. Measure a small, a tuned and a large rung before any default flip; a win at
  one size that regresses another is a NO-GO. Find dark gates by the **log-log exponent between
  consecutive rungs**, there is no other signal, and add an off-lattice rung because a factor-of-2
  ladder cannot localise a cliff that moves inside a doubling.
- **A published baseline cell is historical, not live.** Score a lever as `lever-on minus same-run
  main` on the same card, tree and process. One lever measured 1.318 s against a published cell that
  had already drifted 0.792 s from other landings; its honest effect was 0.526 s. The published cell
  is the number to *update*, not the number to diff against.

## 11. Noise, and when to bisect

A single-shot regression on a perf gate is usually noise. Gate legs that run one cold-inflated timed
call with no warm loop carry **±20-30 % run-to-run noise**, and a 15 % threshold sits inside that
band, so the leg alarms on its own noise forever.

- **Do not bisect first.** Run an **endpoint A/B**: baseline commit vs current HEAD, interleaved reps,
  same host, cache and driver. If the endpoints match, the baseline was the outlier: reseed honestly
  and stop. One flagged 12-17 % regression resolved to 0.265288 vs 0.265299, no guilty commit.
- **Repeat N times and gate a statistic, not a draw.** Median-of-3 fixed the general case at ~3x wall
  cost on the affected legs only.
- **For one-sided noise, the median is the wrong statistic.** Host contention can only slow a rep
  down, never speed one up, so with unlucky timing 2 of 3 reps are contended and the median reports
  the contended value. Three runs of identical code gave -29.2 %, -46.1 % and -24.5 % purely from
  host load, while the quiet run's *first* rep beat baseline. **Use the min of N reps** for
  contention-limited legs: the fastest rep is the closest the run got to an uncontended measurement.
- Bisect only once the signal survives repetition.

## 12. What a performance claim must contain

A number without these is an anecdote. Refuse to publish or act on one missing any line.

- **Hardware**: card model, and how many.
- **Shape**: sequence length / atom count / batch, and the exact fixture file.
- **dtype and fidelity**: bf16/fp32, LoFi/HiFi2/HiFi4, `fp32_dest_acc` on or off.
- **Warm or cold**, stated explicitly, with what was excluded (model load, first compile).
- **N runs, the statistic and why that statistic, and the spread**, plus the **A/A floor**
  measured the same session. Median for symmetric noise, min for one-sided; §11 has the rule.
- **Device time and end-to-end wall**, separately.
- **Which instrument produced each number**, and confirmation that every timed region synced.
- **The exact command**, reproducible from a clean checkout.
- **What it is compared against**: a same-run control arm, not a published cell, and matched batch
  and protocol on both sides.
- **Blank stays blank.** If an instrument was unavailable or a counter unreadable, say so precisely.
  A named negative result saves the next pass a day. Never fill a measurement slot from a spec sheet
  or a prior run's memory.

`06-profiling-instruments.md` covers which instrument answers which question and what each one
silently lies about. `02-parity-and-correctness.md` covers the parity method every lever here must
not break.
