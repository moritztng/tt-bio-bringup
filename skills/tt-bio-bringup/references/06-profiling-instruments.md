# Profiling instruments: which one answers which question, and how each one lies

This document decides, for each performance question you can ask about a model running on
Tenstorrent, exactly which instrument to reach for, how to run it on a real model without
overflowing its buffers, how to read its output into an op census, and the specific false
answer each instrument gives when used for the wrong question. Every instrument here is
wrong about something; the corrections are the point.

**Read this when** you are about to time anything, when you have a profile you do not trust,
or when a measured speedup did not survive contact with the full model.
Method (floor → screen → predict → build) lives in `05-perf-method-and-roofline.md`; this file is the toolbox.

## 1. Question → instrument

| Question | Instrument | Command / knob |
|---|---|---|
| Where does model time go? | device profiler ops report, aggregated to an op census | `./env/bin/python3 -m tracy -r -o OUT --op-support-count N -- /abs/path/script.py` |
| Is this op compute- or bandwidth-bound? | roofline microbenchmark for the two roofs, plus this op's `DEVICE KERNEL DURATION` and its own byte/FLOP arithmetic | `scripts/profiling/roofline_bh.py`, then bytes ÷ duration vs FLOPs ÷ duration |
| Is the host or the device the bottleneck? | bare synced wall clock minus the summed device kernel time; explain the residual with a host trace | `time.perf_counter()` + `ttnn.synchronize_device()`; `py-spy record` / `cProfile` on the same region |
| Why is the second call cheaper than the first? | run the same shape twice in one process and diff; compile lands in the op-to-op gap, not in a kernel | see §5 warm/cold; `~/.cache/tt-metal-cache` and `enable_program_cache()` |
| How many ops does one step dispatch? | `ttnn.graph` capture around one iteration (works on the pip wheel, no Tracy build) | count `function_end` nodes whose `params.name` starts with `ttnn.` |
| What is this op's peak L1 usage? | `ttnn.graph.extract_peak_L1_memory_usage` on a **cold** call | a warm capture reads 0 (§4) |
| Is my optimized path actually firing? | an assertion or an explicit counter in the code, incremented on the line that does the work | never a timing inference, never a flag read (§4) |
| Is dispatch the cost? | eager vs traced A/B on an idle host | `ttnn-trace-capture` procedure; if traced ≈ eager, dispatch was not the problem |
| Is the matrix engine busy? | hardware perf counters, FPU group | `--profiler-capture-perf-counters=fpu,instrn` (budget cost: §3) |

Two rules that sit above the table:

- **A `ttnn` op call is an asynchronous enqueue.** `ttnn.add(a, b)` returns as soon as the command
  is queued. A host timer around it measures enqueue cost. Only `ttnn.synchronize_device(device)`
  immediately before the clock stops makes a host timer mean anything. Measured on one 2048³
  matmul: 0.019 ms/iter unsynced vs 0.145 ms/iter synced, a **7.63x** under-report, and the factor
  grows with the length of the loop you time.
- **Syncs are removed to go faster and added to measure honestly.** These are opposite jobs.
  `ttnn.to_torch` is a blocking drain, so in an unsynced per-op profile it absorbs the device time
  of everything queued ahead of it. The tell is cost inversely correlated with bytes: your "most
  expensive" line moves 0.0 MB. That inversion is not a magnitude error, it reverses the ranking.

## 2. Four clocks, never added together

- **Host wall clock** around an async enqueue: meaningless without a sync.
- **`DEVICE FW DURATION [ns]`**: firmware entry to exit, includes the per-core dispatch prologue.
- **`DEVICE KERNEL DURATION [ns]`**: the kernel zone only. This is "how long did this op compute".
- **`OP TO OP LATENCY [ns]`**: the gap between ops. Dispatch, host Python, and JIT compile all
  land here.

Per-RISC columns split the kernel: `DEVICE BRISC/NCRISC KERNEL DURATION` are dataflow (NoC in/out),
`DEVICE TRISC0/1/2` are unpack/math/pack. **TRISC1 much shorter than BRISC/NCRISC means the op is
waiting on data.** The subtlety: TRISC1 can be resident for the whole kernel and idle inside it. On
a bandwidth-bound `ttnn.add`, TRISC1 ran 226 of the kernel's 227 µs. "TRISC1 short" catches ops that
skip compute entirely; it does not catch a stalled math thread. For that you need counters.

## 3. Profiling a real model without hanging it

The device profiler has a hard wall at **1000 dispatched programs** (`DEFAULT_PROFILER_PROGRAM_
SUPPORT_COUNT`), and a bio model blows through it inside a single step. Measured: an ESMFold-class
model at L=330 dispatches **1290 ttnn ops in one diffusion step**; a whole fold of the same model
dispatches **43 291**. So "profile one step" is mandatory, and even one step needs the budget raised.

Constraints, in the order they bite:

1. **Build.** The pip `ttnn` wheel has the device profiler compiled out, and it still ships a
   host-side `libtracy.so`, so it looks enabled. Three tells:
   `grep -m1 ENABLE_TRACY:BOOL $BUILD/CMakeCache.txt` reads `OFF`; `TT_METAL_DEVICE_PROFILER=1`
   raises `TT_FATAL: ... requires a Tracy-enabled build`; and on a real Tracy build `import ttnn`
   needs the python `tracy` module from `$TT_METAL_HOME/tools`.
   ```sh
   export TT_METAL_HOME=/path/to/your/tt-metal-source-build
   export PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME
   unset LD_LIBRARY_PATH   # a stale one resolves _ttnn.so against the wrong libttnncpp
   ```
2. **Op budget.** `--op-support-count N` (`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`) is the only
   mitigation that recovers a failing run. Set it just above the op count you measured with a graph
   capture. With perf counters on, multiply by **~21x**: a counter-instrumented op eats ~21x the
   marker budget (1695 ops recovered 98 rows at 2000, 374 at 8000, all 1695 at 45 000).
3. **`--dump-device-data-mid-run` does not fix overflow.** Tested at 5000 ops with the default
   budget: identical dropped-marker warnings, identical failure. It changes when data is flushed,
   not the on-device buffer size. It is also mutually exclusive with `--device-trace-profiler` and
   `--profile-dispatch-cores`.
4. **Disk.** ~350 KB of device CSV per dispatched op, and the artifact directory is ~3x that
   because `reports/` duplicates `.logs/`: 5.2 GB for 5000 ops. One diffusion step ≈ 450 MB;
   a whole 43k-op fold ≈ 15 GB of CSV and ~45 GB of artifacts. Delete the directory afterwards.
5. **Run length.** Warm the shape first, then profile the *second* occurrence, and wrap only the
   region you care about. Iteration N of a sampler is representative of N+1.

Failure signature to memorise, because it reads like your script broke:
650 repetitions of `Profiler DRAM buffers were full, markers were dropped!` (one per core×RISC),
`bufferEndIndex = 12000`, then post-processing dies with
`AssertionError: Device data missing: Op <id> not present in cpp_device_perf_report.csv` and you get
no ops report at all. Your workload's own stdout printed fine, which is what makes it confusing.
With counters on the message is different, and more useful:
`Device data mismatch: Expected 1695 but received 98` tells you the ratio to size the budget by.

**Two-pass recipe for a big model.** Pass 1: `ttnn.graph` capture around **one** iteration on the
plain wheel, to get the op count and shapes. Pass 2: device profiler on that same single iteration,
`--op-support-count` set from pass 1. Pass 1 exists to tell pass 2 what to set.

## 4. Reading the output into an op census

The ops report is one CSV row per dispatched op. Aggregate by `OP CODE`:

```python
import csv, collections
rows = list(csv.DictReader(open("ops_perf_results_*.csv")))
K, G = "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"
num = lambda s: int(s) if s.strip().isdigit() else 0
agg = collections.defaultdict(lambda: [0, 0])
for r in rows:
    a = agg[r["OP CODE"]]; a[0] += num(r[K]); a[1] += 1
total = sum(a[0] for a in agg.values())
for op, (ns, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:10]:
    print(f"{op:36s} {ns/1e6:8.3f} ms {n:6d} {100*ns/total:5.1f}% {ns/n/1000:8.1f} us/call")
```

Columns that matter, in order: `OP CODE`, `DEVICE KERNEL DURATION [ns]`, `GLOBAL CALL COUNT`,
`CORE COUNT`, `MATH FIDELITY`, `INPUT_*_{W,Z,Y,X}` (format is `padded[logical]`, parse the leading
integer), `OP TO OP LATENCY [ns]`, and the per-RISC durations. `CORE COUNT` below the full grid is
a second limiter the roofline alone will not name. Per-core and full-grid utilization differ by
exactly the occupancy ratio, so a dominant matmul on 80 of 130 cores reads `130/80` = **1.63x
better per core than it does across the grid**, and a tool that reports only the per-core figure
makes an op with a third of the grid idle look close to its roof. Compute the ratio yourself from
`CORE COUNT`; do not accept either number alone.

**Separating kernel time from dispatch gaps.** Sum both columns. A warm, compute-bound section has
a median `OP TO OP LATENCY` near a microsecond and a gap sum far below the kernel sum. A
dispatch-bound section has many rows with small kernel durations and gaps of the same order or
larger, i.e. hundreds of tiny ops each doing a few microseconds of work. Below N≈1024 on a square
matmul you are not measuring compute at all: 18-24 µs of dispatch dominates the math, and the
fidelity ordering inverts (HiFi4 measured *faster* than LoFi at N=512).

**The gap column is where cold compile hides.** On one cold ops report the median gap was 0.54 µs
but p90 was 180 ms and the max 762 ms, and the gap sum (19.2 s) was a hundred times the kernel sum
(0.188 s). Almost all of that is JIT kernel compilation charged to the gap before the op that
triggered it. Never read a gap distribution from a cold capture.

## 5. The lies, and the correction for each

| Lie | Correction |
|---|---|
| **Isolated per-op timing over-reads the op's real cost.** A per-op `synchronize_device` forces a full device drain that a real chain never pays between consecutive ops. Measured: pieces of one FFN summed to 17.93 ms isolated against a measured chain cost of 14.657 ms; a single `ttnn.slice` read 0.0433 ms/chunk isolated and 0.0222 ms/chunk when issued 52-at-a-time the way the model issues it, a 95 % over-read. | Re-time candidate ops batched back-to-back the way the model issues them, then price the decision. Treat any gate derived from isolated timing as provisional. The bias also runs the *other* way for residency levers: an isolated screen predicted −3.87 ms/step and the built lever delivered −6.974 ms/step, because syncing both sides of one op hides the DRAM-queue relief that keeping bytes resident gives every concurrent op. Isolated screening is unreliable in both directions; only the in-model measurement settles it. |
| **A single op in a loop stays warm, resident and uncontended in a way the real model never is.** It has the whole chip, no queue contention, no competing L1 tenants. Measured: an op that took 21.5 ms in isolation accounted for 3.9 % of the real loop, so the isolated timing overstated its share and the isolated number was the one that got quoted. | Validate every op-level win inside the full step before quoting it. A per-call delta must also be multiplied by the model's own repeat count before comparing it to fold-level headroom (a 0.54 %-per-recycle cost against a 0.33 % margin is not "room" when the model spends 10 recycles). |
| **A cold profile charges compile to the first op.** Cold-to-warm is roughly 13x on wall clock and 60-84 s of kernel compile for a mid-size fold. Three cache layers: on-disk kernel cache, in-process program cache, device DRAM. | Warm at least the shape you are profiling in the same process, then profile the second occurrence. Enable the program cache once and never clear it per input. Discard the first-ever run on a fresh machine entirely. |
| **Summing per-op device time never equals wall clock.** The sum omits dispatch gaps, host Python, host↔device transfers, featurisation, and the queue. | Compute the residual explicitly and explain it: `residual = bare_synced_wall − Σ DEVICE KERNEL DURATION`. That residual, not the op census, is the number that tells you whether to optimise kernels or the host. Take the wall from a **bare** run: the profiler itself adds +20.7 % at 100 ops rising to +36.2 % at 5000, so a profiled wall clock is not comparable to anything. |
| **A counter or flag named after an optimization tells you the path was requested, not that it ran.** A merged L1-residency win never fired in any real fold: its guard tested `M % 32` on the *logical* flattened M, while TILE_LAYOUT pads before `fuse_batch` folds the leading dims, so the real M is `prod(leading) * ceil32(rows)`. Both arms of its validating A/B ran identical code, so the result was bit-exact and looked like a clean pass. A separate case: an upstream fused-kernel fallback counter read 0 for the entire run while the fallback was firing. And a kernel-variant counter tells you which kernel ran, never what dtype flowed through it. | Assert on the line that does the work, not on the flag that requests it. Count admits *and* rejects, and print both. Then prove the guard admits on a real model input, not on a synthetic benchmark shape: standalone harnesses feed N=128/256/320/384, all tile multiples, so a broken tile predicate never trips there. |
| **A number measured on a quietly faulty or thermally limited card is unreproducible anywhere else.** One card was root-caused as silently miscomputing some matmuls at a low, location-keyed rate at every size, with 64 of 130 cores never hit; 15 clean folds on it proved nothing about the next one. Thermal drift is the milder version: all-A-then-all-B legs read +13.3 % where the honest interleaved answer was +5.2 %. | Interleave A/B legs (A,B,A,B) and compare medians, never all-A-then-all-B. Re-run any surprising result on a second card before believing it. Never let a bit-exact or hash-equality check land on a card that has ever failed one. Host contention is one-sided noise (it can only slow a rep down), so for that class the robust statistic is the **min** of N reps, not the median. |

Three more the same table would cover if it were longer: the profiler perturbs what it measures, so
attribute with it and time A/B without it, and never mix the two in one table; `PM FPU UTIL (%)` is
a performance *model*, not a counter, and reads 125-128 % on `ttnn.clone` (an op with no compute
kernel), so read `Avg FPU util on full grid (%)` instead; and FPU utilization is cycles-issued, not
fraction of peak FLOP/s, so raising math fidelity raises utilization 24 %→53 % while *lowering*
delivered throughput 110→81 TFLOP/s. "Utilization is low, so there is headroom" is valid;
"utilization is high, so we are near the roof" is not.

## 6. Host-side profiling

The residual from §5 is host time, and it is usually Python. Sample it directly:

```sh
py-spy record -o /tmp/host.svg --subprocesses -- ./env/bin/tt-bio predict ...
# or, for a single region you can wrap:  cProfile.Profile() around the warm step only
```

What you are looking for is per-call overhead multiplied by call count, not any single slow
function. Concrete shapes that have dominated real models:

- **A Python `for` loop of small device ops.** A slice-plus-concat loop building a windowed K/V
  gather was 85 % of a stage; replacing it with one `ttnn.embedding` took that stage from
  113 → 41 ms/step, bit-identical. The device work was never the cost; issuing it was.
- **Per-iteration `to_torch`/`from_torch`.** Each one is a full device drain. Removing 8 host
  round-trips from a recycling loop was worth 8-12 %.
- **Recompute of step-invariant work inside the loop.** A section that is >25 % of per-step cost
  but whose inputs do not include the step variable is a hoist, not a kernel problem: precomputing
  per-block conditioning biases once took a diffusion step 364 → 113 ms.
- **Host featurisation outside the model entirely.** On one model a large fraction of a 22.0 s fold
  was host featurisation, re-run per fold with no cache. A cost that sits on both arms of a
  comparison pulls any ratio toward 1 and flatters whichever side is slower. Recover it from the two
  ratios rather than timing it by eye: `05-perf-method-and-roofline.md` §9 gives the formula and
  works it out at 4.07 s for that fold.

Because dispatch is CPU-bound host work, **host load contaminates dispatch-heavy arms selectively**.
A trace-on/trace-off comparison on a loaded machine read 10.1 s vs 22.4 s, a fake −55 % win; with
the machine idle the same comparison read 9.75 s vs 9.68 s, i.e. zero. Check that nothing else is
running on the host, not just on the card, before believing any host-vs-device comparison.

## 7. Trace capture as an instrument, not only an optimization

Capturing a trace replays the recorded device instruction stream with zero host dispatch. That makes
the eager-vs-traced delta a direct measurement of **how much of your step was host dispatch**, and it
is a cleaner answer than any counter.

```python
dev.enable_program_cache()
body()                                       # warm: capture disallows binary loads
tid = ttnn.begin_trace_capture(dev, cq_id=0); out = body(); ttnn.end_trace_capture(dev, tid, cq_id=0)
ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
```

Read the result as a diagnosis:

- **traced ≪ eager** ⇒ host-dispatch-bound. Fix by tracing, or better by device residency (removing
  host round-trips), which subsumes it.
- **traced ≈ eager** ⇒ compute- or bandwidth-bound, and no dispatch work will pay. Measured on a
  folding trunk: 17.3 s traced vs 17.4 s eager. On 512-residue diffusion: 31.9 s vs 31.7 s. On a
  five-model campaign of trace ports, every fold came back within noise of 0 %.
- The variable that decides which you get is per-step host dispatch time relative to per-step device
  compute time on *that* device generation, not fold-vs-design and not model family.

Constraints that make the measurement invalid if ignored: the device must be opened with
`trace_region_size` reserved, no allocation is permitted during capture, shapes and addresses must
be static across replays, and there is no loop primitive, so a Python `for` over N steps records N
copies (200 diffusion steps cannot be one trace at any region size). A traced op's output buffer is
owned by the replay: return a copy.

## 8. A profiling session, start to finish

Produces one op census table and one residual line. Roughly 30 minutes on a warm machine.

Steps 0 and 2 use two harnesses that come with tt-bio, `scripts/profiling/roofline_bh.py` and
`scripts/profiling/graph_capture_probe.py`. They are model-agnostic, so you run them, you do not
write them. Everything else in this section you point at your own model.

```bash
# 0. Roofs, once per card. Numbers below are Blackhole p150a, measured, for calibration only.
./env/bin/python3 scripts/profiling/roofline_bh.py --iters 20 --warmup 5 \
    --mm_sizes 4096 8192 12288      # the defaults stop at 4096 and understate the compute roof
#    PEAK_DRAM_BW_GBs 435.2   PEAK_BF16_TFLOPs 161.14 (LoFi) / 100.55 (HiFi4)
#    machine balance  231 FLOP/byte at HiFi4  -> below that, an op cannot be compute-bound

# 1. Bare warm wall clock for the region, profiler OFF, host idle, synced.
#    Warm >=1 iteration in-process first; report the second.
#    -> WALL_MS

# 2. Op count for ONE iteration, pip wheel, no Tracy needed. The probe prints `nodes=<N>`;
#    capture it rather than reading it, because an unset N_OPS makes the budget below 0.
N_OPS=$(./env/bin/python3 scripts/profiling/graph_capture_probe.py --only overhead \
        | grep -oP 'nodes=\K[0-9]+')
echo "N_OPS=${N_OPS:?probe printed no nodes= count; fix step 2 before profiling}"

# 3. Device profile of that same single warm iteration.
env -u LD_LIBRARY_PATH TT_METAL_HOME=$TT_METAL_HOME \
    PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME \
    ./env/bin/python3 -m tracy -r -o /tmp/prof --op-support-count $(( ${N_OPS:?} * 2 )) \
    -- /abs/path/to/profile_target.py
# Not `grep -c ... # must be 0`: grep exits 1 when the count is 0, so the good outcome is the
# failing one, and with a glob -c prints file:count per file rather than one number.
shopt -s nullglob; logs=(/tmp/prof/*.log)
if [ ${#logs[@]} -eq 0 ]; then
    # An unexpanded glob makes grep exit 2 and an "else" branch fire, so "no logs" reads as
    # "no problem". The profiler writing nothing is the worst outcome, not the best one.
    echo "no profiler logs at all: the run wrote nothing, so there is nothing to trust"; false
elif grep -q "markers were dropped" "${logs[@]}"; then
    echo "markers dropped: the trace is incomplete, raise --op-support-count and re-run"; false
else echo "no dropped markers in ${#logs[@]} log(s)"; fi
```

4. Aggregate the ops report with the snippet in §4. That is the census: op, ms, calls, %, µs/call.
5. Residual line: `residual_ms = WALL_MS − Σ DEVICE KERNEL DURATION`, stated as a percentage of
   `WALL_MS`. Use the **bare** wall from step 1, never the profiled one.
6. Route on the residual before touching a kernel. Residual large ⇒ §6 and §7. Residual small ⇒
   take the top census entries, compute each one's arithmetic intensity from its own shapes, and
   place it against the balance from step 0.
7. Delete `/tmp/prof` (gigabytes).

Report it as: which instrument produced each number, that you warmed and synced, the census table,
the residual line, and profiler-perturbed attribution kept in a separate table from bare A/B timing.
If a counter was unreadable or an instrument unavailable, say so precisely and name it. A blank
stays blank: never fill a measurement slot from a spec sheet or from a previous run's memory.
