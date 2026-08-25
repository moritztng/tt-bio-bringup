# Writing a custom Tensix kernel

This document decides whether to write a kernel at all, which of four routes to use, and what to
measure before and after. The default answer is no: a custom kernel is the most expensive lever per
unit of speedup in the whole port, and cheaper moves (a blocking change to an existing op, trace
capture, `ttnn.generic_op`) capture most of what people reach for kernels to get.

Read this when the op census says one op or one chain dominates, the roofline says the current
composition is far from its measured roof, and no `ttnn` op or fusion covers the pattern.

## 1. The decision gate

Do not open an editor on a `.cpp` until all four hold. Write the answers down as numbers.

| Gate | Passing condition | Where the number comes from |
| --- | --- | --- |
| Share | Target op or chain is >=10% of warm end-to-end time at production size. Higher than the campaign-wide effort bar in `05-perf-method-and-roofline.md` §1 on purpose: this is the most expensive lever per unit of effort, so it earns a higher bar than the cheap ones | Op census inside the real forward, not a microbenchmark |
| Headroom | Current composition sits far below its **measured** roof, typically <50% | Roofline with a roof measured on this board, not a datasheet figure |
| No existing op | No `ttnn` op and no fused `ttnn` op covers the pattern | Audit the sequence against the `ttnn` op list, `ttnn.experimental` included |
| Amdahl | `1 / (1 - share)` is a speedup you would ship | Arithmetic. A component at 10.9% of the stack caps a perfect kernel at 1.12x |

Two more that change the plan rather than gating it:

- **Addressable size range.** A fused kernel is usually correct only where its own blocking choice
  coincides with the surrounding op's blocking. Both are one-line functions of the shape, so
  enumerate them across the size axis *before* building; serving 50% of real key-widths and
  declining safely on the rest is a different product from serving all of them.
- **Absorber regime.** Bandwidth-bound or compute-bound, see §6.

### Honest expectations

- **A hand-written matmul loses to `ttnn.matmul` at production sizes.** Measured on Blackhole: 48.9
  TFLOP/s at N=1024 for `ttnn.matmul` against 3.2-3.9 for a careful single-purpose hand-written one,
  ~12x slower, because the library op has L1 reuse and multicast blocking. Substituting it into a
  real fold made the fold slower (105 s vs 86 s). A micro-benchmark "15x" against a whole `ttnn`
  *module* at tiny N is not a comparison against `ttnn.matmul` at production N.
- **If the model is dispatch-bound, no kernel fixes it.** At a 1000-residue fold the triangle
  multiplication's matmuls cost ~2 s of that step's ~48 s; the rest was ~12,000 host dispatches. The
  lever is trace capture (`ttnn.begin_trace_capture` / `ttnn.execute_trace`).
- **Wins that did ship** are 1.06x to 1.34x end-to-end, from bandwidth deletions, not from
  out-computing the library.

## 2. Candidates that are actually worth it

Ranked by hit rate:
1. **A chain of bandwidth-bound elementwise / broadcast / reduction ops that round-trips through
   DRAM between each step.** Layer-norm into a gate into a scale, a residual chain, a permute or
   reblock feeding a matmul. The prize is the deleted round trips; the arithmetic is trivial.
2. **A resident operand currently re-read per call.** Holding a triangle bias resident per head
   inside attention, instead of re-reading it once per batch row, cut 2048 MiB/call to 4 MiB, made
   the attention op 2.43x faster and the whole forward 1.12x-1.34x. Highest-yield pattern in a
   pairformer-style trunk.
3. **Bias + softmax, or a gated projection (`sigmoid(g) * p`) feeding a contraction.** The gate is
   one SFPU pass over tiles the kernel already holds in registers.
4. **Anything whose intermediate is much larger than its input plus output**, triangle
   multiplication and triangle attention above all: `O[i,j,h] = sum_k a[i,k,h] * b[j,k,h]`
   materialises a full `[N,N,C]` intermediate unfused and nothing fused. That size ratio is the
   fusion prize, computable from shapes in a minute.

Non-candidates, from repeated dead ends: packing Q/K/V projections (the reshape/permute/slice to
recover them costs more dispatch than the merged linear saves, 0.93x and 0.58x on two models);
re-implementing a matmul; anything under 5% of the forward, and anything between 5% and the 10% bar
above unless a cheaper lever has already been tried and measured on it; anything running once per request rather
than once per step.

## 3. The four routes

| Route | Time to first result | Deploys on a stock wheel | Full control | Use when |
| --- | --- | --- | --- | --- |
| A. `ttnn.generic_op` | hours | yes | dataflow + compute, within the wheel's kernel API | Default. Almost always start here |
| B. tt-lang Python DSL | hours, simulator-first | no | limited by compiler | Prototyping the math, checking a fusion is expressible |
| C. tt-metal C++ programming example | days | no | total | The DSL refuses to lower your pattern |
| D. `ttnn.experimental` op | days + review | no | total, and maintained | The op will be upstreamed and owned long-term |

### A. `ttnn.generic_op` (start here)

`ttnn.generic_op` takes a `ttnn.ProgramDescriptor` plus `KernelDescriptor`s that name kernel source
files by path, and JIT-compiles them at call time against the ttnn wheel already installed. No
tt-metal source build, no nanobind, no CMake, no dependency bump. This is the route the kernels in
`tt_bio/kernels/` use, and it kills the common assumption that a custom kernel must be upstreamed
before it can reach production.

Method: transcribe an existing op's C++ program factory into a Python `ProgramDescriptor` first,
unmodified, and check it reproduces the native op's time and output bit-exactly. That transcription
costs 1.00x-1.02x of native and is the kill gate for the route: if the faithful transcription does
not match, stop. Only then edit the dataflow. Two traps:

- **Cache the descriptor.** `generic_op` takes the whole program description per call; building it
  in Python cost ~155 us at one production shape against ~91 us of device time, so the call went
  from 91 us to ~246 us, **2.7x slower**, and the op spent more time being described than executed.
  Everything except the buffer addresses is a pure function of (shape, dtype, layout, buffer type,
  core grid): put the addresses in `common_runtime_args`, cache the rest on that tuple, rewrite two
  scalars per call.
- **Resolve the ttnn C++ root properly.** Kernel sources compile in place so sibling includes
  resolve. A pip wheel vendors them at `<pkg>/ttnn/cpp`; a source build keeps them at
  `<checkout>/ttnn/cpp` while importing ttnn from `<checkout>/ttnn/ttnn`. Hardcoding the wheel
  layout makes every kernel fail to JIT-build with "No such file or directory" on a source-built
  environment and only there. Probe for `cpp/ttnn/operations` and take the path that exists
  (`tt_bio/mm_generic.py:ttnn_cpp_root`).

### B. tt-lang Python DSL

Fastest way to find out whether a fusion is even expressible. Simulator first, always: `ttlang-sim
<kernel.py>`; the same binary dispatches to the device when one is visible. Structure is
`@ttl.operation(grid=...)` plus `@ttl.compute()` and `@ttl.datamovement()` blocks over L1 dataflow
buffers. Known compiler limits, which will bite:

- **No 3D batched matmul.** Loop the channel axis host-side.
- **No inline `acc += a @ b`.** Use the carried-accumulator dataflow-buffer pattern.
- **No SFPU/eltwise prologue fused into `@`.** Even `(p*g) @ (p*g)` is rejected, so a fully-fused
  gate-plus-matmul runs in simulation and refuses to lower to hardware. That is exactly the shape a
  trimul-style fusion wants; expect to drop to route A or C to fuse the gate.
- **`transpose` is 2D only.** It crashes on 3D blocks; pre-arrange operands on the host.

Passing in simulation and failing to lower is the normal outcome. The DSL is a correctness prototype
and a feasibility screen, not a delivery vehicle.

### C. tt-metal C++ programming example

Full control, days of work. Create `tt_metal/programming_examples/<name>/` with a `CMakeLists.txt`
doing `add_executable` plus `target_link_libraries(... TT::Metalium)`, add `add_subdirectory` to the
parent `CMakeLists.txt`, then:

```bash
cmake -B build -DBUILD_PROGRAMMING_EXAMPLES=ON     # -B names the build dir; `cmake build` would
                                                  # read `build` as the SOURCE dir and fail.
                                                  # The flag is OFF by default.
cmake --build build --target "metal_example_${NAME:?your example name}"
```

Compute kernels JIT-compile at runtime, so editing a kernel `.cpp` needs no host rebuild. Running
needs both `TT_METAL_HOME` and `TT_METAL_RUNTIME_ROOT` set or the process aborts on an empty root
dir; always wrap in `timeout`. Gotchas costing an afternoon each: include
`"api/dataflow/dataflow_api.h"`, not the bare `"dataflow_api.h"` (the bare include wedged a device);
`init_sfpu` needs `eltwise_unary/eltwise_unary.h`; call `mm_init` once per invocation and do all
gating before all matmul, since interleaving forces mode switches that can hang. L1 is about 1.5 MB
(1,572,864 B) per core as reported, 1,461,760 B per bank as the allocator will actually give you,
and it is the second you budget against (`08-memory-and-residency.md` §1). A bf16 32x32 tile is
2048 B, fp32 4096 B. Use `split_work_to_cores` from
`<tt-metalium/work_split.hpp>` for grid parallelisation.

### D. A real `ttnn.experimental` op

Only if the op will be maintained. Files: `device/<op>_device_operation.{hpp,cpp}`,
`device/<op>_program_factory.{hpp,cpp}`, `device/kernels/{compute,reader,writer}.cpp`,
`<op>.{hpp,cpp}`, `<op>_nanobind.{hpp,cpp}`, `sources.cmake`, `CMakeLists.txt`. The
`DeviceOperation` contract needs five type aliases (`operation_attributes_t`, `tensor_args_t`,
`spec_return_value_t`, `tensor_return_value_t`, `program_factory_t`) and six static methods
(`validate_on_program_cache_miss/hit`, `select_program_factory`, `compute_output_specs`,
`create_output_tensors`, `invoke`). Model it on an existing minimal op, not on the spec.

## 4. Mechanics that bite

- **Registration is four edits, not one.** The op's own `sources.cmake`; an `add_subdirectory` in
  `ttnn/CMakeLists.txt`; the include plus `bind_<name>(mod)` call in the experimental nanobind
  module; and the one everyone misses, the nanobind `.cpp` added to the **top-level**
  `ttnn/sources.cmake`. Miss the last and the build succeeds, then import fails with
  `undefined symbol: bind_<name>`.
- **Rebuild and sync the extension.** The build target is `ttnn`, not `_ttnn`, and the freshly built
  `_ttnn.so` must be copied over the one the Python package imports (the copy under
  `build_Release/lib/` is stale). Verify with `python -c "import ttnn;
  print(hasattr(ttnn.experimental,'<name>'))"` before touching a device.
- **A stale kernel cache makes your edit look like a no-op.** The program cache keys on shape and
  config, not on kernel source content. After editing a kernel, a run that reuses a cached program
  silently executes the old binary, so a fix reads as "no change" and a bug reads as "already
  fixed". Clear or disable the program cache for any debug run before concluding anything about an
  edit. When bisecting a numerical difference, call `disable_and_clear_program_cache()` at the
  diverging call and re-run: if the wrong answer comes back bit-for-bit, the cache is innocent.
- **The first run after a kernel change pays compile time inside the timed region.** Any perf leg
  with `warmup=0, repeat=1` will report a large regression immediately after a kernel lands. Legs
  configured that way have read -16%, -20%, -63% on a merge that was actually neutral or positive.
  Re-run warm before triaging; if the warm re-take passes, raise `warmup` for those legs.

## 5. Correctness for kernels

Validate on device against the CPU PyTorch golden at multiple shapes, and prefer `torch.equal` over
PCC wherever the kernel should be a bit-exact replacement (index reordering, resident operand,
transcription). PCC hides the failures that matter.

- **Fuzz over shapes, do not test one.** A sweep made only of multiples of 32 is structurally blind
  to the whole logical-vs-padded bug class. Lead the sweep with a non-tile-multiple size.
- **The ragged tile tail is where hand-written masking breaks.** A fused attention kernel that left
  ragged tail columns unmasked was 71-76x wrong against fp64 at any non-tile-aligned length, on 100%
  of calls for models that do not pad their token axis. Models that do pad were immune, which is why
  it survived. Screen at the raw ragged length; a screen rounded up to a tile multiple never sees
  it.
- **A fused kernel that allocates its own output must take logical dims from `tensor.shape`, never
  `padded_shape`.** Getting this wrong crashes every forward whose length is not a multiple of 32
  and is invisible at any size where logical equals padded.
- **Read the tile padding when chasing an order-dependent difference.** Some ops read the tile
  padding of the axis they reduce over and some leave their output padding undefined, so stale DRAM
  from an earlier larger forward leaks in. A checksum over the logical region alone calls those two
  tensors identical.
- **A plausible speed is not evidence of correctness.** A kernel that silently inherited a different
  compute kernel config compiled, ran at exactly the expected rate, and was wrong on 88% of
  elements. Only the bit-exact check caught it.
- **Census the call sites, not the model list.** Fixing one caller of a defective primitive leaves
  the others broken. Count ragged vs aligned calls per site; "believed to bucket" is not a census,
  and the counter has inverted the guess before.

## 6. Performance and fusion economics

Measure inside the real model at the production size, warm.

- **Isolated per-op timing over-reads cost, sometimes by ~2x.** An isolated loop pays a
  `synchronize_device` pipeline drain per op that a real chain never pays, and keeps everything warm.
  Measured: isolated per-op timings summing to 17.93 ms against a 14.657 ms chain; one op at 0.0433
  ms/chunk isolated versus 0.0222 ms/chunk issued the way the model issues it, a 95% over-read. Any
  go/no-go gate priced off isolated numbers is provisional.
- **A screen that prices only the prize is half a screen.** Net time is prize minus cost. A fused
  pair-FFN kernel passed GO three times on the prize alone and was dead: fitting its operand into L1
  forces a smaller output block width, whose own measured penalty (2.452 ms/call) landed inside the
  fusion's prize range (1.81-2.89 ms/call) and above its midpoint, so the net was zero to negative
  across the shapes that mattered. Price both halves in one pass, same shape, same L1 budget.
- **Price the comeback.** Fusing DRAM traffic into a transaction-bound kernel does not delete all
  the traffic it appears to: 27-37% of the "deleted" cost came back as new latency, because two page
  reads before one barrier expose roughly twice the reader latency. Budget ~1/3 back, not zero.

### Know the regime before writing anything

Fusing arithmetic into a **bandwidth-bound** kernel is close to free: the absorbed op's math hides
behind DRAM waits. Fusing the same arithmetic into a **compute-bound** kernel un-hides it, the math
must now run in addition rather than underneath, and the subtractive "bytes deleted = prize" screen
is wrong by construction. Screen it in minutes with no prototype:

1. Measure the **absorbing** kernel's roofline colour. One case: the absorber sat at 50% of its DRAM
   roof while the op to be absorbed sat at 88% of its. The absorber had no slack.
2. Price the absorbed op's arithmetic with a switch that already exists in the absorbing kernel and
   toggles exactly one in-DST SFPU pass. Measure the wall-clock delta. That gives you the cost of
   "one more pass over this kernel's tile count" directly. One sigmoid pass measured 0.663 ms/call,
   one integer-round pass 0.198 ms/call.
3. Count the passes the fused op needs. A layer-norm needs at least four row-reading passes and two
   forced circular-buffer materialisations: a full fp32 row exceeds DST and each reduction is a
   barrier. Even at the cheapest pass price the fused arm's compute floor exceeded the whole prize.
   NO-GO in four minutes instead of four days.

Unfusing is not free either: taking an activation off a matmul epilogue rounds its input from the
fp32 accumulator down to storage dtype, and that rounding hit ~3x more elements at ~3x the
per-element error than the approximate-algorithm swap it was blamed on.

## 7. Packaging

Hand-written kernel sources load at runtime by **file path**, not as Python imports, so a missing
file has zero signal until the first eligible on-device call, which then crashes. Dev installs never
catch it: the files are still on disk in the checkout. Recurred three times in one codebase. The fix
is recursive globs, once:

```
# MANIFEST.in
recursive-include tt_bio/kernels *.cpp *.hpp
```
```toml
# pyproject.toml
[tool.setuptools.package-data]
tt_bio = ["kernels/**/*.cpp", "kernels/**/*.hpp"]
```

A glob naming one kernel directory dropped 22 sources across four new directories in a single merge:
66 missing-file failures on the built wheel. Guard with a release-gate leg that **installs the built
wheel** into a clean environment and calls every kernel path, not `pip install -e .`. If a source
with a new extension appears, extend the extension list; never re-add a per-directory path.
Separately, assert at import time that a required custom op exists: an environment whose ttnn
silently resolves to a stock wheel fails, or hangs, on every path with no fallback.

## 8. When it hangs

A buggy kernel hangs the card, and a hung card blocks cluster discovery for every other device on
the host regardless of visibility settings. `SIGINT` the process first, never `SIGTERM` or `SIGKILL`
(both skip the device close and can leave the chip dirty), then
reset; full procedure in `09-devices-and-hardware-operations.md`. Run kernel experiments under `timeout`, and never
scope a wait-loop to a shared script basename: it matches another job's process, or its own, and
waits forever.

## 9. Checklist before merging a kernel

- [ ] Census + roofline + Amdahl numbers recorded, both halves of the screen priced in one pass
- [ ] Addressable size range enumerated; the guard declines safely outside it
- [ ] Bit-exact (`torch.equal`) against the previous path everywhere it should be a no-op
- [ ] Shape sweep includes at least one non-tile-multiple size, at production scale
- [ ] Measured warm, inside the real forward, not in an isolated loop
- [ ] Ships behind an env flag; the default is stated explicitly in the merge message
- [ ] Packaging globs cover the new kernel directory; installed-wheel smoke test passes
- [ ] Perf-gate legs with `warmup=0` re-run warm before any regression is believed
