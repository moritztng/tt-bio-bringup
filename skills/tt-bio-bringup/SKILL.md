---
name: tt-bio-bringup
description: Port a PyTorch biomolecular model (protein folding, structure prediction, protein language model, diffusion structure head, MSA encoder, docking, binder design) to Tenstorrent hardware inside a tt-bio fork, to production quality - complete functionality, parity-verified against a CPU golden, performance optimized against a measured roofline, packaged and gated like the models already in tt-bio. Use when the task is "port model X to ttnn", "bring up our model on Tenstorrent", "add our model to tt-bio", "make our TT port correct", or "make our TT port fast". Also use for any single step of that work: capturing goldens, debugging a parity failure, building an op census, choosing an optimization lever, writing a fused kernel, or designing the release gate.
---

# Bringing a bio model up on Tenstorrent

You are porting a PyTorch model to Tenstorrent hardware inside a fork of `tt-bio`. The finished
state is not "it runs". The finished state is:

1. **Complete.** Every input mode, every size in the supported range, every output the reference produces.
2. **Correct.** Every component matched against a CPU PyTorch golden, with a test suite that fails if it regresses.
3. **Fast.** Each hot op placed on a measured roofline, every lever above the effort bar landed and A/B proven.
4. **Integrated.** One shared mechanism per concern, on the CLI, packaged, documented, gated.

That takes weeks, not hours. The way it goes wrong is never a lack of effort. It goes wrong by
optimizing before measuring, by declaring parity from a test that could not have failed, and by
losing the thread across sessions. This skill exists to stop those three things.

## Ground rules

**The CPU PyTorch reference is the only golden.** You do not need a GPU and you should not use one
as a reference. GPU output is a third number that agrees with neither side and starts arguments.
Run the unmodified reference on CPU in fp32, save the tensors, compare against those.

**Verify against live state, every time.** Re-run the command. Re-read the file. Check the deployed
artifact, not your own diff. Never repeat a status from earlier in the session as if it were fresh,
and never state a number you did not measure this session. Most wasted days in this work trace back
to a stale fact treated as current.

**A claim needs a test that could have failed.** Before trusting any green check, break the thing it
guards and confirm it goes red. A gate that passes when the op it names never executed is the single
most common defect in this entire domain. See `references/12-testing-and-gates.md`.

**Write it down as you learn it.** Every session starts with no memory of the last one. Continuity
lives in the repository: the branch, the commits, `PORT_STATE.md`, and a short note per non-obvious
finding. See `references/14-running-a-long-campaign.md`.

**One mechanism, one place.** A per-model copy of shared logic is a defect, not a shortcut. If you
need a variant of an existing helper, extend the helper with a default-safe parameter.

## Phases

Work them in order. Each phase has an exit gate that is a command, not an opinion. Do not start a
phase before its predecessor's gate is green, and do not start a phase before reading the reference
docs it names.

### Phase 0 - Map the model (no device code)

Read: `references/01-orientation.md`.

Produce `PORT_PLAN.md` in your fork:

- The module tree of the reference, leaf modules first, with parameter count and output shape per module.
- Every tensor axis, marked static or dynamic. Name the token axis, the pair axis, the sample axis,
  the MSA axis, the atom axis. Dynamic axes are where tiling and bucketing decisions get made.
- The op inventory: every distinct torch op used, with a first guess at its ttnn equivalent, and a
  list of the ops with no equivalent. That list is your risk register.
- Control flow: loops (recycling, diffusion steps, sampling), conditionals on data, early exits.
- External inputs the model needs beyond the sequence: MSAs, templates, ligand featurization,
  constraints. Each is a host-side pipeline that must be ported too and is often underestimated.
- Every source of randomness, and how you will make it reproducible on both sides.
- The supported size range you intend to ship, stated as numbers.

**Exit gate:** `PORT_PLAN.md` names, for every module in the tree, the golden that will prove it.
No module without a golden. No "we will figure that out later" entry.

### Phase 1 - Golden capture on CPU

Read: `references/02-parity-and-correctness.md`, `references/03-precision-and-numerics.md`.

Pin the reference at a commit. Make it deterministic. Capture per-module input/output fixtures at
two or three sizes, including one size that is deliberately not a multiple of 32. Record the
reference commit and config inside each fixture.

**Exit gate:** capture runs twice and produces byte-identical fixtures. A test loads each fixture
and re-runs the reference module against it, green. If that test cannot fail, it is not a gate.

### Phase 2 - Skeleton on device

Read: `references/11-tt-bio-integration.md`, `references/04-shapes-tiles-and-bucketing.md`.

Get weights loaded and one forward pass running on hardware at one small size. It is allowed to be
slow. It is allowed to be numerically off. It is not allowed to silently skip parameters.

**Exit gate:** a weight-mapping test asserts every reference parameter is consumed exactly once, and
a forward pass at one size runs to completion on device twice with identical output.

### Phase 3 - Component parity, leaves first

Read: `references/02-parity-and-correctness.md`, `references/03-precision-and-numerics.md`,
`references/13-failure-atlas.md`.

Port and match one module at a time against its captured golden, leaves before blocks, blocks before
the trunk, trunk before the heads, heads before the sampling loop. Inject the device module into the
reference graph so it is tested in situ, not only in isolation. When a module fails, find the first
diverging tensor, then classify: dtype, layout, masking, or a real porting bug.

Do not move on from a failing module. A deviation you leave behind compounds coherently through
depth and reappears at the end as an unattributable end-to-end failure.

**Exit gate:** every component parity test green at a threshold you can justify, plus an end-to-end
parity test, plus at least one task-level metric on real inputs.

### Phase 4 - Generality

Read: `references/04-shapes-tiles-and-bucketing.md`, `references/08-memory-and-residency.md`.

Run the size ladder across the full range you promised, including non-multiple-of-32 lengths, the
smallest input, and the largest that fits. Test every input mode. Find the OOM boundary and document
it. Assert that your optimized paths actually fire at every size, because a residency or budget
guard tuned at one size silently stops firing at another.

**Exit gate:** the ladder is green end to end, and each optimized path has an assertion proving it
fired, not an inference from the timing.

### Phase 5 - Performance

Read: `references/05-perf-method-and-roofline.md`, `references/06-profiling-instruments.md`,
`references/07-optimization-levers.md`, `references/08-memory-and-residency.md`, then
`references/10-custom-kernels.md` only if the census sends you there.

The loop, repeated until the ranked backlog has nothing above the effort bar:

1. Profile the whole model warm. Build an op census: time, count, share.
2. Measure the roofs on your card with microbenchmarks. Do not quote a spec sheet.
3. Place the top ops on the roofline. Compute, for each candidate lever, the ceiling implied by the
   share of time it touches. Write the prediction down before building.
4. Build the highest-value lever. A/B it at matched protocol: same shapes, same batch, same warm
   state, same card, N runs, median.
5. Re-census. Every label expires once a lever removes the traffic that justified it, and levers
   dismissed as too small grow as the others shrink.

**Exit gate:** a performance document with the census before and after, the measured roofs, every
landed lever with its predicted and actual effect, and the exact command to reproduce each number.

### Phase 6 - Integration and gates

Read: `references/11-tt-bio-integration.md`, `references/12-testing-and-gates.md`.

Land it the way the existing models are landed: shared helpers not private copies, registered on the
CLI, tests in the suite, packaging manifest updated, parity and performance docs written, release
gate extended with an arm for every bug that escaped during the port.

**Exit gate:** a fresh clone, a fresh virtualenv, an install from the built artifact, and the full
gate green on that installed package. Not on your working checkout. They differ, and the difference
is what your customers will hit.

### Phase 7 - Keep it

Read: `references/12-testing-and-gates.md`, `references/14-running-a-long-campaign.md`.

Every bug found after this point becomes a permanent gate arm before it is fixed. Rebase on upstream
tt-bio in small steps. Re-measure the performance baseline whenever the hardware or dependency stack
moves, because a recorded baseline is history, not a live comparison.

## Router: symptom to document

| What you are dealing with | Read |
|---|---|
| Where do I even start, what is tt-bio | `references/01-orientation.md` |
| Proving the port is correct, goldens, thresholds | `references/02-parity-and-correctness.md` |
| bf16 vs fp32, fidelity, rounding, drift | `references/03-precision-and-numerics.md` |
| Tiles, padding, masks, bucketing, recompiles | `references/04-shapes-tiles-and-bucketing.md` |
| What to optimize and what the ceiling is | `references/05-perf-method-and-roofline.md` |
| Which profiler, and how each one lies | `references/06-profiling-instruments.md` |
| The catalogue of speedups and when each pays | `references/07-optimization-levers.md` |
| L1, DRAM, sharding, OOM, chunking | `references/08-memory-and-residency.md` |
| Cards, wedges, resets, multi-chip | `references/09-devices-and-hardware-operations.md` |
| Writing a custom Tensix kernel | `references/10-custom-kernels.md` |
| Landing it inside tt-bio properly | `references/11-tt-bio-integration.md` |
| Test suite and release gate design | `references/12-testing-and-gates.md` |
| A specific symptom, indexed by how it looks | `references/13-failure-atlas.md` |
| Running this as a months-long agentic campaign | `references/14-running-a-long-campaign.md` |

When something breaks, go to `13-failure-atlas.md` first. It is indexed by symptom in the words you
would use, and most of what will happen to you is already in it.

## Working with subagents

This work parallelizes along the component tree, and only there. Fan out one agent per independent
module in phase 3, one per candidate lever in phase 5. Do not fan out across phases: phase 4 needs
phase 3 finished, and a perf number measured on a not-yet-correct port is worthless.

Constraints when you do fan out:

- One agent per physical card, enforced by a lease file or an environment restriction. Two processes
  on one card corrupt each other's results, and the failure looks like nondeterminism.
- Each agent in its own git worktree, so concurrent edits cannot collide.
- Agents push branches. One owner merges, and verifies the merge against `git log`, not against the
  agent's report.

`agents/` in this repository holds ready-made subagent definitions for four working roles
(`parity-porter`, `perf-analyst`, `kernel-author`, `gate-auditor`) and one audit role,
`cold-reader`, which does not do the work but reads your own documents and briefs back to you as a
newcomer would. Run the cold reader on your `PORT_PLAN.md` before you hand any of it out.

## Templates

`templates/` holds the documents this workflow expects you to keep: the port plan, the state
document, the parity report, the performance report, and the per-finding note. Copy them into your
fork and fill them in. They are short on purpose.
