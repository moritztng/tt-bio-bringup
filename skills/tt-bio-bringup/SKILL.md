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
lives in the repository: the branch, the commits, `notes/PORT_STATE.md`, and a short note per
non-obvious finding. See `references/14-running-a-long-campaign.md`.

**One mechanism, one place.** A per-model copy of shared logic is a defect, not a shortcut. If you
need a variant of an existing helper, extend the helper with a default-safe parameter.

## Where the reference documents are

This skill ships 15 reference documents and one gate script, and they are not in the repository you
are porting into. Find them once, at the start of the session, and pass the path to any subagent you
spawn:

```bash
SKILL=$(find -L ~/.claude/skills ~/.claude/plugins/cache .claude/skills \
        -type d -name tt-bio-bringup -path '*skills*' 2>/dev/null | head -1)
test -n "$SKILL" && test -f "$SKILL/SKILL.md" && echo "installed at $SKILL" || {
    echo "NOT installed: nothing matched"; false; }
```

Two details, both load-bearing. `-L` follows symlinks, and the personal install is a symlink, so
plain `find -type d` returns nothing on a correctly installed skill. And test the result rather than
reading it, because `find` exits 0 when it matched nothing, so a missing skill looks like a
successful search. `references/`, `templates/` and `gates/` are all under `$SKILL`.

If nothing matched, you are working from a plain clone rather than an install. That is fine: use the
clone's own path as `$SKILL`. Copy `$SKILL/gates/port_gate.py` into your fork as
`scripts/port_gate.py` before Phase 0, because every gate below calls it. It is standard library
only.

## Where the port's own documents live

`notes/` in your fork, committed on your port branch:

- `notes/PORT_PLAN.md`, the Phase 0 plan. Updated when the shape of the work changes, never deleted.
- `notes/PORT_STATE.md`, where the port is right now. One file, amended every session, under two
  screens. The first thing a fresh session reads.
- `notes/state-<workstream>.md`, one per parallel line of work, when you have more than one.
- `notes/findings/`, one short note per non-obvious debugging session.

Upstream tt-bio gitignores `/notes/` because its planning lives in a separate system. You do not have
that system, and your continuity across amnesiac sessions is the whole point, so drop that ignore
rule in your fork and commit these files. Record the difference in `notes/PORT_STATE.md` so the next
upstream rebase does not silently restore it. `notes/` is a directory, so `tests/test_repo_root_clean.py`
does not object: that test allowlists root *files* only.

## Phases

Work them in order. Each phase has an exit gate that is a command. Do not start a phase before its
predecessor's gate exits 0, and do not start a phase before reading the reference docs it names.

Every gate below is written for a model called `yourmodel`; substitute your own name. Gate commands
name `./env/bin/python3` rather than a bare `python3`, because a gate that silently ran under the
system interpreter has measured the wrong thing, and that is not visible in its output.

`$CARD` is the **UMD chip ID** of the card you are working on, from `tt-smi -ls`. Not the
`/dev/tenstorrent/<n>` node number, which is a different number on a multi-card host and pins a
different card; `09-devices-and-hardware-operations.md` §1 has both and the mapping. Set it once per
shell:

```bash
export CARD=0
```

Every device gate below writes `${CARD:?set CARD first}` rather than `$CARD`, and that is
load-bearing. With `CARD` unset, `TT_VISIBLE_DEVICES=$CARD` expands to `TT_VISIBLE_DEVICES=`, which
is *set but empty*, which the conftest reads as a deliberate CPU-only run. Every device test skips and
pytest exits 0. The gate for "every reference parameter is consumed exactly once" then reports green
having opened no device and asserted nothing. `${CARD:?}` makes the shell refuse instead.

`TT_VISIBLE_DEVICES` is not optional on a host with cards. Left out entirely, the conftest refuses the
session rather than guessing, because ttnn brings up every card it can see and an unpinned run takes
the whole box from whoever else is using it.

For each gate, run it, then prove it can fail:

```bash
./env/bin/python3 scripts/port_gate.py prove-red \
    --check   '<the gate command>' \
    --break   '<the smallest edit a real regression would make>' \
    --restore '<the inverse edit>'
```

A gate you have not watched go red is not a gate. `prove-red` exists so that checking costs one
command instead of a decision, which is the only reason it actually gets done.

### Phase 0 - Map the model (no device code)

Read: `references/01-orientation.md`, plus `references/02-parity-and-correctness.md` §1.2-1.4 for
the fixture conventions and §3.4 for how thresholds get chosen, because the plan has to name both,
and `references/15-torch-to-ttnn-op-map.md` for the op-inventory column, which is the one place the
plan asks you for a fact rather than a decision. Do not guess a ttnn op name: §1 of that document
answers it in one command. `examples/worked-example.md`
shows a filled-in plan for a small model, with the gate output, if you would rather see one than read
about one.

Produce `notes/PORT_PLAN.md` from `templates/PORT_PLAN.md`:

- The module tree of the reference, leaf modules first, with parameter count and output shape per
  module, and for each one the golden that will prove it and the threshold it must meet.
- Every tensor axis, marked static or dynamic, with its range and its tile-multiple handling. Dynamic
  axes are where tiling and bucketing decisions get made.
- The op inventory: every distinct torch op used, with a first guess at its ttnn equivalent, and a
  list of the ops with no equivalent. That list is your risk register.
- Control flow: loops (recycling, diffusion steps, sampling), conditionals on data, early exits.
- External inputs the model needs beyond the sequence: MSAs, templates, ligand featurization,
  constraints. Each is a host-side pipeline that must be ported too and is routinely underestimated.
- Every source of randomness, and how you will make it reproducible on both sides.
- The supported size range you intend to ship, stated as numbers.
- The evaluation set Phase 3's task metric will run on: which inputs, which ground truth, and whether
  you are licensed to use them with this model.

**Exit gate:**

```bash
./env/bin/python3 scripts/port_gate.py plan notes/PORT_PLAN.md
```

Exit 0 requires every section present, every table row filled, every module carrying a named golden
and a threshold, a size range with numbers in it, and no placeholder or `TBD` anywhere.

### Phase 1 - Golden capture on CPU

Read: `references/02-parity-and-correctness.md`, `references/03-precision-and-numerics.md`.

Pin the reference at a commit. Make it deterministic. Capture per-module input and output fixtures at
two or three sizes, one of which is deliberately not a multiple of 32. Record the reference commit and
config inside each fixture.

**Exit gate:** the capture reproduces, and a test replays each fixture through the reference and
agrees with it.

```bash
./env/bin/python3 scripts/port_gate.py determinism \
    --run './env/bin/python3 scripts/yourmodel_port/capture.py --len 117 --seed 0' \
    --artifact scripts/yourmodel_port/parity_artifacts/blocks_117.pt
./env/bin/python3 -m pytest tests/test_yourmodel_fixtures.py -q
```

The determinism arm deletes the artifact first, so a run that exits 0 without writing it fails
instead of passing on a stale file from yesterday. Do not add the fixture's `.meta.json` to that
list: it records `runtime_s`, which is a measurement and differs run to run, so hashing it fails
the gate for a reason that is not a defect. The metadata is checked for required keys by the pytest
arm instead.

Prove the pytest arm red by corrupting one tensor inside a fixture. If it stays green it is checking
the fixture's keys, not its contents, which is the most common shape of a fixture test that proves
nothing.

### Phase 2 - Skeleton on device

Read: `references/11-tt-bio-integration.md`, `references/04-shapes-tiles-and-bucketing.md`.

Get weights loaded and one forward pass running on hardware at one small size. It is allowed to be
slow. It is allowed to be numerically off. It is not allowed to silently skip parameters.

**Exit gate:** every reference parameter consumed exactly once, and the same input twice gives the
same bits.

```bash
TT_VISIBLE_DEVICES=${CARD:?set CARD first} ./env/bin/python3 -m pytest tests/test_yourmodel_weights.py -q
./env/bin/python3 scripts/port_gate.py determinism \
    --run 'TT_VISIBLE_DEVICES=${CARD:?set CARD first} ./env/bin/python3 scripts/yourmodel_port/forward.py --len 64 --out /tmp/fw.npy' \
    --artifact /tmp/fw.npy
```

The weight test asserts set equality, `assert consumed == set(state_dict)`, not a loop over the names
it happens to know about. Both directions matter: a remap that drops a parameter and a remap that
invents one are different bugs, and a one-directional check catches only one of them. Prove it red by
deleting one key from your remap.

`$CARD` is single-quoted inside `--run` on purpose. `port_gate.py` runs the command through `bash -c`,
which expands it there from the environment you exported it into.

### Phase 3 - Component parity, leaves first

Read: `references/02-parity-and-correctness.md`, `references/03-precision-and-numerics.md`,
`references/13-failure-atlas.md`.

Port and match one module at a time against its captured golden: leaves before blocks, blocks before
the trunk, trunk before the heads, heads before the sampling loop. Inject the device module into the
reference graph so it is tested in situ, not only in isolation. When a module fails, find the first
diverging tensor, then classify: dtype, layout, masking, or a real porting bug.

Do not move on from a failing module. A deviation you leave behind compounds coherently through depth
and reappears at the end as an unattributable end-to-end failure.

**Exit gate:** every component green at its threshold, an end-to-end parity test, one task-level
metric on real inputs, and the parity document written.

```bash
TT_VISIBLE_DEVICES=${CARD:?set CARD first} ./env/bin/python3 -m pytest tests/test_yourmodel_parity.py -q
TT_VISIBLE_DEVICES=${CARD:?set CARD first} ./env/bin/python3 scripts/yourmodel_port/task_metric.py --set notes/eval_set.txt
./env/bin/python3 scripts/port_gate.py report docs/yourmodel-parity.md \
    --require-heading "Component parity" --require-heading "Negative controls"
```

The report arm is why the parity template has a negative-controls table. It checks four things and
you should know which: the section exists, it holds a table with rows in it, no cell is blank, and
no "Went red?" cell says `no`, `pass` or `passed`. What it cannot check is whether you actually ran
the injection. `port_gate.py prove-red` is what makes that true; the table is where you record it.

### Phase 4 - Generality

Read: `references/04-shapes-tiles-and-bucketing.md`, `references/08-memory-and-residency.md`.

Run the size ladder across the full range you promised, including non-multiple-of-32 lengths, the
smallest input, and the largest that fits. Test every input mode. Find the OOM boundary and document
it. Assert that your optimized paths actually fire at every size, because a residency or budget guard
tuned at one size silently stops firing at another.

**Exit gate:** the ladder is green end to end, and each optimized path has an assertion proving it
fired, not an inference from the timing.

```bash
TT_VISIBLE_DEVICES=${CARD:?set CARD first} ./env/bin/python3 -m pytest tests/test_yourmodel_ladder.py -q
```

The ladder is parameterized over the sizes your plan promised, read from one list that the test and
your docs share. Prove it red by narrowing a residency guard's size window by one: if nothing fails,
your "the fast path fired" assertions are missing and you are reading timing instead.

### Phase 5 - Performance

Read: `references/05-perf-method-and-roofline.md`, `references/06-profiling-instruments.md`,
`references/07-optimization-levers.md`, `references/08-memory-and-residency.md`, then
`references/10-custom-kernels.md` only if the census sends you there.

The loop, repeated until the ranked backlog has nothing above the effort bar (`05` §1 defines it, and
tells you how to set your own):

1. Profile the whole model warm. Build an op census: time, count, share.
2. Measure the roofs on your card with microbenchmarks. Do not quote a spec sheet.
3. Place the top ops on the roofline. Compute, for each candidate lever, the ceiling implied by the
   share of time it touches. Write the prediction down before building.
4. Build the highest-value lever. A/B it at matched protocol: same shapes, same batch, same warm
   state, same card, N runs, median.
5. Re-census. Every label expires once a lever removes the traffic that justified it, and levers
   dismissed as too small grow as the others shrink.

**Exit gate:** the performance document holds the census before and after, the measured roofs, every
landed lever with its predicted and actual effect, every killed lever with the number that killed it,
and a command that reproduces each figure. The gate below enforces the first four: each named section
has to exist and carry a filled table. The reproducing command is on you, and it is the one a reader
six months from now will actually need.

```bash
./env/bin/python3 scripts/port_gate.py report docs/yourmodel-perf.md \
    --require-heading "Measured roofs" --require-heading "Op census" --require-heading "Levers"
TT_VISIBLE_DEVICES=${CARD:?set CARD first} ./env/bin/python3 -m pytest tests/test_yourmodel_parity.py -q
```

Parity is re-run here on purpose. A performance lever that moves the answer is not a performance
lever.

### Phase 6 - Integration and gates

Read: `references/11-tt-bio-integration.md`, `references/12-testing-and-gates.md`.

Land it the way the existing models are landed: shared helpers not private copies, registered on the
CLI, tests in the suite, packaging manifest updated, parity and performance docs written, release gate
extended with an arm for every bug that escaped during the port.

**Exit gate:** the built artifact, installed clean, passes the full gate. Not your working checkout.
They differ, and the difference is what your users hit.

```bash
./env/bin/python3 scripts/packaging_smoke.py
TT_VISIBLE_DEVICES=${CARD:?set CARD first} ./env/bin/python3 scripts/release_gate.py
./env/bin/python3 -m pytest tests/test_perf_model_coverage.py tests/test_repo_root_clean.py -q
```

`packaging_smoke.py`, `release_gate.py` and that coverage test come with tt-bio; you are extending
them, not writing them. Prove the release gate red by reverting one parity fix on a scratch commit.
A gate that has only ever been green on a working tree has never been tested.

### Phase 7 - Keep it

Read: `references/12-testing-and-gates.md`, `references/14-running-a-long-campaign.md`.

Every bug found after this point becomes a permanent gate arm before it is fixed, demonstrated failing
on the pre-fix commit. Rebase on upstream tt-bio in small steps. Re-measure the performance baseline
whenever the hardware or the dependency stack moves, because a recorded baseline is history, not a
live comparison.

**Exit gate**, run on every change and on every upstream rebase:

```bash
TT_VISIBLE_DEVICES=${CARD:?set CARD first} ./env/bin/python3 scripts/release_gate.py
TT_VISIBLE_DEVICES=${CARD:?set CARD first} ./env/bin/python3 scripts/perf_regression.py
```

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
| Which ttnn op replaces this torch op | `references/15-torch-to-ttnn-op-map.md` |
| One small model taken from Phase 0 to Phase 6, filled in | `examples/worked-example.md` |

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
newcomer would. Run the cold reader on your `notes/PORT_PLAN.md` before you hand any of it out.

## Templates

`templates/` holds the documents this workflow expects you to keep: the port plan, the state
document, the parity report, the performance report, the per-finding note and the work brief. Copy
them into your fork and fill them in. They are short on purpose, and three of them carry the
`port_gate.py` command that checks they are actually filled.
