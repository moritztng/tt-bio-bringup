# tt-bio-bringup

A Claude Code skill that ports a PyTorch biomolecular model to Tenstorrent hardware inside a
[tt-bio](https://github.com/moritztng/tt-bio) fork and finishes the job: parity-verified against a
CPU golden, optimized against a measured roofline, packaged and gated like the models already there.

All you need is a PyTorch reference implementation, Tenstorrent cards and no prior TT experience.
The model can be your own, proprietary, or an open one. **No GPU is required at any point** — the
CPU reference run is the golden.

## Quickstart

**1. Install the skill.** In Claude Code:

```
/plugin marketplace add moritztng/tt-bio-bringup
/plugin install tt-bio-bringup@moritztng
```

Then, in the shell you will work from, point `SKILL` at it — everything in the workflow reads that
variable:

```bash
export SKILL=$(find -L ~/.claude/skills ~/.claude/plugins/cache .claude/skills \
        -type d -name tt-bio-bringup -path '*skills*' 2>/dev/null | head -1)
test -f "$SKILL/SKILL.md" && echo "installed at $SKILL" || echo "NOT installed"
```

**2. Get to the starting line.** Fork tt-bio, install it, and fold one model that already ships with
it. Everything from Phase 2 on is uninterpretable until a known-good model folds on your card, so
this is not optional. `references/01-orientation.md` § "Day zero" walks it.

**3. Point it at your model.** From your fork, either drive it yourself:

```
Port the model in ./reference_impl to Tenstorrent. Start at Phase 0 of the tt-bio-bringup skill.
```

or let it run unattended until every gate is green:

```bash
export CARD=0                        # UMD chip id, from `tt-smi -ls`
nohup bash "$SKILL/run.sh" --unattended --model-name yourmodel > port-loop.log 2>&1 &
```

That is the whole setup. Two notes before you start:

- **This is a real port, not a wrapper.** The skill exists to stop the three ways it goes wrong:
  optimizing before measuring, declaring parity from a test that could not have failed, and losing
  the thread across sessions.
- **Phases 0 and 1 need no card**, so if the hardware is still in a box, start today and do day zero
  alongside.

## Install

The plugin above is the recommended path and picks up the five subagents by itself. If you would
rather have the files locally to read or edit, clone instead:

```bash
git clone https://github.com/moritztng/tt-bio-bringup.git
mkdir -p ~/.claude/skills ~/.claude/agents
ln -sfn "$PWD/tt-bio-bringup/skills/tt-bio-bringup" ~/.claude/skills/tt-bio-bringup
cp tt-bio-bringup/agents/*.md ~/.claude/agents/     # the plugin path does this for you
```

The symlink means `git pull` updates the skill in place. To give a team the same setup, commit that
symlink and a submodule of this repo into your own project's `.claude/skills/`; teammates then need
`git clone --recurse-submodules` (or `git submodule update --init`) or the link dangles.

Three details of the `SKILL` one-liner in the Quickstart, each of which got this wrong once:
`-L` is required because the install is a symlink and plain `find -type d` will not descend into
one; the result is tested rather than read, because `find | head -1` exits 0 even when it matched
nothing; and it is exported because the gate script shells out via `bash -c`, which inherits
exported variables only.

If it prints `NOT installed` you are on a plain clone rather than an install, which is fine: point
`SKILL` at the clone's own `skills/tt-bio-bringup` directory. `/skills` should also list
`tt-bio-bringup`.

## How the port runs

Eight phases, each ending in a gate that is a command you can run rather than a paragraph you can
believe. Claude reads `SKILL.md`, writes `notes/PORT_PLAN.md`, and works them in order, loading only
the reference documents each phase names so context stays small.

| | Phase | |
|---|---|---|
| 0 | Map the model | no device code yet |
| 1 | Golden capture on CPU | fp32, unmodified reference |
| 2 | Skeleton on device | |
| 3 | Component parity, leaves first | every module against its golden |
| 4 | Generality | every input mode and size |
| 5 | Performance | roofline first, each lever A/B proven |
| 6 | Integration and gates | CLI, packaging, release gate |
| 7 | Keep it | regressions stay caught |

`skills/tt-bio-bringup/examples/worked-example.md` shows all of it done on a small model, if you
would rather see it than read about it.

### Three things to hold Claude to

1. **No phase starts before the previous gate exits 0.** All eight gates are commands; five call
   `port_gate.py`, the copy you make in your fork.
2. **No performance number without a matched-protocol A/B**: same shapes, batch, warm state and
   card, repeated runs, statistic named. Median for symmetric noise, **min** when the noise is
   one-sided, which host contention is (`05-perf-method-and-roofline.md` §11).
3. **Break every green check once**, to prove it can go red. A test that cannot fail is not
   evidence. `port_gate.py prove-red --check … --break … --restore … --expect-change <file>` runs
   the sequence and reads the exit codes for you. It refuses without `--expect-change`, because a
   break that edited nothing looks exactly like a gate that is decoration.

### About the unattended loop

`run.sh` runs the gates and starts a fresh Claude session on the next phase until all eight are
green. `--unattended` is required and means `--permission-mode bypassPermissions`: sessions run
commands in your fork without asking, so use a branch and a checkout you are willing to have edited.

It stops when the port is green, when three iterations move nothing, when it reaches the dollar
ceiling you gave it, or when a session hits a decision only you can make — and appends which to
`notes/PORT_STATE.md`. The first run writes `notes/PORT_GATES.md`, the eight commands it will hold
the port to. Read that before you walk away.
`references/14-running-a-long-campaign.md` §3.1 covers the rest, including what the loop does not
decide and why tamper detection is not tamper prevention.

## What is in here

```
skills/tt-bio-bringup/
  SKILL.md                  the workflow: 8 phases, each with a machine-checkable exit gate
  references/               15 reference documents, loaded on demand
  templates/                the documents the workflow expects you to keep
  gates/port_gate.py        the gate checker the phases call, standard library only
  examples/                 one small model from Phase 0 to Phase 6, plus a runnable Phase 1 capture
  run.sh                    the unattended loop
agents/                     subagent definitions for five roles: three that work, two that audit
scripts/redaction-check.sh  the check that keeps this repo publishable
scripts/check-example.sh    runs the worked example through the gates it prescribes
```

The reference documents are the distilled part: the accumulated result of porting a series of
protein folding, structure prediction, protein language and diffusion models to this hardware. What
the correctness method actually has to be, which optimizations pay and by how much, and the long
list of ways a port silently goes wrong while every test stays green.

| Document | Subject |
|---|---|
| `01-orientation.md` | The hardware, the stack, what tt-bio is, your first hour |
| `02-parity-and-correctness.md` | Goldens, component-by-component parity, thresholds, fixture blind spots |
| `03-precision-and-numerics.md` | bf16 vs fp32, math fidelity, drift, silent slow paths |
| `04-shapes-tiles-and-bucketing.md` | 32x32 tiles, padding tails, masking, bucketing, recompiles |
| `05-perf-method-and-roofline.md` | Census, measured roofs, predict-then-build, screening traps |
| `06-profiling-instruments.md` | Which profiler answers which question, and how each one lies |
| `07-optimization-levers.md` | The ranked catalogue of speedups, each with its precondition |
| `08-memory-and-residency.md` | L1, DRAM, sharding, OOM by allocation count, chunking |
| `09-devices-and-hardware-operations.md` | Device identity, wedges, resets, faulty cards, multi-chip |
| `10-custom-kernels.md` | When a hand-written Tensix kernel is justified, and how to write one |
| `11-tt-bio-integration.md` | Landing the model in the repo the way the others are landed |
| `12-testing-and-gates.md` | Test layers, and gates that cannot pass vacuously |
| `13-failure-atlas.md` | Symptom-indexed atlas of real failure modes |
| `14-running-a-long-campaign.md` | Running this as a multi-session agentic effort without losing the plot |
| `15-torch-to-ttnn-op-map.md` | Which ttnn op replaces which torch op, and how to check for yourself |

## Scope

Covers protein folding and structure prediction, protein language models, diffusion structure heads,
MSA and template encoders, docking and affinity heads, binder and sequence design loops. Inference
only; training is out of scope.

## License

MIT. See `LICENSE`.
