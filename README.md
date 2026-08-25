# tt-bio-bringup

A Claude Code skill for porting a PyTorch biomolecular model to Tenstorrent hardware inside a
[tt-bio](https://github.com/moritztng/tt-bio) fork, and finishing the job: parity-verified against a
CPU golden, optimized against a measured roofline, packaged and gated like the models already there.

It is written for a team that has a proprietary model, a PyTorch reference implementation,
Tenstorrent hardware, and no prior TT experience. No GPU is required at any point. The CPU reference
run is the golden.

## What is in here

```
skills/tt-bio-bringup/
  SKILL.md                  the workflow: 8 phases, each with a machine-checkable exit gate
  references/               15 reference documents, loaded on demand
  templates/                the documents the workflow expects you to keep
  gates/port_gate.py        the gate checker the phases call, standard library only
  examples/                 one small model through every phase, plus a runnable Phase 1 capture
agents/                     subagent definitions for five recurring roles
scripts/redaction-check.sh  the check that keeps this repo publishable
```

The reference documents are the distilled part. They are the accumulated result of porting a series
of protein folding, structure prediction, protein language and diffusion models to this hardware:
what the correctness method actually has to be, which optimizations pay and by how much, and the
long list of ways a port silently goes wrong while every test stays green.

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
| `14-running-a-long-campaign.md` | Running this as a months-long agentic effort without losing the plot |
| `15-torch-to-ttnn-op-map.md` | Which ttnn op replaces which torch op, and how to check for yourself |

## Install

Two ways. The plugin gets you the skill and all five subagents in one command; the symlink gets you
the skill only.

**As a plugin**, from inside Claude Code:

```
/plugin marketplace add moritztng/tt-bio-bringup
/plugin install tt-bio-bringup@moritztng
```

**As a plain skill**, for one machine:

```bash
git clone https://github.com/moritztng/tt-bio-bringup.git
mkdir -p ~/.claude/skills ~/.claude/agents
ln -s "$PWD/tt-bio-bringup/skills/tt-bio-bringup" ~/.claude/skills/tt-bio-bringup
cp tt-bio-bringup/agents/*.md ~/.claude/agents/
```

Or per project, so the whole team gets it from your own repo:

```bash
mkdir -p .claude/skills .claude/agents
git submodule add https://github.com/moritztng/tt-bio-bringup.git third_party/tt-bio-bringup
ln -s ../../third_party/tt-bio-bringup/skills/tt-bio-bringup .claude/skills/tt-bio-bringup
cp third_party/tt-bio-bringup/agents/*.md .claude/agents/
```

Check it took. Two things about that command: `-L` is required because the install above is a
symlink and plain `find -type d` will not descend into one, and the result has to be tested rather
than read, because `find | head -1` exits 0 when it matched nothing.

```bash
export SKILL=$(find -L ~/.claude/skills ~/.claude/plugins/cache .claude/skills \
        -type d -name tt-bio-bringup -path '*skills*' 2>/dev/null | head -1)
test -n "$SKILL" && test -f "$SKILL/SKILL.md" && echo "installed at $SKILL" || {
    echo "NOT installed: nothing matched"; false; }
```

`/skills` should also list `tt-bio-bringup`.

## Using it

First get to the starting line: fork and install tt-bio, check the cards answer, run its test suite,
fold one existing model. That is `references/01-orientation.md`, "Day zero". Everything from Phase 2 on is
uninterpretable until one existing model folds on your card, so do not skip it. You do not have to
wait for it either: Phase 0 and Phase 1 need torch and nothing else, so if the hardware is still in
a box, start Phase 0 today and do day zero alongside it.

Then, in your fork:

```
Port the model in ./reference_impl to Tenstorrent. Start at Phase 0 of the tt-bio-bringup skill.
```

Claude reads `SKILL.md`, writes `notes/PORT_PLAN.md`, and works the phases. Each phase names the
reference documents to read before starting it, so context stays small until it is needed.
`skills/tt-bio-bringup/examples/worked-example.md` shows the whole thing done on a small model if
you would rather see it than read about it.

Three things to hold Claude to, because they are what make the difference:

1. **No phase starts before the previous gate exits 0**, and a gate is a command you can run, not a
   summary paragraph. All eight are commands, and five of them call `gates/port_gate.py`.
2. **No performance number without a matched-protocol A/B**: same shapes, same batch, same warm
   state, same card, repeated runs, and the statistic named. Median for symmetric noise, **min** when
   the noise is one-sided, which host contention is (`05-perf-method-and-roofline.md` §11).
3. **Break every green check once** to prove it can go red. A test that cannot fail is not evidence.
   `port_gate.py prove-red --check ... --break ... --restore ...` does the whole sequence and reads
   the exit codes for you, so this costs one command rather than a decision.

## Scope

Covers: protein folding and structure prediction, protein language models, diffusion structure
heads, MSA and template encoders, docking and affinity heads, binder and sequence design loops.
Inference only. Training is out of scope.

## License

MIT. See `LICENSE`.
