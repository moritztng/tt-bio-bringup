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
  references/               14 reference documents, loaded on demand
  templates/                the documents the workflow expects you to keep
agents/                     subagent definitions for the four recurring roles
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

## Install

As a plain Claude Code skill:

```bash
git clone https://github.com/moritztng/tt-bio-bringup.git
ln -s "$PWD/tt-bio-bringup/skills/tt-bio-bringup" ~/.claude/skills/tt-bio-bringup
```

Or per project, so your whole team gets it from the repo:

```bash
mkdir -p .claude/skills
git submodule add https://github.com/moritztng/tt-bio-bringup.git third_party/tt-bio-bringup
ln -s ../../third_party/tt-bio-bringup/skills/tt-bio-bringup .claude/skills/tt-bio-bringup
```

The subagent definitions are separate. Copy the ones you want into `.claude/agents/`:

```bash
mkdir -p .claude/agents && cp agents/*.md .claude/agents/
```

## Using it

In your tt-bio fork, with the skill installed:

```
Port the model in ./reference_impl to Tenstorrent. Start at Phase 0 of the tt-bio-bringup skill.
```

Claude reads `SKILL.md`, writes `PORT_PLAN.md`, and works the phases. Each phase names the reference
documents to read before starting it, so context stays small until it is needed.

Three things to hold Claude to, because they are what make the difference:

1. **No phase starts before the previous gate is green**, and a gate is a command you can run, not a
   summary paragraph.
2. **No performance number without a matched-protocol A/B**: same shapes, same batch, same warm
   state, same card, repeated runs, median reported.
3. **Break every green check once** to prove it can go red. A test that cannot fail is not evidence.

## Scope

Covers: protein folding and structure prediction, protein language models, diffusion structure
heads, MSA and template encoders, docking and affinity heads, binder and sequence design loops.
Inference only. Training is out of scope.

## License

MIT. See `LICENSE`.
