# Orientation: the hardware, the stack, and what tt-bio is

This document gives you the mental model you need before writing a line of device code, and tells
you what to do in your first hour. Read it once, at the start of Phase 0.

## The hardware, in the terms that change your code

A Tenstorrent accelerator is a grid of Tensix cores. Each core has its own **L1 SRAM**, a compute
engine that works on **32x32 tiles**, and NoC links to its neighbours and to DRAM on the card. There
is no cache hierarchy that hides data movement from you and no warp scheduler that hides latency by
oversubscription. Where the data lives is part of the program.

Four consequences, and they drive almost every decision in this skill:

1. **Everything is tiles.** A tensor is stored as 32x32 tiles. A dimension of 100 occupies 4 tiles
   and 28 rows of padding. That padding is real memory, it is real compute, and if you do not mask
   it, it is real wrong answers.
2. **L1 is small, per-core, and yours to manage.** Keeping a tensor in L1 instead of DRAM is the
   difference between bandwidth-bound and compute-bound for many ops. It is also how you run out of
   memory.
3. **Ops are dispatched from the host.** A model made of many small ops can spend more time being
   dispatched than being computed. Removing host round-trips and capturing the graph are two of the
   biggest wins available on a naive port.
4. **The card is a shared, stateful resource.** One process per device. A crashed run can leave a
   card in a state where the next run hangs. See `09-devices-and-hardware-operations.md`.

Two chip generations you will meet: Wormhole and Blackhole. They differ in core count, clock, L1 per
core and memory bandwidth, so **every performance number is meaningless without naming the chip**.
Multi-chip systems (a desktop box with several cards, or a Galaxy rack) present as multiple devices
or as one mesh device, depending on how you open them.

## The software stack

- **`tt-metal`** is the low-level framework: kernels in C++, explicit data movement, explicit core
  grids. You go here to write a fused kernel and nowhere else.
- **`ttnn`** is the tensor library on top of it. It looks like a NumPy/torch-shaped API
  (`ttnn.matmul`, `ttnn.softmax`, `ttnn.layer_norm`) but every op takes a memory config and often a
  program config, and those arguments are where the performance is.
- **`tt-smi`** is the device tool: telemetry, health, reset.
- Your model code is ordinary Python that builds ttnn tensors, calls ttnn ops, and moves results
  back to host.

The porting unit is therefore: take a `torch.nn.Module`, write a function that takes ttnn tensors and
device-resident weights and returns ttnn tensors, and prove it matches the module.

## What tt-bio is, and why you are forking it

`tt-bio` is a Python package holding production ports of biomolecular models to Tenstorrent, with a
shared CLI, a shared test suite, shared device and tensor utilities, and a release gate. Every model
in it follows the same shape, and that sameness is the point: one implementation of tiling helpers,
one of device management, one of caching, one of chunking, one of output writing.

You are forking it because your model is proprietary and cannot go upstream. Fork discipline
matters more than usual as a result:

- Add your model the way the existing ones are added. Do not build a parallel private stack beside
  the shared one. When your fork diverges structurally, every upstream fix becomes a manual merge.
- Rebase on upstream in small, frequent steps. A six-month-old fork rebased in one go is a project.
- Keep your model's proprietary parts in clearly separated modules, so the boundary between "our
  model" and "the shared machinery" stays obvious to everyone including your future self.

See `11-tt-bio-integration.md` for the file-by-file conventions.

## Day zero: getting to the starting line

Nothing below is Tenstorrent-specific expertise. It is the install, and it has to work before any of
the rest of this skill means anything.

**1. Fork and clone tt-bio.** Fork `https://github.com/moritztng/tt-bio` on GitHub, then:

```bash
git clone https://github.com/<you>/tt-bio.git
cd tt-bio
git remote add upstream https://github.com/moritztng/tt-bio.git
```

The `upstream` remote is how you rebase later, and adding it now costs nothing.

**2. Build the environment.** Python 3.10 or 3.12; 3.11 is not supported:

```bash
python3.10 -m venv env
source env/bin/activate
pip install -e '.[tenstorrent]'
tt-bio install-deps        # Tenstorrent system dependencies for this release; may ask for sudo
tt-bio --help              # if this prints the CLI, the install took
```

Every command in this skill assumes that `env` is active. Activate it in every new shell, and when
you write a script or a gate that runs as a subprocess, give it the interpreter explicitly
(`./env/bin/python3`) rather than relying on the ambient `python3`. A gate that silently ran under the
system interpreter is a real and recurring way to spend a day.

**3. Check the cards answer.**

```bash
tt-smi -ls
```

It lists every board on the host with its chip generation and its `/dev/tenstorrent/<n>` number. Write
the chip generation and the count into `notes/PORT_STATE.md`. Every performance claim you make from
here on has to name that hardware, because Wormhole and Blackhole numbers are not comparable.

The logical IDs `tt-smi` shows and the `/dev/tenstorrent/<n>` node numbers are not always the same
number, and `TT_VISIBLE_DEVICES` takes the node number. Read both columns before pinning a card.

**4. Run the existing suite, before you change anything.**

```bash
TT_VISIBLE_DEVICES=0 python3 -m pytest tests -q          # pick a card that -ls listed
```

`TT_VISIBLE_DEVICES` is mandatory on a host with cards. Unpinned, pytest refuses the session on
purpose: ttnn brings up every chip it can see, not just the one it computes on, so an unpinned run
takes every card on the box away from whoever else is using it. `TT_VISIBLE_DEVICES=` (set, empty) is
the other legal answer and means "skip the device tests, run everything else".

Record the pass, fail and skip counts. That is your baseline. Without it you will spend a day
debugging a failure that was already there before you arrived.

**5. Fold one existing model, end to end.**

```bash
printf '>t\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\n' > /tmp/t.fasta
TT_VISIBLE_DEVICES=0 tt-bio predict /tmp/t.fasta --model esmfold2-fast
```

This proves the install, the driver, the card and the weights download, all at once, before your own
code can be blamed for any of it. `esmfold2-fast` on purpose: it needs no MSA, so nothing here depends
on reaching an MSA server. The first run downloads a checkpoint, so it is slow once and fast after. If
this fails, stop here: nothing downstream is interpretable.

**6. Pin your own reference.** Clone your PyTorch reference implementation, pin the commit, build it
its own separate virtualenv, and run it once on CPU. Record the exact command and the wall-clock
runtime. Two runs must produce identical output; if they do not, fix that before Phase 1, because a
reference you cannot reproduce cannot be a golden.

If steps 3 to 5 all pass, you are at the starting line.

## Your first hour

Phase 0 of `SKILL.md`. Copy the templates in, write the plan, and run the gate on it:

```bash
SKILL=$(find ~/.claude/skills ~/.claude/plugins/cache .claude/skills -type d \
        -name 'tt-bio-bringup' -path '*skills*' 2>/dev/null | head -1)
mkdir -p notes scripts
cp "$SKILL/templates/PORT_PLAN.md" "$SKILL/templates/PORT_STATE.md" notes/
cp "$SKILL/gates/port_gate.py" scripts/
sed -i '/^\/notes\/$/d' .gitignore        # your fork keeps its planning; upstream's does not

python3 scripts/port_gate.py plan notes/PORT_PLAN.md      # red now, and it says why
```

Then fill `notes/PORT_PLAN.md` in until that command exits 0. It will not exit 0 while any module
lacks a named golden, which is the one thing Phase 0 exists to force.

## The vocabulary you will see in the reference docs

| Term | Meaning |
|---|---|
| Tile | The 32x32 element unit of storage and compute |
| Tilize / untilize | Convert between row-major and tile layout |
| Memory config | Where a tensor lives (DRAM or L1) and how it is laid out (interleaved or sharded) |
| Shard spec | How a tensor is split across cores, and over which core grid |
| Program config | Per-op blocking and parallelization parameters |
| Math fidelity | How many passes the matmul engine makes, trading accuracy for speed. Separate from dtype |
| Op census | A table of device time by op type, the starting point of all optimization |
| Roofline | The compute and bandwidth ceilings of the hardware, measured, with your op placed against them |
| Golden | The CPU PyTorch reference output that defines correctness |
| Parity | Agreement between the device implementation and the golden, at a stated threshold |
| PCC | Pearson correlation between two flattened tensors. A summary metric, and a weak one alone |
| Trace capture | Recording a device program once so replays skip host dispatch |
| Residency | Keeping tensors on the device across calls or loop iterations instead of round-tripping |
| Bucketing | Padding a variable length up to a small set of fixed sizes to avoid recompilation |

## What this skill assumes you do not have

No GPUs. No cloud budget for a reference cluster. No existing Tenstorrent expertise. The whole
method is built around a CPU golden and a card on your desk, because that is the setup that always
exists. If you do have more hardware, it buys you parallelism and a second card to cross-check
surprising results, which is genuinely useful. It does not change the method.
