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

## Your first hour

Do these in order. Every step produces something checked into your fork.

1. **Confirm the hardware answers.** Run the device tool, list the cards, note the chip generation
   and how many you have. Write it in `PORT_STATE.md`. Every performance claim you make later must
   name this hardware.
2. **Run the existing test suite** in your fresh fork, on the machine you will work on. Note how
   many pass, fail and skip before you have changed anything. That is your baseline, and without it
   you will spend a day debugging a failure that was already there.
3. **Run one existing model end to end** on a small input. This proves your install, your driver,
   your card and your environment, all at once, before your own code can be blamed.
4. **Pin your reference.** Clone the PyTorch reference implementation, pin the commit, create a
   virtualenv for it, and run it once on CPU. Record the exact command and the runtime.
5. **Write `PORT_PLAN.md`.** The module tree, the axes, the op inventory, the risk register. Phase 0
   of `SKILL.md` says exactly what goes in it.

If step 2 or 3 fails, stop and fix that. Nothing downstream is interpretable until they pass.

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
