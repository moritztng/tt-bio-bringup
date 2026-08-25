#!/usr/bin/env python3
"""Phase 1 golden capture, complete and runnable, for the worked example.

This is the capture protocol from `references/02-parity-and-correctness.md` §1.2 applied to a
toy reference model, so you can watch the Phase 1 gate go green and red before you have any
device code. It needs torch and nothing else. No Tenstorrent card, no ttnn.

`$REF_PY` is whatever interpreter can import your reference; here that is tt-bio's env, because
this toy needs only torch. `$SKILL` must be exported, not just assigned: port_gate runs `--run`
through `bash -c`, which inherits exported variables only.

    export SKILL=... REF_PY=./env/bin/python3

    "$REF_PY" "$SKILL/examples/minifold_capture.py" --len 117 --out /tmp/artifacts

    "$REF_PY" "$SKILL/gates/port_gate.py" determinism \
        --run '"$REF_PY" "$SKILL/examples/minifold_capture.py" --len 117 --out /tmp/artifacts' \
        --artifact /tmp/artifacts/minifold_117.pt

Then break it, and watch the gate catch it: pass --unpinned to skip the seeding, and the two
runs stop matching. That is the whole point of the phase.

The `.meta.json` beside the fixture is deliberately NOT in that artifact list. It records
`runtime_s`, which is a wall-clock measurement and differs between runs of a real reference,
so hashing it would fail the gate for a reason that is not a defect. Byte-identity is for the
golden; the metadata is checked for required keys instead.

MiniFold below stands in for your reference implementation. It is deliberately the easiest
possible shape (no recycling, no diffusion, no MSA) so nothing distracts from the protocol.
One thing about it is not a simplification: `blocks[i]` is called with `mask=` as a keyword,
which is the dominant convention in real bio models and the reason this capture walks
`**kwargs` as well as `*args`.

Verified against torch 2.8.0 on CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn


# --------------------------------------------------------- the toy reference model


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        # Dropout is here so the --unpinned negative control has something real to trip on:
        # a capture taken without ref.eval() is stochastic, which is a bug people actually ship.
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(4 * d, d))

    def forward(self, x, *, mask=None):
        y = self.norm1(x)
        attended, _ = self.attn(y, y, y, key_padding_mask=mask)
        x = x + attended
        return x + self.ffn(self.norm2(x))


class PairHead(nn.Module):
    """Projects to a pair representation by an outer sum, then to distance bins.

    This is the only part of MiniFold that scales as L squared, and it is here because that
    is the shape of the interesting problem in a real bio model: one allocation that is fine
    at the length you develop at and the thing that OOMs at the length you promised.
    """

    def __init__(self, d: int, bins: int):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.out = nn.Linear(d, bins)

    def forward(self, x):
        h = self.proj(x)                                  # [B, L, d]
        pair = h.unsqueeze(2) + h.unsqueeze(1)            # [B, L, L, d]  <- the L-squared one
        return self.out(pair)                             # [B, L, L, bins]


class MiniFold(nn.Module):
    """Embedding, four blocks, a pair head. Outputs a dict, like most real ones."""

    def __init__(self, d: int = 128, layers: int = 4, heads: int = 4, vocab: int = 22, bins: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.head = PairHead(d, bins)

    def forward(self, tokens, mask=None):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, mask=mask)              # keyword, on purpose: see the module docstring
        return {"logits": self.head(x), "single": x}


# ------------------------------------------------------------- the capture protocol


def to_cpu(x):
    """Recurse into every container the reference might return. Leaves stay leaves."""
    if torch.is_tensor(x):
        x = x.detach().cpu()
        # fp32 for the float leaves, so a bf16 reference does not bake its own rounding into the
        # golden. Integer and bool leaves keep their dtype: token ids and masks are indices, and
        # a float index cannot be replayed through the module it came from.
        return x.to(torch.float32) if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*(to_cpu(v) for v in x))   # a namedtuple takes positional args, not an iterable
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: to_cpu(getattr(x, f.name)) for f in dataclasses.fields(x)}
    return x                                     # int, float, str, None, a config object


def capture(ref: nn.Module, seed: int, *, pin: bool = True, **real_input) -> dict:
    """Per-module inputs and outputs from one forward pass, first call of each module only.

    `pin=False` skips the two lines a rushed capture forgets, the seeding and `eval()`, so you
    can watch the Phase 1 determinism gate catch it. Do not copy that path into a real port.
    """
    if pin:
        torch.manual_seed(seed)
        random.seed(seed)
        torch.use_deterministic_algorithms(True)
        ref.eval()                    # without this, dropout is live and the golden is a sample
    else:
        torch.seed()                  # what "no seeding anywhere in the pipeline" actually means
    cap: dict = {}

    def hook(name: str, mod: nn.Module) -> None:
        forward = mod.forward

        def wrapped(*a, **k):
            out = forward(*a, **k)
            if name + "/out" not in cap:                    # FIRST call only
                cap[name + "/args"] = to_cpu(a)
                cap[name + "/kwargs"] = to_cpu(k)           # keyword inputs too
                cap[name + "/out"] = to_cpu(out)
            return out

        mod.forward = wrapped

    for name, mod in ref.named_modules():
        hook(name or "<root>", mod)
    with torch.no_grad():
        ref(**real_input)
    return cap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--len", type=int, default=117,
                    help="sequence length. 117 is not a multiple of 32, on purpose")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="parity_artifacts")
    ap.add_argument("--unpinned", action="store_true",
                    help="skip the seeding and eval(), so you can watch the gate go red")
    args = ap.parse_args()

    torch.manual_seed(1234)                      # stands in for loading a fixed checkpoint
    ref = MiniFold()

    gen = torch.Generator().manual_seed(7)       # stands in for one real target's features
    real_input = {"tokens": torch.randint(0, 22, (1, args.len), generator=gen),
                  "mask": torch.zeros(1, args.len, dtype=torch.bool)}

    t0 = time.perf_counter()
    cap = capture(ref, args.seed, pin=not args.unpinned, **real_input)
    runtime_s = time.perf_counter() - t0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(cap, out / f"minifold_{args.len}.pt")

    # Provenance travels with the fixture or the fixture is worthless. reference_impl must name
    # the REFERENCE, never your own package: a meta.json pointing at your own repo means a
    # device run was saved as the golden and every later parity check is a tautology.
    (out / f"minifold_{args.len}.meta.json").write_text(json.dumps({
        "model": "minifold",
        "reference_impl": "minifold (this file's MiniFold, standing in for yours)",
        "reference_version": "0.1.0",
        "reference_commit": "0000000",
        "settings": {"length": args.len, "dtype": "fp32", "device": "cpu"},
        "seeds": [args.seed],
        "command": f"python3 minifold_capture.py --len {args.len} --seed {args.seed}",
        "provenance": "CPU torch reference, fp32, deterministic algorithms on",
        "runtime_s": round(runtime_s, 3),
        "invalidation_rule": "regenerate if reference_commit, length or settings change",
    }, indent=1, sort_keys=True) + "\n")

    modules = len({k.split("/")[0] for k in cap})
    print(f"captured {modules} modules, {len(cap)} entries, {runtime_s:.3f}s -> {out}")
    print(f"blocks.0 kwargs captured: {sorted(cap['blocks.0/kwargs'])}")
    print(f"blocks.0 mask is a tensor: {torch.is_tensor(cap['blocks.0/kwargs'].get('mask'))}")


if __name__ == "__main__":
    main()
