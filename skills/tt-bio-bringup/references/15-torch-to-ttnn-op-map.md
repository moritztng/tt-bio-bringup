# The torch to ttnn op map

Phase 0 asks you to name a `ttnn` equivalent for every distinct torch op in your model. This
document is how you answer that column without guessing.

**Read this** while filling the op inventory in `notes/PORT_PLAN.md`, and again in Phase 3 when a
module will not match its golden and you suspect the op you picked is not the op you meant.

## 1. Answer it yourself, in one command

The table below is a starting point and it will age. `ttnn` is the authority, and it answers
directly:

```bash
./env/bin/python3 -c "import ttnn; print(hasattr(ttnn, 'silu'))"
TT_VISIBLE_DEVICES= ./env/bin/python3 -c "import ttnn; print([a for a in dir(ttnn) if 'norm' in a])"
```

Pin `TT_VISIBLE_DEVICES` empty for the second one. Importing `ttnn` unpinned on a host with cards
brings up every card it can see, which takes them from whoever else is using them. Empty means
"deliberately no device", and the import still gives you the full symbol table.

Three namespaces, and it matters which one you find a symbol in:

| Namespace | What lives there | Stability |
| --- | --- | --- |
| `ttnn.*` | the eltwise, reduction, matmul, layout and shape ops | the API you should be writing against |
| `ttnn.transformer.*` | attention: `scaled_dot_product_attention`, head split and concat, `attention_softmax` | stable, but check the mask convention against your reference |
| `ttnn.experimental.*` | fused and hardware-specific kernels, many of them model-specific | may be renamed or removed between wheels. Pin the version and re-check on every bump |

An op that exists is not automatically the op you want. `ttnn.gelu` and `torch.nn.functional.gelu`
disagree on the default approximation, which is the single easiest way to spend a week on a parity
failure that is not a bug. Check the definition, not just the name.

## 2. The map

Every row below was resolved against `ttnn` by attribute lookup, so each named symbol exists. What
the row does **not** promise is that the semantics match your reference: that is what Phase 3's
per-module parity test is for.

**Elementwise and arithmetic**

| torch | ttnn | Note |
| --- | --- | --- |
| `+ - * /` | `ttnn.add` `ttnn.subtract` `ttnn.multiply` `ttnn.divide` | broadcasting rules are not torch's; check the shapes |
| `torch.pow` `sqrt` `rsqrt` `exp` `log` `abs` `neg` `reciprocal` | same names under `ttnn` | |
| `torch.maximum` `minimum` `clamp` | `ttnn.maximum` `ttnn.minimum` `ttnn.clamp` (`ttnn.clip` is the alias) | |
| `torch.erf` `sinh` `cosh` | `ttnn.erf` `ttnn.sinh` `ttnn.cosh` | |

**Activations**

| torch | ttnn | Note |
| --- | --- | --- |
| `F.relu` `F.silu` `F.sigmoid` `torch.tanh` `F.leaky_relu` | `ttnn.relu` `ttnn.silu` `ttnn.sigmoid` `ttnn.tanh` `ttnn.leaky_relu` | |
| `F.gelu` | `ttnn.gelu` | **erf vs tanh approximation. Check both sides.** |
| `F.softmax` | `ttnn.softmax`, or `ttnn.transformer.attention_softmax` when it is an attention mask | |
| `F.log_softmax` | **no direct op.** `ttnn.log(ttnn.softmax(x))`, or keep it on host | the naive composition loses precision where torch's fused version does not |
| `nn.Dropout` | none, and none needed | you are porting inference. `eval()` on the reference, nothing on the device side |

**Normalization**

| torch | ttnn | Note |
| --- | --- | --- |
| `nn.LayerNorm` | `ttnn.layer_norm` | |
| RMSNorm, however your reference spells it | `ttnn.rms_norm` | do not hand-compose `pow`/`mean`/`rsqrt`; the fused op is both faster and better conditioned |
| `nn.GroupNorm` `nn.BatchNorm*` | `ttnn.group_norm` `ttnn.batch_norm` | batch norm in inference is a fixed affine; folding it into the neighbouring linear is usually better |

**Linear algebra**

| torch | ttnn | Note |
| --- | --- | --- |
| `nn.Linear` | `ttnn.linear` | |
| `torch.matmul` `@` `torch.bmm` | `ttnn.matmul` | one op; there is no separate `bmm`. Batched operands are leading dims |
| `torch.outer` | `ttnn.outer` | |
| `nn.Embedding` | `ttnn.embedding` | |
| `F.scaled_dot_product_attention`, `nn.MultiheadAttention` | `ttnn.transformer.scaled_dot_product_attention` plus `split_query_key_value_and_split_heads` and `concatenate_heads` | the mask convention is the thing to check first |

**Shape and layout**

| torch | ttnn | Note |
| --- | --- | --- |
| `transpose` `permute` `reshape` `squeeze` `unsqueeze` | same names under `ttnn` | a permute that moves the channel axis is a full re-tile and is expensive. `04-shapes-tiles-and-bucketing.md` §8 |
| `torch.cat` | `ttnn.concat` | |
| `repeat` `repeat_interleave` | `ttnn.repeat` `ttnn.repeat_interleave` | |
| `x[a:b]` | `ttnn.slice` | **not a view.** It allocates and copies. `08-memory-and-residency.md` |
| `F.pad` | `ttnn.pad` | |
| `torch.split` `chunk` | `ttnn.split` `ttnn.chunk` | |
| `.to(dtype)` | `ttnn.typecast` | |
| no torch equivalent | `ttnn.to_layout` `ttnn.tilize` `ttnn.untilize` | tile vs row-major is a real distinction on this hardware and torch has nothing like it |

**Reductions, comparison, construction**

| torch | ttnn | Note |
| --- | --- | --- |
| `sum` `mean` `max` `min` `prod` `std` `var` `argmax` `cumsum` | same names under `ttnn` | check the `dim`/`keepdim` spelling against the installed signature |
| `sort` `topk` | `ttnn.sort` `ttnn.topk` | |
| `== != > < >= <=` | `ttnn.eq` `ne` `gt` `lt` `ge` `le` | |
| `logical_and/or/not`, `torch.where` | `ttnn.logical_and` `logical_or` `logical_not` `ttnn.where` | |
| `masked_fill` | **no direct op.** `ttnn.where(mask, value, x)`, or an additive mask before the softmax | for an attention mask, additive is the one that stays bit-exact under padding |
| `zeros` `ones` `full` `arange` `zeros_like` `ones_like` | same names under `ttnn` | |
| `tril` `triu` | `ttnn.tril` `ttnn.triu` | |
| `scatter` `gather` | `ttnn.scatter` `ttnn.gather` | per-element cost, not per-tile. A hot gather is a red flag; see `07-optimization-levers.md` |

## 3. When there is no equivalent

Three answers, in order of how much they cost you:

1. **Compose it from ops that do exist.** Most gaps are one or two ops wide.
2. **Keep it on host.** Legitimate for anything that runs once per call and is not on the hot path:
   featurization, a final argmax, a rigid alignment that needs an exact SVD. It costs a device
   round trip, so count it in the census rather than pretending it is free.
3. **Write a kernel.** Last, and only after `10-custom-kernels.md`'s four gates all pass.

Record the answer in the plan's risk register. The op with no clean equivalent is the one that
decides your schedule, so it is worth more thought than the forty that map one to one.

## 4. Counting the torch side

The plan wants a count per op, and torch will tell you rather than making you grep:

```python
import torch
from collections import Counter

class Count(torch.overrides.TorchFunctionMode):
    def __init__(self): self.n = Counter()
    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.n[getattr(func, "__name__", str(func))] += 1
        return func(*args, **(kwargs or {}))

with Count() as c, torch.no_grad():
    model(example_input)
for name, n in c.n.most_common():
    print(f"{n:6d}  {name}")
```

This counts calls in one forward, which is what the plan means. A model with a recycling or
diffusion loop will report the loop body once per trip, so state the trip count beside the number
or the census reads ten times too large. That unit slip has cost real time; see
`05-perf-method-and-roofline.md` §5.
