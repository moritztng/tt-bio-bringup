# Landing your model inside a tt-bio fork

This document decides the shape of your integration for you: your model becomes one more `--model`
choice on the existing CLI, loaded once per worker and dispatched from the same scheduler, writing
the same output files, reusing every shared helper instead of copying it, with its checkpoint in the
weights registry and out of git, its tests discovered by the existing suite, its parity and perf
numbers written down, and its non-Python assets covered by a recursive packaging glob. A file that
only your model uses is fine. A *mechanism* that only your model uses is a defect.

Read this when the port is numerically working in a scratch script and you now need it to look and
behave like the models already in the tree, or when you are about to add a helper, a flag, an env
variable, or a second code path.

---

## 1. Where the files go

Two layouts are in use, both correct. Pick one; do not mix them for a single model.
**Flat prefix modules** for a model assembled from the shared primitives:

| Stage | File |
|---|---|
| host featurization, input parsing | `tt_bio/yourmodel_data.py` |
| heavy host prep done once per target | `tt_bio/yourmodel_host_prep.py` |
| checkpoint key remap (pure torch, no ttnn) | `tt_bio/yourmodel_weights.py` |
| trunk | `tt_bio/yourmodel_trunk.py` |
| per-head modules | `tt_bio/yourmodel_confidence.py`, `tt_bio/yourmodel_template.py` |
| diffusion / structure module | `tt_bio/yourmodel_diffusion_module.py` |
| the sampling loop over that module | `tt_bio/yourmodel_sample_diffusion.py` |
| top-level assembly, the class the worker instantiates | `tt_bio/yourmodel_fold.py` |

**A package** when the port exceeds roughly ten modules or ships its own featurizer tree:
`tt_bio/yourmodel/{featurize,weights,remap,model,sampler,confidence,host}.py` with a docstring-only
`__init__.py` saying what each module is for. OpenFold3 in the tree uses the flat form
(`tt_bio/openfold3_*.py`, 15 files); a later port uses the package form.

Split on **pipeline stage**, never on "utils". A file named `yourmodel_utils.py` is where the
unification rule goes to die: it is the natural home for a private copy of something shared.

To reuse the reference forward rather than reimplement it, add a **runtime wrapper**
(`tt_bio/yourmodel_runtime.py`) reassigning reference submodules to ttnn wrappers through one
declarative spec table plus one generic adapter. The alternative, building ttnn modules at init from
the checkpoint, is the vendored-model style. Do not build both.

Everything that is not the port itself lives elsewhere: parity and bisect scripts under
`scripts/yourmodel_port/`, benchmark drivers and their output under `perf/<task-slug>/`. The repo root
is an allowlist enforced by `tests/test_repo_root_clean.py`.

## 2. The public entry point and CLI registration

**What the rest of the system expects.** Two callables, nothing else:

```python
# tt_bio/yourmodel_fold.py
class YourModel(Module):                       # tt_bio.tenstorrent.Module
    def __init__(self, state_dict, compute_kernel_config, *, num_cycles=4): ...
    def __call__(self, feats, *, n_sample=1, seed=0): ...
    @classmethod
    def load_from_checkpoint(cls, path, **opts): ...   # or a module-level load()
```

`Module.__init__` opens the device via `get_device()`, wraps the state dict in a `WeightScope`, and
gives you `torch_to_tt`, `_lin`, `_split_heads`, `_merge_heads`. Subclass it for every device module;
do not open a device yourself.

**Registration, in order:**

1. Add a row to `_MODEL_RESULTS_PREFIX` in `tt_bio/main.py`. `PREDICT_MODELS` derives from its keys,
   the `--model` `click.Choice` derives from `PREDICT_MODELS`, and the release and perf gates import
   `PREDICT_MODELS`, so one edit registers the model everywhere. Design models go in `DESIGN_MODELS`,
   embedders in `EMBED_MODELS`, scalar predictors in `AFFINITY_MODELS`.
2. Add a branch to `_WorkerState.load_model(cfg)` in `tt_bio/worker.py`, keyed on `cfg["model"]`.
   Loading is keyed on `_hash_run_config(cfg)`, so weights stay resident across runs and users of the
   same model. Build the compute kernel config there (`HiFi4`, `fp32_dest_acc_en=True`,
   `packer_l1_acc=True` is the shipped default; see `03-precision-and-numerics.md`).
3. Add a `_predict_yourmodel_one(path, cfg)` branch to `_WorkerState.predict_one`. It returns a
   result object the shared writer turns into files. It must not write files itself.
4. If the port is ttnn-only, say nothing: `load_model` already refuses a non-Tenstorrent accelerator
   for every model except the one with a torch CPU path.

**The CLI contract you inherit and must not re-invent:**

- Input: `DATA` is a YAML or FASTA file, or a directory of them (`INPUT_SUFFIXES` in
  `tt_bio/runtime.py`). Parse it with the existing chain reader, not a new one.
- Output directory: `predict_results_dir_name(model, stem)` gives `<prefix>_results_<stem>`. Never
  hardcode an output path.
- Output files: structure in `--output_format {pdb,cif}` (default `cif`) plus the per-target
  confidence summary JSON; `--write_pae`, `--write_pde`, `--write_embeddings` are opt-in extras.
- Free flags you inherit: `--accelerator`, `--cache`, `--seed`, `--recycling_steps`,
  `--sampling_steps`, `--diffusion_samples`, `--max_parallel_samples`, `--fast`, `--debug`, `--log`,
  `--devices/--device_ids`, `--num_devices`, `--host_threads`, `--override`, the whole MSA block.
- Nonzero exit when a job fails, and per-stage progress events through the shared progress helper so
  the Rich display and `--debug --log` look identical to every other model.

**A per-model default is a table row, not a branch.** If your inference default differs (sampling-step
count, MSA on or off), put it in the existing per-model resolution function or membership tuple with
a one-sentence reason, beside the other models' values. `if model == "yourmodel"` scattered through
`main.py` is how defaults drift apart.

**Then grep for lists you have not joined yet.** Hardcoded model-name lists fail silently: the model
is absent from the output, never an error. `grep -rn "yourmodel" scripts/ tests/` should hit the
release gate's model dict, the size-ladder list, the perf-gate coverage list and the README `--model`
table. Treat this grep as part of "port done".

## 3. Weights

**Register the checkpoint, do not hardcode a path.** Add an `Artifact` row to `_ROWS` in
`tt_bio/weights.py`: key, the CLI model names that load it, source (`hf-file`, `hf-repo`, `url`,
`manual`), licence, repo/filename or URL, measured `approx_bytes`. The override env var is *derived*
from the key (`TT_BIO_<KEY>`) so it cannot drift. Cache root resolves `--cache`, `$TT_BIO_CACHE`,
`$BOLTZ_CACHE`, `~/.boltz`, in that order.

**Never gate on `path.exists()`.** A download killed mid-flight leaves a truncated multi-GB file that
is then reused forever and surfaces much later as a corrupt-archive error. Use the registry's
intactness check, which compares against the source's recorded size.

**Do the remap once, on host, in a pure-torch module.** `tt_bio/yourmodel_weights.py` renames the
reference's parameter names onto the key names the shared primitives already expect and returns plain
torch tensors; no ttnn import belongs in that file. If your architecture is a family member of one
already ported, rename onto *its* key names and delegate to its remap rather than writing a second
one: the math is identical, only the strings differ.

Two checkpoint traps to check before concluding the port is wrong:

- **EMA weights.** Many checkpoints carry both a live and a shadow/EMA copy, and inference uses the
  EMA. Loading the live copy gives a quietly worse network that reads exactly like a port bug.
- **Config keys the constructor rejects.** Training configs baked into the checkpoint carry
  framework keys (`_target_`, Lightning hparams). Filter to the inference signature.

**Cache the device-ready form.** `Module.torch_to_tt` tiles a host tensor and DMAs it once. For
multi-card fanout, wrap construction in `tt_bio.tenstorrent.weight_cache(cache_dir, "dump"|"load")`:
the first process tiles fp32 to device dtype on host and dumps the tile, peers `load_tensor` it and
pay only the per-card DMA. Bit-exact, because it is the tile `from_torch` would have produced. A
load-time optimization, not a correctness mechanism; measure it before claiming it.

**Checkpoints never enter git.** They live in the cache root. The repo carries a size-guard test; a
committed multi-GB weight file or a directory of regenerable arrays trips it, long after the commit.

## 4. Inference-only discipline

Delete, do not comment out: losses, optimizers, LR schedulers, training loops, Lightning modules,
dataset/dataloader code, augmentation, EMA update logic (keep the EMA *weights*, drop the updater).

The reference still has a job: it is the golden. It lives under `scripts/yourmodel_port/` (harness,
capture, bisect), or vendored under `tt_bio/_vendor/yourmodel/` if production imports part of it. Not
beside your device modules in `tt_bio/`, which is where someone later mistakes it for a supported
second path. When a device module works end to end, delete the scaffolding: the half-ported variant,
the host fallback you no longer take, the flag that selected between them. A second path nothing
exercises is a path nothing tests.

## 5. The unification rule

**One mechanism, one place, shared by every model. A per-model copy of shared logic is a defect, not
a shortcut.**

Stated as failure: a shared engine with N models costs O(N) to apply a fix per-model, O(N) to keep
the copies consistent, and silently reaches N different answers. Worse, the models that never got the
fix are *invisible*, because nothing tells you a model is missing something that exists elsewhere.
Not hypothetical: a token-axis padding convention every older port had adopted by side effect was not
inherited by a later one, that port ran ragged lengths into a reduction over physically padded tiles,
and it produced 72x the reference error at every non-multiple-of-32 length, with no error and no log
line.

- **One shared helper, every model calls it.** Not five near-copies with different variable names.
- **Per-model facts are data.** `tt_bio/token_axis.py` is the model of this: one `TOKEN_AXIS` table
  with status, multiple, site and reason per model, sitting next to the single implementation. An
  exception is a table row with a written reason, not a second code path.
- **Need a variant? Extend the shared helper with a default-safe parameter.** Add the keyword,
  default it to today's behaviour, pass your value from your call site. Forking is never cheaper once
  you count the second bug fixed in only one of the copies.
- **A constant that splits on a model's vintage is a fork.** It encodes the repo's history rather
  than the hardware's, and nobody can later tell whether the old value is right or merely old.
- **A guard must not be opt-in.** `env_flag("TT_BIO_FIX_THE_BUG", False)` protects exactly the people
  who already knew about the bug. Default-safe, or make the primitive refuse the input it would
  mishandle.

**Env flags and shared defaults are the same rule.** A module-level or env-var default read inside a
class that more than one model instantiates is a cross-model landmine: flipping it for model A's
accuracy silently changes model B with zero code touching B. That has cost a >60x slowdown in a
shipped model, found by a human noticing a fold would not finish, not by any gate. Fix pattern: an
explicit constructor keyword that *defaults* to the env var, so existing callers are unchanged and
every other caller can pin its own value.

```python
class Shared:
    def __init__(self, *, diffusion_fp32=None):
        self.diffusion_fp32 = (env_flag("TT_BIO_DIFFUSION_FP32", False)
                               if diffusion_fp32 is None else diffusion_fp32)
```

Before flipping any shared default, `grep -rn "SharedClass(" tt_bio/` and check every caller wants it.
**Elegance here is measurable:** count call sites implementing the mechanism (should trend to one),
distinct compiled shapes before and after, surviving per-model special cases with a written reason
for each.

## 6. Reuse instead of reimplementing

Before you write any helper, grep for it. These already exist:

| You need | Use | Where |
|---|---|---|
| open/close a device, hold a lease | `get_device()`, `cleanup()` | `tt_bio/tenstorrent.py`, `tt_bio/device_lease.py` |
| host weight to device tensor, linear, head split/merge | `Module.torch_to_tt`, `_lin`, `_split_heads`, `_merge_heads` | `tt_bio/tenstorrent.py` |
| navigate a nested checkpoint, standard core grid | `WeightScope`, `CORE_GRID_MAIN` | `tt_bio/tenstorrent.py` |
| tile size, token-axis padding, masking, slicing back | `TILE`, `bucket_multiple`, `pad_amount`, `bucketed_width`, `token_pad_masks_tt`, `TOKEN_AXIS` | `tt_bio/token_axis.py` |
| chunk a big op to an L1 budget | the existing chunk-size resolvers | `tt_bio/tenstorrent.py` |
| fused SDPA, softmax, matmul, reblocking | `sdpa_generic.py`, `softmax_generic.py`, `mm_generic.py`, `reblock_permute.py` | `tt_bio/` |
| rigid alignment, RMSD | `rigid_transform`, `rmsd` | `tt_bio/align.py` |
| content-addressed file cache, atomic publish | `seq_hash`, `cached`, `staged`, `publish_text`, `publish_file` | `tt_bio/cache.py` |
| read a boolean env var | `env_flag` | `tt_bio/envflags.py` |
| checkpoint download, cache, intactness | `resolve`, `fetch`, `status` | `tt_bio/weights.py` |
| job discovery, host thread caps, device enumeration | `discover_jobs`, `host_thread_cap`, `detect_tenstorrent_devices` | `tt_bio/runtime.py` |
| write a structure and its metrics, report progress | the shared writer and reporter | `tt_bio/data/write.py`, `main.py::write_result`, `tt_bio/progress.py` |

Custom kernel sources go under `tt_bio/kernels/<name>/`, loaded at runtime by file path
(`KERNEL_DIR`, ttnn `KernelDescriptor` `FILE_PATH`), with `compute/` and `dataflow/` subdirectories
when the kernel has both. A new kernel directory is a packaging event: see section 9.

## 7. Tests

Name files `tests/test_yourmodel_<component>.py` and pytest finds them (`testpaths = ["tests"]`).
Import shared test helpers as `from conftest import ...`, and load a `scripts/<port>/*.py` harness
through `tests/_port_module.py::port_module("yourmodel_port", "parity_gate")`, not `sys.path.insert`
plus a bare import: four ports ship a `parity_gate.py` and a bare import resolves by collection order.

**Device marker.** Any test that opens a card gets `@pytest.mark.device`. The conftest resolves four
states: no card node present, skip. `TT_VISIBLE_DEVICES` set but empty, skip (a declared CPU-only
run). Card present and pinned, run. Card present and `TT_VISIBLE_DEVICES` unset, **refuse the whole
session, loudly**, because ttnn brings up every chip it can *see*, not just the one it computes on,
so an unpinned pytest on a multi-card host takes every card away from whoever else is using it. A
backstop hook turns an *unmarked* device test that dies on ttnn's no-chips abort into a skip, but
marking is still your job: the marker makes it skip *before* it takes a lease. An autouse fixture
saves and restores the device-selection env vars; a test that sets `TT_VISIBLE_DEVICES` and pops it
instead of restoring leaves the session unpinned, and the failures land nowhere near the cause.

**What to add, minimum:**

- One card-free unit test per remap function: reference key names in, expected shapes out.
- One `@pytest.mark.device` PCC test per component against a captured golden.
- One end-to-end device test at a real size producing a structure metric.
- **A test that fails on a non-multiple-of-32 token axis.** A documented convention is not a guard.
  Assert over the pad constants, and where cheap add a counter check on a tiny fold proving the fused
  path did not silently decline at the padded length.
- Tests asserting a reference capture's *content*, not just its keys. A capture that silently lost a
  feature still has all its keys and still loads, and the parity gate then agrees with the wrong thing.

**Fixtures.** Small inputs in `tests/fixtures/`, sized in kilobytes. Large captured goldens under
`scripts/yourmodel_port/parity_artifacts/`, fetched from release assets by a script rather than
committed, with a clean `pytest.skip(f"{ARTIFACT} not present")` so a fresh clone still runs
everything else.

## 8. Docs

Two documents in `docs/`, both required before merge.

**`docs/yourmodel-parity.md`** records how correctness was established, not that it was: the
reference and its exact provenance, per-component PCC with the input each was measured on, the
end-to-end metric against ground truth with target name and length, the reference's own run-to-run
spread (device results are compared against that, not against zero), which variants are *not* gated
and why, and a "Reproduce" section whose commands run today. State a verdict line.

**`docs/yourmodel-perf.md`** records measured numbers with the **exact command** that produced them:
model, input file and its size, recycling and sampling settings, card type, warm or cold, median of
how many repeats. Then add the model to `docs/perf_baselines.json` and `scripts/perf_regression.py`,
which measures warm steady-state throughput on a small fixed input, excludes model load and
first-kernel compile, and fails against a committed per-card-type baseline.
`tests/test_perf_model_coverage.py` fails if a shipped `--model` is neither perf-gated nor explicitly
exempted with a reason, so skipping this is caught, not silent. Also update the README `--model`
table (one row folded in, not a parallel prose section) and the CHANGELOG. See
`05-perf-method-and-roofline.md` for how to produce a number worth writing down.

## 9. Packaging

Anything your model loads **by path** rather than importing must be listed twice. The failure mode is
zero signal until the first eligible call on a clean `pip install`.

```toml
# pyproject.toml
[tool.setuptools.package-data]
"tt_bio" = ["kernels/**/*.cpp", "kernels/**/*.hpp", "_vendor/**/LICENSE"]
"tt_bio.data" = ["*.json"]
"tt_bio.yourmodel.resources" = ["*.yaml"]
```

```text
# MANIFEST.in, for the sdist. Belt and braces, so both artifacts are independent of
# whatever setuptools decides to include on its own.
recursive-include tt_bio/kernels *.cpp *.hpp
recursive-include tt_bio/_vendor LICENSE
include tt_bio/yourmodel/resources/*.yaml
```

Use **recursive globs**. The enumerated form has dropped kernel sources from the published wheel
three separate times, once losing 22 files across a set of new kernel directories. If a new kernel
source extension appears, extend the extension list; do not re-add a per-directory path.

`scripts/packaging_smoke.py` is the only guard that catches this class: it builds the real wheel and
sdist, and asserts every non-`.py` file tracked under `tt_bio/` ships in both and survives a clean
`pip install --no-deps --target`. A pytest run against the source tree never catches it, the files
are still on disk in editable mode. Card-free and fast, so run it before every tag.

## 10. Dependencies

**Vendor the small host-side reference** into `tt_bio/_vendor/yourmodel/`: featurization, structure
assembly, reference model files. Rewrite absolute imports to the vendored namespace, keep per-file
provenance headers, ship the upstream LICENSE inside the vendored directory, add a NOTICE entry. No
runtime `git clone`, no sibling checkout, no `sys.path` shim, no `MODEL_SRC` env var.

**Declare every import as a real dependency**, including ones that feel transitive. The failure is
not an import error on your module: your module imports fine and the fold dies deep in featurization
instead. Build a venv from `pyproject.toml` alone and fold one target; it surfaces them one package
at a time. Every upper bound carries a comment naming the API that broke and the versions verified
clean, because a bound with no reason gets lifted by the next person and the failure returns.

**A dependency major bump is release-gated, because it can move accuracy.** Numerical libraries,
chemistry toolkits and model frameworks change default algorithms across majors. Before bumping, run
the CPU reference on the same weights and seed under both versions and show the outputs are
bit-identical; if they are not, re-run the parity gate and record the new numbers. "Tests passed" is
not evidence, the tests may not be sensitive to the thing that moved.

## 11. Git hygiene for a fork

- **One branch per workstream**, named for the work. Land a coherent unit, not a week of everything.
- **Never `git add -A`.** Stage named paths. `git add -A` is how build artifacts, scratch scripts, a
  1 GB capture directory and a stray editor file enter the history in one commit.
- **Before proposing a merge, look at *where* the files landed**, not just whether tests pass:
  `git status` and `git diff --stat <upstream>/main`. A port's diff should touch `tt_bio/yourmodel*`,
  the shared modules it genuinely changed, `tests/test_yourmodel_*` and `docs/`, nothing else.
- **A per-directory `.gitignore` is not inherited by siblings.** Three sibling campaign directories
  each carried their own ignore rule; the fourth did not, and a normal commit landed 783 MB of `.npy`
  undetected until an unrelated run tripped the repo-size guard. Write the rule once at the family
  level (`perf/family/**/*.npy`), not once per leaf.
- **Watch for ignore rules that swallow real code.** A generic output-cache rule such as `msa/` hides
  a vendored package directory of the same name, and the symptom is a `ModuleNotFoundError` on a
  clean checkout long after the commit. Re-include explicitly (`!tt_bio/_vendor/**/msa/**`) and check
  `git status --ignored` after vendoring.
- **Planning documents, status updates and task notes stay out of the code repo.** The repo is the
  product; it is what someone clones.
- **Rebase on upstream in small steps.** A fork that skips six upstream releases and then merges once
  produces a conflict set nobody can review, and the resolution silently reverts fixes. Rebase per
  upstream release, run the suite, keep going.
