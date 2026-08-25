# Devices and hardware operations

This document decides how you address, pin, share, diagnose and recover Tenstorrent cards. Four standing
rules: pin every device-touching process to an explicit card before `import ttnn`; verify every per-card
claim against the kernel node the process actually opened; kill device-holding processes SIGINT-first; and
trust no card after a crash until a fast canary workload has opened it and produced a correct result.

Read this when a run hangs with no output, when a result differs between two cards, when
`ttnn.open_device` fails or blocks, when `/dev/tenstorrent` disappears, or before you schedule anything
whose verdict is bit-exact. Measurement discipline lives in `05-perf-method-and-roofline.md`; what counts as
a passing numerical comparison is in `02-parity-and-correctness.md`.

## 1. Device identity: two numberings that are not the same

| Name | Source | Used by |
|---|---|---|
| UMD / logical chip ID | PCI BDF ascending order, as listed by `tt-smi -ls` | `TT_VISIBLE_DEVICES`, `ttnn` device ids, `tt-smi -r`, `--device_ids` |
| Kernel node number | `/dev/tenstorrent/N` | `lsof`/`fuser`, `/proc/<pid>/fd`, dmesg (`tenstorrent!N`) |

They are not an identity mapping on multi-card hosts. Observed on a 4-card box: logical 0 opened
`/dev/tenstorrent/1`, logical 3 opened `/dev/tenstorrent/0`. On a 32-chip Galaxy the permutation was a
block rotation (logical L to physical L+16 for L in 0..15, L-8 for 16..23, L-24 for 24..31). Extract the
live mapping per host; there is no formula worth memorising.

```python
# ttmap.py: which kernel node does this logical id actually open? Run once per candidate id:
#   for v in 0 1 2 3; do TT_VISIBLE_DEVICES=$v python3 ttmap.py; done
import os, ttnn
ttnn.open_device(device_id=0)   # logical 0 *within* the pinned set
fds = (os.readlink(f"/proc/self/fd/{f}") for f in os.listdir("/proc/self/fd"))
print("TTVD", os.environ.get("TT_VISIBLE_DEVICES"),
      sorted({t for t in fds if t.startswith("/dev/tenstorrent/")}))
```

Without opening anything: `grep PCI_SLOT_NAME /sys/class/tenstorrent/tenstorrent\!0/device/uevent` maps
node 0 to a PCI BDF, and `tt-smi -ls` maps that BDF back to a logical id.

A card quarantined as bad and a card blessed as good differ by one integer in the wrong numbering. Before
recording "card 2 reproduces it, card 1 does not", read `/proc/<pid>/environ` for the compute process's own
`TT_VISIBLE_DEVICES` and its device-node fds. Trust the process's environment, not the launcher's intent.

## 2. One device context per process, and pinning is not optional

`ttnn` reads `TT_VISIBLE_DEVICES` **at import time**. Set it before any module that imports `ttnn` loads,
not in `main()` and not from argparse. If you need a CLI flag, pre-scan `sys.argv` at the top of the entry
module and `os.environ.setdefault("TT_VISIBLE_DEVICES", ...)` there, so an explicit env still wins.

- Defer `import ttnn` into functions. If the parent imports `ttnn` at module scope, a spawn-based worker
  can no longer pin itself. Use `multiprocessing.get_context("spawn")`, never fork: spawned children
  re-execute module top and pick up the per-worker env.
- A pinned process sees exactly one device, as logical id 0:
  `TT_VISIBLE_DEVICES=N python3 -c "import ttnn; print(ttnn.GetNumAvailableDevices())"` must print `1`.
- **Unset means "the whole box".** The UMD brings up every *visible* chip, not only the one you open:
  `TT_VISIBLE_DEVICES=0,1` reports a 1x1 mesh while the process holds fds on both `/dev/tenstorrent/0` and
  `/1` for its entire lifetime.
- **CPU-only tests must be pinned too.** A repo-wide `pytest -q` with no pin enumerates and opens every
  card as an import side effect, colliding with whatever real measurement is running. "This task doesn't
  use hardware" is a claim about intent, not about fds.
- A parent that has already opened device N can never fan a shard onto device N: the per-chip UMD mutex
  `CHIP_IN_USE_<n>_PCIe` is held for the whole process lifetime and `close_device()` does not release it.
  Pattern: the parent stays device-free, only spawned per-card children call `get_device()`.

False positive to know: `fuser /dev/tenstorrent/*` showing one process mmapping *all* nodes with flag `m`
is usually the one-time cluster-descriptor topology scan at library init, not a held compute lock. Check
the compute child's env before calling it a collision.

## 3. Device open is serialized host-wide, with no timeout

Device bring-up and teardown run through a UMD cross-process init path that is not concurrency safe:
concurrent opens deadlock in `acquire_mutex`, and a raced open can bring a chip up "remote-only" (no local
dispatch core) which then throws on the first program dispatch. The fix is a single host-wide advisory lock
held across the whole open and close, here `/tmp/tt-bio-device-open.lock` (`_device_init_lock()` in
`tt_bio/tenstorrent.py`). Consequence: **one stalled opener blocks every device open and close on the box,
on every card.** The presentation is a whole-host wedge with idle cards, near-zero load average and
`tt-smi` exiting 0. It looks exactly like hardware failure and is not.

First check: `stat -c %i /tmp/tt-bio-device-open.lock`, then `grep <inode> /proc/locks` (holder is the top
entry, the rest is a FIFO wait queue), then `ps -o pid,ppid,stat,wchan:24,etimes,time -p <holder_pid>`.
`locks_lock_inode_wait` in `wchan` on the waiters plus a holder with PPID 1 and near-zero CPU over a long
elapsed time is the signature. Two shapes that break the obvious diagnosis:

- The holder need not be wedged. The lock fd is inherited by `multiprocessing` spawn and resource_tracker
  children, so a healthy run holds it for its whole duration. Killing the worker's process group does not
  free it: multiprocessing children install SIGTERM handlers. Kill the inheriting child by explicit pid.
  Fix in your own code: set close-on-exec on the lock fd, or close it before fork.
- The deadlock can be circular across two locks: A holds the global open lock via a leaked child fd while
  blocked on a per-chip `CHIP_IN_USE_<n>_PCIe` mutex held by B, itself queued on the global lock. Tell-tale
  log line: `UMD | Waiting for lock 'CHIP_IN_USE_<n>_PCIe' which is currently held by...`. Kill either tree.

Recovery here is a lock release, not a reset. Kill *your own* stalled opener by explicit pid. Do not
`tt-smi -r` a card you do not own to clear a lock-wait: the next opener gets clean exclusive access once
the lock drops.

## 4. Telling a wedge from a slow op

A wedged chip: the process sits in `futex_do_wait` at ~0% CPU, the log is frozen right after the banner, the
device is clocked up at 60-68 W but computing nothing, and nothing under the run directory has been written
for minutes.

```bash
ps -o pid,stat,wchan:24,etime -p <pid>              # Ssl + futex_do_wait
top -b -n2 -d 1.5 -p <pid> | tail -3                # 0.0% CPU across 3+ minutes
find <run_dir>/ -newermt "5 minutes ago" -type f    # nothing = no progress
tt-smi -s | grep -iE '"power"|"aiclk"'              # powered but idle
```

Three discriminators that stop the common misdiagnoses:

1. **Measure CPU on the right process.** In a spawn-based fan-out the parent parks in a `select()` while
   the real work happens in a spawned grandchild. Parent CPU freeze proves nothing; the reliable signal is
   grandchild CPU accrual over several minutes (`py-spy dump`, or `/proc/<pid>/stat` deltas).
2. **A zero-byte log is not a wedge.** A cold first run compiles kernels and can sit at zero bytes for many
   minutes, so a watchdog keyed on "no output by N seconds" kills healthy cold runs. Key it on
   elapsed-vs-expected for a *warm* run, and set `PYTHONUNBUFFERED=1` so log growth is a real stall signal.
3. **Compare against a measured expectation.** Keep a known wall-clock for a small input on this hardware;
   elapsed above ~3x that with flat CPU is a wedge.

Two non-wedge causes present identically and must be excluded first: the global open lock of section 3, and
a stale `multiprocessing.spawn` orphan (PPID 1) holding the device from a crashed earlier run. Find the
latter with `for p in $(pgrep -f python); do ls -l /proc/$p/fd 2>/dev/null | grep -q tenstorrent && ps -o
pid,ppid,etime,cmd -p $p; done`. A passing small run does **not** prove a chip is healthy for a larger or
different workload: cheap models run fine on a chip that wedges a diffusion or design job.

## 5. Recovery ladder

Cheapest first. Do not skip to the bottom.

1. **Free the device.** Nothing below works while a process holds it: SIGINT, wait, then explicit-pid
   SIGKILL for stragglers (section 6). Confirm zero holders: no `/dev/tenstorrent/*` in any `/proc/*/fd`,
   and `/proc/locks` clean for the open lock.
2. **Reset the boards** in logical ids: `tt-smi -r 0,1,2,3`, or `tt-smi -r 0` for one. Bare `tt-smi -r`
   resets everything on the box. Confirm the output reaches `Re-initializing boards after reset....`.
   Never reset a card another process holds: it fails, or it produces a second wedge.
3. **Reload the kernel module** if opens still fail *broadly*: with zero open handles (`lsmod` refcount 0),
   `sudo rmmod tenstorrent && sudo modprobe tenstorrent`, then reset again against the freshly loaded
   module. A board reset does not clear driver-internal device-open state, so driver poison survives it and
   looks exactly like a dead board. On a 32-chip host this cured all 32 chips, four of which had already
   been written off as hardware-faulty.
4. **PCI remove and rescan** for one board failing reinit (`Failed to set initial power state: -22`, or
   tt-smi topology discovery crashing): `echo 1 > /sys/bus/pci/devices/<bdf>/remove; echo 1 >
   /sys/bus/pci/rescan`. Function-level PCI reset alone did not cure this; remove plus rescan did.
5. **Host reboot** only after 1-4 are exhausted.

**Zero processes is not proof of a clean card.** After a SIGTERM landed mid-run on four cards, the process
table and the lease files both read clean, and 20 subsequent legs then died with `device open failed /
Timed out while waiting for active ethernet core`. A clean `ps` proves nothing currently *claims* the card,
not that the firmware state left behind is openable.

**Verify with a fast canary, never with the slow thing that hung**: a slow reproducer cannot distinguish
"recovered" from "wedged again" in useful time. Canary =
`TT_VISIBLE_DEVICES=0 python3 -c "import ttnn; d=ttnn.open_device(device_id=0); ttnn.close_device(d)"`,
then a small known-good end-to-end run whose output you compare against a stored golden.

After killing device processes mid-run the card can leak TLB windows: the next open fails with
`tt_tlb_alloc failed ... error code -12` or `Failed to allocate TLB window` (holders listed in
`/proc/driver/tenstorrent/<node>/pids`). Here a reset *is* warranted, and the first run after clearing a
stale holder may fail once with that error and succeed on retry.

If you are measuring a hang *rate*, reset the card between every trial, hang or not. A dirty card from
trial 1 makes trial 2 stall identically: you will report a manufactured rate, and a working fix as broken.

## 6. Kill safety

`ttnn.close_device` is registered via `atexit`. It runs on normal exit and on SIGINT (which raises
`KeyboardInterrupt`). It does **not** run on SIGTERM or SIGKILL. Killing a process that is inside a device
op with SIGTERM or SIGKILL skips the close, leaves the chip dirty, and can crash the kernel module.

- Cancel with **SIGINT**, wait up to 60 s for teardown, escalate only then.
- GNU `timeout` sends SIGTERM by default. For any bounded run of a device-touching process use
  `timeout -s INT <secs> ...`. Plain SIGTERM wedged cards twice this way.
- Never `pkill -9` during trace capture; it corrupts the device. Never send SIGUSR1 to a running pytest to
  get a traceback: without `faulthandler` enabled Python treats it as fatal and kills the process mid-op.
- A host-deadlocked process (low CPU, `Sl`, no log activity, device *idle*) is safe to SIGKILL: children
  first, then the parent.
- **Kill by verified pid, one at a time.** A batched `kill p1 p2 p3 ...` from an earlier `ps` snapshot
  carries no name context and no confirmation, and pids get reused between snapshot and signal. Re-resolve
  each pid against its cmdline immediately before signalling it.
- `pgrep -f <pattern>` self-matches your own monitoring shell. Filter with
  `ps -eo pid,cmd | grep -E '<pattern>' | grep -v grep`.
- Orphaned `multiprocessing.spawn` and `resource_tracker` children survive the parent and ignore SIGTERM.
  Kill them explicitly with `-KILL`.

Order is always: kill the job, verify the card, then rerun. Not: kill and rerun.

## 7. Driver: `/dev/tenstorrent` vanished after a kernel upgrade

Symptom: `lspci -d 1e52:` still lists the cards, but `/dev/tenstorrent` is gone, `lsmod | grep tenstorrent`
is empty, `modinfo tenstorrent` says not found, and `modprobe tenstorrent` reports "Module not found in
directory /lib/modules/<new-kernel>". Cause: the driver is a DKMS module built per kernel, and an upgrade
without DKMS autoinstall leaves no `.ko` for the running one. Same symptom if the module crashed, typically
after something was SIGKILLed mid device-op.

```bash
lspci -d 1e52:; dkms status; uname -r            # hardware present, DKMS state, running kernel
ls /usr/src/tenstorrent-<version>/dkms.conf      # source present
ls -d /usr/src/linux-headers-$(uname -r)         # headers present
sudo dkms install tenstorrent/<version> -k $(uname -r) && sudo modprobe tenstorrent
ls -l /dev/tenstorrent/ && tt-smi -ls
```

`apt list --installed | grep tenstorrent-dkms` gives `<version>`. If `dkms status` itself errors on a stale
entry, `rm -rf /var/lib/dkms/tenstorrent/<broken_version>` rather than `dkms remove`. Under Secure Boot the
module is MOK-signed automatically. Guard: wire DKMS autoinstall through the apt package, or this recurs on
every kernel upgrade.

## 8. A card can be silently wrong

One card in a fleet was found to compute a subset of matmuls incorrectly at a low rate, silently. The
transfer path was clean (0 mismatched elements across 4 x 256 MB upload/readback in fp32 and bf16) and the
fault was op-specific: `concat` and `layer_norm` bit-stable at every size, while one 512->256 matmul
differed on 15/15 repeats at one size in fp32 and 2/31 in bf16 (bf16 lowered the rate 15-50x, did not
remove it). It was **location-keyed, not data-keyed**: a synthetic-weight probe reproduced the same victim
row clusters a real-weight dump had recorded, and a large fraction of cores were never hit. There was **no
size threshold**; the original "only at 512 residues" framing was one-sample-per-size on a probabilistic
fault, since one size was clean over 31 repeats while a smaller one tripped 1 in 7. A matched card on
another host was 15/15 clean on the same commit and config. Rules that follow:

1. Any surprising correctness or performance result is reproduced on a **second physical card** before it
   is believed. A single-card observation of a "size-conditioned bug" is indistinguishable from a
   probabilistic hardware fault, and a clean run on a suspect card proves nothing about the next one.
   Quarantine is per-card and sticky until the card is replaced.
2. **Never schedule a bit-exact check on an unvetted card.** `torch.equal`, sha and hash-equality gates get
   an explicit pin to a card with a clean history, with the reason written next to the pin. "Any free card"
   scheduling knows a card is idle, not that it is quarantined.
3. Model-level nondeterminism exists independently of bad hardware: one model was not reproducible with a
   fixed seed above 256 residues (up to 3.17 A RMSD between two identical runs at 384) while bit-exact at
   128. Before attributing a hash difference to your change, run the **same-size, same-config,
   fresh-process control twice**. If your signal sits inside that floor, you have measured nothing.

**If you only have one card** (see also `02-parity-and-correctness.md`), the guard is two-sided:
(a) repeat-run determinism, N >= 3 identical runs of unchanged code, establishing the noise floor before any
A/B; and (b) agreement with the CPU PyTorch golden within your parity gate's tolerance. Neither alone
suffices: (a) passes on a card that is consistently wrong, (b) passes on a card that is intermittently
wrong. Record the noise floor in the run log so a later reader can tell a regression from the floor.

## 9. Multi-chip

Two distinct uses of N chips. **Replication** is N independent single-chip processes, each pinned to one
card, pulling targets from a shared queue: the default for throughput, what almost every bring-up should
do, output in per-device shard directories, and N=1 is the same code path rather than a special case.
**Sharding one model across chips** (a mesh device) is only for when one instance does not fit or cannot
saturate a chip; it pulls in fabric, ethernet topology and mesh descriptors, so do not reach for it during
bring-up.

- A shard must be pinned with **both** the CLI device selection and `TT_VISIBLE_DEVICES`. Device detection
  that reads `/dev/tenstorrent` directly otherwise enumerates every card and spawns a worker per card,
  wedging cards other jobs own. Here, `detect_tenstorrent_devices()` in `tt_bio/runtime.py` intersects the
  node list with the ambient `TT_VISIBLE_DEVICES`.
- Setting `TT_VISIBLE_DEVICES=` (empty, not unset) makes that intersection empty, so the process exits
  before opening anything. That is the safe way to restart a service without stealing cards from a running
  job.
- Some board families require a fabric mesh-graph descriptor before any open, or `ttnn.open_device` aborts
  with `Custom fabric mesh graph descriptor path must be specified for CUSTOM cluster type`. Set
  `TT_MESH_GRAPH_DESC_PATH` to the shipped `.textproto` under the installed `ttnn` package, injected
  before shards spawn, not inside them. Wormhole additionally needs ethernet dispatch
  (`ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH)` when `ttnn.get_arch_name() == "wormhole_b0"`), and
  compute-grid geometry differs per board, so hard-coded core-grids and chunk thresholds need per-arch
  values.
- **Stale device caches after a close.** A module-scope dict caching device tensors (gather indices, window
  index tables) survives `cleanup()`. After a model switch closes the mesh, the next use of a cached tensor
  touches a dead mesh and throws `SubDeviceManagerTracker is not initialized ... remote-only MeshDevice N`,
  reproducing 100% of the time on the sequence A, B, A. That number is a per-process mesh-creation ordinal,
  **not** a chip or worker id: do not map it to a card. Fix: a monotonic `device_generation()` counter
  bumped in `cleanup()`, with every module-level device-tensor cache keyed on `(generation, ...)` and
  pruned when it sees a stale generation.
- **An ill-timed reset can break an inter-chip link permanently.** On a 32-chip baseboard, a `warm_reset`
  run to clear a stuck ERISC heartbeat took an inter-group ethernet link down: healthy endpoints went 256
  to 250, and a retry moved the fault rather than restoring it. Every chip still enumerates, opens, clocks
  and reports telemetry, but any mesh spanning the two groups throws `IndexError: unordered_map::at` out of
  cluster-descriptor construction, before any device initializes. Intra-group work is unaffected, so the
  box looks fine. Treat link-level reset tools as destructive writes to shared fabric state: record the
  healthy-endpoint count before and after, and re-check it live before assuming full fan-out.

## 10. Daily hygiene

**Before a benchmark** (`05-perf-method-and-roofline.md` covers what to measure): confirm no other process
holds a device fd (`ls -l /proc/*/fd 2>/dev/null | grep -c tenstorrent`); pin `TT_VISIBLE_DEVICES` and
verify the process sees exactly 1 device; run one warm-up pass and never report the cold first run.

**Before a parity or bit-exact run:** pin to a card with a clean history, reason written next to the pin;
run the unchanged-code control at the same size in a fresh process at least twice and record the floor
before comparing anything; confirm no earlier in-process device test in the same pytest session holds the
card (an in-process `get_device()` holds it for the parent's lifetime and blocks every later
subprocess-launched device test in that session, which reads as a foreign collision and is a self-deadlock,
so keep in-process and subprocess device tests in separate pytest invocations).

**After any crash, kill or timeout:** free holders with SIGINT first, verify zero `/dev/tenstorrent` fds,
check `/proc/locks` for the open lock; reset the affected card, because "no processes" does not mean clean;
canary-open it and run a small known-good workload diffed against its stored golden; only then rerun the
real job. If the canary fails twice after a reset, climb the section 5 ladder instead of repeating it.
