# Running a bring-up as a long agentic campaign

This document decides your operating model: split the port into units of work that each carry a
written brief and one machine-checkable done-condition, prove that done-check goes red before you
trust it green, keep all continuity in the repository because every agent session starts amnesiac,
isolate concurrent agents in their own worktree with an exclusive lease on one card, let agents push
branches while exactly one owner merges, and verify every claim against live state as you make it.

Read this when a port is going to take more than a week, when you are about to run more than one
Claude Code session at a time, or when a session reported something done and it was not.

---

## 1. The unit of work is a written brief

One file per unit of work, written before the session starts, and re-read by every relaunch of that
unit. Six fields, all mandatory:

| Field | Content | Failure if missing |
|---|---|---|
| `GOAL` | one sentence, one outcome | the session invents scope |
| `DONE_CHECK` | one shell command, exits 0 only when the deliverable exists and is correct | the task is declared done while unfinished |
| `COMMANDS` | the exact invocations, with the venv and env vars | each relaunch re-derives them and gets them slightly different |
| `IN SCOPE` | the files it may edit | edits spray across the tree and become unmergeable |
| `OUT OF SCOPE` | what it must not touch, named | "while I was there I also refactored" |
| `STATE_DOC` | the path it appends its findings to | pass N+1 starts from zero |

The brief is agent-facing. Precision beats prose. A unit with no `DONE_CHECK` is not a task, it is a
wish, and it will come back reported complete.

**Never write "merge to main" as an action in a brief**, not even conditionally. Write "report
`VERDICT: <result>`" instead. A task-specific instruction overrides your standing system prompt every
time: three briefs in one campaign said "if X, merge to main", all three agents did, bypassing the
preflight. Two were safe only because nothing raced them.

---

## 2. The done-check is the primitive

A done-check is a command, run by something other than the agent that did the work, exiting 0 only
when the artifact genuinely exists and is correct. Everything else here is scaffolding around it.

Grep for the *shape of a result*, never for a topic keyword:

```bash
# BAD: a session that writes "I did not get to the A/B this pass, here's why" satisfies this
DONE_CHECK: grep -qiE "parity|A/B|PCC" notes/state-<slug>.md

# GOOD: only a table with real numbers in it can produce this line
DONE_CHECK: grep -qE '^VERDICT-P4: (GO|NO-GO) pcc=0\.[0-9]{4}' notes/state-<slug>.md
```

### Seven ways a done-check lies, and the fix for each

| # | Failure | Mechanism | Fix |
|---|---|---|---|
| 1 | Keyword grep on prose | A multi-stage task discusses stage 2 in prose while only running stage 1. Keyword presence cannot distinguish "here is the table" from "here is why I skipped it". | Grep for a labelled number or a table header that only exists once real data lands. |
| 2 | Greps for a name, not a verdict | `grep -q "fused_trimul" report.md` passes when the report says the op was *considered*. | Grep for the decision: `^VERDICT: (GO\|NO-GO)`, plus the measurement that decided it. |
| 3 | Hardcoded path | The check is authored inside the agent's worktree and hardcodes it. Re-run from the merge owner's checkout, the path does not exist, exit 1, reads exactly like a missing deliverable. | Derive the root from the script's own location or one env var. Accept several candidates and pass if any hit. |
| 4 | Reads a mirrored copy | The check reads a state file that a sync process copies between machines. The copy resolves, is non-empty, and is stale, so `test -s` passes and only the verdict grep fails. Silent, indistinguishable from unfinished. | Check the file the writer actually wrote, or make the writer push before it exits. Before treating a red check as real, `diff` the file across every copy. |
| 5 | Prose in a command field | `DONE_CHECK: declared in the state doc by the convention this port already uses` is executed verbatim: `bash: declared: command not found`, exit 127. | Every done-check must be one physical unwrapped line starting with a real command name (`grep`, `test`, `python3`). When copying one forward from a prior pass, copy the literal command and edit only the marker string. |
| 6 | Inverted exit status | `grep -L PAT FILE && echo OK` fires when the pattern **is** present. It can never be satisfied by a correct file, so repeated "fixes" look like a flaky test rather than a backwards one. | For absence, write `! grep -q PAT FILE`. |
| 7 | Depends on reachability | The check runs over ssh to one machine, or curls a service. That machine goes down and the check fails forever, blocking conclusion of finished, verified work. | Make the check local to wherever it runs. If it must be remote, bound it (`timeout 30 ssh -o BatchMode=yes ...`) and distinguish exit 255 (transport) from exit 1 (deliverable missing). |

### The universal remedy: prove it red first

Before you trust a green done-check, break the deliverable and watch it fail.

```bash
CHK='grep -qE "^VERDICT-P4: (GO|NO-GO) pcc=0\.[0-9]{4}" notes/state-x.md'
mv notes/state-x.md /tmp/ && bash -c "$CHK"; echo "broken -> $?"   # MUST be non-zero
mv /tmp/state-x.md notes/  && bash -c "$CHK"; echo "intact -> $?"  # MUST be 0
```

Run it through a fresh `bash -c`, not your interactive prompt: an interactive shell often has `grep`
shadowed by a function routing to a different implementation with different exit semantics, so a
hand-check at the prompt can look right and still be wrong about what the runner does.

Two authoring constraints that prevent whole classes of this:

- **Only require writes to per-agent or append-only files.** Mandating a write to a single-owner file
  with several agents live is a mandated race. Each agent writes its own state doc; shared findings
  go to one append-only file with the agent's slug on each line.
- **Human decisions live in a file nobody rewrites**, or they vanish and get relitigated.

---

## 3. Drive to done

Scope a deliverable as complete or as one-shot, and say which in the brief. If it is complete:

- The unit relaunches across sessions until every part is landed and verified. Number the passes
  (`p1`, `p2`, ...) so the state doc is readable.
- Never conclude while any stated priority is owed, deferred, or pending. A conclusion that says
  "P1 done, P2-P4 owed" makes a human re-queue it, which is the work you were hired to remove.
- Out of context, out of turns, out of time: that is *pause and resume*, not done. Write the state
  doc, name the next action, exit.
- Genuinely blocked on something only a human can decide: escalate with one short question and stop.
  Not a silent stop, not an essay. One line of recommendation, one line of question. Check first
  whether that question is already open from a previous pass: a fresh session has no memory of
  asking it and will ask again.

Be strict because a conclusion marker is usually permanent: a unit that self-concludes one stage
short never gets retried, and nobody notices until someone reads the state doc.

---

## 4. Every session starts amnesiac

Chat history is not storage. The next session sees the repository, the brief, and nothing else, so
anything it needs must be written down in the pass that learns it.

Continuity lives in three places, all in the repo or the worktree:

1. **The branch.** `campaign/<slug>`, one per unit, always pushed before the session ends. Unpushed
   work is lost work.
2. **The commits.** Small, one mechanism each, with the measurement in the message. `boltz2 trimul:
   fuse gate+out, 30.9s -> 24.1s, CIF sha unchanged` is worth more than any note about it.
3. **The state docs.** `notes/PORT_STATE.md` is where the port is, one file for the whole port,
   amended every session, under two screens. Beside it, one `notes/state-<workstream>.md` per
   parallel line of work, append-only, one section per pass: what was measured (with numbers), what
   was decided, what is next, what was ruled out and why. One is the summary a fresh session reads
   first; the others are the detail behind it.

Name a workstream doc after the **chain**, not the pass (`state-trimul-perf.md`, not
`state-trimul-perf-p7.md`), so every pass amends one file. Consequence: tooling that syncs by
matching the pass slug never matches it, so two machines can each accumulate real content in their
own copy. Diff before you copy, never overwrite.

Same discipline for conclusion markers: **existence check, then read, then write**. A marker often
carries the previous pass's full summary, and `echo "merged $sha" > marker` over it is unrecoverable
when that directory is gitignored, which it usually is.

---

## 5. Parallelism: worktree, card, lease

Three isolations, all required. Missing any one produces failures that look like hardware faults.

**Working tree.** One git worktree per concurrent agent:

```bash
git worktree list                                    # check the branch is not checked out already
git worktree add ~/wt/<slug> -b campaign/<slug> origin/main
```

Never `git worktree add -f`. The `-f` overrides the one safety check you want: git normally refuses
to check out a branch already checked out elsewhere. Forcing it moves the branch ref out from under
the other worktree, stranding its index against a new HEAD and producing a large phantom diff in a
checkout nobody touched. For a detached tree, `git worktree add --detach` and merge by explicit sha.

**Hardware, and a lease on it.** One card per agent, plus an exclusive lock taken where the device
is physically opened, so a second opener fails cleanly instead of corrupting both runs:

```bash
exec 9>"$HOME/.tt-leases/card${N}.lock"
flock -n 9 || { echo "card $N leased by someone else"; exit 1; }
TT_VISIBLE_DEVICES=$N python -m tt_bio.cli predict ...
```

The strongest version puts the `flock` inside the device-open helper itself, so it cannot be
forgotten. Three things it must survive:

- **A card assignment is a label, not a sandbox.** An agent told "use card 2" that runs one
  convenience command outside the wrapper (a script copied to `/tmp`, a stray subprocess) opens the
  local default, usually card 0, and kills a co-tenant. A collision on the right card *number* on the
  wrong machine is this. Check `/proc/<pid>/environ` for the real `TT_VISIBLE_DEVICES`, not the lease.
- **A lease held per process poisons subprocess tests.** A test that opens the device in-process
  makes the pytest parent the holder, and every later test in that session that forks a subprocess
  blocks on its own parent. It looks exactly like a foreign holder. Keep in-process device tests in
  their own pytest invocation.
- **A whole-machine benchmark lock with no fairness starves waiters.** A loop reacquiring per
  iteration beats a waiter queued an hour, and the holder is often an orphan of an already-concluded
  agent. Before declaring a machine blocked, check the other cards: usually they are idle.

Independent single-card measurements are N parallel jobs, not one chain: fan them across idle cards
by default. A deliberate multi-card scaling run is a separate experiment.

---

## 6. Merge discipline

Agents push branches, one owner merges, and that owner runs the preflight itself:

```bash
git fetch origin
git log --oneline origin/main..origin/campaign/<slug>          # what actually landed on the remote
git diff --stat origin/main...origin/campaign/<slug>
git merge-base --is-ancestor <sha> origin/main && echo "already merged"
```

Four traps, each of which has silently dropped or duplicated real work:

- **Stale local ref.** An agent commits, pushes, exits without fast-forwarding its own local branch
  pointer. A preflight against the *local* name reports "0 commits ahead, empty diffstat" and reads
  as "nothing to merge" while three real commits sit on the origin ref. Compare against origin.
- **Stale status claims.** "Branch X is staged awaiting approval" gets copied forward from summary
  to summary and survives days past the actual merge. Re-derive any binary status from git before
  repeating it. Two commands, always cheaper than being wrong.
- **A revert does not reach a branch that already merged the bad commit.** `git revert` adds an
  inverse commit on one ref. A feature branch that merged main as its assembly base contains the
  rejected commit as an ancestor, and merging that branch onward re-lands the rejected change with
  every check green. After any revert:

  ```bash
  for b in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin); do
    git merge-base --is-ancestor <rejected-sha> "$b" 2>/dev/null && echo "$b still contains it"
  done
  ```
  For a vendored or deployed copy the acceptance test is byte-identity against source (`diff -r`,
  `sha256sum`), not a green import.
- **Silent duplication.** Two branches fixing the same bug at different lines of the same file
  auto-merge without conflict, keeping both: a list entry added twice, or a redundant mechanism
  sitting dead beside the one actually read. The moment a conflict resolution reveals "the other
  side already fixed this", grep that merge's *full* diff for every other site with the same shape.
  These cluster: one bad merge produced three independent bugs, found across three separate passes,
  all dark until a test exercised the path.

---

## 7. Verification culture

The highest-value rule in this document: verify every claim against live state as you make it.

- **Re-run the command.** Do not repeat a prior status as if it were fresh. A note is a timestamped
  claim about what was true when it was written.
- **Never state a number you did not measure this session.** An arithmetic floor is not a wall-clock
  time. A predicted speedup is not a speedup. If a number's provenance is unclear, retire it.
- **Check the deployed artifact, not your own diff.** A grep proving your change is present proves a
  component, not the composition. Import the installed package and print its `__file__` before
  believing a gate scored your checkout. Three consecutive "it's fixed" claims on one bug, each
  verified by grepping the thing just changed, each grep true, bug unchanged, is the usual shape.
- **Another agent's report is a claim, not a fact.** Especially "I merged it", "I pushed it", "the
  gate is green". Check git, check the remote, check the artifact.
- **A red check on a claim that looks true earns one inspection first.** Rows 3 to 7 of §2 are all
  broken checks that read exactly like missing deliverables. Verify the fact by hand, then fix the
  check.

---

## 8. Knowledge capture

After every non-obvious debugging session, write one short note. This is what makes month two
faster than month one, and it is the first thing dropped under time pressure. Keep the notes in one
directory with an index of one line each, appended in the same commit as the note.

```markdown
---
title: <symptom in one line, in the words you would grep for>
area: numerics | perf | packaging | infra | merge
---

**Symptom.** What was observed, with the number or the error string.
**Mechanism.** Why, at the level of the real system: which op, which layout, which ref, which
process. Not "a race", but which two writers and which file.
**Detection.** The command that separates this from its look-alikes. Name the look-alike.
**Fix.** What changed, at which path, with the measurement after.
**Guard.** The test, assert or default that stops it recurring, and where it lives. If there is no
guard, say so: that is the open item.
```

Two rules that keep the collection useful:

- **One writer.** Several agents writing the knowledge directory concurrently recreates the
  working-tree race worktrees exist to prevent. Agents flag a lesson in their conclusion; one owner
  writes it.
- **A closed negative is knowledge.** Record what failed and the measurement that killed it, or it
  gets re-proposed. A lever killed on a measured accuracy bound comes back weeks later as a *speed*
  proposal on a fresh branch, carrying the milliseconds and not the RMSD that buried it. Grep the
  notes for the site before answering any proposal.

---

## 9. Scope and honesty

Report what happened, including the failures and the steps skipped. A partial result reported as
complete costs more than the work it saved, because the next decision is made on it. Specifically:

- Say which stages ran and which did not, by name.
- Say which numbers are measured and which are estimated, in the same sentence as the number.
- An evidenced no-go is a full pass, not a failure: state the prediction, the measurement, the
  decision. Writing the predicted landing before building (`05-perf-method-and-roofline.md`) is what makes a wrong
  prediction informative rather than embarrassing.
- If you did not run the reference comparison, do not use the word "parity".

---

## 10. What an agent must not decide alone

Escalate, then wait. Each of these is irreversible, public, or changes the model's answer:

| Decision | Why |
|---|---|
| Adding or major-bumping a dependency | Changes the environment for everyone and is hard to unwind after other work lands on it. |
| Anything published: a release, a tag, a package upload, a public repo | Usually irreversible. A pushed tag often auto-publishes with no separate confirmation. |
| Destructive or irreversible actions: force-push, history rewrite, deleting data, stopping a live service | No clean recall path. |
| Changing a shared default that other models inherit | One model's win becomes every other model's silent regression, and nothing reports it. |
| Anything that can change accuracy, including flipping a precision flag, a fusion default, or a bucketing constant | The whole point of the port is that the answer does not change. |
| Merging a whole new model port | Scope and maintenance commitment, not a technical call. |

Everything else, decide and log. An escalation that turns out obvious costs a human's attention
every time; a logged judgment call they disagree with gets corrected afterwards. When unsure whether
something clears the bar, act and log rather than ask.

---

## 11. Cost discipline

Measure spend per landed improvement, not per session. Broad one-op-at-a-time sweeping over about a
hundred agent passes returned roughly 2% end to end on one model; the same lineage run as decompose,
floor, screen, predict, decide, build, one A/B returned 1.23x in about a day. The difference is
method, not budget (`05-perf-method-and-roofline.md`).

- **Price a line of work by its ceiling before starting it.** A component that is 4% of wall time
  has a 4% best case, and no cleverness changes that ranking. Write the ceiling in the brief. Device
  time that can only re-measure a symptom of an already-root-caused defect buys nothing.
- **Kill lines whose ceiling does not justify the effort, in writing, with the number.** Then record
  it (§8) so it does not come back on a different metric.
- **Re-rank the backlog after every landing.** A lever dismissed as too small grows as its
  neighbours shrink, and a label like "DRAM-bound" expires the moment the lever generating that
  traffic is removed. Re-derive the ranking from a fresh profile, not from last week's.

Idle is the cheapest state. An agent with no high-value work should stop, not manufacture busywork,
and a card with nothing valuable to run should be released rather than held warm.
