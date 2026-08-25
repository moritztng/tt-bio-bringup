# Running a bring-up as a long agentic campaign

This document decides your operating model: split the port into units of work that each carry a
written brief and one machine-checkable done-condition, prove that done-check goes red before you
trust it green, keep all continuity in the repository because every session starts amnesiac, isolate
concurrent agents in their own worktree with an exclusive lease on one card, let agents push branches
while exactly one person merges, and verify every claim against live state as you make it.

Read this when a port is going to take more than a week, when you are about to run more than one
Claude Code session at a time, or when a session reported something done and it was not.

**What this assumes you have: Claude Code, git, and at least one card.** Everything up to §5 works
on exactly one; §5 is about running several agents at once and needs one card per agent. On a
single card, run the agents one at a time and keep the lease anyway, because the thing it protects
you from is your own second terminal. No dispatcher, no queue, no
orchestration service. A unit of work is a markdown file plus a branch, you start each session
yourself, and the done-check is a command you run in your own shell. Everything below works that way
on purpose. If you later build automation around it, the primitives do not change.

---

## 1. The unit of work is a written brief

One file per unit of work, `notes/briefs/<slug>.md`, written before the session starts and re-read at
the top of every session that continues it. `templates/work-brief.md` is the form; every heading in it
is load-bearing:

| Heading | Content | Failure if it is missing |
|---|---|---|
| Goal | one sentence, one outcome | the session invents scope |
| Definition of done | one shell command, exits 0 only when the deliverable exists and is correct | the work gets called done while unfinished |
| In scope | the files it may edit | edits spray across the tree and become unmergeable |
| Out of scope | what it must not touch, named | "while I was there I also refactored" |
| Commands to run | the exact invocations, with the interpreter and the env vars | each session re-derives them and gets them slightly different |
| Context the agent will not otherwise have | prior attempts, known traps, why the obvious approach was rejected | the same dead end gets walked twice |
| Hardware | which card, and the lease | two sessions on one card, reported as nondeterminism |
| When blocked | what to escalate and what to do meanwhile | the session either stalls or improvises |
| Report back | where the state doc is, what must be written before finishing | the next session starts from zero |

The brief is agent-facing. Precision beats prose. A unit with no definition-of-done command is not a
task, it is a wish, and it will come back reported complete.

**Never write "merge to main" as an action in a brief**, not even conditionally. Write "report
`VERDICT: <result>`" instead, and merge it yourself after checking. A task-specific instruction
overrides a standing rule every time: three briefs in one campaign said "if X, merge to main", all
three sessions did, bypassing the preflight. Two were safe only because nothing raced them.

---

## 2. The done-check is the primitive

A done-check is a command, run by someone other than the agent that did the work, exiting 0 only when
the artifact genuinely exists and is correct. Everything else here is scaffolding around it. You are
the one who runs it: paste it into a fresh shell in your own checkout, after the session says it is
finished, before you believe it.

Grep for the *shape of a result*, never for a topic keyword:

```bash
# BAD: a session that writes "I did not get to the A/B, here's why" satisfies this
grep -qiE "parity|A/B|PCC" "notes/state-$SLUG.md"

# GOOD: only a line carrying a real measured number can produce this
grep -qE '^VERDICT-P4: (GO|NO-GO) pcc=0\.[0-9]{4}' "notes/state-$SLUG.md"
```

### Seven ways a done-check lies, and the fix for each

| # | Failure | Mechanism | Fix |
|---|---|---|---|
| 1 | Keyword grep on prose | A multi-stage task discusses stage 2 in prose while only running stage 1. Keyword presence cannot distinguish "here is the table" from "here is why I skipped it". | Grep for a labelled number or a table header that only exists once real data lands. |
| 2 | Greps for a name, not a verdict | `grep -q "fused_trimul" report.md` passes when the report says the op was *considered*. | Grep for the decision: `^VERDICT: (GO\|NO-GO)`, plus the measurement that decided it. |
| 3 | Hardcoded path | The check is written inside the agent's worktree and hardcodes it. Re-run from your own checkout the path does not exist, exit 1, which reads exactly like a missing deliverable. | Derive the root from the script's own location or one env var. Accept several candidates and pass if any hit. |
| 4 | Reads a stale copy | The check reads a file that exists in two places, and resolves the one nobody wrote. It is non-empty, so `test -s` passes and only the content grep fails. Indistinguishable from unfinished. | Check the file the writer actually wrote. Before treating a red check as real, confirm you are reading the same path the session wrote. |
| 5 | Prose where a command belongs | "done when the verdict is declared in the state doc by the usual convention" is not runnable. Pasted into a shell it is `bash: declared: command not found`, exit 127, which reads as a failure rather than as a check that was never written. | One physical unwrapped line starting with a real command name (`grep`, `test`, `python3`). Copying one forward, copy the literal command and edit only the marker string. |
| 6 | Inverted exit status | `grep -L PAT FILE && echo OK` fires when the pattern **is** present. It can never be satisfied by a correct file, so repeated "fixes" look like a flaky test rather than a backwards check. | For absence, write `! grep -q PAT FILE`. |
| 7 | Depends on reachability | The check ssh'es to a machine or curls a service. That goes down and the check fails forever, blocking finished, verified work. | Keep it local. If it must be remote, bound it (`timeout 30 ssh -o BatchMode=yes ...`) and distinguish exit 255 (transport) from exit 1 (deliverable missing). |

### The universal remedy: prove it red first

Before you trust a green done-check, break the deliverable and watch it fail.

```bash
CHK='grep -qE "^VERDICT-P4: (GO|NO-GO) pcc=0\.[0-9]{4}" notes/state-x.md'
cp notes/state-x.md /tmp/state-x.bak
sed -i 's/^VERDICT-P4:.*/VERDICT-P4: GO/' notes/state-x.md          # drop the pcc= the check wants
bash -c "$CHK"; echo "broken -> $?"                                  # MUST be non-zero
cp /tmp/state-x.bak notes/state-x.md
bash -c "$CHK"; echo "intact -> $?"                                  # MUST be 0
```

**Edit the file, do not move it away.** Deleting the thing under test makes almost any check
non-zero, and a check that fails because its subject is missing has not been shown to notice a
defect: the whole point here is that a verdict line without a `pcc=` is a real, plausible thing a
session writes. `port_gate.py prove-red` refuses a break that removes a file rather than changing
it, for exactly this reason.

Run it through a fresh `bash -c`, not your interactive prompt: an interactive shell often has `grep`
shadowed by a function with different exit semantics, so a hand-check at the prompt can look right
and still be wrong about what a script does. `scripts/port_gate.py prove-red` is this loop with the
exit codes read for you, and with the refusals above built in:

```bash
python3 scripts/port_gate.py prove-red \
    --check         "$CHK" \
    --break         "sed -i 's/^VERDICT-P4:.*/VERDICT-P4: GO/' notes/state-x.md" \
    --restore       "cp /tmp/state-x.bak notes/state-x.md" \
    --expect-change notes/state-x.md
```

Two authoring constraints that prevent whole classes of this:

- **Only require writes to per-agent or append-only files.** Mandating a write to a single-owner file
  with several agents live is a mandated race. Each agent writes its own state doc; shared findings go
  to one append-only file with the agent's slug on each line.
- **Decisions live in a file nobody rewrites**, or they vanish and get relitigated three weeks later
  by someone who was not in the room.

---

## 3. Drive to done

Scope a deliverable as complete or as one-shot, and say which in the brief. If it is complete:

- The unit gets restarted, by you, until every part is landed and verified. Number the sessions
  (`s1`, `s2`, ...) in the state doc so it stays readable.
- Never report done while any stated priority is owed, deferred or pending. "P1 done, P2 to P4 owed"
  reported as a conclusion means someone has to notice and re-queue it by hand, which is exactly the
  work the brief was supposed to remove.
- Out of context, out of time: that is *pause and resume*, not done. Write the state doc, name the
  next action, stop. The next session picks it up from the file, not from the chat.
- Genuinely blocked on something only a person can decide (§10): say so in one line, with a
  recommendation, and stop. Check the state doc first for whether that question is already open: a
  fresh session has no memory of asking it and will ask again.

Be strict because "done" is sticky. A unit reported complete one stage short does not get retried,
and nobody notices until someone reads the state doc weeks later.

---

## 4. Every session starts amnesiac

Chat history is not storage. The next session sees the repository, the brief, and nothing else, so
anything it needs must be written down in the pass that learns it.

Continuity lives in three places, all in the repo or the worktree:

1. **The branch.** `port/<slug>`, one per unit, always pushed before the session ends. Unpushed work
   is lost work: a session that ends with commits only in a local worktree has produced nothing you
   can act on.
2. **The commits.** Small, one mechanism each, with the measurement in the message.
   `yourmodel trimul: fuse gate+out, 30.9s -> 24.1s, output sha unchanged` is worth more than any
   note written about it afterwards, because it cannot drift away from the change it describes.
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
SLUG=your-workstream-slug
git worktree add ~/wt/"$SLUG" -b port/"$SLUG" origin/main
```

Never `git worktree add -f`. The `-f` overrides the one safety check you want: git normally refuses
to check out a branch already checked out elsewhere. Forcing it moves the branch ref out from under
the other worktree, stranding its index against a new HEAD and producing a large phantom diff in a
checkout nobody touched. For a detached tree, `git worktree add --detach` and merge by explicit sha.

**Hardware, and a lease on it.** One card per agent, plus an exclusive lock taken where the device
is physically opened, so a second opener fails cleanly instead of corrupting both runs:

```bash
N=${CARD:?which card: the UMD chip id from tt-smi -ls}
mkdir -p "$HOME/.tt-leases"                 # exec 9> fails on a missing directory, and then
exec 9>"$HOME/.tt-leases/card${N}.lock"     # flock -n 9 operates on a bad descriptor
flock -n 9 || { echo "card $N leased by someone else"; exit 1; }
TT_VISIBLE_DEVICES=$N ./env/bin/tt-bio predict ...
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

Agents push branches, one person merges, and that person runs the preflight themselves. Not the
agent that did the work: its report that the merge is clean is a claim about a command it ran, and §7
is about why that is not the same thing as the command's output.

```bash
git fetch origin
git log --oneline "origin/main..origin/port/$SLUG"          # what actually landed on the remote
git diff --stat "origin/main...origin/port/$SLUG"
git merge-base --is-ancestor "$SHA" origin/main && echo "already merged"
```

Four traps, each of which has silently dropped or duplicated real work:

- **Stale local ref.** A session commits, pushes, then exits without fast-forwarding its own local
  branch pointer. A preflight against the *local* name reports "0 commits ahead, empty diffstat",
  which reads as "nothing to merge" while three real commits sit on the origin ref. Compare against
  origin, always.
- **Stale status claims.** "Branch X is staged awaiting approval" gets copied forward from summary
  to summary and survives days past the actual merge. Re-derive any binary status from git before
  repeating it. Two commands, always cheaper than being wrong.
- **A revert does not reach a branch that already merged the bad commit.** `git revert` adds an
  inverse commit on one ref. A feature branch that merged main as its assembly base contains the
  rejected commit as an ancestor, and merging that branch onward re-lands the rejected change with
  every check green. After any revert:

  ```bash
  for b in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin); do
    git merge-base --is-ancestor "$REJECTED_SHA" "$b" 2>/dev/null && echo "$b still contains it"
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
- **A session's report is a claim, not a fact.** Especially "I merged it", "I pushed it", "the gate
  is green". Check git, check the remote, check the artifact. This applies to your own earlier
  sessions as much as to a subagent's summary.
- **A red check on a claim that looks true earns one inspection first.** Rows 3 to 7 of §2 are all
  broken checks that read exactly like missing deliverables. Verify the fact by hand, then fix the
  check.

---

## 8. Knowledge capture

After every non-obvious debugging session, write one short note. This is what makes month two
faster than month one, and it is the first thing dropped under time pressure. Keep the notes in one
directory with an index of one line each, appended in the same commit as the note.

`templates/finding-note.md` is the format. Six sections: Symptom, Mechanism, Detection, Fix, Guard,
Generalization. The last one is the reason the note is worth writing, and it is the one that gets
dropped. Fill it in even when the answer is "no wider class, this was specific".

Two of the six are worth spelling out because they are usually written too vaguely to use.
**Mechanism** means at the level of the real system: which op, which layout, which reference, which
process. Not "a race", but which two writers and which file. **Detection** means the command that
separates this from its look-alikes, and it should name the look-alike.

Two rules that keep the collection useful:

- **One writer.** Several agents writing `notes/findings/` concurrently recreates the working-tree
  race that worktrees exist to prevent. An agent names the lesson in its final report; one person
  writes the file.
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

Everything else, decide and log. An escalation that turns out obvious costs someone's attention every
time; a logged judgment call they disagree with gets corrected afterwards, cheaply. When unsure
whether something clears the bar, act and log rather than ask.

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

Idle is the cheapest state. A session with no high-value work left should stop rather than manufacture
busywork, and a card with nothing valuable to run should be released rather than held warm.
