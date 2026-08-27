#!/usr/bin/env bash
# The autonomous bring-up loop. One command, run from the root of your tt-bio fork:
#
#     export CARD=0                                  # UMD chip id, from `tt-smi -ls`
#     bash "$SKILL/run.sh" --unattended --model-name yourmodel
#
# It runs `port_gate.py status --all-phases` itself, and while that is not green it starts a fresh
# Claude session to work the next phase. It stops on exactly four things: the port is done, nothing
# moved for N iterations, the budget ran out, or the agent hit a decision only a person can make.
# Every ending appends its reason to notes/PORT_STATE.md, which is the file you will read.
#
# The loop runs the gate. It never trusts a gate result the agent reports, and it never reads the
# agent's exit code as progress: under the default permission mode `claude -p` denies every tool
# and still exits 0, so an exit code says nothing about whether work happened.
#
# The termination condition is a file the agent could edit. So the loop hashes notes/PORT_GATES.md
# and scripts/port_gate.py before the first iteration and re-checks them before every one, and
# aborts if either moved. That detects tampering; it does not prevent it. See
# references/14-running-a-long-campaign.md section 3.1 for what that does and does not buy you.
#
# Linux, bash 4+, python3 3.10+. Nothing else. Exit codes: 0 done, 2 could not start or aborted,
# 3 gate tampered with, 4 stalled, 5 spend ceiling, 6 out of iterations, 7 blocked on a person.

set -u
set -o pipefail

# Resolve our own directory before anything else, and follow symlinks: the documented install is a
# symlink into ~/.claude/skills, and $0 through one is the link's directory, not the skill's.
SELF="$(readlink -f "$0" 2>/dev/null || echo "$0")"
SELFDIR="$(cd "$(dirname "$SELF")" && pwd)" || exit 2
GATE_SRC="$SELFDIR/gates/port_gate.py"
[ -f "$GATE_SRC" ] || { echo "FATAL: no gates/port_gate.py next to $SELF. Point this at the" >&2
                        echo "run.sh inside the skill directory, not a copy of it." >&2; exit 2; }

MODEL=""
MAX_ITER=500
MAX_USD=500
STALL_N=3
TIMEOUT=21600            # 6h per gate run and per agent iteration; a wedged card hangs forever
UNATTENDED="${PORT_LOOP_UNATTENDED:-0}"
GATES="notes/PORT_GATES.md"
STATE="notes/PORT_STATE.md"
LOOPDIR=".port_loop"
EXTRA=()

usage() {
  cat <<'USAGE'
usage: bash run.sh --unattended --model-name NAME [options] [-- <extra claude args>]

  --unattended         required. The loop runs `claude --permission-mode bypassPermissions`: it
                       executes commands in this directory without asking. That is what unattended
                       means. Run it in a checkout you are willing to have edited.
  --model-name NAME    your model's name, used once to generate notes/PORT_GATES.md.
  --max-iterations N   stop after N iterations (default 500).
  --max-usd X          stop when this run has spent X dollars (default 500).
  --stall N            stop when N consecutive iterations move nothing (default 3).
  --timeout SECONDS    kill a gate command or an agent iteration that runs this long (default
                       21600). A wedged card hangs pytest indefinitely.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --unattended)     UNATTENDED=1 ;;
    --model-name)     MODEL="${2:-}"; shift ;;
    --max-iterations) MAX_ITER="${2:-}"; shift ;;
    --max-usd)        MAX_USD="${2:-}"; shift ;;
    --stall)          STALL_N="${2:-}"; shift ;;
    --timeout)        TIMEOUT="${2:-}"; shift ;;
    --gates)          GATES="${2:-}"; shift ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; EXTRA=("$@"); break ;;
    *)  echo "FATAL: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

die() { echo "FATAL: $*" >&2; exit 2; }

if [ "$UNATTENDED" != "1" ]; then
  cat >&2 <<'MSG'
FATAL: refusing to start without --unattended.

The loop passes `--permission-mode bypassPermissions` to every Claude session it starts, which
means those sessions run commands in this directory without asking you first. There is no useful
attended mode: under the default permission mode a `claude -p` session is denied every tool and
still exits 0, so the loop would spin forever making no progress and reporting no error.

Run it in a checkout you are willing to have edited, on a branch, and read notes/PORT_STATE.md.
MSG
  exit 2
fi

# ---------------------------------------------------------------- tools and working directory

PY="$(command -v python3 2>/dev/null)" || true
[ -n "$PY" ] || die "no python3 on PATH. port_gate.py needs 3.10 or later."

CLAUDE="${PORT_LOOP_CLAUDE:-}"
if [ -z "$CLAUDE" ]; then
  CLAUDE="$(command -v claude 2>/dev/null)" || true
  [ -n "$CLAUDE" ] || die "no \`claude\` on PATH. Install Claude Code: https://claude.com/claude-code"
fi

# Pick the hash tool once, at the top level. Doing it inside sha_of would put the failure path in
# a command substitution, where an exit kills only the subshell: the caller would carry on with an
# empty hash, and an empty hash compares unequal to everything, which reads as a tamper.
if   command -v sha256sum >/dev/null 2>&1; then SHAMODE=sha256sum
elif command -v shasum    >/dev/null 2>&1; then SHAMODE=shasum
elif command -v openssl   >/dev/null 2>&1; then SHAMODE=openssl
else die "no sha256sum, shasum or openssl on PATH. The loop hashes the gate to detect a weakened
one, and a loop that cannot do that is the thing this design exists to avoid."
fi
sha_of() {
  case "$SHAMODE" in
    sha256sum) sha256sum "$1"      2>/dev/null | cut -d' ' -f1 ;;
    shasum)    shasum -a 256 "$1"  2>/dev/null | cut -d' ' -f1 ;;
    openssl)   openssl dgst -sha256 "$1" 2>/dev/null | awk '{print $NF}' ;;
  esac
}

TOUT=()
command -v timeout >/dev/null 2>&1 && TOUT=(timeout -k 30 "$TIMEOUT")

ISGIT=0
if git rev-parse --show-toplevel >/dev/null 2>&1; then
  ISGIT=1
  TOP="$(git rev-parse --show-toplevel)"
  [ "$TOP" = "$PWD" ] || die "run this from the root of your fork ($TOP), not $PWD. Every gate
command is written relative to the root."
else
  echo "WARNING: this is not a git repository. The loop works, but nothing records what each"
  echo "iteration changed, which is the only audit trail an unattended run leaves."
fi
[ -w . ] || die "$PWD is not writable."

# ---------------------------------------------------------------- the fork's copy of the gate

mkdir -p "$LOOPDIR" notes || die "cannot create $LOOPDIR and notes/ in $PWD"
if [ ! -f scripts/port_gate.py ]; then
  mkdir -p scripts && cp "$GATE_SRC" scripts/port_gate.py || die "cannot copy the gate script"
  echo "copied $GATE_SRC to scripts/port_gate.py"
elif [ "$(sha_of scripts/port_gate.py)" != "$(sha_of "$GATE_SRC")" ]; then
  die "scripts/port_gate.py differs from $GATE_SRC.
Refusing to overwrite it: if it was edited, replacing it here would launder the edit. Diff them,
then \`cp \"$GATE_SRC\" scripts/port_gate.py\` yourself."
fi

if [ ! -f "$GATES" ]; then
  [ -n "$MODEL" ] || die "no $GATES yet, so --model-name NAME is required to generate it."
  "$PY" "$GATE_SRC" status --init --model "$MODEL" --gates "$GATES" || exit 2
  echo
  echo "Read $GATES now. It is this port's definition of done, and the loop freezes it below."
  echo
fi

if [ ! -f "$STATE" ]; then
  cp "$SELFDIR/templates/PORT_STATE.md" "$STATE" 2>/dev/null \
    || printf '# Port state\n\n## Right now\n\n- Phase:\n' > "$STATE"
  echo "started $STATE from the template. The first iteration fills it in."
fi

if git check-ignore -q "$LOOPDIR" 2>/dev/null; then
  echo "WARNING: $LOOPDIR is gitignored. The gate baseline then lives outside the history, which"
  echo "is the last place a rewritten gate would still show up. Drop that ignore rule."
fi

[ -n "${CARD:-}" ] || echo "NOTE: CARD is unset. Phases 0 and 1 need no card; from Phase 2 on,
\`export CARD=<UMD chip id from tt-smi -ls>\` before starting the loop."
[ -n "${REF_PY:-}" ] || echo "NOTE: REF_PY is unset. Phase 1's capture runs under your reference's
interpreter; \`export REF_PY=...\` before the loop reaches it."

# ---------------------------------------------------------------- baseline and tamper detection
#
# On the first run this records the gate's hashes. On every later run, including a restart after a
# kill, it reads them back: that is what makes the check survive an interruption instead of
# re-baselining whatever it finds.

BASE="$LOOPDIR/baseline.sha256"
if [ -f "$BASE" ]; then
  MEM_GATES="$(awk '$1=="gates"{print $2}' "$BASE")"
  MEM_GATEPY="$(awk '$1=="port_gate"{print $2}' "$BASE")"
  [ -n "$MEM_GATES" ] && [ -n "$MEM_GATEPY" ] || die "$BASE is unreadable. Delete it only if you
are certain the gate is the one you meant to freeze."
  echo "gate baseline read from $BASE (frozen on an earlier run)"
else
  MEM_GATES="$(sha_of "$GATES")"
  MEM_GATEPY="$(sha_of scripts/port_gate.py)"
  mkdir -p "$LOOPDIR/baseline"
  cp "$GATES" "$LOOPDIR/baseline/PORT_GATES.md"
  printf 'gates %s\nport_gate %s\n' "$MEM_GATES" "$MEM_GATEPY" > "$BASE"
  echo "gate baseline frozen in $BASE"
  git add "$BASE" "$LOOPDIR/baseline/PORT_GATES.md" 2>/dev/null \
    && git commit -q -m "Freeze the port gate the autonomous loop runs against" 2>/dev/null \
    && echo "committed the baseline, so a rewritten gate shows up in the history"
fi

ITER=0
SPENT=0
SAME=0
LAST_FP=""
FAILS=0
PASSED=0
BEST=0
SINCE=0

end() {   # end REASON EXITCODE MESSAGE
  {
    printf '\n## Loop ended %s\n\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'LOOP-ENDED: %s\n' "$1"
    printf '%s\n' "$3"
    printf '\nIteration %s, %s of 8 phase gates green, $%s spent this run.\n' \
           "$ITER" "$PASSED" "$SPENT"
    printf 'The gate that decides this: python3 scripts/port_gate.py status --all-phases\n'
  } >> "$STATE"
  printf '\n================ LOOP ENDED: %s ================\n%s\n' "$1" "$3"
  printf 'iteration %s, %s/8 phases green, $%s spent. Reason appended to %s.\n' \
         "$ITER" "$PASSED" "$SPENT" "$STATE"
  exit "$2"
}

tamper_check() {
  local now_g now_p
  now_g="$(sha_of "$GATES")"
  now_p="$(sha_of scripts/port_gate.py)"
  if [ "$now_g" != "$MEM_GATES" ] || [ "$now_p" != "$MEM_GATEPY" ]; then
    echo "=== the gate moved ==="
    diff -u "$LOOPDIR/baseline/PORT_GATES.md" "$GATES" || true
    end TAMPER 3 "$GATES or scripts/port_gate.py changed after the loop froze it. The loop's exit
condition is not something the port is allowed to edit, so this is an abort, not a pass. Restore
both from $LOOPDIR/baseline/ (and \`cp \"\$SKILL/gates/port_gate.py\" scripts/\`), or if the change
was a deliberate correction, delete $BASE to re-freeze and say so in $STATE."
  fi
}

# What counts as an iteration having moved something: a commit landed, a file appeared or changed
# status, or a phase gate advanced.
#
# Deliberately NOT the contents of notes/PORT_STATE.md. A session that rewrites it with a fresh
# timestamp and does nothing else is the adversary this check exists for, and a content hash cannot
# tell a timestamp from a fact. `git status --short` can: it moves when a file appears, disappears
# or changes staging state, and does not move when one already-modified file is rewritten. So an
# iteration that only churns text is caught, and one that produces a file or a commit is not.
fingerprint() {
  if [ "$ISGIT" = "1" ]; then
    printf '%s\n%s\n%s\n' "$(git rev-parse HEAD 2>/dev/null || echo none)" \
      "$(git status --short 2>/dev/null | grep -v -e "$LOOPDIR/" -e __pycache__ -e .pytest_cache \
         | sort)" "$PASSED"
  else
    printf '%s\n%s\n' "$(find . -path ./.git -prune -o -path "./$LOOPDIR" -prune -o \
                           -name __pycache__ -prune -o -name .pytest_cache -prune -o \
                           -type f -printf '%p %s\n' 2>/dev/null | sort)" "$PASSED"
  fi
}

read_iter_json() {   # cost <TAB> subtype <TAB> denials <TAB> is_error <TAB> turns <TAB> result
  "$PY" - "$1" <<'PYJSON'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("0\tunreadable\t0\ttrue\t0\tno JSON: the session produced no result object")
    raise SystemExit(0)
if isinstance(d, list):
    d = d[-1] if d else {}
res = " ".join(str(d.get("result", "")).split())[:400]
print("%s\t%s\t%s\t%s\t%s\t%s" % (d.get("total_cost_usd") or 0, d.get("subtype", "?"),
      len(d.get("permission_denials") or []), str(bool(d.get("is_error"))).lower(),
      d.get("num_turns") or 0, res))
PYJSON
}

prune_iters() {
  ls -1t "$LOOPDIR"/iter-*.json 2>/dev/null | tail -n +51 | while IFS= read -r f; do rm -f "$f"; done
}

if grep -qE '^BLOCKED: ' "$STATE"; then
  echo "FATAL: $STATE carries a BLOCKED: line. An earlier iteration hit a decision only you" >&2
  echo "can make. Answer it, delete the line, and start the loop again." >&2
  grep -E '^BLOCKED: ' "$STATE" >&2
  exit 7
fi

echo
echo "loop starting in $PWD"
echo "  gate     $GATES ($MEM_GATES)"
echo "  agent    $CLAUDE --permission-mode bypassPermissions"
echo "  limits   $MAX_ITER iterations, \$$MAX_USD, stall after $STALL_N, ${TIMEOUT}s per step"
echo

HAVE_BUDGET=""
"$CLAUDE" --help 2>/dev/null | grep -q -- --max-budget-usd && HAVE_BUDGET=1

# Baseline the stall fingerprint on the tree as it is now, so the first iteration is compared
# against something and "three iterations moved nothing" counts three, not four.
LAST_FP="$(fingerprint)"

while : ; do
  ITER=$((ITER + 1))
  tamper_check

  echo
  echo "======== iteration $ITER  $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"
  # Through tee, not a command substitution: a phase gate can be a pytest run that takes an hour,
  # and a customer tailing the log wants to see it while it happens.
  "${TOUT[@]+"${TOUT[@]}"}" "$PY" "$GATE_SRC" status --all-phases \
      --gates "$GATES" --repo . --timeout "$TIMEOUT" 2>&1 | tee "$LOOPDIR/status.txt"
  RC=${PIPESTATUS[0]}
  STATUS_OUT="$(cat "$LOOPDIR/status.txt")"
  PASSED="$(printf '%s\n' "$STATUS_OUT" | sed -n 's/^ALL-PHASES: \([0-9]*\)\/8 PASS$/\1/p' | tail -1)"
  [ -n "$PASSED" ] || PASSED=0

  if [ "$RC" -eq 0 ]; then
    end GREEN 0 "Every phase gate exits 0: complete, parity-verified, and optimized against a
measured roofline, each proven by a command in $GATES that you can re-run yourself."
  fi
  if [ "$RC" -eq 2 ]; then
    end ABORT 2 "The gate could not run, so it measured nothing, and the loop will not treat that
as \"keep working\". The output above names the reason."
  fi
  if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then
    end ABORT 2 "The gate ran for ${TIMEOUT}s without finishing and was killed. A wedged card is
the usual cause: references/09-devices-and-hardware-operations.md has the reset."
  fi

  if [ "$ITER" -gt "$MAX_ITER" ]; then
    end MAX-ITERATIONS 6 "Reached the $MAX_ITER-iteration ceiling with $PASSED of 8 gates green.
Read the phase lines above, decide whether it is progressing, and raise --max-iterations or fix
what it is stuck on."
  fi
  REMAIN="$(awk -v m="$MAX_USD" -v s="$SPENT" 'BEGIN{printf "%.4f", m - s}')"
  if awk -v r="$REMAIN" 'BEGIN{exit !(r <= 0.01)}'; then
    end SPEND-CEILING 5 "This run reached the ceiling it was given, \$$MAX_USD, with $PASSED of 8
gates green and \$$SPENT spent. Nothing is lost: raise --max-usd and start the loop again, it
resumes from $STATE."
  fi

  PROMPT="You are continuing a Tenstorrent model port. This is an unattended loop: a fresh session
each iteration, with no memory of the last one.

Read notes/PORT_STATE.md first, then $SELFDIR/SKILL.md, then the reference documents the next
phase names. Work the next phase that is not green. Do not start a later phase first.

The loop just ran the gates itself. This is the real output, not a summary:

$STATUS_OUT

Rules:
- The loop runs the gates, not you. Reporting a gate as passed does not make it pass.
- Do not edit notes/PORT_GATES.md or scripts/port_gate.py. The loop hashes both and aborts if
  either moves. If a gate is genuinely wrong, say so under BLOCKED: below instead of editing it.
- Before you finish: amend notes/PORT_STATE.md with what you measured, what you decided and what
  is next, and commit your work. An iteration that changes no file has produced nothing, and three
  of those in a row stop the loop.
- If you are blocked on something only a person can decide ($SELFDIR/references/14-running-a-long-campaign.md
  section 10), put a line starting with \"BLOCKED: \" at the top of notes/PORT_STATE.md with the
  question and your recommendation, and stop. The loop halts on it and reports it."

  ARGS=(-p "$PROMPT" --permission-mode bypassPermissions --output-format json)
  [ -n "$HAVE_BUDGET" ] && ARGS+=(--max-budget-usd "$REMAIN")
  ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

  JSON="$LOOPDIR/iter-$ITER.json"
  "${TOUT[@]+"${TOUT[@]}"}" "$CLAUDE" "${ARGS[@]}" > "$JSON" 2> "$LOOPDIR/iter-$ITER.err"
  ARC=$?

  IFS=$'\t' read -r COST SUBTYPE DENIALS ISERR TURNS RESULT < <(read_iter_json "$JSON")
  SPENT="$(awk -v a="$SPENT" -v b="$COST" 'BEGIN{printf "%.4f", a + b}')"
  echo "iteration $ITER: exit $ARC, $SUBTYPE, $TURNS turns, \$$COST (\$$SPENT total)"
  [ -n "$RESULT" ] && echo "  agent said: $RESULT"

  if [ "${DENIALS:-0}" -gt 0 ]; then
    end ABORT 2 "The session was denied $DENIALS tool call(s), so it could not do the work and the
loop cannot make progress. That means the permission mode is not the one this loop passes; check
for a settings.json or a hook in this repository that overrides it."
  fi
  if [ "$ARC" -eq 124 ] || [ "$ARC" -eq 137 ]; then
    end ABORT 2 "The agent ran for ${TIMEOUT}s without finishing and was killed. See $JSON and
$LOOPDIR/iter-$ITER.err."
  fi
  if [ "$ARC" -ne 0 ] && [ "$SUBTYPE" != "error_max_budget_usd" ]; then
    FAILS=$((FAILS + 1))
    if [ "$FAILS" -ge 2 ]; then
      end ABORT 2 "Two Claude sessions in a row failed to run (exit $ARC, $SUBTYPE). This is the
CLI or the account, not the port. See $JSON and $LOOPDIR/iter-$ITER.err."
    fi
    echo "  session failed; retrying once"
  else
    FAILS=0
  fi

  prune_iters

  if grep -qE '^BLOCKED: ' "$STATE"; then
    end BLOCKED 7 "The agent hit a decision only a person can make and wrote it at the top of
$STATE:
$(grep -E '^BLOCKED: ' "$STATE" | head -3)
Answer it, delete the line, and start the loop again."
  fi

  if [ "$PASSED" -gt "$BEST" ]; then
    BEST="$PASSED"
    SINCE=0
  else
    SINCE=$((SINCE + 1))
    [ "$SINCE" -ge 5 ] && echo "  $SINCE iterations since a phase gate last advanced (now $BEST/8)"
  fi

  FP="$(fingerprint)"
  if [ "$FP" = "$LAST_FP" ]; then
    SAME=$((SAME + 1))
    echo "  nothing moved: no commit, no file appeared, no gate advanced ($SAME/$STALL_N)"
    if [ "$SAME" -ge "$STALL_N" ]; then
      end STALLED 4 "$STALL_N iterations in a row moved nothing: no commit landed, no file appeared
or disappeared, and no phase gate advanced past $PASSED of 8. It is stuck, not working. The phase
lines above name the command it cannot get past; $LOOPDIR/iter-$ITER.json has what the last session
said. $SINCE iterations have passed since a gate last advanced."
    fi
  else
    SAME=0
  fi
  LAST_FP="$FP"
done
