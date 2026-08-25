#!/usr/bin/env bash
# The worked example claims its tables pass the gates it prescribes. This checks that they do.
#
# Scope, stated because the label on a check has to match what it measured: the TABLES and the
# pasted gate output come from the example. The plan's six prose sections (Reference, Target,
# Control flow, Host-side pipelines, and the two below) are written here, because the example
# presents those as narrative rather than as a filled-in plan. So a green run means "every table
# the example shows is accepted by the gate that judges it, and the output it pastes is the
# output that gate produces". It does not mean the example contains a complete plan.
#
# Why it exists: the example is the highest-trust document here, a reader copies its shape, and
# nothing tied its tables to the gate that judges them. A cold read found four of them rejected by
# the gates the same document tells you to run. That is this repository's own headline defect,
# "a green check nobody has watched go red", committed against itself.
#
# Exit 0 every extracted document passes its gate, 1 one does not, 2 the check could not run.
set -u
cd "$(dirname "$0")/.." || exit 2

PY=${PY:-python3}
GATE=skills/tt-bio-bringup/gates/port_gate.py
EX=skills/tt-bio-bringup/examples/worked-example.md
TMP=$(mktemp -d) || exit 2
trap 'rm -rf "$TMP"' EXIT

[ -f "$GATE" ] && [ -f "$EX" ] || { echo "missing $GATE or $EX"; exit 2; }

# Pull one markdown table out of the example by a string in its header row.
# Fenced blocks hold pasted gate output, which contains table headers as text. Skip them, or the
# extraction picks up a line of a GATE 1 message instead of the table it names.
table() {
    awk -v key="$1" '
        /^[[:space:]]*```/ { fence = !fence; next }
        fence { next }
        $0 ~ /\|/ && !/^ *#/ && index($0, key) { on = 1 }
        on && /\|/ && !/^ *#/ { print; next }
        on { exit }
    ' "$EX"
}

# How many tables in the example have a header containing this key. More than one and table()
# silently picks the first, which is the same defeat the plan gate has for a decoy module tree.
count_tables() {
    awk -v key="$1" '
        /^[[:space:]]*```/ { fence = !fence; next }
        fence { next }
        /\|/ && !/^ *#/ && index($0, key) && !prev { n++ }
        { prev = (/\|/ && !/^ *#/) }
        END { print n + 0 }
    ' "$EX"
}

# table() finding nothing looked exactly like table() finding an empty section, so a deleted
# table in the example produced a plan the gate happily passed. Fail loudly instead.
need_table() {
    n=$(count_tables "$1")
    if [ "$n" -gt 1 ]; then
        printf '  FAIL  %s tables in the example have a header containing %s\n' "$n" "$1" >&2
        printf '        table() takes the first, so this check would judge whichever one comes\n' >&2
        printf '        first and say nothing about the other. Make the headers distinct.\n' >&2
        exit 1
    fi
    out=$(table "$1")
    if [ -z "$out" ]; then
        printf '  FAIL  no table in the example whose header contains %s\n' "$1" >&2
        printf '        The extraction found nothing, which is not the same as finding an empty\n' >&2
        printf '        table. Either the example lost it or this key no longer matches.\n' >&2
        exit 1
    fi
    printf '%s\n' "$out"
}

fail=0
check() {   # check <label> <expected-rc> <ignored> <gate args...>
    # $3 is a decoy kept so the call sites line up visually; the gate arguments start at $4.
    "$PY" "$GATE" "${@:4}" >"$TMP/out" 2>&1
    rc=$?
    if [ "$rc" -eq "$2" ]; then
        printf '  ok    %s (exit %s)\n' "$1" "$rc"
    else
        printf '  FAIL  %s: expected exit %s, got %s\n' "$1" "$2" "$rc"
        sed 's/^/        /' "$TMP/out"
        fail=1
    fi
}

# 1. The Phase 0 module tree, axes, op inventory and randomness tables, assembled into a plan
#    that keeps the example's own cells and adds only the prose sections the plan template has.
{
    echo "# Port plan: minifold"
    echo; echo "## Reference"
    echo "- Repository and pinned commit: examples/minifold_capture.py, this repository"
    echo "- Config / checkpoint used, with its hash: random init, torch.manual_seed(1234)"
    echo "- Command that runs it on CPU, and its runtime: minifold_capture.py --len 117, 0.006 s"
    echo "- Known reference-side nondeterminism, and how it is pinned: dropout, pinned by eval() and seed 0"
    echo; echo "## Target"
    echo "- Chip generation and card count: Blackhole p150a, 1 card, unconfirmed"
    echo "- Supported size range to ship (state numbers, not \"large\"): L = 16 to 512"
    echo "- Input modes to support: token ids plus an optional boolean mask"
    echo "- Outputs to produce, in what formats: logits and single, as a .pt"
    echo; echo "## Module tree"; echo
    need_table "Golden fixture"
    echo; echo "## Axes"; echo
    need_table "Tile-multiple handling"
    echo; echo "## Op inventory"; echo
    need_table "ttnn equivalent"
    echo; echo "## Control flow"
    echo "One loop over 4 blocks, trip count fixed at 4, independent of the data."
    echo; echo "## Host-side pipelines"
    echo "Tokenization only: a 22-symbol vocabulary lookup and an optional padding mask."
    echo; echo "## Randomness"; echo
    need_table "How both sides will share draws"
    echo; echo "## Evaluation set"; echo
    echo "| Item | Decision |"; echo "|---|---|"
    echo "| Inputs, named | 3 sequences at L = 64, 117 and 380 |"
    echo "| Ground truth, and where it lives | published contact maps, notes/eval/ |"
    echo "| The metric, and what a domain expert considers passing | top-L/5 contact precision, within 2 points of CPU |"
    echo "| Licensed for use with this model | yes, confirmed before Phase 0 closed |"
    echo "| Who curates it | the porting engineer |"
    echo; echo "## Risks, ranked"
    echo "1. The head materialises [1, L, L, 128] before projecting to 16 bins."
    echo "2. F.gelu has two definitions and ttnn defaults to the tanh approximation."
} > "$TMP/plan.md"
check "Phase 0 plan, from the example's own tables" 0 "$TMP/plan.md" plan "$TMP/plan.md"

# 1b. The Phase 0 report arm, which the example shows as a markdown block rather than a table.
#     Extract it and run the same command the example prints.
awk '/^## Environment$/{on=1} on&&/^```$/{exit} on{print}' "$EX" > "$TMP/state_env.md"
{ echo "# Port state: minifold"; echo; cat "$TMP/state_env.md"; echo
  echo "## Decisions taken"; echo
  echo "| Date | Decision | Why | What would reverse it |"; echo "|---|---|---|---|"
  echo "| 2026-08-25 | one venv, not two | the reference has no conflicting pins | a pin conflict |"
} > "$TMP/state.md"
check "Phase 0 report arm, from the example's own block" 0 "$TMP/state.md" \
      report "$TMP/state.md" --no-tables --require-heading "Environment" \
      --require-heading "Decisions taken"

# 1c. EVERY table the example shows under a heading a gate judges, not only the nine this script
#     extracts by key. Both halves are collected by position, not by a header string: a second
#     table under either heading was invisible to all of the checks above, because they pull
#     tables the script names and the gate never saw the rest. Naming one more header key would
#     have fixed one planted table and missed the next one.
{
    echo "# Parity report: minifold"
    echo; echo "## Component parity"; echo
    # From the Phase 3 heading to the Negative controls paragraph.
    awk '
        /^[[:space:]]*```/ { fence = !fence; next }
        fence { next }
        /^### Phase 3: component parity/ { on = 1; next }
        on && /^\*\*Negative controls\*\*/ { exit }
        on && /^#/ { exit }
        on && /\|/ && !/^ *#/ { print }
    ' "$EX"
    echo; echo "## Negative controls"; echo
    # And from that paragraph to the next heading. Stopping at a blank line would collect only
    # the first table, which is how a planted second one stayed invisible: the point of this
    # block is that it does not choose.
    awk '
        /^[[:space:]]*```/ { fence = !fence; next }
        fence { next }
        /^\*\*Negative controls\*\*/ { on = 1; next }
        on && /^#/ { exit }
        on && /\|/ && !/^ *#/ { print }
    ' "$EX"
} > "$TMP/parity_all.md"
check "every table the example puts under a heading Phase 3 judges" 0 "$TMP/parity_all.md" \
      report "$TMP/parity_all.md" --require-heading "Component parity" \
      --require-heading "Negative controls"

# 1d. EVERY table in the document, heading or no heading. Blocks 1, 2 and 3 pull tables by
#     header string, and 1c fixed that for the two Phase 3 headings only, which left the same
#     hole open under Phase 0 and Phase 5: a second table planted after the Levers table left all
#     of these checks green because none of them extracts it. This block does not name anything.
#     It cannot judge which heading a table sits under, so it catches the content defects
#     (blank cell, TBD, "lots", a short row, a verdict that says the control did not fire) and
#     leaves the heading-sensitive ones to the blocks above. A table written as HTML is
#     invisible to this and to the gate alike, which is a reason not to write one.
{
    echo "# Every table in the worked example"
    echo; echo "## Tables"; echo
    awk '/^[[:space:]]*```/ { fence = !fence; next } fence { next } /\|/ && !/^ *#/ { print }' "$EX"
} > "$TMP/alltables.md"
check "every table in the example, extracted by nothing but a pipe" 0 "$TMP/alltables.md" \
      report "$TMP/alltables.md" --require-heading "Tables"

# 2. The Phase 3 component-parity table, under the headings the Phase 3 gate requires.
{
    echo "# Parity report: minifold"
    echo; echo "## Component parity"; echo
    need_table "Threshold (measured bf16 envelope)"
    echo; echo "## Negative controls"; echo
    need_table "Went red?"
} > "$TMP/parity.md"
check "Phase 3 parity report, from the example's own tables" 0 "$TMP/parity.md" \
      report "$TMP/parity.md" --require-heading "Component parity" --require-heading "Negative controls"

# 3. The Phase 5 census and lever tables, under the headings the Phase 5 gate requires.
{
    echo "# Performance report: minifold"
    echo; echo "## Measured roofs"; echo
    need_table "| Roof | Method | Value |"
    echo; echo "## Op census"; echo
    need_table "Share of device"
    echo; echo "## Levers"; echo
    need_table "Share of wall it touches"
} > "$TMP/perf.md"
check "Phase 5 perf report, from the example's own tables" 0 "$TMP/perf.md" \
      report "$TMP/perf.md" --require-heading "Measured roofs" --require-heading "Op census" \
      --require-heading "Levers"

# 4. The templates must still be red, or the three passes above prove nothing.
check "blank plan template is red" 1 x plan skills/tt-bio-bringup/templates/PORT_PLAN.md
check "blank parity template is red" 1 x report skills/tt-bio-bringup/templates/parity-report.md \
      --require-heading "Component parity" --require-heading "Negative controls"
check "blank perf template is red" 1 x report skills/tt-bio-bringup/templates/perf-report.md \
      --require-heading "Measured roofs" --require-heading "Op census" --require-heading "Levers"
# The example's Environment block deliberately shortens two labels, so testing only the example
# would not have caught an 80-character cap that exempted exactly the template's two long ones.
check "blank PORT_STATE template is red" 1 x report skills/tt-bio-bringup/templates/PORT_STATE.md \
      --no-tables --require-heading "Environment" --require-heading "Decisions taken"

# 5. The example pastes the gate's own output for the blank template. That output carries line
#    numbers, so every edit to the template invalidates it, and it has gone stale twice. Compare
#    the problem count and the first line number against a live run.
"$PY" "$GATE" plan skills/tt-bio-bringup/templates/PORT_PLAN.md > "$TMP/live" 2>&1
live_n=$(sed -n '1s/.*finished\. \([0-9]*\) problem.*/\1/p' "$TMP/live")
doc_n=$(grep -m1 -o "is not finished\. [0-9]* problem" "$EX" | grep -o "[0-9]*")
# Every PORT_PLAN.md:<n> the example pastes, against every one a live run prints. The count alone
# is not enough: an edit low in the template shifts line numbers without changing how many
# problems there are, and that is the stale-output case that actually happened.
live_l=$("$PY" - "$TMP/live" <<'PYEOF'
import re, sys
print(" ".join(sorted(set(re.findall(r"PORT_PLAN\.md:(\d+):", open(sys.argv[1]).read())), key=int)))
PYEOF
)
doc_l=$("$PY" - "$EX" <<'PYEOF'
import re, sys
t = open(sys.argv[1]).read()
i = t.find("GATE 1: phase 0 plan notes/PORT_PLAN.md is not finished.")
blk = t[i:t.find("```", i)] if i >= 0 else ""
print(" ".join(sorted(set(re.findall(r"PORT_PLAN\.md:(\d+):", blk)), key=int)))
PYEOF
)
# The problem TEXT too, not only the count and the line numbers. Rewriting a pasted problem into
# something the gate never says used to pass, and three of the pasted problems carry no line
# number at all, so they were unchecked entirely. Both texts go through FILES, not through shell
# interpolation: the gate's own messages contain quotes, and interpolating them into a Python
# string is how the first version of this check produced a false failure.
"$PY" - "$EX" "$TMP/live" > "$TMP/textdiff" <<'PYEOF'
import re, sys

def joined(lines):
    out = []
    for line in lines:
        if line.startswith("    ") and out:      # a wrapped continuation of the line above
            out[-1] += " " + line.strip()
        elif line.startswith("  "):
            out.append(line[2:].rstrip())
    return out

doc_text = open(sys.argv[1]).read()
i = doc_text.find("GATE 1: phase 0 plan notes/PORT_PLAN.md is not finished.")
doc = joined(doc_text[i:doc_text.find("```", i)].splitlines()[1:]) if i >= 0 else []
live = joined(open(sys.argv[2]).read().splitlines()[1:])

# The live run reads the template, the example pastes notes/PORT_PLAN.md. Same document, two
# paths, so normalise the prefix away before comparing the message itself.
def norm(s):
    s = re.sub(r"\S*PORT_PLAN\.md", "PLAN", s)
    return re.sub(r"\s+", " ", s).strip().rstrip(".")
live_n = [norm(l) for l in live]
for d in doc:
    if d.strip().startswith("..."):              # a deliberate elision, not a claim
        continue
    if not any(norm(d) in l for l in live_n):
        print(d)
PYEOF
if [ -s "$TMP/textdiff" ]; then
    printf '  FAIL  the example pastes problem text the gate does not produce:\n'
    sed 's/^/          /' "$TMP/textdiff"
    fail=1
elif [ "$live_n" = "$doc_n" ] && [ "$live_l" = "$doc_l" ]; then
    printf '  ok    pasted gate output still matches a live run (%s problems, lines %s)\n' "$live_n" "$live_l"
else
    printf '  FAIL  pasted gate output is stale.\n'
    printf '        problems: example says %s, live run says %s\n' "$doc_n" "$live_n"
    printf '        lines:    example says [%s], live run says [%s]\n' "$doc_l" "$live_l"
    printf '        Re-paste it. A document that says its output came from a run has to mean it.\n'
    fail=1
fi

# The GATE 0 lines the example pastes, too. Only the GATE 1 block was compared, so a verdict
# message the tool cannot produce sat in the example for as long as it took someone to notice.
for want in "$(printf '%s\n' "GATE 0: phase 0 plan")" "$(printf '%s\n' "GATE 0: report")"; do
    "$PY" - "$EX" "$want" <<'PYEOF' || fail=1
import re, sys
doc = open(sys.argv[1]).read()
prefix = sys.argv[2]
# the tool's own wording, from the source, so this cannot drift silently
src = open("skills/tt-bio-bringup/gates/port_gate.py").read()
m = re.search(r'"GATE 0: \{what\} \{path\} ([^"]*)"', src)
phrase = m.group(1).split("{")[0].strip() if m else None
if phrase is None:
    print("  FAIL  could not find the GATE 0 wording in port_gate.py"); sys.exit(1)
for line in doc.splitlines():
    if line.startswith(prefix) and phrase not in line:
        print(f"  FAIL  the example pastes a GATE 0 line the gate cannot produce:")
        print(f"          {line.strip()}")
        print(f"        the gate says: ... {phrase} ...")
        sys.exit(1)
sys.exit(0)
PYEOF
done
[ "$fail" -eq 0 ] && echo "worked example passes the gates it prescribes" && exit 0
echo
echo "The worked example does not pass its own gates. Fix the example, not the gate:"
echo "a reader copies the example's shape and meets the gate's opinion of it."
exit 1
