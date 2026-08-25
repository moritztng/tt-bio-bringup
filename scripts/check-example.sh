#!/usr/bin/env bash
# The worked example claims its tables pass the gates it prescribes. This checks that they do.
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
        /^```/ { fence = !fence; next }
        fence { next }
        $0 ~ /^\|/ && index($0, key) { on = 1 }
        on && /^\|/ { print; next }
        on { exit }
    ' "$EX"
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
    table "Golden fixture"
    echo; echo "## Axes"; echo
    table "Tile-multiple handling"
    echo; echo "## Op inventory"; echo
    table "ttnn equivalent"
    echo; echo "## Control flow"
    echo "One loop over 4 blocks, trip count fixed at 4, independent of the data."
    echo; echo "## Host-side pipelines"
    echo "Tokenization only: a 22-symbol vocabulary lookup and an optional padding mask."
    echo; echo "## Randomness"; echo
    table "How both sides will share draws"
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

# 2. The Phase 3 component-parity table, under the headings the Phase 3 gate requires.
{
    echo "# Parity report: minifold"
    echo; echo "## Component parity"; echo
    table "Threshold (measured bf16 envelope)"
    echo; echo "## Negative controls"; echo
    table "Went red?"
} > "$TMP/parity.md"
check "Phase 3 parity report, from the example's own tables" 0 "$TMP/parity.md" \
      report "$TMP/parity.md" --require-heading "Component parity" --require-heading "Negative controls"

# 3. The Phase 5 census and lever tables, under the headings the Phase 5 gate requires.
{
    echo "# Performance report: minifold"
    echo; echo "## Measured roofs"; echo
    table "| Roof | Method | Value |"
    echo; echo "## Op census"; echo
    table "Share of wall"
    echo; echo "## Levers"; echo
    table "Share of wall it touches"
} > "$TMP/perf.md"
check "Phase 5 perf report, from the example's own tables" 0 "$TMP/perf.md" \
      report "$TMP/perf.md" --require-heading "Measured roofs" --require-heading "Op census" \
      --require-heading "Levers"

# 4. The templates must still be red, or the three passes above prove nothing.
check "blank plan template is red" 1 x plan skills/tt-bio-bringup/templates/PORT_PLAN.md
check "blank parity template is red" 1 x report skills/tt-bio-bringup/templates/parity-report.md \
      --require-heading "Component parity" --require-heading "Negative controls"

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
if [ "$live_n" = "$doc_n" ] && [ "$live_l" = "$doc_l" ]; then
    printf '  ok    pasted gate output still matches a live run (%s problems, lines %s)\n' "$live_n" "$live_l"
else
    printf '  FAIL  pasted gate output is stale.\n'
    printf '        problems: example says %s, live run says %s\n' "$doc_n" "$live_n"
    printf '        lines:    example says [%s], live run says [%s]\n' "$doc_l" "$live_l"
    printf '        Re-paste it. A document that says its output came from a run has to mean it.\n'
    fail=1
fi

[ "$fail" -eq 0 ] && echo "worked example passes the gates it prescribes" && exit 0
echo
echo "The worked example does not pass its own gates. Fix the example, not the gate:"
echo "a reader copies the example's shape and meets the gate's opinion of it."
exit 1
