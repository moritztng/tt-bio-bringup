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
check() {   # check <label> <expected-rc> <file>
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
    echo "| Roof | Method | Value |"; echo "|---|---|---|"
    echo "| Peak matmul throughput | N=8192 bf16 HiFi4 | 100.6 TFLOP/s |"
    echo "| DRAM bandwidth | 8192^2 bf16 add | 435.2 GB/s |"
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

[ "$fail" -eq 0 ] && echo "worked example passes the gates it prescribes" && exit 0
echo
echo "The worked example does not pass its own gates. Fix the example, not the gate:"
echo "a reader copies the example's shape and meets the gate's opinion of it."
exit 1
