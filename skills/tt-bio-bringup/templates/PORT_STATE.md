# Port state: <model name>

The file a fresh session reads first. Keep it under two screens. Facts only, each with the date it
was established. Delete what is no longer true rather than appending a correction.

## Right now

- Phase:
- Current gate, and whether it is green:
- Branch:
- The one thing blocking progress:

## Environment

- Chip generation, card count, host:
<!-- ttnn has no __version__ attribute and tt-metal is not a distribution name, so read the
     version with: ./env/bin/python3 -c "import importlib.metadata as m; print(m.version('ttnn'))"
     Answers go on the same line as the label, after the colon. -->
- Package versions that matter (ttnn / tt-metal / torch), pinned where:
- The interpreter each gate runs under ($REF_PY for the capture, ./env/bin/python3 for pytest):
- The effort bar for this campaign (05-perf-method-and-roofline.md section 1), as two numbers:
- Baseline test-suite result before any of my changes (pass / fail / skip):

## Established facts

<!-- No "none yet" row here on purpose, unlike Open questions and Killed lines of work. By the time
     the Phase 0 gate runs you have established at least one fact (the reference runs on CPU, and
     what it produced), so an empty table here is a prompt, not a state the phase can end in. -->

| Date | Fact | How it was verified |
|---|---|---|

## Open questions

| Question | Who or what can answer it | Blocking? |
|---|---|---|
| none yet | none yet | none yet |

## Decisions taken

<!-- Same: no "none yet" row. Choosing the fork point and the interpreter each gate runs under are
     decisions, and you have made both before this gate runs. -->

| Date | Decision | Why | What would reverse it |
|---|---|---|---|

## Killed lines of work

Record every abandoned lever with its number, so it is not re-proposed later under a different
metric.

| Date | Lever | Predicted ceiling | Why killed |
|---|---|---|---|
| none yet | none yet | none yet | none yet |
