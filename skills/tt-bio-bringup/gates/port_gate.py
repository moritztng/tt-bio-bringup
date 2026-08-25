#!/usr/bin/env python3
"""Machine-checkable exit gates for a tt-bio bring-up.

Copy this file into your fork as ``scripts/port_gate.py``. Every phase gate in the
tt-bio-bringup skill is a shell command, and five of them call this script:

    python3 scripts/port_gate.py plan PORT_PLAN.md
    python3 scripts/port_gate.py report docs/yourmodel-perf.md --require-heading "Op census"
    python3 scripts/port_gate.py determinism --run "<capture cmd>" --artifact <path>
    python3 scripts/port_gate.py prove-red --check "<cmd>" --break "<cmd>" --restore "<cmd>" \\
        --expect-change <the file the break edits>

Exit 0 the gate passed, 1 the gate failed and names why, 2 it could not run (missing
file, bad arguments) and measured nothing. Keep 1 and 2 distinct: a bad tree and a bad
invocation are different problems.

Standard library only, so it runs in any environment your port already has.

What this script cannot do: it cannot tell you your parity threshold is scientifically
right, or that your census is honest. It checks that the artifact exists, is filled in,
is reproducible, and that the check guarding it can go red. That is the mechanical half.
The other half is `prove-red`, pointed at your own tests.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)   # keep our labels interleaved with children's output

PLACEHOLDER = re.compile(r"<[a-z][a-z0-9 _/.\-]*>", re.I)
#: Literal angle-bracket names the reference documents require, not slots to fill.
LITERAL_ANGLE = {"<root>"}
#: Angle brackets holding a real clause, e.g. "<x> is <y>", are prose rather than a template slot.
#: A stopword alone is not enough: "<url of the repo>" and "<hash of the checkpoint>" are exactly
#: the holes this check exists to find, and every one of them contains "of" or "the".
NOT_A_PLACEHOLDER = re.compile(r"\b(is|are|was|were|means|becomes|equals|if|then|when)\b", re.I)
DEFERRED_HARD = re.compile(r"\b(TBD|TODO|FIXME|XXX|coming soon"
                           r"|to be (decided|determined|measured|chosen|picked))\b", re.I)
#: Softer phrasings. Only a hole when they are the *value* of something: in a prose paragraph
#: "we will measure the exact boundary in Phase 2" is a plan, and SKILL.md instructs it.
DEFERRED_SOFT = re.compile(
    r"\b(figure (this |it )?out( later)?|decide later"
    r"|we will (decide|pick|choose|look|figure|work)"
    r"|(decide|choose|pick|measure|figure|sort) (this |it |that )?(out )?later"
    r"|pick one (in|during|at) |whatever .{0,30}turns out"
    r"|seems right|good enough for now|not sure yet|no idea"
    r")\b", re.I)
#: A cell that says nothing. "lots", "some", "a few": not a value.
VAGUE_CELL = re.compile(
    r"^(lots?|some|many|a few|several|various|misc|assorted|big|small|large|huge|tiny"
    r"|maybe|probably|roughly|approx|\?+|-+|\.+|etc\.?|see above|as needed|standard|default)$", re.I)
#: The escape hatch. A blank cell is indistinguishable from an unfinished one, so it stays a
#: failure, but "nothing here yet" is a real answer in early phases and needs a way to be said.
#: Deliberately explicit: you have to type it, so the record shows you decided rather than forgot.
NOT_YET = re.compile(r"^(none yet|not yet|none needed|n/a, .+|unconfirmed(,.*)?|"
                     r"first pass, no fix needed|no fix needed)$", re.I)
#: A named golden is a file or a path, not a promise. Accept an explicit exactness note too.
GOLDEN_OK = re.compile(r"[\w/.-]+\.(pt|pth|npz|npy|json|safetensors|h5|pkl)\b"
                       r"|[\w.-]+/[\w.-]+/[\w./-]+"
                       r"|\bnone needed\b", re.I)
#: A path-shaped answer that is really a shrug. "n/a" reads as a directory to a loose regex.
GOLDEN_NOT_OK = re.compile(r"^(n/?a|na|none|-+|\?+|tbd/.*|todo/.*"
                           r"|to/be/[a-z]+|[a-z]+/to/[a-z]+)$", re.I)
#: Columns whose whole purpose is to record that something failed on purpose.
MUST_BE_YES = re.compile(r"went red|goes red|red\?|fails\?|did it fail", re.I)
#: "pass" is not an answer here: the column records that the test FAILED when broken. Matched as
#: a PREFIX, so "yes - red at 0.712" and "red on commit abc123" are answers, not violations.
AFFIRMATIVE = re.compile(r"^(yes|y|red|true|✓|✔|confirmed|went red|fail(s|ed)( as expected)?)\b",
                         re.I)
#: ...but a bare negative verdict as its own clause overrides the prefix, so "red herring, no" is
#: rejected. Scanned clause by clause, not across the whole cell: "yes, red at 0.31 (not a fluke)"
#: and "yes, went from pass to fail" are answers, and a substring scan rejected both.
NEGATIVE_VERDICT = re.compile(r"^(no|nope|not yet|never|didn'?t|did not|pass(ed)?|green|"
                              r"unknown|n/?a|none)$", re.I)


def negated(cell: str) -> bool:
    return any(NEGATIVE_VERDICT.fullmatch(c.strip(" .!")) for c in re.split(r"[,;()]", cell))
#: A threshold is a number, or a named exactness criterion.
#: A digit alone is not a threshold: "fp32" and "v2" have one. Want a real number, or a
#: named exactness criterion.
THRESHOLD_OK = re.compile(r"[0-9]*\.[0-9]+|[0-9]+e-?[0-9]+|\b[0-9]+ ?%"
                          r"|\bbit[- ]exact\b|\bmaxdiff ?(of )?0\b|\bexact(ly)? 0\b"
                          r"|\bidentical\b|\bbyte[- ]identical\b", re.I)


# --------------------------------------------------------------------------- report


class Table:
    """One markdown pipe table: its header cells and its data rows."""

    def __init__(self, line_no: int, header: list[str]):
        self.line_no = line_no
        self.header = header
        self.rows: list[tuple[int, list[str]]] = []


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?[\s:|-]*-[\s:|-]*\|?", line.strip())) and "-" in line


def parse_tables(text: str) -> list[Table]:
    lines = text.splitlines()
    tables: list[Table] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and _is_separator(lines[i + 1]):
            t = Table(i + 1, _cells(lines[i]))
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                t.rows.append((i + 1, _cells(lines[i])))
                i += 1
            tables.append(t)
            continue
        i += 1
    return tables


def strip_code_blocks(text: str) -> str:
    """Blank out fenced blocks so an example inside one is not read as content."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            out.append("")
            continue
        out.append("" if fenced else line)
    return "\n".join(out)


def strip_inline_code(line: str) -> str:
    """Blank out `code spans`, so a documented filename convention is not read as a hole."""
    return re.sub(r"`[^`]*`", "``", line)


def check_document(path: Path, required_headings: list[str], require_tables: bool) -> list[str]:
    """Generic filled-in-template check. Returns a list of problems."""
    text = path.read_text(encoding="utf-8", errors="replace")
    prose = strip_code_blocks(text)
    problems: list[str] = []

    if not text.strip():
        return [f"{path} is empty."]
    if not any(l.startswith("#") for l in prose.splitlines()):
        problems.append(f"{path} has no headings, so it is not the document this gate checks.")

    headings = [h.strip("# ").strip().lower() for h in prose.splitlines() if h.startswith("#")]
    claimed: set[int] = set()
    for want in required_headings:
        # A distinct heading per requirement. One line naming all ten sections used to satisfy
        # all ten, after which _section() returned the whole document for each of them.
        hit = next((i for i, h in enumerate(headings)
                    if want.lower() in h and i not in claimed), None)
        if hit is None:
            if any(want.lower() in h for h in headings):
                problems.append(f"{want!r} shares a heading with another required section. "
                                "Each one needs its own heading, or the section checks below all "
                                "read the same block of text.")
            else:
                problems.append(f"missing a heading containing {want!r}")
        else:
            claimed.add(hit)

    for n, line in enumerate(prose.splitlines(), 1):
        bare = strip_inline_code(line)
        for m in PLACEHOLDER.finditer(bare):
            if m.group(0).lower() in LITERAL_ANGLE:
                continue                       # a mandated key, e.g. the fixture's <root>
            if NOT_A_PLACEHOLDER.search(m.group(0)):
                continue                       # prose in angle brackets, not a template slot
            problems.append(f"{path}:{n}: unfilled placeholder {m.group(0)!r}")
        m = DEFERRED_HARD.search(line) or DEFERRED_SOFT.search(bare)
        if m:
            problems.append(f"{path}:{n}: deferred entry {m.group(0)!r}, so this is not finished")

    # Which section each line sits in, so a table can be judged by where it is and not only by
    # what its columns are called.
    section_of: dict[int, str] = {}
    current = ""
    for n, line in enumerate(prose.splitlines(), 1):
        if line.startswith("#"):
            current = line.strip("# ").strip().lower()
        section_of[n] = current

    tables = parse_tables(prose)
    if require_tables and not tables:
        problems.append(f"{path}: no filled table anywhere, so nothing here is a record of anything")
    for t in tables:
        in_controls_section = "negative control" in section_of.get(t.line_no, "")
        if not t.rows:
            problems.append(
                f"{path}:{t.line_no}: table [{' | '.join(t.header)}] has no data rows. If it is "
                "genuinely empty at this phase, say so in a row rather than leaving it blank: a "
                "blank table and a forgotten table look identical.")
            continue
        for n, row in t.rows:
            if len(row) < len(t.header):
                missing = t.header[len(row):]
                problems.append(f"{path}:{n}: row stops after {len(row)} of {len(t.header)} "
                                f"columns, so {', '.join(missing)!r} is absent rather than empty")
            blank = [t.header[i] if i < len(t.header) else f"col{i + 1}"
                     for i, c in enumerate(row) if not c]
            if blank:
                problems.append(f"{path}:{n}: empty cell(s) under {', '.join(blank)!r}")
            for i, cell in enumerate(row):
                col = t.header[i] if i < len(t.header) else f"col{i + 1}"
                raw = cell.strip()
                v = raw.strip("*`").strip()
                # A cell holding only decoration is a blank cell wearing a disguise: stripping
                # `*` and backticks left nothing, and every check below is guarded on `v`.
                if raw and not v:
                    problems.append(f"{path}:{n}: {raw!r} under {col!r} is punctuation, not an "
                                    "answer. A cell that survives stripping as empty is empty.")
                    continue
                # "none yet" is an answer in most columns and never in this one: a control you
                # have not run yet is the thing the column exists to make visible.
                if v and NOT_YET.fullmatch(v) and not MUST_BE_YES.search(col):
                    continue                       # an explicit "nothing here yet" is an answer
                if v and DEFERRED_SOFT.search(v):
                    problems.append(f"{path}:{n}: {col!r} says {v!r}, which defers the answer "
                                    "rather than giving it")
                if v and VAGUE_CELL.fullmatch(v):
                    problems.append(f"{path}:{n}: {cell.strip()!r} under {col!r} is not a value")
                # A "did it go red?" column answered "no" is the finding, not a filled cell.
                # Keyed on the column name OR on being the last column of a negative-controls
                # table, because renaming the column used to disable this check entirely.
                is_verdict = MUST_BE_YES.search(col) or (
                    in_controls_section and i == len(t.header) - 1)
                if v and is_verdict and (
                        not AFFIRMATIVE.match(v) or negated(v)):
                    problems.append(
                        f"{path}:{n}: {col!r} says {v!r}. This column records that the test FAILED "
                        "when you broke it, so it has to start with yes, red, true or confirmed, "
                        "and it must not also say no, pass or green. A control that did not fire "
                        "is the finding, not a filled cell.")
    return problems


# ----------------------------------------------------------------------------- plan

def _section(text: str, name: str) -> str | None:
    """The body of the ``## <name>`` section, up to the next heading of the same level."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        # Substring, matching how required headings are checked at :138 exactly. Prefix matching
        # was not enough: "## The Target" satisfies the heading check and missed this one, so the
        # section's own checks vanished silently. Two matchers for one name have to agree.
        stripped = line.strip()
        if stripped.startswith("#") and name.lower() in stripped.lstrip("#").strip().lower():
            body = []
            for nxt in lines[i + 1:]:
                if nxt.startswith("#" * line.count("#", 0, 6)) and nxt.lstrip("#").strip():
                    if len(nxt) - len(nxt.lstrip("#")) <= line.count("#"):
                        break
                body.append(nxt)
            return "\n".join(body)
    return None


PLAN_HEADINGS = ["Reference", "Target", "Module tree", "Axes", "Op inventory", "Control flow",
                 "Host-side pipelines", "Randomness", "Evaluation set", "Risks"]

#: A leaf-first module tree has leaves, the blocks they compose into, and the whole model.
MIN_TREE_ROWS = 3


def check_section(path: Path, text: str, name: str) -> list[str]:
    """A required section has to say something, and its `- Label:` bullets have to be answered.

    Shared by both gates on purpose: the Phase 0 report arm exists to make `$REF_PY` and the
    effort bar findable, and it used to pass with exactly those two bullets blank because only
    the plan gate looked.
    """
    body = _section(text, name)
    if body is None:
        return []                          # already reported as a missing heading
    problems = []
    stripped = strip_code_blocks(body)
    bullets = [l for l in stripped.splitlines() if re.match(r"\s*[-*]\s", l)]
    empty = [l.strip() for l in bullets if re.match(r"\s*[-*]\s*[^:]{1,80}:\s*$", l)]
    if empty:
        problems.append(f"{path}: {name!r} has {len(empty)} unanswered item(s), "
                        f"first is {empty[0]!r}")
    # A table is content too: Randomness is legitimately a table and no prose.
    content = [l for l in stripped.splitlines()
               if l.strip() and not l.startswith("#")
               and not re.fullmatch(r"\|[\s|:-]*\|", l.strip())]
    if not content:
        problems.append(f"{path}: {name!r} is a heading with nothing under it.")
    return problems


def gate_plan(args) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"GATE 2: {path} does not exist. Phase 0 has produced nothing.")
        return 2

    problems = check_document(path, PLAN_HEADINGS, require_tables=True)
    text = strip_code_blocks(path.read_text(encoding="utf-8", errors="replace"))
    tables = parse_tables(text)

    # The module tree is the load-bearing table: no module without a named golden.
    tree = next((t for t in tables
                 if any("module" in h.lower() for h in t.header)
                 and any("golden" in h.lower() for h in t.header)), None)
    if tree is None:
        problems.append(f"{path}: no module-tree table with both a Module and a Golden column. "
                        "Every module needs the golden that will prove it, named, in Phase 0.")
    else:
        cols = {}
        for want in ("module", "golden", "threshold"):
            idx = next((i for i, h in enumerate(tree.header) if want in h.lower()), None)
            if idx is None:
                problems.append(f"{path}:{tree.line_no}: module tree has no {want} column")
            cols[want] = idx

        if len(tree.rows) < MIN_TREE_ROWS:
            problems.append(
                f"{path}:{tree.line_no}: module tree has {len(tree.rows)} row(s). "
                f"Phase 0 is a leaf-first decomposition, so it needs at least {MIN_TREE_ROWS}: "
                "the leaves, the blocks they compose into, and the whole model. One row naming "
                "the whole model is the thing this phase exists to replace.")

        for n, row in tree.rows:
            def cell(key):
                i = cols.get(key)
                return row[i].strip(" *`") if i is not None and i < len(row) else ""
            mod, golden, thresh = cell("module"), cell("golden"), cell("threshold")
            if golden and (GOLDEN_NOT_OK.match(golden) or not GOLDEN_OK.search(golden)):
                problems.append(
                    f"{path}:{n}: module {mod!r} names its golden as {golden!r}, which is not a "
                    "fixture. Name the file the test will load, or say 'none needed' and why.")
            if thresh and re.match(r"^(see|per|as (in|per)|cf\.?)\b", thresh, re.I):
                problems.append(
                    f"{path}:{n}: module {mod!r} has threshold {thresh!r}, which is a pointer, not "
                    "a threshold. Write the number here; the test loads this column, not that "
                    "document.")
            elif thresh and not THRESHOLD_OK.search(thresh):
                problems.append(
                    f"{path}:{n}: module {mod!r} has threshold {thresh!r}, which has no number in "
                    "it. A threshold is a number, or a named exactness criterion like 'maxdiff 0'.")

    # Four required sections are prose, not tables, and nothing checked them: the pinned
    # commit, the checkpoint hash, the CPU command and the nondeterminism note that Phase 1's
    # golden depends on. The gate printed "every section present" over four empty ones.
    for name in ("Reference", "Target", "Control flow", "Host-side pipelines", "Randomness",
                 "Risks"):
        problems += check_section(path, text, name)

    target = _section(text, "target")
    if target is None:
        pass                              # already reported as a missing heading
    else:
        size = next((l for l in target.splitlines() if "size range" in l.lower()), None)
        if size is None:
            problems.append(f"{path}: the Target section has no 'size range' line.")
        elif len(re.findall(r"\d+", size)) < 2:
            problems.append(
                f"{path}: the supported size range reads {size.strip(' -*')!r}. That is one number "
                "or none. State both ends, because Phase 4 runs a ladder across exactly this range "
                "and Phase 2 has to prove the top of it fits.")

    return _verdict("phase 0 plan", path, problems)


def gate_report(args) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"GATE 2: {path} does not exist.")
        return 2
    problems = check_document(path, args.require_heading or [], require_tables=not args.no_tables)

    # A required heading with nothing under it used to pass, because the table check only asked
    # whether the document contained a table *somewhere*. So a perf report with Measured roofs,
    # Op census and Levers all empty, plus one unrelated table, read GATE 0. Every section named
    # on the command line is named because its rows are the evidence.
    text = path.read_text(encoding="utf-8", errors="replace")
    for want in args.require_heading or []:
        problems += check_section(path, text, want)
        if args.no_tables:
            continue                              # this document is prose, --no-tables said so
        body = _section(text, want)
        if body is None:
            continue
        if not [tb for tb in parse_tables(strip_code_blocks(body)) if tb.rows]:
            problems.append(
                f"{path}: {want!r} has no table with data rows under it. The heading is not the "
                "evidence; the rows are. If this section is genuinely prose, pass --no-tables.")
    return _verdict("report", path, problems)


def _verdict(what: str, path: Path, problems: list[str]) -> int:
    if problems:
        print(f"GATE 1: {what} {path} is not finished. {len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"GATE 0: {what} {path} is complete: every section present, every table row filled, "
          "no placeholders, nothing deferred.")
    return 0


# ---------------------------------------------------------------------- determinism


def _sh(cmd: str) -> int:
    """Run through bash -c so the semantics match what a person types."""
    return subprocess.run(["bash", "-c", cmd]).returncode


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_determinism(args) -> int:
    artifacts = [Path(a) for a in args.artifact]
    runs: list[dict[Path, str]] = []
    for a in artifacts:
        if a.is_dir():
            print(f"GATE 2: {a} is a directory. --artifact takes the file whose bytes must match.")
            return 2
    # Move the existing artifacts aside rather than deleting them. A capture is a CPU reference
    # run that can take thousands of seconds, `02` says to gitignore parity_artifacts/, and the
    # commonest way to get here is a --run that never starts (an unexported $SKILL, a typo). If
    # we unlink first and the command then exits 127, the gate has destroyed the thing it exists
    # to protect and reported "measured nothing".
    stash = {}
    for a in artifacts:
        if a.is_file():
            keep = a.with_suffix(a.suffix + ".prove-red-backup")
            a.replace(keep)
            stash[a] = keep

    def _restore_stash():
        for a, keep in stash.items():
            if keep.is_file() and not a.is_file():
                keep.replace(a)
            elif keep.is_file():
                keep.unlink()

    for attempt in (1, 2):
        for a in artifacts:
            if a.is_file():
                a.unlink()
        print(f"--- run {attempt}: {args.run}")
        rc = _sh(args.run)
        if rc != 0:
            _restore_stash()
            print(f"GATE 2: the command exited {rc} on run {attempt}. Nothing was compared, and "
                  "your existing artifact was put back.")
            return 2
        missing = [a for a in artifacts if not a.is_file()]
        if missing:
            _restore_stash()
            print(f"GATE 1: run {attempt} exited 0 but did not write "
                  f"{', '.join(str(m) for m in missing)}. "
                  "An exit code is not an artifact. Your existing artifact was put back.")
            return 1
        runs.append({a: _sha(a) for a in artifacts})

    for keep in stash.values():                # both runs wrote; the backups are stale now
        if keep.is_file():
            keep.unlink()

    differing = [a for a in artifacts if runs[0][a] != runs[1][a]]
    for a in artifacts:
        mark = "DIFFERS" if a in differing else "same"
        print(f"  {mark:8s} {a}  {runs[0][a][:16]}  {runs[1][a][:16]}")
    if differing:
        print(f"GATE 1: {len(differing)} artifact(s) changed between two identical runs. "
              "Pin the seeds, the thread count and the iteration order before going on: "
              "a golden you cannot reproduce cannot prove anything later.")
        return 1
    print(f"GATE 0: {len(artifacts)} artifact(s) byte-identical across two runs.")
    return 0


# ------------------------------------------------------------------------ prove-red


def _fingerprint(paths):
    return {q: (_sha(Path(q)) if Path(q).is_file() else None) for q in paths}


#: Exit codes that mean "the check never ran", not "the check failed". pytest uses 2 for a
#: collection error, 3 internal, 4 usage (the file is not there), 5 nothing collected; the shell
#: uses 126 not-executable and 127 not-found. A break that produces one of these has not shown
#: the check can fail, it has shown the check can be prevented from running, which is the
#: vacuous pass this whole subcommand exists to catch.
CANNOT_RUN = {
    2: "pytest: collection error, or our own gate saying it could not run",
    3: "pytest: internal error",
    4: "pytest: usage error, usually the test file is not where you said",
    5: "pytest: no tests were collected",
    126: "shell: command found but not executable",
    127: "shell: command not found",
}


def gate_prove_red(args) -> int:
    watched = args.expect_change or []
    if not watched and not args.no_expect_change:
        print("GATE 2: pass --expect-change <the file the break edits>.")
        print("  Without it this subcommand cannot tell a real injection from a break that did\n"
              "  nothing, and it will certify a check that only tests whether a file exists:\n"
              "    --check 'test -f model.py' --break 'mv model.py model.bak'\n"
              "  reads exactly like a working gate. That is the vacuous check this tool exists to\n"
              "  catch, so the flag is required rather than advised.")
        print("  If your break genuinely edits nothing on disk (it flips an env var, say), pass\n"
              "  --no-expect-change and say in the report why.")
        return 2
    print(f"--- check, as it stands: {args.check}")
    before = _sh(args.check)
    if before != 0:
        print(f"GATE 2: the check exits {before} before anything was broken. "
              "Fix the check or the code first; there is nothing to prove yet.")
        return 2

    was = _fingerprint(watched)
    print(f"--- break it: {getattr(args, 'break')}")
    if _sh(getattr(args, "break")) != 0:
        print("GATE 2: the break command itself failed, so the fault was never injected.")
        return 2

    # An exit code is not an edit. `sed -i s/pattern/x/` exits 0 when the pattern does not
    # match, and a stale break command is the usual reason this check reads backwards: nothing
    # was injected, the gate stayed green, and the gate got blamed for it.
    if watched:
        now = _fingerprint(watched)
        unchanged = [q for q in watched if was[q] == now[q]]
        if unchanged:
            print(f"GATE 2: the break command exited 0 but did not change "
                  f"{', '.join(unchanged)}. Nothing was injected, so this run says nothing about "
                  "the check. Usually the pattern no longer matches the file.")
            return 2
    # Deleting the thing under test is not injecting a fault into it. The fingerprint above
    # counts a vanished file as "changed", which is true and useless: the check then fails
    # because there is nothing to check.
    if watched:
        gone = [q for q in watched if was[q] is not None and not Path(q).is_file()]
        if gone:
            print(f"GATE 2: the break removed {', '.join(gone)} rather than changing it. A check "
                  "that fails because the file is missing has not been shown to fail on a defect. "
                  "Edit the file, do not move it.")
            return 2

    print(f"--- check, with the fault injected: {args.check}")
    broken = _sh(args.check)

    print(f"--- restore: {args.restore}")
    restore_rc = _sh(args.restore)
    print(f"--- check, restored: {args.check}")
    after = _sh(args.check)

    if restore_rc != 0 or after != 0:
        print(f"GATE 2: restore exited {restore_rc} and the restored check exits {after}. "
              "Your tree may still hold the injected fault. Fix that before reading the result.")
        return 2

    # The rule applied to the break, applied to the restore: an exit code is not an edit. A
    # restore that exits 0 and leaves the file changed would otherwise print GATE 0 and hand
    # back a dirty checkout, with the green coming from a tree nobody meant to ship.
    if watched:
        final = _fingerprint(watched)
        dirty = [q for q in watched if was[q] != final[q]]
        if dirty:
            print(f"GATE 2: the check went red and back to green, but {', '.join(dirty)} no "
                  "longer matches what it was before the break. The restore exited 0 without "
                  "fully undoing the edit, so your tree is dirty and this green is not about "
                  "the tree you started with.")
            return 2
    if broken == 0:
        print("GATE 1: the check stayed green with the fault injected.")
        if not watched:
            print("  First confirm the break actually edited something. A `sed -i` whose pattern no\n"
                  "  longer matches exits 0 having changed nothing, and that reads identically to\n"
                  "  this. Re-run with --expect-change <the file you meant to edit>.")
        print("  If the injection was real, the check is decoration. Find out what it asserts:\n"
              "  existence instead of content, a committed verdict re-read instead of recomputed,\n"
              "  the installed package instead of your checkout, or an inverted exit status.")
        return 1
    expected = args.red_exit
    if expected is not None and broken != expected:
        print(f"GATE 1: with the fault injected the check exits {broken}, and you said its "
              f"failure code is {expected}. Whatever {broken} means here, it is not this check "
              "reporting a defect.")
        return 1
    if expected is None and broken in CANNOT_RUN:
        print(f"GATE 2: with the fault injected the check exits {broken}, which means it did not "
              f"run: {CANNOT_RUN[broken]}.")
        print("  A check that cannot run is not a check that failed. Injecting a fault that stops\n"
              "  the check from starting proves nothing about what it asserts, and this is the\n"
              "  most common way a prove-red run reads green while measuring nothing.")
        print("  Break the code under test, not the harness. If this exit code really is how your\n"
              f"  check reports a defect, say so with --red-exit {broken}.")
        return 2
    print(f"GATE 0: green (0) -> fault injected -> red ({broken}) -> restored -> green (0). "
          "This check can fail, so its green means something.")
    return 0


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="Phase 0: PORT_PLAN.md is complete and every module has a golden")
    p.add_argument("path", nargs="?", default="PORT_PLAN.md")
    p.set_defaults(fn=gate_plan)

    p = sub.add_parser("report", help="a filled-in report: no placeholders, no empty table rows")
    p.add_argument("path")
    p.add_argument("--require-heading", action="append", metavar="TEXT")
    p.add_argument("--no-tables", action="store_true", help="do not require any table")
    p.set_defaults(fn=gate_report)

    p = sub.add_parser("determinism", help="run a command twice, require byte-identical artifacts")
    p.add_argument("--run", required=True, metavar="CMD")
    p.add_argument("--artifact", required=True, action="append", metavar="PATH")
    p.set_defaults(fn=gate_determinism)

    p = sub.add_parser("prove-red", help="prove a check can fail: green, break, red, restore, green")
    p.add_argument("--check", required=True, metavar="CMD")
    p.add_argument("--break", required=True, metavar="CMD", dest="break")
    p.add_argument("--restore", required=True, metavar="CMD")
    p.add_argument("--no-expect-change", action="store_true",
                   help="the break edits nothing on disk, so skip the did-it-change check. Use "
                        "only when that is true; without it --expect-change is required.")
    p.add_argument("--red-exit", type=int, metavar="N",
                   help="the exit code this check uses to report a defect, when it is not the "
                        "usual 1. Without it, an exit code that means 'could not run' (2-5, 126, "
                        "127) is refused rather than counted as red.")
    p.add_argument("--expect-change", action="append", metavar="PATH",
                   help="a file the break must actually modify; refuses with exit 2 if it did not. "
                        "Use it whenever the break is a sed or a patch.")
    p.set_defaults(fn=gate_prove_red)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
