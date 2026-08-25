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
#: Prose words inside angle brackets mean it is a sentence, not an unfilled template slot.
NOT_A_PLACEHOLDER = re.compile(r"\b(and|or|not|if|then|is|in|to|of|the)\b", re.I)
DEFERRED = re.compile(
    r"\b(TBD|TODO|FIXME|XXX"
    r"|figure (this |it )?out( later)?|decide later|to be decided|coming soon"
    r"|we will (decide|pick|choose|look|figure|work|measure|find)"
    r"|(will|to) be (decided|determined|measured|chosen|picked)"
    r"|pick one (in|during|at) |whatever .{0,30}turns out"
    r"|(decide|choose|pick|measure|figure|sort) (this |it |that )?(out )?later"
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
GOLDEN_NOT_OK = re.compile(r"^(n/?a|na|none|-+|\?+|tbd/.*|todo/.*)$", re.I)
#: Columns whose whole purpose is to record that something failed on purpose.
MUST_BE_YES = re.compile(r"went red|goes red|red\?|fails\?|did it fail", re.I)
#: "pass" is not an answer here: the column records that the test FAILED when broken.
AFFIRMATIVE = re.compile(r"(yes|y|red|true|✓|✔|confirmed|went red|failed as expected)"
                         r"(\s*[,(].*)?", re.I)
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

    headings = {h.strip("# ").strip().lower() for h in prose.splitlines() if h.startswith("#")}
    for want in required_headings:
        if not any(want.lower() in h for h in headings):
            problems.append(f"missing a heading containing {want!r}")

    for n, line in enumerate(prose.splitlines(), 1):
        bare = strip_inline_code(line)
        for m in PLACEHOLDER.finditer(bare):
            if NOT_A_PLACEHOLDER.search(m.group(0)):
                continue                       # prose in angle brackets, not a template slot
            problems.append(f"{path}:{n}: unfilled placeholder {m.group(0)!r}")
        m = DEFERRED.search(bare)
        if m:
            problems.append(f"{path}:{n}: deferred entry {m.group(0)!r}, so this is not finished")

    tables = parse_tables(prose)
    if require_tables and not tables:
        problems.append(f"{path}: no filled table anywhere, so nothing here is a record of anything")
    for t in tables:
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
                v = cell.strip().strip("*`").strip()
                if v and NOT_YET.fullmatch(v):
                    continue                       # an explicit "nothing here yet" is an answer
                if v and VAGUE_CELL.fullmatch(v):
                    problems.append(f"{path}:{n}: {cell.strip()!r} under {col!r} is not a value")
                # A "did it go red?" column answered "no" is the finding, not a filled cell.
                if v and MUST_BE_YES.search(col) and not AFFIRMATIVE.fullmatch(v):
                    problems.append(
                        f"{path}:{n}: {col!r} says {v!r}. A negative control that did not go red is "
                        "not a control: the test it guards has not been shown able to fail.")
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
            if thresh and not THRESHOLD_OK.search(thresh):
                problems.append(
                    f"{path}:{n}: module {mod!r} has threshold {thresh!r}, which has no number in "
                    "it. A threshold is a number, or a named exactness criterion like 'maxdiff 0'.")

    # Four required sections are prose, not tables, and nothing checked them: the pinned
    # commit, the checkpoint hash, the CPU command and the nondeterminism note that Phase 1's
    # golden depends on. The gate printed "every section present" over four empty ones.
    for name in ("Reference", "Target", "Control flow", "Host-side pipelines", "Randomness",
                 "Risks"):
        body = _section(text, name)
        if body is None:
            continue                      # already reported as a missing heading
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
            problems.append(f"{path}: {name!r} is empty. It is one of the sections this gate "
                            "says is present, so it has to say something.")

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
        body = _section(text, want)
        if body is None:
            continue                              # already reported as a missing heading
        stripped = strip_code_blocks(body)
        content = [l for l in stripped.splitlines()
                   if l.strip() and not l.startswith("#")
                   and not re.fullmatch(r"\|[\s|:-]*\|", l.strip())]
        if not content:
            problems.append(f"{path}: {want!r} is a heading with nothing under it.")
            continue
        if args.no_tables:
            continue                              # this document is prose, --no-tables said so
        if not [tb for tb in parse_tables(stripped) if tb.rows]:
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
    for attempt in (1, 2):
        for a in artifacts:
            if a.is_file():
                a.unlink()
        print(f"--- run {attempt}: {args.run}")
        rc = _sh(args.run)
        if rc != 0:
            print(f"GATE 2: the command exited {rc} on run {attempt}. Nothing was compared.")
            return 2
        missing = [a for a in artifacts if not a.is_file()]
        if missing:
            print(f"GATE 1: run {attempt} exited 0 but did not write "
                  f"{', '.join(str(m) for m in missing)}. "
                  "An exit code is not an artifact.")
            return 1
        runs.append({a: _sha(a) for a in artifacts})

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


def gate_prove_red(args) -> int:
    watched = args.expect_change or []
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
    p.add_argument("--expect-change", action="append", metavar="PATH",
                   help="a file the break must actually modify; refuses with exit 2 if it did not. "
                        "Use it whenever the break is a sed or a patch.")
    p.set_defaults(fn=gate_prove_red)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
