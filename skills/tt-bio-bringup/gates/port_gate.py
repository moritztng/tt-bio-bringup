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
import os
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)   # keep our labels interleaved with children's output

PLACEHOLDER = re.compile(r"<[a-z][a-z0-9 _/.,'\-]*>", re.I)
#: Literal angle-bracket names the reference documents require, not slots to fill.
#: `<br>` is the only way to get a line break inside a GFM table cell, and every template here is
#: a table, so a reader who uses one was told their document had a hole it did not have.
LITERAL_ANGLE = {"<root>", "<br>", "<br/>", "<br />", "<sub>", "</sub>", "<sup>", "</sup>",
                 "<kbd>", "</kbd>", "<details>", "</details>", "<summary>", "</summary>",
                 "<code>", "</code>", "<b>", "</b>", "<i>", "</i>", "<em>", "</em>"}
#: Angle brackets holding a real clause, e.g. "<x> is <y>", are prose rather than a template slot.
#: A stopword alone is not enough: "<url of the repo>" and "<hash of the checkpoint>" are exactly
#: the holes this check exists to find, and every one of them contains "of" or "the".
NOT_A_PLACEHOLDER = re.compile(r"\b(is|are|was|were|means|becomes|equals)\b\s+\S+\s+\S+", re.I)
DEFERRED_HARD = re.compile(r"\b(TBD|TODO|FIXME|XXX|coming soon)\b", re.I)
#: Softer phrasings. Only a hole when they are the *value* of something: in a prose paragraph
#: "we will measure the exact boundary in Phase 2" is a plan, and SKILL.md instructs it.
DEFERRED_SOFT = re.compile(
    r"\b(to be (decided|determined|measured|chosen|picked)\b(?!.{0,40}\bphase\b)"
    r"|figure (this |it )?out( later)?|decide later"
    # "We will choose bf16 for the trunk" names the answer, so it is a decision, not a deferral.
    # The lookahead used to forgive only a destination ("in Phase 2"), which meant a plan written
    # in the future tense -- the tense a plan is written in -- read as unfinished. This arm now
    # fires only when the verb has no object at all, or a placeholder one: "we will decide",
    # "we will choose something", "we will figure it out later".
    r"|we will (decide|pick|choose|look|figure|work)\b"
    r"(?=\s*(later|then|soon|eventually|next time)\b|\s*[.,;)]|\s*$"
    r"|\s+(it|this|that|something|one|some)\b\s*(out\b)?\s*"
    r"(later|then|soon|[.,;)]|$))"
    r"|(decide|choose|pick|measure|figure|sort) (this |it |that )?(out )?later"
    r"|pick one (in|during|at) |whatever .{0,30}turns out"
    r"|seems right|good enough for now|not sure yet|no idea"
    r")\b", re.I)
#: A cell that says nothing. "lots", "some", "a few": not a value.
VAGUE_CELL = re.compile(
    r"^(n/?a|_+|lots?|some|many|a few|several|various|misc|assorted|big|small|large|huge|tiny"
    r"|maybe|probably|roughly|approx|\?+|[-\u2010-\u2015\u2212]+|\.+|etc\.?|see above|as needed"
    r"|standard|default)$", re.I)
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
GOLDEN_NOT_OK = re.compile(r"^(n/?a|na|none|-+|\?+"
                           r"|(tbd|todo|not|to|ask|later|unknown|maybe|decide|pick)/.*"
                           r"|.*/(yet|decided|known|later|tbd|todo|someone|author)(/.*)?)$", re.I)
#: Columns whose whole purpose is to record that something failed on purpose. Matched by NAME
#: only. Position is not a fallback: taking the last column meant renaming the verdict header
#: pointed the check at whatever came last, which both passed a table saying every control stayed
#: green and rejected an honest one that had a Notes column after the verdict.
MUST_BE_YES = re.compile(r"went red|goes red|red\?|fails?\?|did it fail|detected\?|fired\?|"
                         r"caught\?", re.I)
#: "Verdict" only means "did the control fire" under Negative controls. Read as a verdict column
#: everywhere, it rejected honest work: a component-parity table with a natural Verdict column
#: saying "pass" was told the column "records that the test FAILED when you broke it".
VERDICT_WORD = re.compile(r"verdict", re.I)


def verdict_column(header: str, in_controls: bool) -> bool:
    return bool(MUST_BE_YES.search(header) or (in_controls and VERDICT_WORD.search(header)))
#: "pass" is not an answer here: the column records that the test FAILED when broken. Matched as
#: a PREFIX, so "yes - red at 0.712" and "red on commit abc123" are answers, not violations.
#: The trailing \b is why ✓ and ✔ never matched: a word boundary after a non-word character needs
#: a word character next, so "✓ red" and a bare "✓" were both refused by a regex advertising them.
AFFIRMATIVE = re.compile(r"^(?:yes|y|red|true|confirmed|went red|fail(?:s|ed)(?: as expected)?)\b"
                         r"|^[✓✔]", re.I)
#: ...but a bare negative verdict as its own clause overrides the prefix, so "red herring, no" is
#: rejected. Scanned clause by clause, not across the whole cell: "yes, red at 0.31 (not a fluke)"
#: and "yes, went from pass to fail" are answers, and a substring scan rejected both.
NEGATIVE_VERDICT = re.compile(r"^(no|nope|not yet|never|didn'?t|did not|pass(ed)?|green|"
                              r"unknown|n/?a|none)$", re.I)
#: The verdict column is a VERDICT, not a note, and it is checked as a closed field rather than
#: parsed as English. Two adversarial rounds tuned a prose classifier here and each traded one
#: error class for the other: the round that stopped accepting "yes, the gate was green" started
#: rejecting "yes, red. Before the break it was green", and the round that fixed that started
#: accepting "yes, the gate passed; maxdiff 3e-4". Measured on the last set, both directions were
#: wrong on every input: 5 of 5 honest verdicts rejected, 6 of 6 dishonest ones accepted.
#:
#: A false positive is the worse of the two. A reader whose honest, more detailed verdict is
#: refused learns to write the bare word "yes", which every version of this check accepts.
#:
#: So the cell is capped instead. Inside 40 characters there is no room for a subordinate clause
#: about the restore, the clean tree or the other test, which is where every false positive came
#: from, and the word list below becomes decidable. Anything longer is not judged for truth at
#: all: it is returned with "move the detail to another column", which is an instruction the
#: reader can follow rather than an accusation the reader knows to be false.
VERDICT_MAX = 40
#: Blacklisting negative words did not work either: capping the cell made the evasions shorter,
#: not impossible. "yes, no fault injected", "yes (not run yet)", "red already, break not needed"
#: and "y, 0 of 3 controls fired" are all inside 40 characters and all passed a 12-word list.
#: So the field is a WHITELIST now. An affirmative, then only numbers and a small set of words
#: that measure or name a transition. Anything else is not called a lie, it is returned with
#: "put it in a Notes column", which is an instruction rather than an accusation.
VERDICT_WORD_OK = re.compile(
    r"^(?:red|green|pass(?:ing|ed|es)?|fail(?:s|ed|ing|ure)?|"
    r"pcc|maxdiff|rmsd|mae|diff|delta|tol|atol|rtol|sigma|abs|rel|exit|code|"
    r"at|to|from|on|in|for|of|vs|both|all|each|and|then|now|commit|seed|seeds|"
    r"step|steps|row|rows|test|tests|case|cases|run|runs|"
    r"went|fell|drop(?:ped|s)?|rose|jumped|nan|inf|"
    r"a|ang|angstrom|nm|deg|ms|us|ns|s|kb|mb|gb|bits?|bytes?|ulp"
    r")$", re.I)
#: Zero means "no difference" for these and nothing of the kind for pcc, a seed or a step index.
DIFFERENCE_METRIC = {"maxdiff", "diff", "delta", "rmsd", "mae", "abs", "rel", "tol", "atol",
                     "rtol", "exit", "code"}
#: One means "the two sides agree" for these, and zero means they have nothing in common.
AGREEMENT_METRIC = {"pcc", "corr", "cosine"}
#: A zero after one of these is an ordinal, not a measurement.
INDEX_WORD = {"seed", "seeds", "step", "steps", "row", "rows", "run", "runs", "case", "cases",
              "test", "tests", "commit", "block", "layer", "index", "idx"}



#: A test id, a path or a symbol. Naming the test that went red is detail worth having, and an
#: identifier is distinguishable from prose: it carries an underscore, a separator or a file
#: extension, which "injected", "skipped" and "needed" do not.
#: A bare underscore is not enough: "no_fault_injected" and "exit_code_0" are prose wearing an
#: identifier's coat, and they passed. An identifier needs a separator prose does not use.
VERDICT_CODE_OK = re.compile(r"[/]|::|\.(?:py|pt|pth|json|npz|npy|md|sh|cpp|h)$|^`.+`$"
                             r"|^\w+(?:\.\w+){1,}$", re.I)
#: A number, a scientific-notation number, a percentage, or a commit-shaped hash.
VERDICT_NUM_OK = re.compile(r"^(?:[0-9][0-9a-fx.eE+_-]*%?|[0-9a-f]{6,}|[<>=~+-]|%)$", re.I)
#: Naming the transition is the most informative thing a verdict can do, so "green -> red" and
#: "went from pass to fail" are answers. "red -> red" is not: if the cell mentions a green state
#: at all, a red one has to come after it.
GREENISH = re.compile(r"\b(green|pass(?:ing|ed|es)?)\b", re.I)
REDDISH = re.compile(r"\b(red|fail(?:s|ed|ing|ure)?)\b", re.I)


def verdict_problem(cell: str) -> str | None:
    """None if this reads as "the control fired", else why it does not."""
    if len(cell) > VERDICT_MAX:
        return (f"holds {len(cell)} characters. This column is a verdict, not a note: keep it "
                f"under {VERDICT_MAX} and put the explanation in a column of its own. A sentence "
                "here is read by a person and checked by nobody.")
    if not AFFIRMATIVE.match(cell):
        return ("does not start with an affirmative. This column records that the test FAILED "
                "when you broke it, so it has to start with yes, red, true or confirmed.")
    rest = AFFIRMATIVE.sub("", cell, count=1)
    # Tokens are validated across the whole cell; VALUES are judged only on the segment after the
    # last transition marker. "yes, pcc 1.0 -> 0.31" is the most informative verdict a numeric
    # control can give, and the left side of an honest one is always the agreeing value: a
    # bit-exact golden starts at maxdiff 0, a matched module at pcc 1.0, a passing test at exit 0.
    # Judging every number in the cell condemned exactly those, telling the author their proof was
    # the thing it disproves.
    tail = re.split(r"->|\u2192|\bto\b", rest)[-1]
    carried = None
    for w in re.findall(r"[A-Za-z_]+", rest[:len(rest) - len(tail)]):
        if w.lower() in DIFFERENCE_METRIC or w.lower() in AGREEMENT_METRIC:
            carried = w.lower()
    tail_start = len(rest) - len(tail)
    nums: list[tuple[str | None, float, int]] = []
    prev: str | None = None
    for m in re.finditer(r"[^\s,;:()\[\]/]+", re.sub(r"->|\u2192", " ", rest)):
        tok = m.group(0).strip(".!\u2013\u2014")
        if not tok:
            continue
        # Classify before measuring. Scanning the whole cell for digits read the 0 out of
        # `blocks.0.ffn` and called the verdict a zero.
        if VERDICT_NUM_OK.match(tok):
            try:
                nums.append((prev, float(tok.rstrip("%")), m.start()))
            except ValueError:
                try:
                    # 0x0 is a zero. Swallowing the parse failure let it through the rule below.
                    nums.append((prev, float(int(tok, 0)), m.start()))
                except ValueError:
                    pass
            prev = tok
            continue
        # After the number test, or "1.0" reads as a dotted identifier and never reaches the
        # rules below. Only a token that is not a number can be a path.
        if VERDICT_CODE_OK.search(tok):
            continue
        if VERDICT_WORD_OK.match(tok):
            prev = tok
            continue
        return (f"contains {tok!r}. Past the affirmative this column takes numbers and words that "
                "measure or name the transition (pcc, maxdiff, red, green -> red, at, commit). "
                "Anything you want to say in a sentence goes in a Notes column, where a person "
                "will read it, instead of here, where this gate cannot check it.")
    # "red -> red" is a transition from nothing: it says the control was already failing.
    # Searched in `cell`, not `rest`: AFFIRMATIVE had already eaten the leading "red", so a bare
    # "red -> red" left only " -> red" and the check could not see the pair it exists to catch.
    if re.search(r"\b(?:red|fail(?:s|ed|ing|ure)?)\b\s*(?:->|\u2192|to)\s*"
                 r"\b(?:red|fail(?:s|ed|ing|ure)?)\b", cell, re.I):
        return ("names the same state on both sides of the transition, so nothing moved. A control "
                "that was already red before the break proves nothing about the break.")
    # A number is only evidence if it moved, but only for metrics where zero means "no
    # difference". A blanket zero rule condemned "yes, pcc 0.0", which is MAXIMAL divergence and
    # the loudest possible red, and "yes, red at seed 0", where the 0 is an index. Which it is
    # depends on the word in front of it.
    tailnums = [(p, n) for p, n, at in nums if at >= tail_start]
    if not tailnums:
        # No number after the arrow. Either there is no arrow, in which case judge what there is,
        # or the broken run produced something unnumbered. "yes, maxdiff 0 -> nan" is the loudest
        # possible red, and falling back to the last number in the cell judged the PRE-break 0 and
        # told its author the fault changed nothing.
        if tail_start > 0 and re.search(r"\S", tail):
            tailnums = []
        else:
            tailnums = [(p, n) for p, n, _ in nums]
    for prev, n in tailnums:
        # `carried` is the metric named before the arrow. In "pcc 0.99 -> 1.0" the token directly
        # before 1.0 is 0.99, not a metric, so without the fallback the value after the transition
        # was judged with no idea what it measured.
        word = (prev or "").lower()
        if word not in DIFFERENCE_METRIC and word not in AGREEMENT_METRIC and word not in INDEX_WORD:
            word = carried or word
        # Zero is the control not firing everywhere EXCEPT as a PCC (where 0 is total divergence,
        # the loudest red there is) and as an index (seed 0, step 0, row 0). Anything else --
        # "maxdiff 0", "exit code 0", "0 of 3 runs", "fail 0", a bare 0 -- says nothing moved.
        if n == 0.0 and word not in AGREEMENT_METRIC and word not in INDEX_WORD:
            return ("names a zero as the value the broken run produced. A zero difference, a zero "
                    "exit code and a zero count all say the injected fault changed nothing. If "
                    "the zero is an index, put the word in front of it (seed 0, step 0, row 0).")
        if n == 1.0 and word in AGREEMENT_METRIC:
            return (f"ends at {word} 1, which is the two sides agreeing, and that is what a "
                    "control that did not fire looks like. Record the value the broken run "
                    "produced.")
    g, r = GREENISH.search(rest), REDDISH.search(rest)
    if g and not (r and r.start() > g.start()):
        return ("names a green or passing state with no red one after it. A control that did not "
                "fire is the finding, not a filled cell.")
    if any(NEGATIVE_VERDICT.fullmatch(c.strip(" .!:")) for c in re.split(r"[,;:()]", cell)):
        return "reads as a negative verdict."
    return None


#: A threshold is a number, or a named exactness criterion.
#: A digit alone is not a threshold: "fp32" and "v2" have one. Want a real number, or a
#: named exactness criterion.
THRESHOLD_OK = re.compile(r"[0-9]*\.[0-9]+|[0-9]+e-?[0-9]+|\b[0-9]+ ?%|^\s*[0-9]+\s*$"
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
    r"""Split a row into cells the way a renderer does.

    ``\|`` is a literal pipe inside a cell, not a separator. Splitting on every ``|`` shifted
    every cell after the escape one place left, so the gate read a different column from the one
    a reader sees: a verdict of "no" slid out of the Went red? column and the row passed.
    """
    body = line.strip()
    body = re.sub(r"^\|", "", body)
    body = re.sub(r"(?<!\\)\|$", "", body)
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", body)]


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?[\s:|-]*-[\s:|-]*\|?", line.strip())) and "-" in line


def _is_row(line: str) -> bool:
    """A table row. GFM makes the leading and trailing pipes optional, and renderers agree.

    Requiring a leading `|` meant a row written without one ended the table early and every
    check below simply never saw it: a control that stayed green, a module with no golden, a
    threshold that is a pointer. It rendered as an ordinary row the whole time.
    """
    s = line.strip()
    return bool(s) and "|" in s and not s.startswith("#") and not s.startswith("```")


def parse_tables(text: str) -> list[Table]:
    lines = text.splitlines()
    tables: list[Table] = []
    i = 0
    while i < len(lines):
        if _is_row(lines[i]) and i + 1 < len(lines) and _is_separator(lines[i + 1]):
            t = Table(i + 1, _cells(lines[i]))
            i += 2
            while i < len(lines) and _is_row(lines[i]) and not _is_separator(lines[i]):
                if i + 1 < len(lines) and _is_separator(lines[i + 1]):
                    # A row followed by a separator is the header of a NEW table, not a data row
                    # of this one. Without this break the second header was eaten as a row, its
                    # separator stopped the loop, and every row after it belonged to no table at
                    # all: a second verdict table appended directly under the first was read by
                    # nobody, and a 'no' in it passed the gate.
                    break
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

    # Hard deferral tokens are checked on the WHOLE file, fenced blocks included. Everything
    # else reads `prose`, because an example inside a fence is not a hole; but a plan that
    # parks "TODO, to be decided" in a code block has parked it all the same.
    for n, line in enumerate(text.splitlines(), 1):
        m = DEFERRED_HARD.search(line)
        if m:
            problems.append(f"{path}:{n}: deferred entry {m.group(0)!r}, so this is not finished")

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
        m = DEFERRED_SOFT.search(bare)
        if m:
            problems.append(f"{path}:{n}: deferred entry {m.group(0)!r}, so this is not finished")

    # Which section each line sits in, so a table can be judged by where it is and not only by
    # what its columns are called.
    # Every heading in scope at each line, not only the nearest one: an `###` under
    # `## Negative controls` used to move the section off and switch the verdict check off.
    section_of: dict[int, str] = {}
    stack: list[tuple[int, str]] = []
    for n, line in enumerate(prose.splitlines(), 1):
        s = line.strip()
        if s.startswith("#") and s.lstrip("#").strip():
            depth = len(s) - len(s.lstrip("#"))
            while stack and stack[-1][0] >= depth:
                stack.pop()
            stack.append((depth, s.lstrip("#").strip().lower()))
        section_of[n] = " / ".join(h for _, h in stack)

    tables = parse_tables(prose)
    if require_tables and not tables:
        problems.append(f"{path}: no filled table anywhere, so nothing here is a record of anything")
    for t in tables:
        in_controls_section = "negative control" in section_of.get(t.line_no, "")
        if in_controls_section and t.rows and not any(
                verdict_column(h, in_controls_section) for h in t.header):
            problems.append(
                f"{path}:{t.line_no}: the table under Negative controls has no column this gate "
                f"can read as the verdict. Headers are [{' | '.join(t.header)}]. Name one of them "
                "'Went red?', or 'Detected?', or 'Verdict'. Guessing by position is how a table "
                "saying every control stayed green got certified.")
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
            elif len(row) > len(t.header):
                # A renderer drops the surplus, so a value parked there is invisible on the page
                # and invisible to a reviewer, while still sitting in the file.
                problems.append(f"{path}:{n}: row has {len(row)} cells against {len(t.header)} "
                                f"columns. The extra {', '.join(row[len(t.header):])!r} is "
                                "dropped when this renders, so nobody reading the document sees "
                                "it. Escape a literal pipe as \\| .")
            blank = [t.header[i] if i < len(t.header) else f"col{i + 1}"
                     for i, c in enumerate(row) if not c]
            if blank:
                problems.append(f"{path}:{n}: empty cell(s) under {', '.join(blank)!r}")
            for i, cell in enumerate(row):
                col = t.header[i] if i < len(t.header) else f"col{i + 1}"
                raw = cell.strip()
                # Zero-width space, word joiner and BOM are invisible to a reader and were a
                # filled value to this check: a cell holding only U+200B passed.
                v = re.sub(r"[\u00ad\u200b\u200c\u200d\u2060\ufeff]|<!--.*?-->|&nbsp;",
                           "", raw, flags=re.I | re.S).strip("*`").strip()
                # A cell holding only decoration is a blank cell wearing a disguise: stripping
                # `*` and backticks left nothing, and every check below is guarded on `v`.
                if raw and not v:
                    problems.append(f"{path}:{n}: {raw!r} under {col!r} is punctuation, not an "
                                    "answer. A cell that survives stripping as empty is empty.")
                    continue
                # Which cell carries the verdict, decided before any escape hatch applies.
                is_verdict = verdict_column(col, in_controls_section)
                # "none yet" is an answer in most columns and never in this one: a control you
                # have not run yet is the thing the column exists to make visible. This has to be
                # tested after is_verdict, or renaming the column re-opens the escape.
                if v and NOT_YET.fullmatch(v) and not is_verdict:
                    continue                       # an explicit "nothing here yet" is an answer
                if v and DEFERRED_SOFT.search(v):
                    problems.append(f"{path}:{n}: {col!r} says {v!r}, which defers the answer "
                                    "rather than giving it")
                if v and VAGUE_CELL.fullmatch(v):
                    problems.append(f"{path}:{n}: {cell.strip()!r} under {col!r} is not a value")
                # A "did it go red?" cell answered "no" is the finding, not a filled cell.
                if v and is_verdict:
                    why = verdict_problem(v)
                    if why:
                        problems.append(f"{path}:{n}: {col!r} {why}")
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
            depth = len(stripped) - len(stripped.lstrip("#"))
            body = []
            for nxt in lines[i + 1:]:
                s = nxt.strip()
                if s.startswith("#") and s.lstrip("#").strip():
                    # Any heading at this depth or shallower ends the section. Testing only
                    # same-depth folded a following "# Appendix" into the body, so a table there
                    # satisfied this section's checks.
                    if len(s) - len(s.lstrip("#")) <= depth:
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
    # A numbered item with nothing after the number is an empty item, and "Risks, ranked" is a
    # numbered list, so "1." and "2." alone used to count as content for the whole section.
    numbered_empty = [l.strip() for l in stripped.splitlines()
                      if re.match(r"\s*\d+[.)]\s*$", l)]
    if numbered_empty:
        problems.append(f"{path}: {name!r} has {len(numbered_empty)} numbered item(s) with "
                        "nothing after the number.")
    # No length cap on the label. An 80-character cap exempted the two PORT_STATE bullets this
    # check exists for, at 92 and 91 characters, and nothing said so.
    # "- Label:" with nothing after it, or with only a dash, underscore or ellipsis after it.
    # Anchored on the LAST colon, not the first: a label that itself contains a colon, which is
    # ordinary in "- The interpreter ($REF_PY for x, ./env/bin/python3 for y):", used to escape.
    # Emphasis around the label is invisible to a reader and used to be invisible to this check:
    # "- **Pinned commit:**" ends in '**', so the pattern did not reach the end of the line and
    # the bullet read as answered. Every doc in this repo writes labels that way. Trailing
    # emphasis, backticks and zero-width characters are stripped before matching.
    def _bare(line: str) -> str:
        # An HTML comment and an &nbsp; render as nothing, so a bullet "answered" with either is
        # blank on the page. The comment strip further down runs too late to help here, and the
        # dash class below was ASCII plus two of the six unicode dashes: U+00AD renders as
        # nothing at all and U+2212 renders exactly like a hyphen.
        line = re.sub(r"<!--.*?-->", "", line, flags=re.S)
        line = re.sub(r"&nbsp;|&#160;|&#xa0;", "", line, flags=re.I)
        # Emphasis is stripped EVERYWHERE inside the bullet, not only at end of line: with a bold
        # label the closing ** sits between the colon and the answer ("- **Commit:** \u2015"), so a
        # trailing-only strip left a '*' where the match needed whitespace and the bullet read as
        # answered. The list MARKER is split off first and put back untouched, because '*' is a
        # legal bullet marker: stripping it globally deleted the marker, the empty-bullet pattern
        # then failed to match its own input, and every unanswered bullet in an asterisk-marked
        # list went unreported. The blank template lost 2 of its 15 findings that way.
        m = re.match(r"(\s*[-*]\s+)(.*)", line)
        if not m:
            return line
        marker, body = m.group(1), m.group(2)
        body = re.sub(r"[*`]", "", body)
        body = re.sub(r"[_\u00ad\u200b-\u200d\u2060\ufeff\s]+$", "", body)
        return marker + body

    empty = [l.strip() for l in bullets
             if re.match(r"\s*[-*]\s+.*:\s*([-_.\u00ad\u2010-\u2015\u2212]+|\.{2,})?\s*$",
                         _bare(l))]
    if empty:
        problems.append(f"{path}: {name!r} has {len(empty)} unanswered item(s), "
                        f"first is {empty[0]!r}")
    # A table is content too: Randomness is legitimately a table and no prose. An HTML comment
    # is not: "<!-- deliberately left blank -->" is a blank section wearing a note.
    stripped = re.sub(r"<!--.*?-->", "", stripped, flags=re.S)
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
    candidates = [t for t in tables
                  if any("module" in h.lower() for h in t.header)
                  and any("golden" in h.lower() for h in t.header)]
    if len(candidates) > 1:
        problems.append(
            f"{path}: {len(candidates)} tables have both a Module and a Golden column, at lines "
            f"{', '.join(str(c.line_no) for c in candidates)}. Only one of them can be the module "
            "tree, and checking the first would let a small compliant table hide a large "
            "unfinished one. Delete or rename the others.")
    tree = candidates[0] if candidates else None
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

        names = [row[cols["module"]].strip(" *`") for n, row in tree.rows
                 if cols.get("module") is not None and cols["module"] < len(row)]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            problems.append(
                f"{path}:{tree.line_no}: the module tree names {', '.join(sorted(dupes))} more "
                "than once. A decomposition has each module in it once; repeating a row pads the "
                "count without adding a component to prove.")

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
    for name in PLAN_HEADINGS:            # all ten, not the six this used to name
        problems += check_section(path, text, name)

    target = _section(text, "target")
    if target is None:
        pass                              # already reported as a missing heading
    else:
        size = next((l for l in target.splitlines() if "size range" in l.lower()), None)
        if size is None:
            problems.append(f"{path}: the Target section has no 'size range' line.")
        elif re.search(r"\b(see|cf\.?)\b|\b(section|§)\s*[0-9]", size, re.I) or \
                len(re.findall(r"\d+", size)) < 2:
            problems.append(
                f"{path}: the supported size range reads {size.strip(' -*')!r}. That is one number "
                "or none. State both ends, because Phase 4 runs a ladder across exactly this range "
                "and Phase 2 has to prove the top of it fits.")

    return _verdict("phase 0 plan", path, problems,
                    "Every section present, every table row filled, no placeholders, "
                    "nothing deferred.")


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
    # Fences stripped BEFORE any section lookup. A `# comment` inside a ```bash block is not a
    # heading, and reading raw text made one act as one: it could end a section early (rejecting
    # a filled report) or start a fake one (passing a report whose real section is empty).
    # gate_plan already stripped first, so only this path was affected.
    text = strip_code_blocks(path.read_text(encoding="utf-8", errors="replace"))
    for want in args.require_heading or []:
        problems += check_section(path, text, want)
        if args.no_tables:
            continue                              # this document is prose, --no-tables said so
        body = _section(text, want)
        if body is None:
            continue
        if not [tb for tb in parse_tables(body) if tb.rows]:
            problems.append(
                f"{path}: {want!r} has no table with data rows under it. The heading is not the "
                "evidence; the rows are. If this section is genuinely prose, pass --no-tables.")
    asked = ", ".join(repr(h) for h in (args.require_heading or [])) or "no sections named"
    return _verdict("report", path, problems,
                    f"Checked: {asked}"
                    + ("; tables not required (--no-tables)." if args.no_tables else "; each with a filled table."))


def _verdict(what: str, path: Path, problems: list[str], detail: str = "") -> int:
    if problems:
        print(f"GATE 1: {what} {path} is not finished. {len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"GATE 0: {what} {path} passed every check that was asked for. {detail}")
    return 0


# ---------------------------------------------------------------------- determinism


def _sh(cmd: str) -> int:
    """Run through bash -c so the semantics match what a person types."""
    return subprocess.run(["bash", "-c", cmd]).returncode


class CannotRead(Exception):
    """A file this gate must read is not readable. That is exit 2, not a failed gate."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    try:
        f = path.open("rb")
    except OSError as e:
        raise CannotRead(f"cannot read {path}: {e}") from e
    with f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_determinism(args) -> int:
    artifacts = [Path(a) for a in args.artifact]
    for a in artifacts:
        if a.is_dir():
            print(f"GATE 2: {a} is a directory. --artifact takes the file whose bytes must match.")
            return 2
        if a.is_symlink():
            # We move artifacts aside and let the run write fresh ones. Through a symlink that
            # silently replaces the link with a regular file and leaves the real target stale,
            # which is a change to the reader's tree that nothing here announced.
            print(f"GATE 2: {a} is a symlink. Name the file it points at "
                  f"({a.resolve()}), because this gate moves the artifact aside and lets your "
                  "command write a fresh one, which would replace the link rather than follow it.")
            return 2
    # Move the existing artifacts aside rather than deleting them. A capture is a CPU reference
    # run that can take thousands of seconds, `02` says to gitignore parity_artifacts/, and the
    # commonest way to get here is a --run that never starts (an unexported $SKILL, a typo). If
    # we unlink first and the command then exits 127, the gate has destroyed the thing it exists
    # to protect and reported "measured nothing".
    # Check EVERY backup path before moving ANY file. Checking as we went meant a refusal on the
    # second artifact returned 2 with the first one already renamed and nothing said so: the
    # message named only the file it refused over, and the reader who followed its instructions
    # left the other artifact orphaned under a backup name. A gate that half-moves the tree and
    # reports "measured nothing" is the same defect this whole block was written to avoid.
    for a in artifacts:
        if not a.is_file():
            continue
        keep = a.with_suffix(a.suffix + ".determinism-backup")
        if keep.exists():
            # Never overwrite a backup. One already here means a previous run was interrupted
            # and that file, not this one, is the golden.
            print(f"GATE 2: {keep} already exists, so a previous run of this gate did not "
                  "finish. That file is your artifact from before that run; this one is "
                  "whatever the interrupted run left. Decide which you want, remove the "
                  "backup, and re-run. Nothing has been moved.")
            return 2
    stash = {}
    for a in artifacts:
        if a.is_file():
            keep = a.with_suffix(a.suffix + ".determinism-backup")
            a.replace(keep)
            stash[a] = keep

    def _restore_stash():
        """Put the originals back, unconditionally.

        The first version only restored when the path was free, and deleted the backup
        otherwise. A run that failed *after* writing something therefore left its partial
        output in place and destroyed the original, under a message saying the original had
        been put back. That is worse than not stashing at all.
        """
        for a, keep in stash.items():
            if not keep.is_file():
                continue
            if a.is_file():
                a.unlink()                     # whatever the failed run left is not the golden
            keep.replace(a)

    try:
        return _determinism_runs(args, artifacts, stash, _restore_stash)
    except BaseException:
        # Ctrl-C during a capture that takes thousands of seconds used to leave the artifact
        # hidden under a .determinism-backup and say nothing, and the next run then stashed the
        # fresh file over it and deleted it on success.
        _restore_stash()
        print("\nInterrupted. Any artifact that existed before this run has been put back.")
        raise


def _determinism_runs(args, artifacts, stash, _restore_stash) -> int:
    runs: list[dict[Path, str]] = []
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

    differing = [a for a in artifacts if runs[0][a] != runs[1][a]]
    if differing:
        # A red verdict means the tree now holds run 2 of a capture that does not reproduce.
        # Keeping that and deleting the backup destroys the golden, which is exactly what the
        # worked example's own --unpinned negative control does. Put the original back.
        _restore_stash()
    else:
        for keep in stash.values():            # deterministic: the fresh pair is the real one
            if keep.is_file():
                keep.unlink()
    for a in artifacts:
        mark = "DIFFERS" if a in differing else "same"
        print(f"  {mark:8s} {a}  {runs[0][a][:16]}  {runs[1][a][:16]}")
    if differing:
        print(f"GATE 1: {len(differing)} artifact(s) changed between two identical runs. "
              "Pin the seeds, the thread count and the iteration order before going on: "
              "a golden you cannot reproduce cannot prove anything later."
              + (" Any artifact that existed before this run has been put back, so the "
                 "nondeterministic output is not now your golden." if stash else ""))
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


#: A check that only asks whether a path exists cannot be shown to detect a defect: the only
#: fault it can see is the file going missing, which is not a defect in the file.
EXISTENCE_ONLY = re.compile(
    r"^\s*(?:test|\[\[?)\s+!?\s*-[efsrwxdLhpSGONk]\s+\S+\s*\]?\]?\s*$"
    # Reading a file and throwing the bytes away asks the same question in a longer way, and
    # SKILL.md names it in the same breath as `test -f`. It was not implemented.
    r"|^\s*(?:cat|head|tail|wc|stat|ls|file)\s+[^|;&]*>\s*/dev/null(?:\s+2>&1)?\s*;?\s*$"
    r"|^\s*(?:cat|head|tail)\s+\S+\s*;?\s*$", re.I)


def gate_prove_red(args) -> int:
    watched = args.expect_change or []

    # Every path-looking token the check names. If the break changes one of these and you did
    # not list it, the run proved that some *other* file moved while the check went red for a
    # reason nobody recorded. `--check 'cat model.py >/dev/null' --break 'mv model.py x; echo >>s'
    # --expect-change s` used to pass: the fault was deleting the file the check reads.
    def _paths_in(cmd: str) -> list[str]:
        out = []
        for tok in re.findall(r"[\w./~-]+", cmd):
            if "/" in tok or re.search(r"\.\w{1,6}$", tok):
                q = Path(os.path.expanduser(tok))
                if q.exists():
                    out.append(str(q))
        return out

    absent = [q for q in watched if not Path(q).exists()]
    if absent:
        print(f"GATE 2: --expect-change names {', '.join(absent)}, which does not exist.")
        print("  Name the file the break edits, as it is spelled right now. A path that is not\n"
              "  there cannot be shown to change, and the run would report that nothing was\n"
              "  injected while your break edited something else.")
        return 2

    if EXISTENCE_ONLY.match(args.check):
        print(f"GATE 2: {args.check!r} only asks whether a path exists.")
        print("  There is no fault you can inject into that file that it will notice, because the\n"
              "  only thing it reads is the presence of the name. Breaking it means deleting the\n"
              "  file, and a check that fails because its subject is gone has not been shown to\n"
              "  detect anything. This is the existence-checked-contents-depended-on failure by\n"
              "  construction: make the check read the contents, then prove that red.")
        return 2

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
    # Compare resolved paths. "--expect-change /tmp/x/f.py" with "--check 'cd /tmp/x && grep f.py'"
    # is one file named two ways, and comparing the strings refused a completely ordinary
    # invocation as though the break had touched something undeclared.
    def _real(q: str) -> str:
        try:
            return str(Path(q).resolve())
        except OSError:
            return str(Path(q))

    watched_real = {_real(w) for w in watched}
    check_paths = [q for q in _paths_in(args.check) if _real(q) not in watched_real]
    check_before = _fingerprint(check_paths)
    def _bail(msg: str) -> int:
        """Refuse, but put the tree back first. Three of these used to return without running
        --restore, so a break that edited the file and then exited non-zero left the fault in
        place under a message saying nothing had been injected."""
        print(msg)
        print(f"--- restore: {args.restore}")
        rc = _sh(args.restore)
        after = _fingerprint(watched)
        dirty = [q for q in watched if was.get(q) != after.get(q)]
        if rc != 0 or dirty:
            print(f"  and the restore left the tree dirty (exit {rc}"
                  + (f", {', '.join(dirty)} still differs" if dirty else "")
                  + "). Check it by hand before running anything else.")
        else:
            # "the tree" overclaimed: this only ever compared the files passed to --expect-change,
            # and the refusal directly above fires precisely because an UNDECLARED file moved. It
            # printed "the tree is back as it was" over a file it had not looked at.
            print("  the files you declared are back as they were." if watched else
                  "  the restore ran; nothing was declared, so there is nothing to compare.")
        return 2

    print(f"--- break it: {getattr(args, 'break')}")
    if _sh(getattr(args, "break")) != 0:
        return _bail("GATE 2: the break command exited non-zero. Whatever it did or did not "
                     "inject, this run says nothing about the check.")

    # Everything past this point runs with the fault injected, so every exit has to go through
    # _bail. A CannotRead raised here (the break chmod'd a watched file to 000, say) unwound
    # straight to main(), which printed GATE 2 and exited with the fault still in the tree.

    try:
        # An exit code is not an edit. `sed -i s/pattern/x/` exits 0 when the pattern does not
        # match, and a stale break command is the usual reason this check reads backwards: nothing
        # was injected, the gate stayed green, and the gate got blamed for it.
        if watched:
            now = _fingerprint(watched)
            unchanged = [q for q in watched if was[q] == now[q]]
            if unchanged:
                changed = [q for q in watched if q not in unchanged]
                return _bail(
                    f"GATE 2: the break command exited 0 but did not change "
                    f"{', '.join(unchanged)}. Usually the pattern no longer matches the file."
                    + (f" It did change {', '.join(changed)}, so something was injected and this run "
                       "still says nothing about the check." if changed
                       else " Nothing was injected."))
        # Deleting the thing under test is not injecting a fault into it. The fingerprint above
        # counts a vanished file as "changed", which is true and useless: the check then fails
        # because there is nothing to check.
        if watched:
            gone = [q for q in watched if was[q] is not None and not Path(q).is_file()]
            if gone:
                return _bail(
                    f"GATE 2: the break removed {', '.join(gone)} rather than changing it, so it is "
                    "not on disk right now. A check that fails because the file is missing has not "
                    "been shown to fail on a defect. Edit the file, do not move it.")

        # A file the check reads, changed by the break, and not declared.
        moved = [q for q in check_paths if _fingerprint([q])[q] != check_before[q]]
        if moved:
            print(f"GATE 2: the break changed {', '.join(moved)}, which your check reads, and you did "
                  "not list it in --expect-change.")
            # Through _bail, not a bare _sh: every other refusal past the injection reports
            # whether the restore actually worked, and this one exited 2 saying nothing, so it
            # could leave the fault in the tree while claiming to have put it back.
            return _bail("  Either that is the file you meant to break, in which case declare it "
                         "and the\n  deleted-rather-than-edited guard applies to it, or the break "
                         "is touching\n  something it should not and the red you are about to see "
                         "is not the red you\n  think it is.")

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

    except CannotRead as e:
        return _bail(f"GATE 2: {e}")

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
                        "only when that is true; without it --expect-change is required. It does "
                        "not exempt you from the existence-only-check refusal.")
    p.add_argument("--red-exit", type=int, metavar="N",
                   help="the exit code this check uses to report a defect, when it is not the "
                        "usual 1. Without it, an exit code that means 'could not run' (2-5, 126, "
                        "127) is refused rather than counted as red.")
    p.add_argument("--expect-change", action="append", metavar="PATH",
                   help="a file the break must actually modify; refuses with exit 2 if it did not. "
                        "Use it whenever the break is a sed or a patch.")
    p.set_defaults(fn=gate_prove_red)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except CannotRead as e:
        # Exit 1 is reserved for "the gate failed and names why". A file the gate cannot open is
        # not a failed gate, it is a gate that did not run, and the difference is what tells a
        # reader whether to fix the port or fix the permissions.
        print(f"GATE 2: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
