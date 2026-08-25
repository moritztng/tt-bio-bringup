#!/usr/bin/env bash
# Fails if anything in this repository looks like private information.
# This repo is public. Run before every push. Exit 0 = clean, exit 1 = a hit to review.
#
# Prove it works before you trust it: plant a fake credential in a NEW, UNTRACKED file, run this,
# confirm exit 1, remove it, confirm exit 0. A check nobody has seen go red is not a check.
#
# The canary must be untracked, because that is the state a file is in right before `git add`,
# and it is the case this script got wrong once: `git ls-files` alone lists tracked files only,
# so the scan silently narrowed to the index and reported clean while seeing nothing new.
set -u
# Byte semantics everywhere. In a UTF-8 locale PCRE gives up on a line containing an invalid byte,
# and sed does too, so a path or a line holding one was silently not matched while the summary
# counted it: "we\xffird-NAME-at-DOMAIN.md" scanned clean beside an identically shaped valid name that hit.
LC_ALL=C
export LC_ALL
# Resolve the script's own directory to an absolute path BEFORE the cd. Both are derived from
# "$(dirname "$0")", and when $0 is relative the second one resolves against the NEW cwd: run
# this from inside scripts/ and redaction-local.txt was looked for at ../scripts/, not found,
# and every org-specific pattern was dropped with no message. The run printed "clean" and
# exited 0 having never scanned for a single hostname or staff name. That is this script's own
# headline defect, in this script.
# readlink -f, because $0 through a symlink is the LINK's directory: `ln -s .../redaction-check.sh
# ~/bin/rc.sh && rc.sh` cd'd to ~/, scanned an unrelated tree, and printed "clean (1 files)".
SELF="$(readlink -f "$0" 2>/dev/null || echo "$0")"
SELFDIR="$(cd "$(dirname "$SELF")" && pwd)" || exit 2
cd "$SELFDIR/.." || exit 2
# And prove the cd landed where this script lives, rather than trusting the arithmetic above.
# A scan of the wrong tree is the one failure this script cannot report, because everything it
# looks at comes from the tree it landed in.
[ -f scripts/redaction-check.sh ] || {
  echo "FATAL: cd landed in $(pwd), which has no scripts/redaction-check.sh. Refusing to scan a" >&2
  echo "tree that is not this repo: a clean verdict over the wrong files is worse than no run." >&2
  exit 2
}

# Every pattern below is PCRE (lookahead exclusions). A grep without -P does not narrow the
# scan, it empties it: each invocation exits 2, the hit list is empty, and this script prints
# "clean" having matched nothing. Refuse instead of reporting a green it did not measure.
if ! echo x | grep -qP 'x(?!y)' 2>/dev/null; then
  echo "FATAL: this grep has no -P (PCRE). Every pattern here needs it, and without it this"
  echo "script would report clean while scanning nothing. Install GNU grep (macOS: brew install"
  echo "grep, then use ggrep) and re-run." >&2
  exit 2
fi

# In a git checkout, -c -o --exclude-standard is tracked + untracked, honouring .gitignore, which
# is the set that is about to be published. Outside one (a tarball download), fall back to find
# and prune the same build noise .gitignore would have: a .pyc carries the absolute path of the
# machine that built it, and the two code paths scanning different file sets is how this check
# went blind once before.
# NUL-separated end to end, in a FILE rather than a variable: a filename containing a newline used
# to be split into two paths that do not exist, grep's error went to /dev/null, and the file was
# never scanned while the summary counted it. It has to be a file because bash command
# substitution silently strips NUL bytes, which is the same class of loss one layer up.
FILELIST=$(mktemp) || exit 2
trap 'rm -f "$FILELIST"' EXIT
{ git ls-files -zco --exclude-standard 2>/dev/null \
  || find . -type f -not -path './.git/*' -not -path '*/__pycache__/*' -not -name '*.pyc' -print0
} | grep -zv '^\(\./\)\?scripts/redaction-local\.txt$' > "$FILELIST"
# The local denylist is by construction a file full of the strings we are searching for. git never
# lists it because it is gitignored; find does, and then every pattern in it hits itself.
NFILES=$(tr -cd '\0' < "$FILELIST" | wc -c)
[ "$NFILES" -eq 0 ] && { echo "no files to scan"; exit 2; }

# Each entry: <label>|<extended regex>. Case-insensitive unless the pattern needs case.
#
# These are the generic shapes. Your own hostnames, staff names and project codenames are NOT
# here on purpose: writing them into a file you are about to publish is the leak. Put them in
# an untracked `scripts/redaction-local.txt`, one extended regex per line, `#` for comments:
#
#     \brack[0-9]\b|bigbox|\bsharedaccount\b   # your hostnames and service accounts
#     surname|othersurname                     # your staff
#     projectcodename|anothercodename          # your internal names
#
# The file is read if it exists and is gitignored. Everyone who clones this gets the generic
# patterns; nobody inherits anyone else's vocabulary.
PATTERNS=(
  'private ssh key|BEGIN [A-Z ]*PRIVATE KEY'
  'key or credential file|\b(id_rsa|id_dsa|id_ecdsa|id_ed25519)\b|\.(pem|p12|pfx|jks|keystore|kdbx)\b|\bcredentials\.json\b|\.npmrc\b|\.pypirc\b'
  'github token|gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,}'
  'llm api key|\bsk-[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9]){19,}'
  'aws key|\bAKIA[0-9A-Z]{16}\b'
  'slack/telegram token|xox[abprs]-[A-Za-z0-9-]{10,}|bot[0-9]{8,}:[A-Za-z0-9_-]{30,}'
  'assigned secret|(api[_-]?key|secret|password|passwd|auth[_-]?token|access[_-]?token)[[:space:]]*[:=][[:space:]]*["'"'"'`]?[^[:space:]"'"'"'`]{8,}'
  # The assigned form above needs 8+ characters after the '=', so it reads "API_KEY=xyz" as a
  # placeholder and stays quiet. The word itself is what the brief names, and it appears nowhere
  # in this repo, so match it bare and let a human judge the hit.
  'the words api key|api[_ -]?key'
  'personal messaging|\btelegram\b|\bt\.me/|\bwhatsapp\b|\bsignal\.me\b'
  'private ipv4|\b(10|192\.168|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}\b'
  'tailnet ipv4|\b100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}\b'
  'mac address|\b([0-9a-f]{2}:){5}[0-9a-f]{2}\b'
  'home path|/home/[a-z][a-z0-9_-]*|/Users/[a-z][a-z0-9_-]*'
  'agent or private state dir|\.claude/(?!skills|agents|plugins)|\.config/|\.env\b|\.claude\.json|\.(ssh|aws|gnupg|kube|config)\b|\.ssh/|\.aws/|\.gnupg/|\.kube/|\.netrc|/dev/tenstorrent/[0-9]+[[:space:]]*#'
  'internal host|\.(local|lan|internal|intranet|corp|home|localdomain)\b|\bbmc\b|\bilo\b|\bidrac\b'
  'private ipv6|\bfd[0-9a-f]{2}:[0-9a-f:]{2,}|\bfe80::[0-9a-f:]{2,}'
  'webhook url|hooks\.slack\.com/|discord\.com/api/webhooks/|\.webhook\.office\.com/'
  'email address|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
  'non-public git remote|git@(?!github\.com)[a-z0-9][a-z0-9.-]*[:/]|\b(gitlab|bitbucket)\.[a-z]|\bgit\+ssh://'
  # "budget" and "token" are ordinary vocabulary here (L1 budget, sequence token), so matching
  # them bare would fire on 15 tracked files and train a reader to ignore every hit. The money
  # arm therefore needs a currency within 24 characters of the word, and the plural forms the
  # earlier version missed are in: "3 FTEs" and "budget: 40k dollars" both scanned clean.
  'money or budget|\$[0-9]{2,}|\$[0-9]+[.,][0-9]|\$[0-9]+ ?(k|m|bn|million|billion)\b|\bUSD ?[0-9]|\b[0-9]{3,} ?(dollars|euros|pounds)\b|\bheadcount\b|\b[0-9]+ ?(FTEs?|engineers?)\b|\bbudget(ed|s)?\b[^.\n]{0,24}(\$|£|\bUSD\b|\bEUR\b|\bGBP\b|dollars|euros|pounds|\bmillion\b)|[0-9]+ ?(k|m) ?(dollars|euros|pounds|usd|eur|gbp)\b|\b(GBP|USD|EUR) ?[0-9]'
  # The assigned form needs 8+ characters after the '=', so "password: hunter2" was a placeholder
  # to it, and the prose form has no '=' at all.
  'secret in prose|\b(password|passwd|passphrase|api key|secret|credential)s?\b[[:space:]]+(is|was|are|were)[[:space:]]+\S{4,}'
  'the words password or secret|\b(password|passwd|passphrase)\b|\bsecret\b'
  'schedule or commitment|\bQ[1-4] 20[0-9]{2}\b|by end of (Q[1-4]|January|February|March|April|May|June|July|August|September|October|November|December)|\bby [0-9]{4}-[0-9]{2}-[0-9]{2}\b|\bdeadline\b|\bship date\b'
)

LOCAL="$SELFDIR/redaction-local.txt"
if [ -f "$LOCAL" ]; then
  n=0
  # `|| [ -n "$line" ]`: read returns non-zero on a final line with no newline, so the loop body
  # never ran for it. Writing this file with `cat > ...` and Ctrl-D without a final Enter dropped
  # the last pattern, and the run printed "clean". The local denylist is the only arm that scans
  # for the org's own hostnames and staff names, so losing one of them loses the whole point.
  while IFS= read -r line || [ -n "$line" ]; do
    # Only a '#' preceded by whitespace is a comment. Stripping at the first '#' anywhere
    # silently truncated "bldg#3|projectzeus" to "bldg", and the "loaded 1 pattern" line then
    # asserted a pattern that was not the one written.
    # A whole-line comment is a comment. Only a '#' with whitespace before it ends a pattern;
    # stripping at any '#' truncated "bldg#3|projectzeus" to "bldg" while the summary said
    # otherwise, and not stripping a leading '#' loaded the comment itself as a pattern.
    printf '%s' "$line" | grep -q '^[[:space:]]*#' && continue
    line=$(printf '%s' "$line" | sed -e 's/[[:space:]]\+#.*$//' -e 's/[[:space:]]*$//')
    [ -z "$line" ] && continue
    # Validate it. An invalid PCRE makes grep error on every file, the error goes to /dev/null,
    # the pattern matches nothing, and the run still says clean. That is the whole failure this
    # script exists to prevent, arriving through the file the reader is told to write.
    if ! printf 'x\n' | grep -qPi -- "$line" 2>/dev/null && \
       ! printf 'x\n' | grep -vqPi -- "$line" 2>/dev/null; then
      echo "FATAL: $(basename "$LOCAL") line $((n + 1)) is not a valid pattern:" >&2
      printf '  %s\n' "$line" >&2
      printf '%s\n' "$(printf 'x\n' | grep -Pi -- "$line" 2>&1 >/dev/null | head -1)" >&2
      echo "  Fix it or remove it. A pattern that cannot compile catches nothing and this" >&2
      echo "  script would otherwise report clean." >&2
      exit 2
    fi
    PATTERNS+=("local denylist|$line"); n=$((n + 1))
  done < "$LOCAL"
  echo "loaded $n local pattern(s) from $(basename "$LOCAL")"
else
  # Say it. Absent and not-found-because-the-path-was-wrong look identical from the outside,
  # and a run that scans for no org-specific name at all should not look like a full one.
  echo "no $LOCAL: scanning for generic shapes only, no hostnames or staff names" >&2
fi

fail=0
for entry in "${PATTERNS[@]}"; do
  label=${entry%%|*}
  rx=${entry#*|}
  # Allowlisted strings are blanked out of each LINE before matching, not used to drop the line.
  # Dropping the line meant a single github.com/moritztng URL anywhere on it suppressed every
  # pattern for that line, and these documents tell the reader to write those URLs: a planted AWS
  # key on the same line as the repo's own URL read clean. The scanner's own file is still
  # skipped by name, which is a whole-file exemption and deliberate.
  # Contents, and then the PATHS. A path is published exactly as the bytes inside the file are,
  # and scanning only contents meant "notes/rack3-alice-NAME-at-DOMAIN.md" shipped and read clean.
  #
  # The allowlist is applied to each file's text BEFORE matching, one file at a time. The earlier
  # shape matched first and re-applied the pattern to grep's own "path:line:" output, which meant
  # any pattern anchored at ^ could never survive the second pass: "^rack[0-9]" loaded, validated,
  # was counted in the summary, and could not fire. Blanking first removes the second pass.
  hits=$( { while IFS= read -r -d "" f; do
              # This file holds the patterns, so scanning the pattern list against itself is all
              # false positives. Everything else in it is ordinary published text, and a credential
              # typed into the comments or the code shipped unseen: this is the one file a reader
              # is told to edit. So the exemption is now the PATTERNS=( ... ) region only, cut out
              # of the stream, and the rest of the file is scanned like any other.
              cut=cat
              case $f in
                ./scripts/redaction-check.sh|scripts/redaction-check.sh)
                  cut="sed /^PATTERNS=($/,/^)$/d" ;;
              esac
              if [ ! -r "$f" ]; then
                printf 'UNREADABLE: %s\n' "$f"; continue
              fi
              $cut -- "$f" 2>/dev/null \
                | sed -e 's|github\.com/moritztng|ALLOWED|gI' \
                      -e 's|git@github\.com|ALLOWED|gI' \
                      -e 's|example\.com|ALLOWED|gI' \
                | grep -anPi -- "$rx" 2>/dev/null | f="$f" awk '{ print ENVIRON["f"] ":" $0 }'
            done < "$FILELIST"
            grep -zv '^\(\./\)\?scripts/redaction-check\.sh$' < "$FILELIST" \
              | tr '\0' '\n' \
              | sed -e 's|github\.com/moritztng|ALLOWED|gI' \
                    -e 's|git@github\.com|ALLOWED|gI' \
                    -e 's|example\.com|ALLOWED|gI' \
              | grep -aPi -- "$rx" | sed 's|^|FILENAME: |'
          } | tr -d '\000' | cut -c1-200 )
  unreadable=$(printf '%s\n' "$hits" | grep -a '^UNREADABLE: ' | sort -u)
  if [ -n "$unreadable" ]; then
    echo "FATAL: cannot read these files, so they were counted and not scanned:" >&2
    printf '%s\n' "$unreadable" | sed 's/^/  /' >&2
    exit 2
  fi
  if [ -n "$hits" ]; then
    echo "REDACTION HIT [$label]"
    printf '%s\n' "$hits" | head -20 | sed 's/^/  /'
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "redaction check clean ($NFILES files, ${#PATTERNS[@]} patterns)"
  exit 0
fi
echo
echo "Review every hit above. Remove it or, if it is genuinely public, add a narrow exclusion here."
exit 1
