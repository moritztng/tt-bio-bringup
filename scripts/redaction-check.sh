#!/usr/bin/env bash
# Fails if anything in this repository looks like private information.
# This repo is public. Run before every push. Exit 0 = clean, exit 1 = a hit to review.
#
# Prove it works before you trust it: plant a fake secret in a NEW, UNTRACKED file, run this,
# confirm exit 1, remove it, confirm exit 0. A check nobody has seen go red is not a check.
#
# The canary must be untracked, because that is the state a file is in right before `git add`,
# and it is the case this script got wrong once: `git ls-files` alone lists tracked files only,
# so the scan silently narrowed to the index and reported clean while seeing nothing new.
set -u
cd "$(dirname "$0")/.." || exit 2

# Every pattern below is PCRE (lookahead exclusions). A grep without -P does not narrow the
# scan, it empties it: each invocation exits 2, the hit list is empty, and this script prints
# "clean" having matched nothing. Refuse instead of reporting a green it did not measure.
if ! echo x | grep -qP 'x(?!y)' 2>/dev/null; then
  echo "FATAL: this grep has no -P (PCRE). Every pattern here needs it, and without it this"
  echo "script would report clean while scanning nothing. Install GNU grep (macOS: brew install"
  echo "grep, then use ggrep) and re-run." >&2
  exit 2
fi

FILES=$(git ls-files -co --exclude-standard 2>/dev/null || find . -type f -not -path './.git/*')
[ -z "$FILES" ] && { echo "no files to scan"; exit 2; }

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
  'github token|gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,}'
  'llm api key|\bsk-[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9]){19,}'
  'aws key|\bAKIA[0-9A-Z]{16}\b'
  'slack/telegram token|xox[abprs]-[A-Za-z0-9-]{10,}|bot[0-9]{8,}:[A-Za-z0-9_-]{30,}'
  'assigned secret|(api[_-]?key|secret|password|passwd|auth[_-]?token|access[_-]?token)[[:space:]]*[:=][[:space:]]*["'"'"'`]?[A-Za-z0-9/+_.-]{12,}'
  'private ipv4|\b(10|192\.168|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}\b'
  'tailnet ipv4|\b100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}\b'
  'mac address|\b([0-9a-f]{2}:){5}[0-9a-f]{2}\b'
  'home path|/home/[a-z][a-z0-9_-]*/|/Users/[a-z][a-z0-9_-]*/'
  'agent state dir|\.coworker|\.claude/(?!skills|agents|plugins)|\.ssh/|/dev/tenstorrent/[0-9]+[[:space:]]*#'
  'internal host|\.local\b|\bbmc\b|\bilo\b|\bidrac\b'
  'email address|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
  'non-public git remote|git@(?!github\.com)[a-z0-9][a-z0-9.-]*[:/]|\b(gitlab|bitbucket)\.[a-z]|\bgit\+ssh://'
  'money or budget|\$[0-9]{2,}|\$[0-9]+[.,][0-9]|\$[0-9]+ ?(k|m|bn|million|billion)\b|\bUSD ?[0-9]|\bheadcount\b|\b[0-9]+ (FTE|engineers)\b'
  'schedule or commitment|\bQ[1-4] 20[0-9]{2}\b|by end of (Q[1-4]|January|February|March|April|May|June|July|August|September|October|November|December)|\bdeadline\b|\bship date\b'
)

LOCAL="$(dirname "$0")/redaction-local.txt"
if [ -f "$LOCAL" ]; then
  n=0
  while IFS= read -r line; do
    line=${line%%#*}; line=$(printf '%s' "$line" | sed 's/[[:space:]]*$//')
    [ -z "$line" ] && continue
    PATTERNS+=("local denylist|$line"); n=$((n + 1))
  done < "$LOCAL"
  echo "loaded $n local pattern(s) from $(basename "$LOCAL")"
fi

fail=0
for entry in "${PATTERNS[@]}"; do
  label=${entry%%|*}
  rx=${entry#*|}
  hits=$(printf '%s\n' "$FILES" | tr '\n' '\0' \
    | xargs -0 grep -anPi -- "$rx" 2>/dev/null \
    | grep -avE '^(./)?scripts/redaction-check\.sh:' \
    | grep -avi 'github\.com/moritztng' \
    | grep -avi 'git@github\.com' \
    | grep -avi 'example\.com' \
    | tr -d '\000' | cut -c1-200 )
  if [ -n "$hits" ]; then
    echo "REDACTION HIT [$label]"
    printf '%s\n' "$hits" | head -20 | sed 's/^/  /'
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "redaction check clean ($(printf '%s\n' "$FILES" | wc -l) files, ${#PATTERNS[@]} patterns)"
  exit 0
fi
echo
echo "Review every hit above. Remove it or, if it is genuinely public, add a narrow exclusion here."
exit 1
