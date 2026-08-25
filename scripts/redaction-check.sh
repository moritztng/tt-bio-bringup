#!/usr/bin/env bash
# Fails if anything in this repository looks like private information.
# This repo is public. Run before every push. Exit 0 = clean, exit 1 = a hit to review.
#
# Prove it works before you trust it: plant a fake secret in a file, run this, confirm exit 1,
# remove it, confirm exit 0. A check nobody has seen go red is not a check.
set -u
cd "$(dirname "$0")/.." || exit 2

FILES=$(git ls-files 2>/dev/null || find . -type f -not -path './.git/*')
[ -z "$FILES" ] && { echo "no files to scan"; exit 2; }

# Each entry: <label>|<extended regex>. Case-insensitive unless the pattern needs case.
PATTERNS=(
  'private ssh key|BEGIN [A-Z ]*PRIVATE KEY'
  'github token|gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,}'
  'openai-style key|\bsk-[A-Za-z0-9]{20,}'
  'aws key|\bAKIA[0-9A-Z]{16}\b'
  'slack/telegram token|xox[abprs]-[A-Za-z0-9-]{10,}|bot[0-9]{8,}:[A-Za-z0-9_-]{30,}'
  'assigned secret|(api[_-]?key|secret|password|passwd|auth[_-]?token|access[_-]?token)[[:space:]]*[:=][[:space:]]*[A-Za-z0-9/+_-]{12,}'
  'private ipv4|\b(10|192\.168|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}\b'
  'tailnet ipv4|\b100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}\b'
  'mac address|\b([0-9a-f]{2}:){5}[0-9a-f]{2}\b'
  'home path|/home/[a-z][a-z0-9_-]*/|/Users/[a-z][a-z0-9_-]*/'
  'agent state dir|\.coworker|\.claude/projects|\.ssh/|/dev/tenstorrent/[0-9]+[[:space:]]*#'
  'internal host|\bqb[0-9]\b|quietbox|\bttuser\b|\.local\b|bmc'
  'email address|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
  'personal name|moritz(?!tng)|thuening|thüning'
)

fail=0
for entry in "${PATTERNS[@]}"; do
  label=${entry%%|*}
  rx=${entry#*|}
  hits=$(printf '%s\n' "$FILES" | tr '\n' '\0' \
    | xargs -0 grep -InPi -- "$rx" 2>/dev/null \
    | grep -vE '^(./)?scripts/redaction-check\.sh:' \
    | grep -vi 'github\.com/moritztng' \
    | grep -vi 'example\.com' )
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
