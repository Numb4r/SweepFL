#!/bin/bash
# Auto-commit Claude's changes at end of each turn (Stop hook)
REPO="/home/yuri/Documentos/Keys_definitivo/keys"
cd "$REPO" || exit 0

# Stage modifications/deletions to tracked files only
git add -u 2>/dev/null || true

# Nothing staged → nothing to commit
git diff --cached --quiet && exit 0

# Count changed files and parse insertions/deletions
FILE_COUNT=$(git diff --cached --name-only | wc -l | tr -d ' ')
STAT=$(git diff --cached --shortstat | grep -oE '[0-9]+ insertion|[0-9]+ deletion' | \
  awk '{if(/insert/) i=$1; if(/delet/) d=$1} END {r=""; if(i+0>0) r=r"+"i; if(d+0>0) r=r"-"d; print r}')

if [ "$FILE_COUNT" -eq 1 ]; then
    FILE=$(git diff --cached --name-only)
    BASENAME=$(basename "$FILE")
    MSG="Update $BASENAME"
elif [ "$FILE_COUNT" -le 3 ]; then
    BASENAMES=$(git diff --cached --name-only | xargs -I{} basename {} | tr '\n' ' ' | sed 's/ *$//')
    MSG="Update $BASENAMES"
else
    TOP_DIR=$(git diff --cached --name-only | awk -F/ 'NF>1{print $1}' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
    [ -z "$TOP_DIR" ] && TOP_DIR="files"
    MSG="Update $FILE_COUNT files in $TOP_DIR"
fi

[ -n "$STAT" ] && MSG="$MSG ($STAT)"

git commit -m "$MSG"
