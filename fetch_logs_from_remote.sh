#!/bin/bash
# Downloads logs/ from the remote cluster to the local logs/ directory.
# Usage: ./fetch_logs_from_remote.sh [--dry-run]

REMOTE_USER="yuri.faria"
REMOTE_HOST="ssh.recod.ic.unicamp.br"
REMOTE_DIR="~/keys"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

RSYNC_OPTS=(
    -avz
    --progress
    --update          # skip files newer locally; never deletes local-only files
    --exclude="__pycache__/"
    --exclude="*.pyc"
    --exclude=".git/"
)

if [[ "$1" == "--dry-run" ]]; then
    RSYNC_OPTS+=(--dry-run)
    echo "[dry-run] No files will be transferred."
fi

mkdir -p "$LOCAL_DIR/logs"

echo "Fetching logs/  $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/logs/ → $LOCAL_DIR/logs/"
echo

rsync "${RSYNC_OPTS[@]}" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/logs/" \
    "$LOCAL_DIR/logs/"

STATUS=$?
if [[ $STATUS -eq 0 ]]; then
    echo
    echo "Fetch complete."
else
    echo
    echo "Fetch failed (rsync exit code $STATUS)." >&2
    exit $STATUS
fi
