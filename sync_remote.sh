#!/bin/bash
# Syncs the local project to ~/keys on the remote server.
# Usage: ./sync_remote.sh [--dry-run]

REMOTE_USER="yuri.faria"
REMOTE_HOST="ssh.recod.ic.unicamp.br"
REMOTE_DIR="~/keys"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

RSYNC_OPTS=(
    -avz
    --progress
    --delete
    --exclude=".venv/"
    --exclude="client/TenSEAL/"
    --exclude="client/tenseal.tar"
    --exclude="client/rasp_cli.tar"
    --exclude="__pycache__/"
    --exclude="*.pyc"
    --exclude="*.csv"
    --exclude=".git/"
    --exclude="logs/"
    --exclude="mv_logs/"
    --exclude='client/data/'
)

if [[ "$1" == "--dry-run" ]]; then
    RSYNC_OPTS+=(--dry-run)
    echo "[dry-run] No files will be transferred."
fi

echo "Syncing $LOCAL_DIR → $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
echo

# Ensure the remote destination exists
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

rsync "${RSYNC_OPTS[@]}" "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

STATUS=$?
if [[ $STATUS -eq 0 ]]; then
    echo
    echo "Sync complete."
else
    echo
    echo "Sync failed (rsync exit code $STATUS)." >&2
    exit $STATUS
fi
