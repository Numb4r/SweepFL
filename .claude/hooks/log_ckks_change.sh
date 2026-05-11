#!/bin/bash
# Triggered on PostToolUse for Edit|Write; creates a change log when ckks library files are modified.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('file_path', ''))
except Exception:
    print('')
" 2>/dev/null)

# Only act on ckks_backend.py or anything inside ckks_project/
if ! echo "$FILE_PATH" | grep -qE "(ckks_backend\.py|/ckks_project/)"; then
    exit 0
fi

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
HUMAN_TIME=$(date '+%Y-%m-%d %H:%M:%S')
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/ckks_change_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ckks_change_${TIMESTAMP}.md"

cat > "$LOG_FILE" << LOGEOF
# CKKS Library Change — $HUMAN_TIME

**Arquivo modificado:** $FILE_PATH

## Descrição das mudanças
<!-- TODO: preencher com o que foi alterado -->

## Prompt para aplicar no projeto da biblioteca

> Aplique as seguintes alterações no projeto \`ckks_project/\`:
>
> <!-- TODO: descrever exatamente o que precisa ser replicado em ckks_project/ -->
>
> **Arquivo equivalente em ckks_project/:** <!-- TODO: identificar o caminho correspondente -->

## Contexto adicional
<!-- TODO: motivação, impacto, dependências -->
LOGEOF

# Return additionalContext so Claude is reminded to fill in the log
python3 -c "
import json
msg = (
    'LEMBRETE: você modificou um arquivo da biblioteca ckks ($FILE_PATH). '
    'Um log foi criado em ckks_change_logs/ckks_change_${TIMESTAMP}.md — '
    'preencha agora: (1) Descrição das mudanças, (2) Prompt para replicar em ckks_project/.'
)
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'PostToolUse',
        'additionalContext': msg
    }
}))
"
