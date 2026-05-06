#!/bin/bash
LOGFOLDER=$1
FOLDER="./logs/$LOGFOLDER"
INTERVAL=20
NTFY_TOPIC="HIIAC_YURI_PRINCIPAL"
NTFY_TITLE="Erro crítico detectado!"
NTFY_URL="https://ntfy.sh/$NTFY_TOPIC"

declare -A alerted

echo "Monitorando logs em $FOLDER a cada $INTERVAL segundos..."

while true; do
    for file in "$FOLDER"/*.log; do
        [ -f "$file" ] || continue

        # verifica se já enviamos alerta
        [ -z "${alerted[$file]}" ] || continue

        # procura qualquer traceback ou ERROR
        if grep -Ei "ERROR:|Traceback \(most recent call last\):" "$file" >/dev/null; then
            
            # agora trata caso de GrpcBridgeClosed
            if grep -q "GrpcBridgeClosed" "$file"; then
                # procura se houve algum sucesso após o GRPC closed
                # linhas de sucesso: fit_round received results ou EvaluateRes
                last_success=$(grep -nE "fit_round .* received .* results|EvaluateRes" "$file" | tail -n1 | cut -d: -f1)
                last_bridge=$(grep -n "GrpcBridgeClosed" "$file" | tail -n1 | cut -d: -f1)

                if [ -z "$last_success" ] || [ "$last_bridge" -gt "$last_success" ]; then
                    # GRPC travou de verdade (nenhum sucesso depois)
                    echo "Erro crítico (GrpcBridgeClosed) detectado em $file, enviando notificação..."
                    curl -d "title=$NTFY_TITLE&message=$file" "$NTFY_URL"
                    alerted[$file]=1
                else
                    # GRPC foi recoverable, ignora
                    echo "GrpcBridgeClosed detectado mas recoverable em $file"
                fi
            else
                # qualquer outro traceback/ERROR
                echo "Erro crítico detectado em $file, enviando notificação..."
                curl -d "title=$NTFY_TITLE&message=$file" "$NTFY_URL"
                alerted[$file]=1
            fi
        fi
    done

    sleep $INTERVAL
done