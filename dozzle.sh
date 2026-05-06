if [ "$(docker inspect -f '{{.State.Running}}' dozzle 2>/dev/null)" != "true" ]; then
    echo "Dozzle não está rodando. Subindo..."
    # tmux new-session -d -s dozzle "
    docker start dozzle || docker run -d \
        --name dozzle \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -p 127.0.0.1:8080:8080 \
        amir20/dozzle:latest
        # "
        
else
    echo "Dozzle já está rodando."
fi
