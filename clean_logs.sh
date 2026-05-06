# docker run -it \
#    -v $PWD/logs:/logs \
#    flhe:client \
#     bash -c 'rm -rf logs/MNIST && chmod 777 -R  logs/models/*'


# docker run -it \
#    -v $PWD/logs:/logs \
#    flhe:client \
#     bash -c 'rm -rf logs/debug'


# docker run -it \
#    -v $PWD/logs:/logs \
#    flhe:client \
#     bash -c 'rm -rf  logs/models/model_*'

# docker run -it \
#    -v $PWD/logs:/logs \
#    flhe:client \
#     bash -c 'chmod 777 -R  logs/models/*'

docker run -it \
   -v $PWD/client:/client \
   -v $PWD/logs:/logs \
   -v $PWD/mv_logs:/mv_logs \
   flhe:client \
   bash -c 'chmod 777 -R mv_logs/'
   # bash -c 'chmod 777 -R client/data/ && echo "Permissões corrigidas para client/data"'
   #  bash -c 'chmod -R 777  logs/* && echo "Permissões corrigidas para logs e logs/models"'

