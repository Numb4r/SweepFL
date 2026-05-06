# FROM python:3.11.2
FROM tensorflow/tensorflow:2.13.0-gpu
#FROM arm64v8/python:3.9

RUN pip install -U pip
RUN pip install tenseal pandas numpy  flwr=="1.4.0" flwr-datasets flwr_datasets[vision] matplotlib seaborn gmpy2

CMD ["python", "-u", "server/server.py"]


#docker build -t allanmsouza/flhe:client .