FROM tensorflow/tensorflow:2.13.0-gpu

RUN pip install -U pip

# Build dependencies for ckks_project C++ extension
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pybind11

# Install ckks_project (compiles ckks_python .so + installs ckks_federated package)
COPY ckks_project/ckks_project.zip /tmp/ckks_project.zip
RUN cd /tmp && \
    unzip -q ckks_project.zip -d ckks_project && \
    cd ckks_project && \
    sed -i '1s/^/from __future__ import annotations\n/' ckks_federated/__init__.py && \
    pip install . && \
    cd / && rm -rf /tmp/ckks_project /tmp/ckks_project.zip

# Alias: ckks_backend.py imports from ckks_python_simplify → ckks_federated
RUN python3 -c "import site,os; sp=site.getsitepackages()[0]; os.symlink(sp+'/ckks_federated',sp+'/ckks_python_simplify')"

RUN pip install "pandas<2.0.0" numpy flwr=="1.4.0" tenseal flwr-datasets flwr_datasets[vision] matplotlib gmpy2

CMD ["python", "-u", "client/client.py"]
