# FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
# FROM nvidia/cuda:11.6.1-runtime-ubuntu20.04
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install basics
RUN apt update && apt install -y \
    wget build-essential libssl-dev libffi-dev \
    zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev libgdbm-dev libnss3-dev liblzma-dev \
    tk-dev uuid-dev git && \
    apt clean

# for OpenCV
RUN apt update && apt install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

# # Install Python 3.10
# RUN wget https://www.python.org/ftp/python/3.10.15/Python-3.10.15.tgz && \
#     tar -xvf Python-3.10.15.tgz && \
#     cd Python-3.10.15 && \
#     ./configure --enable-optimizations && \
#     make -j"$(nproc)" && \
#     make altinstall && \
#     cd .. && rm -rf Python-3.10.15*

# # Set as standard python
# # RUN ln -s /usr/local/bin/python3.10 /usr/bin/python
# RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1

# Install Python 3.9
RUN wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz && \
    tar -xvf Python-3.9.18.tgz && \
    cd Python-3.9.18 && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd .. && rm -rf Python-3.9.18*
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.9 1
ARG VLLM_VERSION=0.4.0
ARG PYTHON_VERSION=39
ENV VLLM_VERSION=${VLLM_VERSION}
ENV PYTHON_VERSION=${PYTHON_VERSION}

# Update + Ipynb dependencies
RUN python -m ensurepip && python -m pip install --upgrade pip
RUN pip install ipykernel jupyter notebook ipython

# Others
RUN pip install "numpy<2"

# Install PyTorch with CUDA 13.0 support
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118


# Install flash-attn + vLLM
RUN pip install wheel setuptools packaging ninja
# RUN pip install flash-attn --no-build-isolation
#  -> requires ncc -> apt-get install -y cuda-toolkit-12-1
# RUN pip install vllm
RUN pip install https://github.com/vllm-project/vllm/releases/download/v0.4.0/vllm-0.4.0+cu118-cp39-cp39-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

# Clone Bergen repo
RUN git clone https://github.com/naver/bergen.git /workspace/bergen
# fix broken dependencies
# RUN sed -i "/tensorflow/d" /workspace/bergen/requirements.txt && \
#     sed -i "/protobuf/d" /workspace/bergen/requirements.txt

# Install Bergen dependencies
RUN python -m pip install -r /workspace/bergen/requirements.txt

# Install RAGBench
RUN git clone https://github.com/rungalileo/ragbench.git /workspace/ragbench

# Install example RAG dependencies
RUN python -m pip install transformers faiss-cpu accelerate prime_printer


