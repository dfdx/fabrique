
# devel version with DNN is required for JIT compilation in some cases
# FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04 AS build-base
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build-base


## Basic system setup

ENV user=devpod
SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND=noninteractive \
    TERM=linux

ENV TERM=xterm-color

ENV LANGUAGE=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LC_CTYPE=en_US.UTF-8 \
    LC_MESSAGES=en_US.UTF-8

RUN apt update && apt install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        gpg \
        gpg-agent \
        less \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        llvm \
        locales \
        tk-dev \
        tzdata \
        unzip \
        vim \
        wget \
        xz-utils \
        zlib1g-dev \
        zstd \
    && sed -i "s/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g" /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && apt clean


## System packages

RUN apt-get update
RUN apt-get install -y git openssh-server
RUN apt-get install -y python3 python3-pip python-is-python3
RUN apt-get install -y jq
RUN pip install yq==3.1.1

## Install PostgreSQL client

RUN apt install -y postgresql-common
RUN /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y

RUN apt update
RUN apt install -y postgresql postgresql-14-pgvector

## Add user & enable sudo

RUN useradd -ms /bin/bash ${user}
RUN usermod -aG sudo ${user}

RUN apt-get install -y sudo
RUN echo "${user} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER ${user}
WORKDIR /home/${user}


## Python packages

RUN pip install --upgrade pip
RUN pip install wheel

# jax needs to be installed first because otherwise torch dependencies will force older version of jax
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install flax optax

RUN pip install sentencepiece tokenizers

RUN pip install safetensors huggingface-hub

# fix torch version because updating it often leads to painful results
# RUN pip install torch==2.0.0 torchvision lightning
# RUN pip install transformers accelerate datasets
# RUN pip install einops


## Post-install setup

RUN mkdir -p ${HOME}/.cache
RUN echo 'export PATH=${PATH}:${HOME}/.local/bin' >> ${HOME}/.bashrc

# ensure libraries see CUDA
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}' >> ${HOME}/.bashrc
RUN echo 'export PATH=/usr/local/cuda/bin:${PATH}' >> ${HOME}/.bashrc


FROM build-base AS build-dev

RUN pip install pytest ipython mypy black isort

FROM build-dev AS build-test

RUN pip install torch
RUN pip install transformers

CMD ["echo", "Explore!"]