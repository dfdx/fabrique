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
# RUN apt-get install -y jq
# RUN pip install yq==3.1.1

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


# add specific version of JAX directry to the container
RUN pip install jax["cuda12"]==0.5.0

COPY --chown=${user}:${user} ./pyproject.toml /home/${user}/
RUN pip install pip-tools
RUN python -m piptools compile --extra dev -o requirements.txt pyproject.toml
RUN pip install -r requirements.txt



## Post-install setup

RUN mkdir -p ${HOME}/.cache
RUN echo 'export PATH=${PATH}:${HOME}/.local/bin' >> ${HOME}/.bashrc

# ensure libraries see CUDA
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}' >> ${HOME}/.bashrc
RUN echo 'export PATH=/usr/local/cuda/bin:${PATH}' >> ${HOME}/.bashrc


FROM build-base AS build-dev

# RUN pip install pytest ipython mypy black isort
# RUN pip install tensorflow tensorboard-plugin-profile

FROM build-dev AS build-test

# RUN pip install torch
# RUN pip install transformers

CMD ["echo", "Explore!"]