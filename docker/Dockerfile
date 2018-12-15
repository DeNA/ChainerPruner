FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04


RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    # misc \
    tzdata \
    ca-certificates \
    # for pyenv \
    build-essential \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    git \
    # computer vision \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ARG TIMEZONE
RUN ln -sf /usr/share/zoneinfo/${TIMEZONE} /etc/localtime

ARG UID
RUN useradd docker -l -u ${UID} -s /bin/bash -m

USER docker

ARG PYTHON_VERSION
ENV PYENV_ROOT /home/docker/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash && \
    pyenv update && \
    pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION}

RUN pip install --no-cache-dir --upgrade setuptools pip

RUN pip install \
    chainer==5.1.0 \
    cupy-cuda90==5.1.0 \
    scipy \
    chainer_computational_cost

COPY requirements.txt /home/docker/requirements.txt
RUN pip install -r /home/docker/requirements.txt


WORKDIR /work

EXPOSE 8888

ENV PYTHONPATH /work/:${PYTHONPATH}
ENV JUPYTER_PATH ${PYTHONPATH}:${JUPYTER_PATH}

ENTRYPOINT ["/bin/bash", "--login"]
