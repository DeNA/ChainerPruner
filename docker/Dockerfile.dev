FROM denaai/chainerpruner:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LANGUAGE en_US

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY ./docker/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

ENV PYTHONPATH /work/:${PYTHONPATH}
ENV JUPYTER_PATH ${PYTHONPATH}:${JUPYTER_PATH}
