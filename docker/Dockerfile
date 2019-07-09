FROM chainer/chainer:v7.0.0a1-python3

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LANGUAGE en_US

RUN pip3 install -U pip setuptools
RUN pip3 install \
    networkx \
    chainercv \
    scipy \
    chainer_computational_cost

RUN pip3 install torch torchvision

RUN pip3 install git+https://github.com/DeNA/ChainerPruner