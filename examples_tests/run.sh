#!/usr/bin/env bash

# https://github.com/chainer/chainercv/blob/master/.pfnci/examples_tests.sh

# ./examples_tests/run.sh

set -e
set -u
set -o pipefail

IMAGE=gpu
GPU=0

for SCRIPT in $(find examples_tests/ -type f -name '*.sh')
do
    docker-compose run \
        -e GPU=$GPU \
        $IMAGE sh -ex ${SCRIPT}
done