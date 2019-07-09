#!/usr/bin/env python
# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# Original code forked from MIT licensed chainer project
# https://github.com/chainer/chainer/blob/v5.1.0/examples/mnist/train_mnist.py
#
# Original code forked from MIT licensed keras project
# https://github.com/keras-team/keras/blob/2.2.4/examples/mnist_cnn.py

import logging
import argparse

import pandas as pd
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainerpruner.pruning.psfp.chainer import ProgressiveSoftFilterPruningExtension
from chainerpruner.utils import calc_computational_cost


# Network definition
class ConvNet(chainer.Chain):

    def __init__(self, n_out):
        super(ConvNet, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.conv1 = L.Convolution2D(None, 32, ksize=3)
            self.bn1 = L.BatchNormalization(32)
            self.conv2 = L.Convolution2D(None, 64, ksize=3)
            self.bn2 = L.BatchNormalization(64)
            self.fc1 = L.Linear(None, 128)
            self.fc2 = L.Linear(None, n_out)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, ksize=2)
        h = F.dropout(h, ratio=0.25)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.fc1(h))
        h = F.dropout(h, ratio=0.5)
        h = F.relu(self.fc2(h))
        return h


def train_mnist(gpu, batchsize, out, epoch, frequency, plot,
                resume, target_layers, pruning_rate):
    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batchsize))
    print('# epoch: {}'.format(epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(ConvNet(10))
    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.AdaDelta()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(ndim=3)

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model, device=gpu)
    trainer.extend(evaluator)

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = epoch if frequency == -1 else max(1, frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # dummy input
    x, _ = train[0]
    x = x[np.newaxis, ...]  # CHW -> NCHW
    x = chainer.Variable(x)
    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        x.to_gpu()
    psfp = ProgressiveSoftFilterPruningExtension(model.predictor, x, target_layers, pruning_rate=pruning_rate,
                                                 stop_trigger=(epoch, 'epoch'),
                                                 rebuild=True)
    trainer.extend(psfp)

    if resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(resume, trainer)

    cch_before = calc_computational_cost(model.predictor, x).total_report

    # Run the training
    trainer.run()

    cch_after = calc_computational_cost(model.predictor, x).total_report

    df = pd.DataFrame([cch_before, cch_after], index=['before', 'after'])
    print(df)

    ret = evaluator()
    print(pd.DataFrame([ret]))

    return model


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    # for channel pruning
    parser.add_argument('--pruning-percent', type=float, default=0.8,
                        help='pruning percent.')
    parser.add_argument('--reinitialize', action='store_true',
                        help='reinitialize model')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    target_layers = ['/conv1', '/conv2']

    train_mnist(args.gpu, args.batchsize, args.out,
                args.epoch, args.frequency, args.plot, args.resume,
                target_layers, pruning_rate=args.pruning_percent)


if __name__ == '__main__':
    main()
