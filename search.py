# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import logging
import time
from argparse import ArgumentParser

import numpy as np
from numpy.lib import utils
import torch
import torch.nn as nn


import datasets
from model import CNN
from darts import DartsTrainer
from utils import accuracy, set_logger, set_random



parser = ArgumentParser('darts')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--layers', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--log-frequency', default=10, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--channels', default=16, type=int)
parser.add_argument('--unrolled', default=False, action='store_true')
parser.add_argument('--visualization', default=False, action='store_true')
parser.add_argument('--v1', default=False, action='store_true')
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--seed', default=98765, type=int)
parser.add_argument('--curriculum', default=False, action='store_true')
parser.add_argument('--no-save', default=False, action='store_true')
args = parser.parse_args()

if args.seed > 0:
    set_random(args.seed)

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')
local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
logger = None if args.no_save else \
    set_logger('nni', log_file=os.path.join('checkpoints', local_time + '.log'))

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

dataset_train, dataset_valid = datasets.get_dataset(os.path.join('data', args.dataset))

model = CNN(32, 3, args.channels, 10, args.layers) # (32, 3, 16, 10, 8)
criterion = nn.CrossEntropyLoss(reduction='none')  # loss vector

optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)


trainer = DartsTrainer(
    model=model,
    loss=criterion,
    metrics=lambda output, target: accuracy(output, target, topk=(1,)),
    optimizer=optim,
    num_epochs=args.epochs,
    dataset=dataset_train,
    batch_size=args.batch_size,
    logger=logger,
    log_frequency=args.log_frequency,
    unrolled=args.unrolled,
    curriculum=args.curriculum,
    device=device
)
try:
    trainer.fit()
except KeyboardInterrupt:
    pass
final_architecture = trainer.export()
print('Final architecture:', final_architecture)
if not args.no_save:
    with open(os.path.join('checkpoints', local_time + '.json'), 'w') as checkpoint:
        json.dump(final_architecture, checkpoint)
