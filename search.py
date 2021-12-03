# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import logging
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn


import datasets
from model import CNN
from darts import DartsTrainer
from utils import accuracy



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
args = parser.parse_args()

if args.seed > 0:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

logger = logging.getLogger('nni')
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

dataset_train, dataset_valid = datasets.get_dataset(os.path.join('data', args.dataset))

model = CNN(32, 3, args.channels, 10, args.layers)
criterion = nn.CrossEntropyLoss()

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
    log_frequency=args.log_frequency,
    unrolled=args.unrolled,
    device=device
)
try:
    trainer.fit()
except KeyboardInterrupt:
    pass
final_architecture = trainer.export()
print('Final architecture:', trainer.export())

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')
fname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.json'
with open(os.path.join('checkpoints', fname), 'w') as checkpoint:
    json.dump(trainer.export(), checkpoint)
