import argparse
import math
import random

import h5py
import numpy as np
import os
import sys
import torch
import provider
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import models.model_cls
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

parser = argparse.ArgumentParser(description='PyTorch Point Cloud Classification Model')
parser.add_argument('--cuda', type=str, default='false', help='use CUDA')
parser.add_argument('--num_point', type=int, default=1024)
parser.add_argument('--max_epoch', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--decay_step', type=int, default=200000)
parser.add_argument('--decay_rate', type=float, default=0.7)
parser.add_argument('--model', default='model_cls')
parser.add_argument('--log_dir', default='log')
args = parser.parse_args()

args.manualSeed = random.randint(1, 10000)
torch.manual_seed(args.manualSeed)
args.device = None
if args.cuda == 'true':
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        print("cuda not available")
else:
    args.device = torch.device('cpu')
print(args.device)
DEVICE = torch.device(args.device)

BATCH_SIZE = args.batch_size
NUM_POINT = args.num_point
MAX_EPOCH = args.max_epoch
BASE_LEARNING_RATE = args.learning_rate
MOMENTUM = args.momentum
OPTIMIZER = args.optimizer
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate

print('\nbatch size', BATCH_SIZE, '\nnum point', NUM_POINT, \
      '\nmax epoch', MAX_EPOCH, '\nbase learning rate', \
      BASE_LEARNING_RATE, '\nmomentum', MOMENTUM, '\noptimizer', \
      OPTIMIZER, '\ndecay step', \
      DECAY_STEP, '\ndecay rate', DECAY_RATE)
print(args)
LOG_DIR = args.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args) + '\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

input_transform = models.model_cls.input_transform_net()
pre_feature = models.model_cls.pre_feature_transfrom_net()
feature_transform = models.model_cls.feature_transform_net()
output = models.model_cls.output_net()

model = models.model_cls.point_cls()


optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def exp_lr_decay(init_lr, global_step, decay_steps, decay_rate, staircase=True):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if staircase:
        lr = init_lr * decay_rate ** (global_step // decay_steps)
    else:
        lr = init_lr * decay_rate ** (global_step / decay_steps)
    return lr


def get_learning_rate(batch):
    learning_rate = exp_lr_decay(BASE_LEARNING_RATE, batch * BATCH_SIZE, DECAY_STEP, DECAY_RATE, staircase=True)
    '''
     BASE_LEARNING_RATE,  # Base learning rate.
     batch * BATCH_SIZE,  # Current index into the dataset.
     DECAY_STEP,          # Decay step.
     DECAY_RATE,          # Decay rate
    '''
    learning_rate = max(learning_rate, 0.00001)
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = exp_lr_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    '''
    BN_INIT_DECAY,
    batch*BATCH_SIZE,
    BN_DECAY_DECAY_STEP,
    BN_DECAY_DECAY_RATE,
    '''
    bn_decay = min(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


for epoch in range(args.max_epoch):
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(args.batch_size):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            jittered_data = torch.from_numpy(jittered_data).float()
            label=current_label[start_idx:end_idx]
            print(jittered_data.type())
            optimizer.zero_grad()
            model.train()
            criterion = nn.CrossEntropyLoss()
            pred = model(jittered_data)
            print(pred.size())
            loss= criterion(pred,label)
            loss.backward()
            optimizer.step()