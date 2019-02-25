import argparse
import math
import h5py
import numpy as np
import importlib
import os
import sys
import torch
import provider
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

parser = argparse.ArgumentParser(description='PyTorch Point Cloud Classification Model')
parser.add_argument('--cuda', type=str, default='false', help='use CUDA')
parser.add_argument('--num_point', type=int, default=1024)
parser.add_argument('--max_epoch', type=int, default=250)
parser.add_argument('--batch_size', type=int , default=32)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--decay_step', type=int, default=200000)
parser.add_argument('--decay_rate', type=float, default=0.7)
parser.add_argument('--model', default='model_cls')
parser.add_argument('--log_dir', default='log')
args = parser.parse_args()

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

print('\nbatch size', BATCH_SIZE, '\nnum point', NUM_POINT,\
       '\nmax epoch', MAX_EPOCH, '\nbase learning rate',\
      BASE_LEARNING_RATE ,'\nmomentum', MOMENTUM, '\noptimizer',\
      OPTIMIZER, '\ndecay step',\
      DECAY_STEP,'\ndecay rate', DECAY_RATE)

MODEL = importlib.import_module(args.model)
MODEL_FILE = os.path.join(BASE_DIR,'models', args.model+'.py')
LOG_DIR = args.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR,'log_train.txt'),'w')
LOG_FOUT.write(str(args)+'\n')


