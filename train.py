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

args = parser.parse_args()

args.device = None
if args.cuda=='true':
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        print("cuda not available")
else:
    args.device = torch.device('cpu')
print(args.device)

