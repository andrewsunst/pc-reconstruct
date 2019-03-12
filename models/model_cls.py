import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import  torch
import sys
import os
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class input_transform_net(nn.Module):
    def __init__(self):
        super(input_transform_net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, (1,3), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 128,(1,1), stride=(1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024,(1,1),stride=(1,1))
        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,9)
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(256)

    def forward(self,x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))






def _variable_with_weight_decay(shape, stddev, wd, use_xavier=True):
    return 0
