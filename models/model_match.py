import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import sys
import os
from torch.autograd import Variable


import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import sys
import os
from torch.autograd import Variable
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class input_transform_net(nn.Module):

    def __init__(self):
        super(input_transform_net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 128, (1, 1), stride=(1, 1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1, 1), stride=(1, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.mp = nn.MaxPool2d((1024, 1), stride=(2, 2))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp(x)
        """maxpooling output N,C,H,W (32,1024,1,1)"""
        x = x.view(32, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, 3, 3)
        return x


class pre_feature_transfrom_net(nn.Module):

    def __init__(self):
        super(pre_feature_transfrom_net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x


class feature_transform_net(nn.Module):
    def __init__(self):
        super(feature_transform_net, self).__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, (1, 1), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 128, (1, 1), stride=(1, 1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1, 1), stride=(1, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4096)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.mp = nn.MaxPool2d((1024, 1), stride=(2, 2))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp(x)
        x = x.view(32, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(32, 64, 64)
        return x


class output_net(nn.Module):
    def __init__(self):
        super(output_net, self).__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, (1, 1), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 128, (1, 1), stride=(1, 1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1, 1), stride=(1, 1))
        self.fc1 = nn.Linear(1024, 1000)
        self.fc2 = nn.Linear(1000, 900)
        self.fc3 = nn.Linear(900, 864)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(900)
        self.mp = nn.MaxPool2d((1024, 1), stride=(2, 2))
        self.dp1 = nn.Dropout(p=0.7)
        self.dp2 = nn.Dropout(p=0.7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp(x)
        """maxpooling output N,C,H,W (32,1024,1,1)"""
        x = x.view(32, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(self.dp1(x))))
        x = self.fc3(self.dp2(x))
        return x


def _variable_with_weight_decay(shape, stddev, wd, use_xavier=True):
    return 0


class point_cls(nn.Module):
    def __init__(self):
        super(point_cls, self).__init__()
        self.inputrans = input_transform_net()
        self.prefeature = pre_feature_transfrom_net()
        self.featuretransform = feature_transform_net()
        self.output = output_net()

    def forward(self, x):
        point_with_channel = x.view(32, 1, 1024, 3)
        out = self.inputrans(point_with_channel)
        data = torch.matmul(x, out)
        data=data.view(32,1,1024,3)
        pre_out = self.prefeature(data)
        out2 = self.featuretransform(pre_out)
        pre_out = pre_out.squeeze()
        pre_out = pre_out.permute(0, 2, 1)
        out2 = out2.permute(0, 2, 1)
        net_transformed = torch.matmul(pre_out, out2)
        net_transformed = net_transformed.permute(0, 2, 1)
        net_transformed = net_transformed.view(32, 64, 1024, 1)
        output = self.output(net_transformed)
        return output


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 1, 1024, 3))
    print(sim_data.type())
    trans = input_transform_net()
    out = trans(sim_data)
    print('input_transform_net', out.size())

    sim_data1 = Variable(torch.rand(32, 1024, 3))
    sim_data1 = torch.matmul(sim_data1, out)
    sim_data1 = sim_data1.view(32, 1, 1024, 3)
    pre = pre_feature_transfrom_net()
    out1 = pre(sim_data1)
    print('pre_feature_transfrom_net', out1.size())

    feature = feature_transform_net()
    out2 = feature(out1)
    print('feature trans net', out2.size())

    """feature trans mul pre feature trans"""
    out1 = out1.squeeze()
    out1 = out1.permute(0, 2, 1)
    out2 = out2.permute(0, 2, 1)
    print(out1.size(), out2.size())
    net_transformed = torch.matmul(out1, out2)
    print('net_transformed', net_transformed.size())
    net_transformed = net_transformed.permute(0, 2, 1)
    net_transformed = net_transformed.view(32, 64, 1024, 1)
    print('net_transformed torch style', net_transformed.size())

    outputnet = output_net()
    out3 = outputnet(net_transformed)
    print('final out put', out3.size())

    sim_data2 = Variable(torch.rand(32, 1024, 3))
    pointcls=point_cls()
    out4=pointcls(sim_data2)
    print(out4.size())