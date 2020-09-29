import torch
import numpy as np
import os
import sys
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))


class get_transform(nn.Module):
    def __init__(self, num_classes=16):
        super(get_transform, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 1024, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d((2048, 1), stride=(2, 2))
        )
        self.linear1 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(128, 9, bias=True)
        )

    def forward(self, x):
        point_cloud = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = x.view(32, 1024)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(32, 3, 3)

        return x


class get_transform_K(nn.Module):
    def __init__(self):
        super(get_transform_K, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d((2048, 1), stride=(2, 2))
        )
        self.linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 16384)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(32, 1024)
        x = self.linear(x)
        x = x.view(32, 128, 128)
        return x


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.get_transform = get_transform()
        self.get_transform_K = get_transform_K()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cnn2 = nn.Sequential(

            nn.Conv2d(64, 128, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.cnn3 = nn.Sequential(

            nn.Conv2d(128, 128, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(128, 512, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.cnn5 = nn.Sequential(

            nn.Conv2d(512, 2048, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(2048),
            nn.ReLU(),

        )
        self.maxpool = nn.MaxPool2d((2048, 1), stride=(2, 2))
        self.linear = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(256, 16)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(4944, 256, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Conv2d(256, 256, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Conv2d(256, 128, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 50, (1, 1), stride=(1, 1)),
            nn.ReLU()
        )

    def forward(self, x, y):
        pc = x
        pc = pc.view(32, 2048, 3)
        x = x.view(32, 1, 2048, 3)
        x = self.get_transform(x)
        pc_transformed = torch.matmul(pc, x)

        input_image = pc_transformed.view(32, 1, 2048, 3)
        out1 = self.cnn1(input_image)
        out2 = self.cnn2(out1)
        out3 = self.cnn3(out2)
        transform = self.get_transform_K(out3)
        squeezed_out3 = out3.view(32, 2048, 128)
        net_transformed = torch.matmul(squeezed_out3, transform)
        net_transformed = net_transformed.view(32, 128, 2048, 1)
        out4 = self.cnn4(net_transformed)
        out5 = self.cnn5(out4)
        out_max = self.maxpool(out5)
        net = out_max.view(32, 2048)
        net = self.linear(net)
        one_hot_label_expand = y.view(32, 1, 1, 16)
        out_max = torch.cat([out_max.view(32, 1, 1, 2048), one_hot_label_expand], dim=3)
        expand = out_max.repeat([1, 2048, 1, 1])
        concat = torch.cat(
            [expand, out1.view(32, 2048, 1, 64), out2.view(32, 2048, 1, 128), out3.view(32, 2048, 1, 128),
             out4.view(32, 2048, 1, 512), out5.view(32, 2048, 1, 2048)], dim=3)
        concat = concat.view(32, 4944, 2048, 1)
        net2 = self.net2(concat)
        net2 = net2.view(32, 2048, 50)
        return net, net2, transform


def get_loss(l_pred, seg_pred, label, seg, weight, end_point):
    label = label.type(torch.long)
    per_instance_label_loss = sparse_softmax_cross_entropy_with_logits1(l_pred, label)
    label_loss = torch.mean(per_instance_label_loss)
    per_instance_seg_loss = sparse_softmax_cross_entropy_with_logits2(seg_pred, seg)
    per_instance_seg_loss = torch.mean(per_instance_seg_loss, dim=1)
    seg_loss = torch.mean(per_instance_seg_loss)
    per_instance_seg_pred_res = torch.argmax(seg_pred, 2)
    K = end_point.shape[1]
    mat_diff = torch.matmul(end_point, end_point.permute(0, 2, 1) - torch.tensor(np.eye(K), dtype=torch.float))
    mat_diff_loss = torch.sum(mat_diff ** 2) / 2
    total_loss = weight * seg_loss + (1 - weight) * label_loss + mat_diff_loss * 1e-3

    return total_loss, label_loss, per_instance_label_loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res


def sparse_softmax_cross_entropy_with_logits2(input, target):
    input = F.softmax(input, dim=-1)
    target = target.view(32, 2048, 1)
    num_classes = 50
    f = torch.arange(num_classes).reshape(1, num_classes)
    f=f.float()
    one_hot_target = (target == f).float()
    loss = -torch.sum(one_hot_target * torch.log(input), [-1])

    return loss


def sparse_softmax_cross_entropy_with_logits1(input, target):
    input = F.softmax(input, dim=-1)
    target = target.view(32, 1)
    num_classes = 16
    f = torch.arange(num_classes).reshape(1, num_classes)
    one_hot_target = (target == f).float()
    loss = -torch.sum(one_hot_target * torch.log(input), [-1])

    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 2048, 3))
    input_label = Variable(torch.rand(32, 16))
    model = get_model()
    x, y, z = model(sim_data, input_label)
    print(str(x.shape) + str(y.shape) + str(z.shape))

    l_pred = Variable(torch.rand(32, 16))
    seg_pred = Variable(torch.rand(32, 2048, 50))
    label = torch.LongTensor(32).random_(0, 16)
    seg = torch.LongTensor(32, 2048).random_(0, 50)
    end_points = Variable(torch.rand(32, 128, 128))
    print(str(get_loss(l_pred, seg_pred, label, seg, 1.0, end_points)))
