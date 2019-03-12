import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import  torch
import sys
import os
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def input_transform_net(point_cloud, K=3):

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = point_cloud[:, :, :, None]


def _variable_with_weight_decay(shape, stddev, wd, use_xavier=True):



def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=F.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_h, kernel_w,
                    num_in_channels, num_output_channels]



