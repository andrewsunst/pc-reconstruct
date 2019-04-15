import provider
import os
import numpy as np
from collections import Counter
import torch
import argparse
import models.model_match
import torch.optim as optim
import math
import torch.nn as nn
from random import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

parser = argparse.ArgumentParser(description='PyTorch Point Cloud Classification Model')
parser.add_argument('--cuda', type=str, default='false', help='use CUDA')
parser.add_argument('--num_point', type=int, default=1024)
parser.add_argument('--max_epoch', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--decay_step', type=int, default=200000)
parser.add_argument('--decay_rate', type=float, default=0.7)
parser.add_argument('--model', default='model_cls')
parser.add_argument('--log_dir', default='log')
args = parser.parse_args()
BATCH_SIZE = args.batch_size
NUM_POINT = args.num_point
MAX_EPOCH = args.max_epoch
BASE_LEARNING_RATE = args.learning_rate
MOMENTUM = args.momentum
OPTIMIZER = args.optimizer
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate
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
LOG_DIR = args.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_match.txt'), 'w')
LOG_FOUT.write(str(args) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
train_file_idxs = np.arange(0, len(TRAIN_FILES))

# put things together
data = []
label = []
for fn in range(len(TRAIN_FILES)):
    current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
    current_data = current_data[:, 0:NUM_POINT, :]
    data.append(current_data)
    label.append(current_label)
templen = 0
for l1 in range(len(data)):
    templen += data[l1].shape[0]
data_complete = np.ndarray(shape=(templen, NUM_POINT, 3), dtype=float, order='F')
label_complete = np.ndarray(shape=(templen, 1), dtype=float, order='F')
counter = 0
for i in range(len(data)):
    for j in range(len(data[i])):
        data_complete[counter] = data[i][j]
        label_complete[counter] = label[i][j]
        counter += 1
label_complete = np.squeeze(label_complete)
# find the biggest type
type_count = Counter(label_complete)
most_common = type_count.most_common(1)
length = most_common[0][1]
pick_ones = np.ndarray(shape=(length, NUM_POINT, 3), dtype=float, order='F')
counter = 0
for i in range(len(label_complete)):
    if label_complete[i] == 8.0:
        pick_ones[counter] = data_complete[i]
        counter += 1

pick_ones = pick_ones[0:864]
labels = np.arange(0, 864)
model = models.model_match.point_cls()
model.to(args.device)

pick = randint(0, 864)
pick_points = data_complete[pick]
pick_label = labels[pick]
test_set = np.ndarray(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=float, order='F')
test_label = np.ndarray(shape=(BATCH_SIZE), dtype=float, order='F')
for i in range(BATCH_SIZE):
    test_set[i] = pick_points
    test_label[i] = pick_label
test_set = torch.from_numpy(test_set).float()
test_set = test_set.to(args.device)
test_label = torch.from_numpy(test_label).float()
test_label = test_label.to(args.device)
test_label = test_label.long()
for epoch in range(MAX_EPOCH):
    current_data, current_label, _ = provider.shuffle_data(pick_ones, labels)
    current_label = np.squeeze(current_label)
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    log_string('epoch No.' + str(epoch))
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
        jittered_data = provider.jitter_point_cloud(rotated_data)
        jittered_data = torch.from_numpy(jittered_data).float()
        jittered_data = jittered_data.to(args.device)
        label = current_label[start_idx:end_idx]
        label = torch.from_numpy(label).float()
        label = label.to(args.device)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate * math.pow(DECAY_RATE, (batch_idx * BATCH_SIZE) / DECAY_STEP))
        optimizer.zero_grad()
        model.train()
        criterion = nn.CrossEntropyLoss()
        pred = model(jittered_data)
        label = label.long()
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        loss_sum += loss
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))


    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            jittered_data = torch.from_numpy(jittered_data).float()
            jittered_data = jittered_data.to(args.device)
            label = current_label[start_idx:end_idx]
            label = torch.from_numpy(label).float()
            label = label.long()
            label = label.to(args.device)
            model = model.eval()
            pred_val = model(jittered_data)
            loss = criterion(pred_val, label)
            pred_choice = pred_val.data.max(1)[1]
            correct = pred_choice.eq(label.data).sum()
        log_string('correct  ' + str(correct))

# pick = randint(0, counter - 1)
# pick_points = data_complete[pick]
# x = pick_points[:, 0]
# y = pick_points[:, 1]
# z = pick_points[:, 2]
# fig = plt.figure()
#
# ax = Axes3D(fig)
# ax.scatter(x, y, z)
# plt.title('raw points')
# plt.show()
#
# tri = Delaunay(np.array([x, y]).T)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap=plt.cm.Spectral)
# plt.title('Delaunay mesh')
# plt.show()
