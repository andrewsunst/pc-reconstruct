import argparse
import torch
import numpy as np
import json
import os
import sys
import datetime

# start timing
from torch import optim

begin_time = datetime.datetime.now()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider
import models.model_seg

# settings for model
parser = argparse.ArgumentParser(description='PyTorch Point Cloud Classification Model')
parser.add_argument('--cuda', type=str, default='false', help='use CUDA')
parser.add_argument('--point_num', type=int, default=2048)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--output_dir', type=str, default='train_results',
                    help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
args = parser.parse_args()

hdf5_data_dir = os.path.join(BASE_DIR, './data/hdf5_data')
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
point_num = args.point_num
batch_size = args.batch
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
color_map = json.load(open(color_map_file, 'r'))

all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
fin = open(all_obj_cats_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
fin.close()

all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
NUM_CATEGORIES = 16
NUM_PART_CATS = len(all_cats)  # 50

print('#### Batch Size: {0}'.format(batch_size))
print('#### Point Number: {0}'.format(point_num))
print('#### Training using GPU: {0}'.format(args.cuda))

DECAY_STEP = 16881 * 20
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-5

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.001
MOMENTUM = 0.9
TRAINING_EPOCHES = args.epoch
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'val_hdf5_file_list.txt')
MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')

train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
num_train_file = len(train_file_list)
test_file_list = provider.getDataFiles(TESTING_FILE_LIST)
num_test_file = len(test_file_list)

if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)


def printout(flog, data):
    print(data)
    flog.write(data + '\n')


def convert_label_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot


def update_lr(optimizer, epoch):
    optimizer.param_groups[0]['lr'] = BASE_LEARNING_RATE * DECAY_RATE ** (epoch * batch_size / DECAY_STEP)


model = models.model_seg.get_model()
model = model.to(args.device)

optimizer = optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE)

flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')

for epoch in range(args.epoch):

    printout(flog, '\n<<< Testing on the test dataset')
    with torch.no_grad():
        model = model.eval()
        # initialize the loss output
        total_loss = 0.0
        total_label_loss = 0.0
        total_seg_loss = 0.0
        total_label_acc = 0.0
        total_seg_acc = 0.0
        total_seen = 0

        total_label_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
        total_seg_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
        total_seen_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.int32)

        for i in range(num_test_file):
            cur_test_filename = os.path.join(hdf5_data_dir, test_file_list[i])
            printout(flog, 'Loading test file ' + cur_test_filename)
            cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_test_filename)
            cur_labels = np.squeeze(cur_labels)
            cur_labels_one_hot = convert_label_to_one_hot(cur_labels)

            num_data = len(cur_labels)
            num_batch = num_data // batch_size

            for j in range(num_batch):
                begidx = j * batch_size
                endidx = (j + 1) * batch_size
                pointclouds_ph = cur_data[begidx:endidx, ...]
                labels_ph = cur_labels[begidx:endidx, ...]
                input_label_ph = cur_labels_one_hot[begidx:endidx, ...]
                seg_ph = cur_seg[begidx:endidx, ...]

                pointclouds_ph = torch.from_numpy(pointclouds_ph)
                input_label_ph = torch.from_numpy(input_label_ph)
                labels_ph = torch.from_numpy(labels_ph)
                seg_ph = torch.from_numpy(seg_ph)

                pointclouds_ph = pointclouds_ph.float()
                input_label_ph = input_label_ph.float()
                labels_ph = labels_ph.float()
                seg_ph = seg_ph.float()

                pointclouds_ph = pointclouds_ph.to(args.device)
                input_label_ph = input_label_ph.to(args.device)
                labels_ph = labels_ph.to(args.device)
                seg_ph = seg_ph.to(args.device)

                labels_pred, seg_pred, end_points = model(pointclouds_ph, input_label_ph)
                total_loss, label_loss, per_instance_label_loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res = models.model_seg.get_loss(
                    labels_pred, seg_pred, labels_ph, seg_ph, 1.0, end_points)
                tensor_cur_seg = torch.from_numpy(cur_seg)
                tensor_cur_seg = tensor_cur_seg.long()
                midstep = per_instance_seg_pred_res.cpu() == tensor_cur_seg[begidx: endidx, :]
                midstep = midstep.data.numpy()
                per_instance_part_acc = np.mean(midstep, axis=1)
                average_part_acc = np.mean(per_instance_part_acc)
                total_seen += 1
                total_loss += total_loss
                total_label_loss += label_loss
                total_seg_loss += seg_loss

                per_instance_label_pred = np.argmax(labels_pred, axis=1)
                total_label_acc += np.mean(np.float32(per_instance_label_pred.data.numpy() == cur_labels[begidx:endidx, ...]))
                total_seg_acc += average_part_acc
                for shape_idx in range(begidx, endidx):
                    total_seen_per_cat[cur_labels[shape_idx]] += 1
                    total_label_acc_per_cat[cur_labels[shape_idx]] += np.int32(
                        per_instance_label_pred[shape_idx - begidx].cpu() == cur_labels[shape_idx])
                    total_seg_acc_per_cat[cur_labels[shape_idx]] += per_instance_part_acc[shape_idx - begidx]
        total_loss = total_loss * 1.0 / total_seen
        total_label_loss = total_label_loss * 1.0 / total_seen
        total_seg_loss = total_seg_loss * 1.0 / total_seen
        total_label_acc = total_label_acc * 1.0 / total_seen
        total_seg_acc = total_seg_acc * 1.0 / total_seen
        printout(flog, '\tTesting Total Mean_loss: %f' % total_loss)
        printout(flog, '\t\tTesting Label Mean_loss: %f' % total_label_loss)
        printout(flog, '\t\tTesting Label Accuracy: %f' % total_label_acc)
        printout(flog, '\t\tTesting Seg Mean_loss: %f' % total_seg_loss)
        printout(flog, '\t\tTesting Seg Accuracy: %f' % total_seg_acc)
    printout(flog, '\n<<< Training on the test dataset ...')

    current_lr = optimizer.param_groups[0]['lr']
    if current_lr > 0.00001:
        update_lr(optimizer, epoch)
        print('\nLearning rate updated')
    print('\nLearning rate for next epoch is: %0.9f' % optimizer.param_groups[0]['lr'])
