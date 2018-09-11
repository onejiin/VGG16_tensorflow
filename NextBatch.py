import os
import sys
import numpy as np
import h5py
import scipy.io as sio
import cv2
import tensorflow as tf

__author__ = 'wjung'


class Dataset:
    def __init__(self, train_list, test_list):

        # 1. Load training features (path) and labels
        self.feature_dimension = 0
        self.train_size = 0
        self.total_train_num = 0
        self.train_image = []
        self.train_label = []
        with open(train_list) as f:
            lines = f.readlines()
            for l in lines:
                items = l.strip('\n')
                self.train_image.append(items)
                self.train_size += 1
                set_x = cv2.imread(items)
                # set_x = sio.loadmat(items)['featureTrain']
                # self.total_train_num += set_x.shape[0]
                # self.feature_dimension = set_x.shape[1]
                # #items_lab = items[:-11] + 'label.mat'
                # items_lab = 'VAD' + items[8:]
                # self.train_label.append(items_lab)
        label_list = train_list[:-8] + 'label.txt'
        with open(label_list) as f:
            lines = f.readlines()
            for l in lines:
                label = l.strip('\n')
                self.train_label.append(label)

        # 2. Load testing features (path) and labels
        self.test_size = 0
        self.total_test_num = 0
        self.test_image = []
        self.test_label = []
        with open(test_list) as f:
            lines = f.readlines()
            for l in lines:
                items = l.strip('\n')
                self.test_image.append(items)
                self.test_size += 1
        label_list = test_list[:-8] + 'label.txt'
        with open(label_list) as f:
            lines = f.readlines()
            for l in lines:
                label = l.strip('\n')
                self.test_label.append(label)

        # 3. Init params
        self.train_ptr = 0
        self.test_ptr = 0
        self.last_train_size = 0
        self.last_test_size = 0

        # hdf5 file read
        # file count in hdf5 list
        self.train_hdf5_lines = 0
        self.test_hdf5_lines = 0
        # data count in one hdf5 file
        self.train_1024_lines = 0
        self.test_1024_lines = 0

    def batchsize_data(self, batch_size, train_file, train_label):

        # CNN input  : [[batch_size] [width] [height] [channel]]
        # CNN output : [[batch_size] [width] [height] [channel]]
        imgs = np.ndarray([batch_size, 224, 224, 3])
        labels = np.ndarray([batch_size, 2])
        #
        for i in range(batch_size):
            # data
            img = cv2.imread(train_file[i])
            label = train_label[i]
            if img.shape[0] != 224 and img.shape[1] != 224:
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            imgs[i] = np.reshape(img, [1, 224, 224, 3])
            # label
            onehot = np.zeros((1, 2))
            onehot[np.arange(1), np.array(int(label))] = 1
            labels[i] = onehot
            # labels[i] = tf.one_hot(label, 1000)
        #
        return imgs, labels

    def next_batch(self, batch_size, phase):
        if self.train_1024_lines + batch_size < self.train_size: #total_saving_data:
            # file read as bach size
            if phase == 'train':
                train_file = self.train_image[self.train_1024_lines:self.train_1024_lines+batch_size]
                train_label = self.train_label[self.train_1024_lines:self.train_1024_lines+batch_size]
            elif phase == 'test':
                train_file = self.test_image[self.train_1024_lines:self.train_1024_lines + batch_size]
                train_label = self.test_label[self.train_1024_lines:self.train_1024_lines + batch_size]

            features, labels = self.batchsize_data(batch_size, train_file, train_label)
            self.train_1024_lines += batch_size
        else:
            # curr_idx = self.train_size - self.train_1024_lines
            # next_idx = batch_size - curr_idx                        # next_hdf5[0 ~ next_idx]
            curr_idx = self.train_1024_lines
            next_idx = batch_size - (self.train_size - curr_idx)

            # file read as bach size
            if phase == 'train':
                train_file = self.train_image[curr_idx:]
                train_file += self.train_image[:next_idx]
                train_label = self.train_label[curr_idx:]
                train_label += self.train_label[:next_idx]
            elif phase == 'test':
                train_file = self.test_image[curr_idx:]
                train_file += self.test_image[:next_idx]
                train_label = self.test_label[curr_idx:]
                train_label += self.test_label[:next_idx]

            features, labels = self.batchsize_data(batch_size, train_file, train_label)

            if phase == 'train':
                self.train_1024_lines = 0   # new file start
                self.train_hdf5_lines += 1  # next hdf5 file
                if self.train_hdf5_lines == self.train_size:
                    self.train_hdf5_lines = 0

            elif phase == 'test':
                self.train_1024_lines = 0  # new file start
                self.test_hdf5_lines += 1  # next hdf5 file
                if self.test_hdf5_lines == self.test_size:
                    self.test_hdf5_lines = 0

        self.train_ptr += 1

        return features, labels
