# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as cPickle
from argparse import ArgumentParser
from pprint import pprint

import h5py
import numpy as np
from sklearn.utils import shuffle
from tqdm import trange

from config import load_global_conf, get_global_conf

parser = ArgumentParser()
parser.add_argument('--conf', default='conf/demo.yml', help='Config file')
args = parser.parse_args()

load_global_conf(args.conf)
conf = get_global_conf()

print("Experiment parameters:")
pprint(conf)


def process_2012(data, ith, offset, num_classes, num_per_classes_train, num_per_classes_val):
    cls_num_begin = ith * num_classes
    cls_num_end = cls_num_begin + num_classes

    train_feature, train_label, eval_feature, eval_label = [], [], [], []

    f = data['feat']

    for clazz in trange(cls_num_begin, cls_num_end, desc="inner"):
        # get from raw data
        fd = data[f[clazz][0]][()].T

        # random select
        index = np.random.permutation(fd.shape[0])

        train_feature.extend(fd[index[:num_per_classes_train]])
        eval_feature.extend(fd[index[num_per_classes_train:num_per_classes_train + num_per_classes_val]])
        train_label.extend([clazz + offset * num_classes] * num_per_classes_train)
        eval_label.extend([clazz + offset * num_classes] * num_per_classes_val)

    # Make it np.array
    train_feature = np.array(train_feature)
    train_label = np.array(train_label)
    eval_feature = np.array(eval_feature)
    eval_label = np.array(eval_label)

    # Shuffle
    train_feature, train_label = shuffle(train_feature, train_label)
    eval_feature, eval_label = shuffle(eval_feature, eval_label)

    return {"train_feature": train_feature, "train_label": train_label,
            "eval_feature": eval_feature, "eval_label": eval_label}


def process_2010(data, ith, offset, num_classes, num_per_classes_train, num_per_classes_val):
    cls_num_begin = ith * num_classes
    cls_num_end = cls_num_begin + num_classes

    train_feature, train_label, eval_feature, eval_label = [], [], [], []

    l = np.array(data['cls_label'][()][0], dtype=np.int)
    f = data['feat'][()].T

    for clazz in trange(cls_num_begin, cls_num_end, desc="inner"):
        # get from raw data
        index = shuffle(np.where(l == clazz + 1)[0])

        train_feature.extend(f[index[:num_per_classes_train]])
        eval_feature.extend(f[index[num_per_classes_train:num_per_classes_train + num_per_classes_val]])
        train_label.extend([clazz + offset * num_classes] * num_per_classes_train)
        eval_label.extend([clazz + offset * num_classes] * num_per_classes_val)

    # Make it np.array
    train_feature = np.array(train_feature)
    train_label = np.array(train_label)
    eval_feature = np.array(eval_feature)
    eval_label = np.array(eval_label)

    # Shuffle
    train_feature, train_label = shuffle(train_feature, train_label)
    eval_feature, eval_label = shuffle(eval_feature, eval_label)

    return {"train_feature": train_feature, "train_label": train_label,
            "eval_feature": eval_feature, "eval_label": eval_label}


def main():
    content_2012 = h5py.File(os.path.join(conf.raw_data_path, 'ImageNet_2012_VGG.mat'))
    content_2010 = h5py.File(os.path.join(conf.raw_data_path, 'ImageNet_2010_VGG.mat'))['imgtag']['te']

    data = []

    # Generate first group from 2012
    tmp = process_2012(content_2012, 0, 0, conf.nb_cl, conf.nb_train, conf.nb_val)
    data.append(tmp)

    # Generate next groups from 2010
    for ith in trange(conf.nb_groups - 1, desc="outer"):
        tmp = process_2010(content_2010, ith, 1, conf.nb_cl, conf.nb_train, conf.nb_val)
        data.append(tmp)

    # Save
    os.makedirs(conf.datapath, exist_ok=True)
    fnm = 'data_{}_{}_{}_{}_{}.pkl'.format(conf.name, conf.nb_groups, conf.nb_cl, conf.nb_train, conf.nb_val)
    cPickle.dump(data, open(os.path.join(conf.datapath, fnm), 'wb'))


if '__main__' == __name__:
    main()
