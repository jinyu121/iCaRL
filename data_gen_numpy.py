# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as cPickle
from argparse import ArgumentParser
from pprint import pprint

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


def process(ith, feature_train, label_train, feature_test, label_test,
            num_classes, num_per_classes_train, num_per_classes_val):
    cls_num_begin = ith * num_classes
    cls_num_end = cls_num_begin + num_classes

    train_feature, train_label, eval_feature, eval_label = [], [], [], []

    for clazz in trange(cls_num_begin, cls_num_end, desc="Class"):
        index_train = shuffle(np.where(label_train == clazz)[0])[:num_per_classes_train]
        train_feature.append(feature_train[index_train, :])
        train_label.extend([clazz] * num_per_classes_train)

        index_val = shuffle(np.where(label_test == clazz)[0])[:num_per_classes_val]
        eval_feature.append(feature_test[index_val, :])
        eval_label.extend([clazz] * num_per_classes_val)

    # Make it np.array
    train_feature = np.concatenate(train_feature)
    train_label = np.array(train_label)
    eval_feature = np.concatenate(eval_feature)
    eval_label = np.array(eval_label)

    # Shuffle
    train_feature, train_label = shuffle(train_feature, train_label)
    eval_feature, eval_label = shuffle(eval_feature, eval_label)

    return {"train_feature": train_feature, "train_label": train_label,
            "eval_feature": eval_feature, "eval_label": eval_label}


def main():
    train_feature = np.load(os.path.join(conf.raw_data_path, 'features_train.npy'))
    train_label = np.load(os.path.join(conf.raw_data_path, 'targets_train.npy'))
    eval_feature = np.load(os.path.join(conf.raw_data_path, 'features_test.npy'))
    eval_label = np.load(os.path.join(conf.raw_data_path, 'targets_test.npy'))

    data = []

    for ith in trange(conf.nb_groups, desc="group"):
        tmp = process(ith, train_feature, train_label, eval_feature, eval_label, conf.nb_cl, conf.nb_train, conf.nb_val)
        data.append(tmp)

    # Save
    os.makedirs(conf.datapath, exist_ok=True)
    fnm = 'data_{}_{}_{}_{}_{}.pkl'.format(conf.name, conf.nb_groups, conf.nb_cl, conf.nb_train, conf.nb_val)
    cPickle.dump(data, open(os.path.join(conf.datapath, fnm), 'wb'))


if '__main__' == __name__:
    main()
