# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import trange, tqdm

try:
    import cPickle
except:
    import pickle as cPickle

from utils import prepare_files, top_k_accuracy
from config import load_global_conf, get_global_conf

parser = ArgumentParser()
parser.add_argument('--conf', default='conf/demo.yml', help='Config file')
args = parser.parse_args()

load_global_conf(args.conf)
conf = get_global_conf()

print("Experiment parameters:")
pprint(conf)

######### Modifiable Settings ##########
nb_val = conf.nb_val  # Validation samples per class
nb_cl = conf.nb_cl  # Classes per group
nb_groups = conf.nb_groups  # Number of groups
nb_proto = conf.nb_proto  # Number of prototypes per class: total protoset memory/ total number of classes
top = conf.top  # Choose to evaluate the top X accuracy
test_group = conf.test_group  # 'all' / 'this' / '0'
feature_dim = conf.feature_dim
save_path = conf.save_path
########################################

os.makedirs(save_path, exist_ok=True)

### Initialization of some variables ###
class_means = np.zeros((feature_dim, nb_groups * nb_cl, 2, nb_groups))
loss_batch = []
files_protoset = [list() for _ in range(nb_groups * nb_cl)]

### Preparing the files for the training/validation ###

# Random mixing
np.random.seed(1993)

# Preparing the files per group of classes
print("Loading dataset ...\n")
fnm = 'data_{}_{}_{}_{}_{}.pkl'.format(conf.name, conf.nb_groups, conf.nb_cl, conf.nb_train, conf.nb_val)
universal_data = cPickle.load(open(os.path.join(conf.datapath, fnm), 'rb'))
index_train, features_train, labels_train, \
index_valid, features_valid, labels_valid = prepare_files(universal_data)

### Start of the main algorithm ###
for itera in trange(nb_groups, desc="Group"):

    # Train
    tqdm.write('Group #{}, total classes {}'.format(itera + 1, (itera + 1) * nb_cl))

    # Reducing number of exemplars for the previous classes
    nb_protos_cl = int(np.ceil(nb_proto * nb_groups * 1. / (itera + 1)))
    files_from_cl = index_train[itera]

    # Load the training samples of the current batch of classes in the feature space to apply the herding algorithm
    Dtot, processed_files, label_dico = features_train[files_from_cl, :], files_from_cl, labels_train[files_from_cl]

    Dtot = Dtot.T / np.linalg.norm(Dtot.T, axis=0)

    # Herding procedure : ranking of the potential exemplars
    # Exemplars selection
    for iter_dico in range(nb_cl):
        now_cls = itera * nb_cl + iter_dico
        ind_cl = np.where(label_dico == now_cls)[0]
        D = Dtot[:, ind_cl]
        files_iter = processed_files[ind_cl]
        mu = np.mean(D, axis=1)
        w_t = mu
        step_t = 0
        while not (len(files_protoset[now_cls]) == nb_protos_cl) and step_t < 1.1 * nb_protos_cl:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            w_t = w_t + mu - D[:, ind_max]
            step_t += 1
            if files_iter[ind_max] not in files_protoset[now_cls]:
                files_protoset[now_cls].append(files_iter[ind_max])

    # Compute theoretical class means for NCM and mean-of-exemplars for iCaRL
    for iteration2 in range(itera + 1):
        files_from_cl = index_train[iteration2]

        Dtot, processed_files, label_dico = features_train[files_from_cl, :], files_from_cl, labels_train[files_from_cl]

        Dtot = Dtot.T / np.linalg.norm(Dtot.T, axis=0)

        for iter_dico in range(nb_cl):
            now_cls = iteration2 * nb_cl + iter_dico
            ind_cl = np.where(label_dico == now_cls)[0]
            D = Dtot[:, ind_cl]
            files_iter = processed_files[ind_cl]

            # Normal NCM mean
            tmp_mean = np.mean(D, axis=1)
            class_means[:, now_cls, 1, itera] = tmp_mean / np.linalg.norm(tmp_mean)

            # iCaRL approximated mean (mean-of-exemplars)
            # use only the first exemplars of the old classes: nb_protos_cl controls the number of exemplars per class
            ind_herding = np.array(
                [np.where(files_iter == files_protoset[now_cls][i])[0][0] for i in
                 range(min(nb_protos_cl, len(files_protoset[now_cls])))])
            D_tmp = D[:, ind_herding]
            tmp_mean = np.mean(D_tmp, axis=1)
            class_means[:, now_cls, 0, itera] = tmp_mean / np.linalg.norm(tmp_mean)

    # Pickle class means and protoset
    info = "{}_cls_{}_proto_{}_group_{}_tot_{}".format(conf.name, nb_cl, nb_proto, itera, nb_cl * (itera + 1))
    with open(os.path.join(save_path, 'class_means_{}.pickle'.format(info)), 'wb') as fp:
        cPickle.dump(class_means, fp)
    with open(os.path.join(save_path, 'files_protoset_{}.pickle'.format(info)), 'wb') as fp:
        cPickle.dump(files_protoset, fp)

    # Test

    ## Get test data group
    if 'all' == test_group:
        eval_groups = [x for x in range(itera + 1)]
    elif 'this' == test_group:
        eval_groups = [itera]
    else:
        eval_groups = [int(test_group)]

    files_from_cl = np.concatenate([index_valid[i] for i in eval_groups])

    tqdm.write("Evaluation on groups {} ".format(eval_groups))
    mapped_prototypes, l = features_valid[files_from_cl, :], labels_valid[files_from_cl]

    pred_inter = mapped_prototypes.T / np.linalg.norm(mapped_prototypes.T, axis=0)

    sqd_icarl = -cdist(class_means[:, :, 0, itera].T, pred_inter.T, 'sqeuclidean').T
    tqdm.write('iCaRL accuracy:\t Top 1: {} / Top {}: {}'.format(top_k_accuracy(l, sqd_icarl, 1),
                                                                 top,
                                                                 top_k_accuracy(l, sqd_icarl, top)))

    sqd_ncm = -cdist(class_means[:, :, 1, itera].T, pred_inter.T, 'sqeuclidean').T
    tqdm.write('NCM accuracy:\t Top 1: {} / Top {}: {}'.format(top_k_accuracy(l, sqd_ncm, 1),
                                                               top,
                                                               top_k_accuracy(l, sqd_ncm, top)))

    tqdm.write("")
