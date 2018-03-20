import numpy as np

try:
    import cPickle
except:
    import pickle as cPickle


def prepare_files_fake(universal_data):
    train_index, train_feature, train_label, valid_index, valid_feature, valid_label = [], [], [], [], [], []
    sta_train, fin_train, sta_valid, fin_valid = 0, 0, 0, 0
    for i, data in enumerate(universal_data):
        tf, tl, ef, el = data['train_feature'], data['train_label'], data['eval_feature'], data['eval_label']

        fin_train = sta_train + len(tl)
        train_index.append(np.arange(sta_train, fin_train))
        train_feature.append(tf)
        train_label.append(tl)
        sta_train = fin_train

        fin_valid = sta_valid + len(el)
        valid_index.append(np.arange(sta_valid, fin_valid))
        valid_feature.append(ef)
        valid_label.append(el)
        sta_valid = fin_valid

    return train_index, \
           np.concatenate(train_feature), \
           np.concatenate(train_label), \
           valid_index, \
           np.concatenate(valid_feature), \
           np.concatenate(valid_label)
