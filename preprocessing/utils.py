import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def get_var_feature(data, col):
    key2index = {}

    def split(x):
        if isinstance(x, float):
            x = "item_-1"
        key_ans = x.split(" & ")
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    var_feature_list = list(map(split, data[col].values))
    var_feature_length = np.array(list(map(len, var_feature_list)))
    max_len = max(var_feature_length)
    var_feature = pad_sequences(var_feature_list, maxlen=max_len, padding='post', )
    return key2index, var_feature, max_len


def get_test_var_feature(data, col, key2index, max_len):
    def split(x):
        if isinstance(x, float):
            x = "item_-1"
        key_ans = x.split(" & ")
        for key in key_ans:
            if key not in key2index:
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    test_hist = list(map(split, data[col].values))
    test_hist = pad_sequences(test_hist, maxlen=max_len, padding='post', )
    return test_hist

def varlenSparseFeature(feat, feat_num, max_len, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param max_len: sequence max length
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat_name': feat, 'feat_num': feat_num, 'max_len': max_len, 'embed_dim': embed_dim}
