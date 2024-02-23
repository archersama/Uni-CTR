import pandas as pd
import torch

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from configs import config as cfg

from inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from callbacks import EarlyStopping, ModelCheckpoint
from utils import setup_seed

from models.autoint import AutoInt

def data_process(data_path, data_source):
    if data_source == 'amazon':
        data = pd.read_csv(data_path, encoding='utf-8')
        data['rating'] = data['overall'].apply(lambda x: 1 if x > 3 else 0)
        data = data.drop(data[data['overall'] == 3].index).reset_index()
        data = data.sort_values(by='unixReviewTime', ascending=True)
        train = data.iloc[:int(len(data) * 0.9)].copy()
        test = data.iloc[int(len(data) * 0.9):].copy()
        return train, test, data
    elif data_source =='welink':
        data = pd.read_csv(data_path, encoding='utf-8')
        data = data.sort_values(by='event_time', ascending=True)
        # data = data[data['scope'] == 2]
        train = data.iloc[:int(len(data) * 0.9)].copy()
        test = data.iloc[int(len(data) * 0.9):].copy()
        return train, test, data



def split_data(data, ration):
    train_index, test_index = [], []
    all_user_id = data['user_id'].unique()
    for i in range(len(all_user_id)):
        index = data[data.user_id == all_user_id[i]].index.tolist()
        slice_index = int(len(index) * ration)
        train_index.extend(index[:slice_index])
        test_index.extend(index[slice_index:])
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    return train, test

def get_var_feature(data, col):
    key2index = {}

    def split(x):
        if isinstance(x, float):
            x = "item_-1"
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding"
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    var_feature = list(map(split, data[col].values))
    var_feature_length = np.array(list(map(len, var_feature)))
    max_len = max(var_feature_length)
    var_feature = pad_sequences(var_feature, maxlen=max_len, padding='post', )
    return key2index, var_feature, max_len

def get_test_var_feature(data, col, key2index, max_len):
    print("user_hist_list: \n")
    def split(x):
        if isinstance(x, float):
            x = "item_-1"
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding"
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    test_hist = list(map(split, data[col].values))
    test_hist = pad_sequences(test_hist, maxlen=max_len, padding='post')

    return test_hist


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


if __name__=='__main__':

    # 1. load data
    print("1")
    embedding_dim = 32
    epoch = 10
    batch_size = 2048
    seed = 2012
    setup_seed(seed)
    lr = 0.00005
    dropout = 0.3

    data_path = cfg.struct_path
    data_source = cfg.dataset

    train, test, data = data_process(data_path, data_source)

    if data_source == 'amazon':
        sparse_features = ['user_id', 'asin', 'brand']
        dense_features = ['new_price']
        target = ['rating']
        item_num = len(data['asin'].value_counts()) + 5

    # 2. Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(data[feat])
        train[feat] = lbe.transform(train[feat])
        test[feat] = lbe.transform(test[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    if len(dense_features) != 0:
        mms.fit(data[dense_features])
        train[dense_features] = mms.transform(train[dense_features])
        test[dense_features] = mms.transform(test[dense_features])

    # 2. Preprocess the sequence feature

    sparse_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=embedding_dim)
                              for i, feat in enumerate(sparse_features)]

    if len(dense_features) != 0:
        dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
    if data_source != "taobao":
        user_key2index, train_user_hist, user_maxlen = get_var_feature(train, 'user_hist')
        user_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('user_hist',
                                                                   vocabulary_size=item_num,
                                                                   embedding_dim=32),
                                                        maxlen=user_maxlen, combiner='mean', length_name=None)]


    # 3.generate input data for model

    sparse_feature_columns += user_varlen_feature_columns

    if len(dense_features) != 0:
        linear_feature_columns = sparse_feature_columns + dense_feature_columns
        dnn_feature_columns = sparse_feature_columns + dense_feature_columns
    else:
        linear_feature_columns = sparse_feature_columns
        dnn_feature_columns = sparse_feature_columns

    train_model_input = {name: train[name] for name in sparse_features + dense_features}

    if data_source != "taobao":
        train_model_input['user_hist'] = train_user_hist

    # 4. Define Model,train,predict and evaluate
    device = 'cpu'
    if torch.cuda.is_available():
        print("cuda is available")
        device = 'cuda:0'

    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=1, patience=3, mode='max', baseline=None)
    mdckpt = ModelCheckpoint(filepath='base_model.ckpt', monitor='val_auc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

    model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout, device=device)
    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy', 'logloss'])
    model.fit(train_model_input, train[target].values, batch_size=batch_size, epochs=epoch, verbose=2, validation_split=0.1, callbacks=[es, mdckpt])

    model.load_state_dict(torch.load('base_model.ckpt'))
    model.eval()

    test_model_input = {name: test[name] for name in sparse_features + dense_features}

    if data_source != "taobao":
        test_user_hist = get_test_var_feature(test, 'user_hist', user_key2index, user_maxlen)
        test_model_input['user_hist'] = test_user_hist

    pred_ts = model.predict(test_model_input, batch_size=batch_size)

    print("test LogLoss", round(log_loss(test[target].values, pred_ts), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ts), 4))

