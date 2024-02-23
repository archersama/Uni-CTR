import os
import sys

from models.star import STAR

# 在服务器上使用这段代码
# sys.path.insert(0, '/root/multi-domain')

import pandas as pd
from torch import nn
import torch.utils.data
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

import configs.config_multi_domain as cfg
from layers import AvgMeter, get_lr
from models.mmoe import MMOE
from models.ple import PLE
from preprocessing.utils import sparseFeature, get_var_feature, get_test_var_feature, varlenSparseFeature
from utils import setup_seed


warnings.filterwarnings("ignore")


def create_dataset(embed_dim, data_path, scenarios):
    data_df = pd.read_csv(data_path, sep='\t', encoding='utf-8')

    data_df['rating'] = data_df['overall'].apply(lambda x: 1 if x > 3 else 0)
    data_df['title'] = data_df['title'].fillna('-1')
    data_df['brand'] = data_df['brand'].fillna('-1')
    data_df = data_df.drop(data_df[data_df['rating'] == 3].index).reset_index()
    sparse_features = ['user_id', 'asin', 'brand', 'title']
    dense_features = ['new_price']
    item_num = len(data_df['asin'].value_counts()) + 5

    features = sparse_features + dense_features
    data_df[sparse_features] = data_df[sparse_features].fillna(-1)
    data_df[dense_features] = data_df[dense_features].fillna(0)

    # Bin continuous data into intervals.
    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    data_df[dense_features] = est.fit_transform(data_df[dense_features])

    for feat in sparse_features:
        lbe = LabelEncoder()
        data_df[feat] = lbe.fit_transform(data_df[feat])

    feature_columns = [sparseFeature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim) for feat in features]
    data_df.sort_values(by=['unixReviewTime'], ascending=True).reset_index()

    text_data_scene = [data_df[data_df['scenario'] == i] for i in scenarios]
    train_text_scene = [i.iloc[:int(len(i) * 0.9)].copy() for i in text_data_scene]
    test_text_scene = [i.iloc[int(len(i) * 0.9):].copy() for i in text_data_scene]
    valid_text_scene = [i.iloc[int(len(i) * 0.9):].copy() for i in train_text_scene]
    train_text_scene = [i.iloc[:int(len(i) * 0.9)].copy() for i in train_text_scene]

    train = pd.concat(train_text_scene)
    test = pd.concat(test_text_scene)
    valid = pd.concat(valid_text_scene)

    user_key2index, train_user_hist, user_maxlen = get_var_feature(train, 'user_hist')

    valid_user_hist = get_test_var_feature(valid, 'user_hist', user_key2index, user_maxlen)
    test_user_hist = get_test_var_feature(test, 'user_hist', user_key2index, user_maxlen)

    user_hist_feature_columns = varlenSparseFeature('user_hist',
                                                    len(user_key2index) + 1,
                                                    user_maxlen,
                                                    embed_dim=embed_dim)
    var_len_list = [train_user_hist, valid_user_hist, test_user_hist]
    varlen_feature_columns = [user_hist_feature_columns]
    train_x = train[features].values.astype('int32')
    train_scenario = train['scenario'].values.astype('int32')
    train_y = train['rating'].values.astype('int32')
    valid_x = valid[features].values.astype('int32')
    valid_y = valid['rating'].values.astype('int32')
    valid_scenario = valid['scenario'].values.astype('int32')
    test_x = test[features].values.astype('int32')
    test_y = test['rating'].values.astype('int32')
    test_scenario = test['scenario'].values.astype('int32')

    return (feature_columns, varlen_feature_columns,
            (train_x, train_y, train_scenario),
            (valid_x, valid_y, valid_scenario),
            var_len_list,
            (test_x, test_y, test_scenario))


class MMoeDataset(torch.utils.data.Dataset):
    def __init__(self, struct_data, label):
        self.rec_data, self.varlen_data, self.scenario = struct_data
        self.label = label

    def __getitem__(self, idx):
        item = {
            'rec_data': torch.tensor(self.rec_data[idx], dtype=torch.int),
            'varlen_data': torch.tensor(self.varlen_data[idx], dtype=torch.int),
            'scenario': torch.tensor(self.scenario[idx], dtype=torch.int),
            'label': torch.tensor(self.label[idx], dtype=torch.int),
        }
        return item
    def __len__(self):
        return len(self.label)


def build_loaders(input, label, mode):
    dataset = MMoeDataset(input, label)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True if mode == 'train' else False
    )
    return dataloader



def train_epoch(model, train_loader, optimizer, lr_scheduler, step, focalloss: bool, loss_fnc):
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    scaler = GradScaler()
    if cfg.mixed_precision:
        for batch in tqdm_object:
            rec_data = batch['rec_data'].to(cfg.device)
            varlen_data = batch['varlen_data'].to(cfg.device)
            scenario = batch['scenario'].unsqueeze(1).to(cfg.device)
            label = batch['label'].unsqueeze(1).to(cfg.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output, _ = model(rec_data, varlen_data, scenario)
                if focalloss:
                    loss = loss_fnc(output, label.float(), scenario)
                else:
                    loss = loss_fnc(output, label.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step == "batch":
                lr_scheduler.step()

            count = rec_data.size(0)
            loss_meter.update(loss.item(), count)

            auc = roc_auc_score(label.detach().cpu().numpy(), torch.sigmoid(output).detach().cpu().numpy())
            acc_meter.update(auc, count)

            tqdm_object.set_postfix(train_loss=loss_meter.avg, train_auc=acc_meter.avg, lr=get_lr(optimizer))

    return loss_meter, acc_meter


def valid_epoch(model, valid_loader, loss_fnc, scenarios):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    predicts = []
    labels = []

    scenario_predicts = [[] for _ in range(len(scenarios))]
    scenario_labels = [[] for _ in range(len(scenarios))]

    with torch.no_grad():
        for batch in tqdm_object:
            rec_data = batch['rec_data'].to(cfg.device)
            varlen_data = batch['varlen_data'].to(cfg.device)
            scenario = batch['scenario'].unsqueeze(1).to(cfg.device)
            label = batch['label'].unsqueeze(1).to(cfg.device)

            output, _ = model(rec_data, varlen_data, scenario)
            loss = loss_fnc(output, label.float())

            scenario_cpu = scenario.cpu().data.numpy()

            if cfg.mixed_precision:
                a = torch.sigmoid(output).cpu().data.numpy()
                predicts.extend(a)
                for i in range(len(scenarios)):
                    scenario_predicts[i].extend(a[scenario_cpu == scenarios[i]])

            b = label.cpu().data.numpy()
            labels.extend(b)
            for i in range(len(scenarios)):
                scenario_labels[i].extend(b[scenario_cpu == scenarios[i]])

            loss_meter.update(loss.item(), rec_data.size(0))
            tqdm_object.set_postfix(valid_loss=loss_meter.avg)

        print("valid_auc", roc_auc_score(labels, predicts))
        for i in range(len(scenarios)):
            print("valid_auc_scenario_{}".format(scenarios[i]), roc_auc_score(scenario_labels[i], scenario_predicts[i]))

    return loss_meter


def test_epoch(model, test_loader, loss_fnc, scenarios):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    predicts = []
    labels = []

    scenario_predicts = [[] for _ in range(len(scenarios))]
    scenario_labels = [[] for _ in range(len(scenarios))]

    with torch.no_grad():
        for batch in tqdm_object:
            rec_data = batch['rec_data'].to(cfg.device)
            varlen_data = batch['varlen_data'].to(cfg.device)
            scenario = batch['scenario'].unsqueeze(1).to(cfg.device)
            label = batch['label'].unsqueeze(1).to(cfg.device)

            output, _ = model(rec_data, varlen_data, scenario)
            loss = loss_fnc(output, label.float())

            scenario_cpu = scenario.cpu().data.numpy()

            if cfg.mixed_precision:
                a = torch.sigmoid(output).cpu().data.numpy()
                predicts.extend(a)
                for i in range(len(scenarios)):
                    scenario_predicts[i].extend(a[scenario_cpu == scenarios[i]])

            b = label.cpu().data.numpy()
            labels.extend(b)
            for i in range(len(scenarios)):
                scenario_labels[i].extend(b[scenario_cpu == scenarios[i]])

            loss_meter.update(loss.item(), rec_data.size(0))
            tqdm_object.set_postfix(test_loss=loss_meter.avg)

        print("test_auc", roc_auc_score(labels, predicts))
        for i in range(len(scenarios)):
            print("test_auc_scenario_{}".format(scenarios[i]), roc_auc_score(scenario_labels[i], scenario_predicts[i]))

    return loss_meter


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def compute_weights(self, scene_ids):
        # 计算每个场景id的频率
        unique_ids, counts = torch.unique(scene_ids, return_counts=True)
        freq = counts.float() / scene_ids.size(0)
        # 计算每个场景id的权重
        weights_dict = {id.item(): 1.0 / freq[i] for i, id in enumerate(unique_ids)}
        # 根据场景id的顺序返回权重
        return torch.tensor([weights_dict[id.item()] for id in scene_ids]).to(cfg.device)

    def forward(self, logits, labels, scene_ids):
        weights = self.compute_weights(scene_ids).reshape(-1, 1)
        bce_loss = self.bce_with_logits(logits, labels)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return (focal_loss * weights).mean()



def main(model_name="ple"):
    data_source = cfg.dataset
    seed = 2012
    setup_seed(seed)

    embed_dim = cfg.embed_dim
    (
        feature_columns,
        varlen_feature_columns,
        (train_x, train_y, train_scenario),
        (valid_x, valid_y, valid_scenario),
        var_len_list,
        (test_x, test_y, test_scenario)
    ) = create_dataset(embed_dim, cfg.data_path, cfg.scenarios)

    train_struct_data = (train_x, var_len_list[0], train_scenario)
    valid_struct_data = (valid_x, var_len_list[1], valid_scenario)
    test_struct_data = (test_x, var_len_list[2], test_scenario)

    train_loader = build_loaders(train_struct_data, train_y, 'train')
    valid_loader = build_loaders(valid_struct_data, valid_y, 'valid')
    test_loader = build_loaders(test_struct_data, test_y, 'test')

    if model_name == "ple":
        model = PLE(
            feature_columns=feature_columns,
            sequence_feature_columns=varlen_feature_columns,
            scenarios=cfg.scenarios,
            input_dim=embed_dim*cfg.multiplier,
            shared_expert_num=1,
            specific_expert_num=1,
            num_levels=1,
            expert_dnn_hidden_units=(512, 256),
            gate_dnn_hidden_units=(512, ),
            tower_dnn_hidden_units=(512, ),
            init_std=0.0001,
            task_names=('amazon fashion', 'musical instruments', 'gift cards'),
            device='cuda:0',
            gpus=cfg.device_ids,
        )
    elif model_name == "mmoe":
        model = MMOE(
            feature_columns=feature_columns,
            sequence_feature_columns=varlen_feature_columns,
            scenarios=cfg.scenarios,
            input_dims=embed_dim*cfg.multiplier,
            num_experts=3,
            tower_input_dims=512,
            tower_hidden_dims=512,
            hidden_dims=(1024, 1024),
        )
    elif model_name == "star":
        model = STAR(
            feature_columns=feature_columns,
            sequence_feature_columns=varlen_feature_columns,
            input_dims=embed_dim*cfg.multiplier,
        )
    else:
        raise NotImplementedError

    model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.4)


    # 定义线性下降的学习率调整函数
    lambda_lr = lambda epoch: (1 - epoch / cfg.epochs) * (cfg.max_lr - cfg.lr) + cfg.lr

    # 创建线性下降的学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.lr, max_lr=cfg.max_lr, step_size_up=5, mode="triangular2", cycle_momentum=False)

    if cfg.mixed_precision:
        loss_fnc = nn.BCEWithLogitsLoss()
        loss_fn_focal = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        loss_fnc = nn.BCELoss()
        loss_fn_focal = FocalLoss(alpha=0.25, gamma=2.0)

    step = "epoch"
    writer = SummaryWriter('./logs')

    best_loss = float('inf')
    print("Begin training...")
    for epoch in range(cfg.epochs):
        model.train()
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, lr_scheduler, step, cfg.focal_loss,
                                            loss_fnc=loss_fn_focal if cfg.focal_loss else loss_fnc)
        lr_scheduler.step()
        writer.add_scalar('train_loss', train_loss.avg, epoch)

        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader, loss_fnc, cfg.scenarios)
            writer.add_scalar('valid_loss', valid_loss.avg, epoch)
            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(
                    {
                        'epoch': epoch,
                        'model': model.state_dict(),
                    },
                    os.path.join(cfg.save_path, "best_finetune_" + model_name + ".pt")
                )
                print("Model saved!")
                test_epoch(model, test_loader, loss_fnc, cfg.scenarios)


if __name__ == "__main__":
    main("star")

