import os
import sys
import random
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, get_linear_schedule_with_warmup, LlamaModel

from torchsummary import summary
from tqdm import tqdm

from layers.core import AvgMeter
from layers.utils import get_lr


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Logger(object):
    def __init__(self, filename="output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        if not self.log.closed:
            self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()


def make_train_valid_dfs(data_path, data_source, all_scenarios, test_scenarios=None):
    df = pd.read_csv(data_path, sep='\t')

    if data_source == "amazon":
        df['label'] = df['rating']
        df = df.drop(df[df['overall'] == 3].index).reset_index()

        # 按照时间顺序排序
        df = df.sort_values(by='unixReviewTime', ascending=True)
        if test_scenarios is None:
            test_scenarios = all_scenarios
        text_data_scene = [df[df['scenario'] == i] for i in test_scenarios]
        train_text_scene = [i.iloc[:int(len(i) * 0.9)].copy() for i in text_data_scene]
        test_text_scene = [i.iloc[int(len(i) * 0.9):].copy() for i in text_data_scene]
        valid_text_scene = [i.iloc[int(len(i) * 0.9):].copy() for i in train_text_scene]
        train_text_scene = [i.iloc[:int(len(i) * 0.9)].copy() for i in train_text_scene]

        train_data = pd.concat(train_text_scene)
        valid_data = pd.concat(valid_text_scene)
        test_data = pd.concat(test_text_scene)

        return train_data, valid_data, test_data


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, text_data, text_label, text_scenario, tokenizer):
        self.cfg = cfg
        self.text_data = list(text_data)
        self.text_label = text_label
        self.text_scenario = text_scenario
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.text_data[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'scenario': torch.tensor(self.text_scenario[idx], dtype=torch.int),
            'label': torch.tensor(self.text_label[idx], dtype=torch.int)
        }
        return item

    def __len__(self):
        return len(self.text_data)


def build_loader(cfg, text_input, tokenizer, mode):
    dataset = BertDataset(
        cfg,
        text_input['content'].values,
        text_input['label'].values,
        text_input['scenario'].values,
        tokenizer=tokenizer,
    )
    # 在分布式训练环境中
    sampler = DistributedSampler(dataset, shuffle=True if mode == "train" else False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.nlp_finetune_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
    )
    return dataloader


def train_epoch(cfg, model, train_loader, optimizer, lr_scheduler, step, loss_fnc, new_scenario_id=None):
    loss_meter = AvgMeter()
    auc_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    scaler = GradScaler() if cfg.mixed_precision else None

    if cfg.mixed_precision:
        for batch in tqdm_object:

            ids = batch['input_ids'].to(cfg.device)
            mask = batch['attention_mask'].to(cfg.device)
            scenario = batch['scenario'].unsqueeze(1).to(cfg.device)
            label = batch['label'].unsqueeze(1).to(cfg.device)

            with autocast():
                # print(summary(model, input_size=(ids.shape, mask.shape, scenario.shape)))
                output, _, general_out = model(
                    input_ids=ids,
                    attention_mask=mask,
                    scenario_id=scenario,
                )
                loss = loss_fnc(output, label.float())
                general_loss = loss_fnc(general_out, label.float())
                total_loss = loss + general_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if step == "batch":
                lr_scheduler.step()

            count = batch['label'].size(0)
            loss_meter.update(loss.item(), count)

            try:
                auc = roc_auc_score(label.detach().cpu().numpy(), torch.sigmoid(output).detach().cpu().numpy())
                auc_meter.update(auc, count)

                tqdm_object.set_postfix(train_loss=loss_meter.avg, train_auc=auc_meter.avg, lr=get_lr(optimizer))
            except:
                tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    else:
        for i, batch in enumerate(tqdm_object):

            accumulation_steps = 4

            ids = batch['input_ids'].to(cfg.device)
            mask = batch['attention_mask'].to(cfg.device)
            scenario = batch['scenario'].to(cfg.device)
            label = batch['label'].unsqueeze(1).to(cfg.device)

            output, _, general_out = model(
                input_ids=ids,
                attention_mask=mask,
                scenario_id=scenario,
            )
            loss = loss_fnc(output, label.float())
            general_loss = loss_fnc(general_out, label.float())
            total_loss = loss + general_loss

            total_loss = total_loss / accumulation_steps
            total_loss.sum().backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if step == "batch":
                    lr_scheduler.step()

            count = batch['label'].size(0)
            loss_meter.update(loss.item(), count)
            auc = roc_auc_score(label.detach().cpu().numpy(), torch.sigmoid(output).detach().cpu().numpy())
            tqdm_object.set_postfix(train_loss=loss_meter.avg, train_auc=auc, lr=get_lr(optimizer))

    return loss_meter, auc_meter


def valid_epoch(cfg, model, valid_loader, loss_fnc, scenarios):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    predicts = []
    labels = []

    scenario_predicts = [[] for _ in range(len(scenarios))]
    scenario_labels = [[] for _ in range(len(scenarios))]

    for batch in tqdm_object:
        ids = batch['input_ids'].to(cfg.device)
        mask = batch['attention_mask'].to(cfg.device)
        scenario = batch['scenario'].unsqueeze(1).to(cfg.device)
        label = batch['label'].unsqueeze(1).to(cfg.device)
        output, _, general_out = model(
            input_ids=ids,
            attention_mask=mask,
            scenario_id=scenario,
        )
        loss = loss_fnc(output, label.float())
        count = batch['label'].size(0)

        scenario_cpu = scenario.cpu().data.numpy()

        if cfg.mixed_precision:
            a = torch.sigmoid(output).cpu().data.numpy()
            predicts.extend(a)
            for i in range(len(scenarios)):
                scenario_predicts[i].extend(a[scenario_cpu == scenarios[i]])
        else:
            predicts.extend(output.cpu().data.numpy())
            for i in range(len(scenarios)):
                scenario_predicts[i].extend(a[scenario_cpu == scenarios[i]])

        b = label.cpu().data.numpy()
        labels.extend(b)
        for i in range(len(scenarios)):
            scenario_labels[i].extend(b[scenario_cpu == scenarios[i]])

        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    print("Valid auc: ", roc_auc_score(labels, predicts))

    for i in range(len(scenarios)):
        print("valid_auc_scenario_" + str(scenarios[i]) + ": ", roc_auc_score(scenario_labels[i], scenario_predicts[i]))

    return loss_meter, roc_auc_score(labels, predicts)


def test_epoch(cfg, model, loss_fnc, all_scenarios, scenarios):

    train_text, valid_text, test_text = make_train_valid_dfs(cfg, cfg.data_path, cfg.dataset, all_scenarios)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_loader = build_loader(cfg, test_text, tokenizer, mode="test")

    loss_meter = AvgMeter()

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    predicts = []
    labels = []

    scenario_predicts = [[] for _ in range(len(scenarios))]
    scenario_labels = [[] for _ in range(len(scenarios))]

    shared_predicts = [[] for _ in range(len(all_scenarios))]
    shared_labels = [[] for _ in range(len(all_scenarios))]

    for batch in tqdm_object:
        ids = batch['input_ids'].to(cfg.device)
        mask = batch['attention_mask'].to(cfg.device)
        scenario = batch['scenario'].unsqueeze(1).to(cfg.device)
        label = batch['label'].unsqueeze(1).to(cfg.device)
        output, _, general_out = model(
            input_ids=ids,
            attention_mask=mask,
            scenario_id=scenario,
        )
        loss = loss_fnc(output, label.float())
        count = batch['label'].size(0)

        scenario_cpu = scenario.cpu().data.numpy()

        if cfg.mixed_precision:
            a = torch.sigmoid(output).cpu().data.numpy()
            predicts.extend(a)
            for i in range(len(scenarios)):
                scenario_predicts[i].extend(a[scenario_cpu == scenarios[i]])

            a_s = torch.sigmoid(general_out).cpu().data.numpy()
            for i in range(len(all_scenarios)):
                shared_predicts[i].extend(a_s[scenario_cpu == all_scenarios[i]])
        else:
            predicts.extend(output.cpu().data.numpy())

        b = label.cpu().data.numpy()
        labels.extend(b)
        for i in range(len(scenarios)):
            scenario_labels[i].extend(b[scenario_cpu == scenarios[i]])

        for i in range(len(all_scenarios)):
            shared_labels[i].extend(b[scenario_cpu == all_scenarios[i]])

        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(test_loss=loss_meter.avg)

    print("test auc: ", roc_auc_score(labels, predicts))

    for i in range(len(scenarios)):
        # 计算每个场景的AUC
        print("test_auc_scenario_" + str(scenarios[i]) + ": ", roc_auc_score(scenario_labels[i], scenario_predicts[i]))
        # 计算每个场景的Loss
        print("test_loss_scenario_" + str(scenarios[i]) + ": ", loss_fnc(torch.tensor(scenario_predicts[i], dtype=torch.float), torch.tensor(scenario_labels[i], dtype=torch.float)))

    for i in range(len(all_scenarios)):
        # 计算每个场景的AUC
        print("test_shared_auc_scenario_" + str(all_scenarios[i]) + ": ", roc_auc_score(shared_labels[i], shared_predicts[i]))
        # 计算每个场景的Loss
        print("test_shared_loss_scenario_" + str(all_scenarios[i]) + ": ", loss_fnc(torch.tensor(shared_predicts[i], dtype=torch.float), torch.tensor(shared_labels[i], dtype=torch.float)))

    return loss_meter


def train(cfg, model, rank=None, optimizer=None, lr_scheduler=None, step="epoch", new_scenario_id=None):
    seed = cfg.seed
    data_source = cfg.dataset
    load_path = cfg.load_path
    save_path = cfg.save_path

    n = 1
    while True:
        folder_name = os.path.join(save_path, 'v' + str(n))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            version = n
            save_path = folder_name
            break
        n += 1

    sys.stdout = Logger(os.path.join(save_path, 'output.txt'))
    print("load_path: ", load_path)
    print("save_path: ", save_path)
    for name, value in vars(cfg).items():
        if not name.startswith("__"):
            print(f"{name}: {value}")

    train_text, valid_text, test_text = make_train_valid_dfs(cfg, cfg.data_path, data_source)
    setup_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = build_loader(cfg, train_text, tokenizer, mode="train")
    valid_loader = build_loader(cfg, valid_text, tokenizer, mode="valid")
    test_loader = build_loader(cfg, test_text, tokenizer, mode="test")

    # model = NLP_Model(cfg.text_encoder_model, cfg.scenarios)
    # summary(model)

    if new_scenario_id is not None:
        # 添加新场景并冻结原有模型的参数
        for param in model.parameters():
            param.requires_grad = False
        model.add_new_scenario(new_scenario_id)

    if cfg.use_special_token:
        print("len_tokenizer: ", len(tokenizer))
        print(tokenizer.tokenize("user1"))
        model.nlp_model.resize_token_embeddings(len(tokenizer))

    # Load model to GPU or multiple GPUs if available
    # Using Distributed Data Parallel
    if cfg.distributed:
        device_id = rank % torch.cuda.device_count()
        print("Using Distributed Data Parallel")
        model.to(device_id)
        model = DDP(model, device_ids=[device_id])

        # 加载模型
        if cfg.pretrained:
            print("load model from %s ..." % load_path)
            # 获得上一次的epoch
            current_epoch = torch.load(load_path, map_location=f'cuda:{device_id}')['epoch']
            total_epochs = current_epoch + cfg.epochs
            # 获得模型参数
            model_dict = torch.load(load_path, map_location=f'cuda:{device_id}')['model']
            # 载入参数
            model.module.load_state_dict(model_dict)
            print("load model success!")
        else:
            current_epoch = 0
            total_epochs = cfg.epochs
    else:
        print("Using Data Parallel")
        model.to(cfg.device)
        model = nn.DataParallel(model, device_ids=cfg.device_ids)

        # 加载模型
        if cfg.pretrained:
            print("load model from %s ..." % load_path)
            # 获得上一次的epoch
            current_epoch = torch.load(load_path)['epoch']
            total_epochs = current_epoch + cfg.epochs
            # 获得模型参数
            model_dict = torch.load(load_path)['model']
            # 载入参数
            model.module.load_state_dict(model_dict)
            print("load model success!")
        else:
            current_epoch = 0
            total_epochs = cfg.epochs

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if lr_scheduler is None:
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.lr, max_lr=cfg.max_lr, step_size_up=5, mode="triangular2", cycle_momentum=False)

    if cfg.mixed_precision:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.BCELoss()

    print("Begin training ...")

    writer = SummaryWriter('./logs')

    best_loss = float('inf')
    best_auc = float('inf')

    for epoch in range(current_epoch, total_epochs):
        print(f"Epoch {epoch + 1}")
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        model.train()
        train_loss, train_auc = train_epoch(cfg, model, train_loader, optimizer, lr_scheduler, step, loss_fn)
        lr_scheduler.step()
        writer.add_scalar('train_loss', train_loss.avg, epoch)

        model.eval()
        with torch.no_grad():
            valid_loss, valid_auc = valid_epoch(cfg, model, valid_loader, loss_fn, cfg.scenarios)
            writer.add_scalar('valid_loss', valid_loss.avg, epoch)

            if valid_loss.avg < best_loss or valid_auc > best_auc:
                best_loss = valid_loss.avg
                best_auc = valid_auc
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                }, os.path.join(save_path, "epoch" + str(((epoch + 1) // 10) * 10) + '.pt'))
                print("Model saved!")

            test_loss = test_epoch(cfg, model, loss_fn, cfg.scenarios, [0, 1, 2, 3, 4])
            writer.add_scalar('test_loss', test_loss.avg, epoch)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        test_data: DataLoader,
        train_scenarios: list,
        loss: torch.nn.modules.loss.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        step: str,
        mixed_precision: bool,
        save_every: int,
        patience: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_scenarios = train_scenarios
        self.train_rounds = len(self.train_scenarios)
        self.current_round = 0
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.mixed_precision = mixed_precision
        self.save_every = save_every
        self.patience = patience
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        loss_meter = AvgMeter()
        auc_meter = AvgMeter()
        tqdm_object = tqdm(self.train_data, total=len(self.train_data))
        scaler = GradScaler() if self.mixed_precision else None

        accumulation_steps = 4

        for i, batch in enumerate(tqdm_object):

            ids = batch['input_ids'].to(self.gpu_id)
            mask = batch['attention_mask'].to(self.gpu_id)

            label = batch['label'].unsqueeze(1).to(self.gpu_id)

            if self.mixed_precision:
                scenario = batch['scenario'].unsqueeze(1).to(self.gpu_id)

                with autocast():
                    # print(summary(model, input_size=(ids.shape, mask.shape, scenario.shape)))
                    output, _, general_out = self.model(
                        input_ids=ids,
                        attention_mask=mask,
                        scenario_id=scenario,
                    )
                    specific_loss = self.loss(output, label.float())
                    general_loss = self.loss(general_out, label.float())
                    total_loss = specific_loss + general_loss

                # 缩放损失并进行反向传播
                total_loss = total_loss / accumulation_steps
                scaler.scale(total_loss).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_data):
                    # 更新模型参数
                    scaler.step(self.optimizer)
                    scaler.update()
                    # 清除梯度
                    self.optimizer.zero_grad()
                    # 更新学习率
                    if self.step == "batch":
                        self.lr_scheduler.step()

                count = batch['label'].size(0)
                loss_meter.update(specific_loss.item(), count)

                try:
                    auc = roc_auc_score(label.detach().cpu().numpy(), torch.sigmoid(output).detach().cpu().numpy())
                    auc_meter.update(auc, count)
                    tqdm_object.set_postfix(train_loss=loss_meter.avg, train_auc=auc_meter.avg, lr=get_lr(self.optimizer))
                except:
                    tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(self.optimizer))

            else:
                scenario = batch['scenario'].to(self.gpu_id)

                output, _, general_out = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    scenario_id=scenario,
                )
                specific_loss = self.loss(output, label.float())
                general_loss = self.loss(general_out, label.float())
                total_loss = specific_loss + general_loss

                total_loss = total_loss / accumulation_steps
                total_loss.backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_data):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.step == "batch":
                        self.lr_scheduler.step()

                count = batch['label'].size(0)
                loss_meter.update(specific_loss.item(), count)

                auc = roc_auc_score(label.detach().cpu().numpy(), torch.sigmoid(output).detach().cpu().numpy())
                tqdm_object.set_postfix(train_loss=loss_meter.avg, train_auc=auc, lr=get_lr(self.optimizer))

        return loss_meter, auc_meter

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        print("Begin training ...")
        writer = SummaryWriter('./logs')

        # Early stopping parameters
        epochs_no_improve = 0
        early_stop = False
        best_loss = float('inf')
        best_auc = 0.5

        self.model.train()
        for epoch in range(self.epochs_run, max_epochs):
            train_loss, train_auc = self._run_epoch(epoch)
            writer.add_scalar('train_loss', train_loss.avg, epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

            self.model.eval()
            with torch.no_grad():
                valid_loss, valid_auc = self.valid()
                writer.add_scalar('valid_loss', valid_loss.avg, epoch)

                # Check if the validation loss improved
                if valid_loss.avg < best_loss or valid_auc > best_auc:
                    best_loss = valid_loss.avg
                    best_auc = valid_auc

                    self._save_snapshot(epoch)
                    epochs_no_improve = 0  # Reset the counter
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    print("Early stopping triggered")
                    early_stop = True
                    break  # Break the loop
                else:
                    test_loss = test_epoch(cfg, model, loss_fn, cfg.scenarios, [1, ])
                    writer.add_scalar('test_loss', test_loss.avg, epoch)


            if early_stop:
                break

        self.current_round += 1

    def valid(self):

        valid_scenarios = self.train_scenarios[self.current_round]
        valid_scenario_num = len(valid_scenarios)
        loss_meter = AvgMeter()

        tqdm_object = tqdm(self.valid_data, total=len(self.valid_data))
        predicts = []
        labels = []

        scenario_predicts = [[] for _ in range(valid_scenario_num)]
        scenario_labels = [[] for _ in range(valid_scenario_num)]

        for batch in tqdm_object:
            ids = batch['input_ids'].to(self.gpu_id)
            mask = batch['attention_mask'].to(self.gpu_id)
            scenario = batch['scenario'].unsqueeze(1).to(self.gpu_id)
            label = batch['label'].unsqueeze(1).to(self.gpu_id)
            output, _, general_out = self.model(
                input_ids=ids,
                attention_mask=mask,
                scenario_id=scenario,
            )
            specific_loss = self.loss(output, label.float())
            count = batch['label'].size(0)

            scenario_cpu = scenario.cpu().data.numpy()

            if self.mixed_precision:
                a = torch.sigmoid(output).cpu().data.numpy()
                predicts.extend(a)
                for i in range(valid_scenario_num):
                    scenario_predicts[i].extend(a[scenario_cpu == valid_scenarios[i]])
            else:
                predicts.extend(output.cpu().data.numpy())
                for i in range(valid_scenario_num):
                    scenario_predicts[i].extend(a[scenario_cpu == valid_scenarios[i]])

            b = label.cpu().data.numpy()
            labels.extend(b)
            for i in range(valid_scenario_num):
                scenario_labels[i].extend(b[scenario_cpu == valid_scenarios[i]])

            loss_meter.update(specific_loss.item(), count)
            tqdm_object.set_postfix(valid_loss=loss_meter.avg)

        print("Valid auc: ", roc_auc_score(labels, predicts))

        for i in range(valid_scenario_num):
            print("valid_auc_scenario_" + str(valid_scenarios[i]) + ": ",
                  roc_auc_score(scenario_labels[i], scenario_predicts[i]))

        return loss_meter, roc_auc_score(labels, predicts)

    def test(self):
        train_text, valid_text, test_text = make_train_valid_dfs(cfg, cfg.data_path, cfg.dataset, all_scenarios)

        tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_loader = build_loader(cfg, test_text, tokenizer, mode="test")

        loss_meter = AvgMeter()

        tqdm_object = tqdm(test_loader, total=len(test_loader))
        predicts = []
        labels = []

        scenario_predicts = [[] for _ in range(len(scenarios))]
        scenario_labels = [[] for _ in range(len(scenarios))]

        shared_predicts = [[] for _ in range(len(all_scenarios))]
        shared_labels = [[] for _ in range(len(all_scenarios))]

        for batch in tqdm_object:
            ids = batch['input_ids'].to(cfg.device)
            mask = batch['attention_mask'].to(cfg.device)
            scenario = batch['scenario'].unsqueeze(1).to(cfg.device)
            label = batch['label'].unsqueeze(1).to(cfg.device)
            output, _, general_out = model(
                input_ids=ids,
                attention_mask=mask,
                scenario_id=scenario,
            )
            loss = loss_fnc(output, label.float())
            count = batch['label'].size(0)

            scenario_cpu = scenario.cpu().data.numpy()

            if cfg.mixed_precision:
                a = torch.sigmoid(output).cpu().data.numpy()
                predicts.extend(a)
                for i in range(len(scenarios)):
                    scenario_predicts[i].extend(a[scenario_cpu == scenarios[i]])

                a_s = torch.sigmoid(general_out).cpu().data.numpy()
                for i in range(len(all_scenarios)):
                    shared_predicts[i].extend(a_s[scenario_cpu == all_scenarios[i]])
            else:
                predicts.extend(output.cpu().data.numpy())

            b = label.cpu().data.numpy()
            labels.extend(b)
            for i in range(len(scenarios)):
                scenario_labels[i].extend(b[scenario_cpu == scenarios[i]])

            for i in range(len(all_scenarios)):
                shared_labels[i].extend(b[scenario_cpu == all_scenarios[i]])

            loss_meter.update(loss.item(), count)

            tqdm_object.set_postfix(test_loss=loss_meter.avg)

        print("test auc: ", roc_auc_score(labels, predicts))

        for i in range(len(scenarios)):
            # 计算每个场景的AUC
            print("test_auc_scenario_" + str(scenarios[i]) + ": ",
                  roc_auc_score(scenario_labels[i], scenario_predicts[i]))
            # 计算每个场景的Loss
            print("test_loss_scenario_" + str(scenarios[i]) + ": ",
                  loss_fnc(torch.tensor(scenario_predicts[i], dtype=torch.float),
                           torch.tensor(scenario_labels[i], dtype=torch.float)))

        for i in range(len(all_scenarios)):
            # 计算每个场景的AUC
            print("test_shared_auc_scenario_" + str(all_scenarios[i]) + ": ",
                  roc_auc_score(shared_labels[i], shared_predicts[i]))
            # 计算每个场景的Loss
            print("test_shared_loss_scenario_" + str(all_scenarios[i]) + ": ",
                  loss_fnc(torch.tensor(shared_predicts[i], dtype=torch.float),
                           torch.tensor(shared_labels[i], dtype=torch.float)))

        return loss_meter

