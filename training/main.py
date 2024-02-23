import argparse
import sys
sys.path.insert(0, '/root/multi-domain')

import configs.config as cfg

from transformers import (DebertaV2Model, GPT2Model)

from peft import LoraConfig
from peft import get_peft_model
from layers.core import TransformerLayer, Expert, AttentionPooling, Lora


import torch.distributed as dist
import warnings

warnings.simplefilter('ignore')

from utils import *


def choose_block(cfg):
    if cfg.ladder_block == "wo_block":
        return None
    elif cfg.ladder_block == "w_lora":
        return Lora(
            input_dim=cfg.text_embedding_dim,
            output_dim=cfg.text_embedding_dim,
            r=cfg.r,
            use_dnn=True,
            dnn_hidden_states=[512, 256, 128],
        )
    elif cfg.ladder_block == "w_self_attention":
        return nn.MultiheadAttention(
            embed_dim=cfg.text_embedding_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
    elif cfg.ladder_block == "w_transformer_block":
        return TransformerLayer(dim=cfg.text_embedding_dim, num_heads=cfg.num_heads, narrowed_ratio=cfg.narrowed_ratio)
    else:
        raise ValueError("Invalid ladder block")


class NLP_Model(nn.Module):
    def __init__(self, cfg):
        super(NLP_Model, self).__init__()
        self.cfg = cfg
        self.text_encoder_model = cfg.text_encoder_model
        self.scenarios = cfg.scenarios
        self.num_scenarios = len(self.scenarios)

        self.nlp_model = AutoModel.from_pretrained(
            self.text_encoder_model,
            # trust_remote_code=True,
            output_hidden_states=True,
            # device_map="auto",
        )
        # summary(self.nlp_model)

        # 输出模型类名
        print("NLP model is", self.nlp_model.__class__.__name__)

        # Get transformer layers
        if isinstance(self.nlp_model, (BertModel, DebertaV2Model)):
            self.num_hidden_layers = len(self.nlp_model.encoder.layer)
            print("NLP model is Bert or DebertaV2")
        elif isinstance(self.nlp_model, GPT2Model):
            self.num_hidden_layers = len(self.nlp_model.h)
            print("NLP model is GPT2")
        elif isinstance(self.nlp_model, LlamaModel):
            self.num_hidden_layers = len(self.nlp_model.layers)
            print("NLP model is Llama")
        else:
            self.num_hidden_layers = 28

        if cfg.use_peft:
            self.peft_config = LoraConfig(
                r=4, lora_alpha=8, lora_dropout=0.1, bias="all"
            )
            self.nlp_model = get_peft_model(self.nlp_model, self.peft_config)
            # self.nlp_model.print_trainable_parameters()
        else:
            for p in self.nlp_model.parameters():
                p.requires_grad = False

        self.expert_modules = nn.ModuleList(
            Expert(cfg.text_embedding_dim, 1, hidden_size=[1024, 512, 256])
            for _ in range(self.num_scenarios)
        )
        self.general_expert = Expert(cfg.text_embedding_dim, 1, hidden_size=[1024, 512, 256])

        self.ladder_frequency = cfg.ladder_frequency

        self.attention_modules = nn.ModuleList(
            nn.ModuleList(
                choose_block(cfg)
                for _ in range(int(self.num_hidden_layers / self.ladder_frequency + 1))
            )
            for _ in range(self.num_scenarios + 1)
        )

        self.attn_pooling = nn.ModuleList(
            AttentionPooling(self.nlp_model.config.hidden_size)
            for _ in range(self.num_scenarios + 1)
        )

    def add_new_scenario(self, new_scenario_id):
        # 添加新的专家模块和attention pooling
        new_expert_module = Expert(self.cfg.text_embedding_dim, 1)
        new_attention_pooling = AttentionPooling(self.nlp_model.config.hidden_size)

        self.expert_modules.append(new_expert_module)
        self.attn_pooling.append(new_attention_pooling)
        self.scenarios.append(new_scenario_id)
        self.num_scenarios += 1

    def freeze_scenario(self, index_list):
        for index in index_list:
            # 确保索引在范围内
            if index < 0 or index >= len(self.expert_modules):
                print(f"Index {index} is out of range for expert_modules.")
                return

            # 冻结指定索引的expert_module
            for param in self.expert_modules[index].parameters():
                param.requires_grad = False
            # 冻结指定索引的expert_module
            for param in self.attention_modules[index].parameters():
                param.requires_grad = False
            # 冻结指定索引的expert_module
            for param in self.attn_pooling[index].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, scenario_id):

        output = self.nlp_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output.hidden_states

        # User & Product
        last_hidden_state = hidden_states[-1]

        if self.cfg.ladder_block == "wo_block":
            final_embedding = [self.attn_pooling[i](last_hidden_state) for i in range(self.num_scenarios + 1)]
            final_domain_embedding = final_embedding[:self.num_scenarios]
            final_shared_embedding = final_embedding[-1]

        else:
            intermediate = [None for _ in range(self.num_scenarios + 1)]

            for i in range(int(self.num_hidden_layers / self.ladder_frequency + 1)):
                if i == 0:
                    for j in range(self.num_scenarios + 1):
                        if self.cfg.ladder_block == "w_lora" or self.cfg.ladder_block == "w_transformer_block":
                            intermediate[j] = self.attention_modules[j][i](hidden_states[i])
                        elif self.cfg.ladder_block == "w_self_attention":
                            intermediate[j], _ = self.attention_modules[j][i](hidden_states[i], hidden_states[i],
                                                                              hidden_states[i])
                else:
                    if self.cfg.ladder_block == "w_lora" or self.cfg.ladder_block == "w_transformer_block":
                        for j in range(self.num_scenarios + 1):
                            intermediate[j] = self.attention_modules[j][i](
                                hidden_states[i * self.ladder_frequency] + intermediate[j])
                    elif self.cfg.ladder_block == "w_self_attention":
                        for j in range(self.num_scenarios + 1):
                            inter = hidden_states[i * self.ladder_frequency] + intermediate[j]
                            intermediate[j], _ = self.attention_modules[j][i](inter, inter, inter)

            final_embeddings = [self.attn_pooling[j](intermediate[j] + last_hidden_state)
                                for j in range(self.num_scenarios + 1)]

            final_domain_embedding = final_embeddings[:self.num_scenarios]
            final_shared_embedding = final_embeddings[-1]

        if self.cfg.mixed_precision:
            out_multi_head = [self.expert_modules[i](final_domain_embedding[i]) for i in self.scenarios]
            general_output = self.general_expert(final_shared_embedding)
        else:
            out_multi_head = [torch.sigmoid(self.expert_modules[i](final_domain_embedding[i])) for i in self.scenarios]
            general_output = torch.sigmoid(self.general_expert(final_shared_embedding))

        scenario_mask = [(scenario_id == i) for i in self.scenarios]

        out = torch.zeros_like(out_multi_head[0])
        for i in self.scenarios:
            out = torch.where(scenario_mask[i], out_multi_head[i], out)
        return out, out_multi_head, general_output


def save_snapshot(trained_model, epoch, saved_path):
    snapshot = {}
    snapshot["MODEL_STATE"] = trained_model.module.state_dict()
    snapshot["EPOCHS_RUN"] = epoch
    torch.save(snapshot, saved_path)
    print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt")


def load_snapshot(initialized_model, snapshot_path):
    snapshot = torch.load(snapshot_path)
    initialized_model.load_state_dict(snapshot["MODEL_STATE"])
    epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Resuming training from snapshot at Epoch {epochs_run}")
    return epochs_run


if __name__ == '__main__':

    """初始化seed和路径"""
    seed = cfg.seed
    setup_seed(seed)

    n = 1
    while True:
        folder_name = os.path.join(cfg.save_path, cfg, 'v' + str(n))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            version = n
            save_path = folder_name
            break
        n += 1

    sys.stdout = Logger(os.path.join(save_path, 'output.txt'))
    print("load_path: ", cfg.load_path)
    print("save_path: ", save_path)
    for name, value in vars(cfg).items():
        if not name.startswith("__"):
            print(f"{name}: {value}")




    """加载模型"""

    # 初始化分布式训练
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    cfg.device = torch.device("cuda", rank)
    cfg.distributed = True

    # 创建模型
    model = NLP_Model(cfg)

    # Load model to GPU or multiple GPUs if available
    # Using Distributed Data Parallel
    device_id = rank % torch.cuda.device_count()
    print("Using Distributed Data Parallel")
    model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # 加载模型
    if cfg.pretrained:
        current_epoch = load_snapshot(model, cfg.load_path)
    else:
        current_epoch = 0







    """加载数据集"""
    train_scenarios = [1,]
    train_text, valid_text, test_text = make_train_valid_dfs(cfg, cfg.data_path, cfg.dataset, scenarios=train_scenarios)
    model.module.freeze_scenario([item for item in cfg.scenarios if item not in train_scenarios])

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.text_tokenizer,
        local_files_only=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = build_loader(cfg, train_text, tokenizer, mode="train")
    valid_loader = build_loader(cfg, valid_text, tokenizer, mode="valid")
    test_loader = build_loader(cfg, test_text, tokenizer, mode="test")

    if cfg.use_special_token:
        print("len_tokenizer: ", len(tokenizer))
        print(tokenizer.tokenize("user1"))
        model.nlp_model.resize_token_embeddings(len(tokenizer))

    """优化器初始化"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.lr, max_lr=cfg.max_lr, step_size_up=5, mode="triangular2", cycle_momentum=False)

    if cfg.mixed_precision:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.BCELoss()


    trainer = Trainer(
        model=model,
        train_data=train_loader,
        loss=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        step=cfg.lr_step,
        mixed_precision=cfg.mixed_precision,
        save_every=cfg.save_every_n_epoch,
        snapshot_path=cfg.load_path if cfg.pretrained else None,
    )

    trainer.train(max_epochs=cfg.max_epochs)







    """模型训练"""
    print("Begin training ...")

    writer = SummaryWriter('./logs')

    lr_step = "epoch"

    # Early stopping parameters
    epochs_no_improve = 0
    early_stop = False

    best_loss = float('inf')
    best_auc = 0.5

    for epoch in range(current_epoch, cfg.epochs):
        print(f"Epoch {epoch + 1}")
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        model.train()
        train_loss, train_auc = train_epoch(cfg, model, train_loader, optimizer, lr_scheduler, lr_step, loss_fn)
        lr_scheduler.step()
        writer.add_scalar('train_loss', train_loss.avg, epoch)

        model.eval()
        with torch.no_grad():
            valid_loss, valid_auc = valid_epoch(cfg, model, valid_loader, loss_fn, train_scenarios)
            writer.add_scalar('valid_loss', valid_loss.avg, epoch)

            # Check if the validation loss improved
            if valid_loss.avg < best_loss or valid_auc > best_auc:
                best_loss = valid_loss.avg
                best_auc = valid_auc

                save_snapshot(model, epoch, os.path.join(save_path, "epoch" + str(((epoch + 1) // 10) * 10) + '.pt'))
                epochs_no_improve = 0  # Reset the counter
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= cfg.patience:
                print("Early stopping triggered")
                early_stop = True
                break  # Break the loop

            test_loss = test_epoch(cfg, model, loss_fn, cfg.scenarios, [1,])
            writer.add_scalar('test_loss', test_loss.avg, epoch)

        if early_stop:
            break










    # """加载数据集"""
    # train_scenarios = [1,]
    # train_text, valid_text, test_text = make_train_valid_dfs(cfg, cfg.data_path, data_source, scenarios=train_scenarios)
    # model.module.freeze_scenario([item for item in cfg.scenarios if item not in train_scenarios])
    # 
    # tokenizer = AutoTokenizer.from_pretrained(
    #     cfg.text_tokenizer,
    #     local_files_only=True,
    #     trust_remote_code=True
    # )
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # 
    # train_loader = build_loader(cfg, train_text, tokenizer, mode="train")
    # valid_loader = build_loader(cfg, valid_text, tokenizer, mode="valid")
    # test_loader = build_loader(cfg, test_text, tokenizer, mode="test")
    # 
    # if cfg.use_special_token:
    #     print("len_tokenizer: ", len(tokenizer))
    #     print(tokenizer.tokenize("user1"))
    #     model.nlp_model.resize_token_embeddings(len(tokenizer))
    # 
    # """优化器初始化"""
    # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.lr, max_lr=cfg.max_lr, step_size_up=5, mode="triangular2", cycle_momentum=False)
    # 
    # if cfg.mixed_precision:
    #     loss_fn = nn.BCEWithLogitsLoss()
    # else:
    #     loss_fn = nn.BCELoss()
    # 
    # """模型训练"""
    # print("Begin training ...")
    # 
    # writer = SummaryWriter('./logs')
    # 
    # lr_step = "epoch"
    # 
    # # Early stopping parameters
    # patience = 10  # You can adjust this
    # epochs_no_improve = 0
    # early_stop = False
    # 
    # best_loss = float('inf')
    # best_auc = 0.5
    # 
    # for epoch in range(current_epoch, cfg.epochs):
    #     print(f"Epoch {epoch + 1}")
    #     print(optimizer.state_dict()['param_groups'][0]['lr'])
    # 
    #     model.train()
    #     train_loss, train_auc = train_epoch(cfg, model, train_loader, optimizer, lr_scheduler, lr_step, loss_fn)
    #     lr_scheduler.step()
    #     writer.add_scalar('train_loss', train_loss.avg, epoch)
    # 
    #     model.eval()
    #     with torch.no_grad():
    #         valid_loss, valid_auc = valid_epoch(cfg, model, valid_loader, loss_fn, train_scenarios)
    #         writer.add_scalar('valid_loss', valid_loss.avg, epoch)
    # 
    #         # Check if the validation loss improved
    #         if valid_loss.avg < best_loss or valid_auc > best_auc:
    #             best_loss = valid_loss.avg
    #             best_auc = valid_auc
    # 
    #             save_snapshot(model, epoch, os.path.join(save_path, "epoch" + str(((epoch + 1) // 10) * 10) + '.pt'))
    #             epochs_no_improve = 0  # Reset the counter
    #         else:
    #             epochs_no_improve += 1
    # 
    #         if epochs_no_improve >= patience:
    #             print("Early stopping triggered")
    #             early_stop = True
    #             break  # Break the loop
    # 
    #         test_loss = test_epoch(cfg, model, loss_fn, cfg.scenarios, [1,])
    #         writer.add_scalar('test_loss', test_loss.avg, epoch)
    # 
    #     if early_stop:
    #         break

    # """添加新场景，测试可扩展性"""
    # if cfg.new_scenario_ids is not None:
    #     # 添加新场景并冻结原有模型的参数
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     for new_scenario_id in cfg.new_scenario_ids:
    #         model.add_new_scenario(new_scenario_id)

