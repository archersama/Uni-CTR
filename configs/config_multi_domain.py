import os
import torch

# LLM
seed = 2012

dataset = "amazon"
dataset_mode = "standard_dataset"
if dataset_mode == "standard_dataset":
    data_path = "../datasets/amazon_review_data/hybrid_data/hybrid_5_id.csv"
    text_path = "datasets/amazon_review_data/text_data/text_5_id.txt"
    struct_path = "datasets/amazon_review_data/filtered_data/filtered_5_id.csv"
elif dataset_mode == "w_clicked_product_title":
    data_path = "datasets/amazon_review_data/hybrid_data/hybrid_5_title.csv"
    text_path = "datasets/amazon_review_data/text_data/text_5_title.txt"
    struct_path = "datasets/amazon_review_data/filtered_data/filtered_5_title.csv"

'''Scenarios

0: Amazon Fashion
1: Digital Music
2: Musical Instruments
3: Gift Cards
4: All Beauty
'''

scenarios = [0, 1, 2, 3]
device_ids = [0,1]
device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")

num_workers = 0
weight_decay = 0.002

# epoch 0
lr = 5e-5
max_lr = 1e-3

r = 4
num_heads = 2
narrowed_ratio = 0.25

pretrained = False
load_path = os.path.join("saved_models", str(dataset), "multi_domain", "traditional", "epoch60.pt")
save_path = os.path.join("../saved_models", str(dataset), "multi_domain", "traditional")

epochs = 50
dropout = 0.2

mixed_precision = False
clip_grad = False
clip_value = 1.0

use_special_token = False

# 多场景传统模型参数
focal_loss = False
batch_size = 2048

multiplier = 6

embed_dim = 32