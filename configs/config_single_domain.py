import os
import torch

# LLM
seed = 2012

dataset = "amazon"
dataset_mode = "standard_dataset"
if dataset_mode == "standard_dataset":
    data_path = "./datasets/amazon_review_data/hybrid_data/hybrid_5_id.csv"
    text_path = "./datasets/amazon_review_data/text_data/text_5_id.txt"
    struct_path = "./datasets/amazon_review_data/filtered_data/filtered_5_id.csv"
elif dataset_mode == "w_clicked_product_title":
    data_path = "./datasets/amazon_review_data/hybrid_data/hybrid_5_title.csv"
    text_path = "./datasets/amazon_review_data/text_data/text_5_title.txt"
    struct_path = "./datasets/amazon_review_data/filtered_data/filtered_5_title.csv"

'''Scenarios
0: Amazon Fashion
1: Digital Music
2: Musical Instruments
3: Gift Cards
4: All Beauty
'''

scenarios = [0, 2, 3]
device_ids = [0]
device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")

num_workers = 0
weight_decay = 0.001

# # epoch 60
# lr = 1e-7
# max_lr = 0.00015625

# # epoch 50
# lr = 1e-7
# max_lr = 0.00015625

# # epoch 60
# lr = 1e-7
# max_lr = 0.0003125

# epoch 0
lr = 5e-5
max_lr = 5e-3

text_encoder_models = [
    # Name, num_hidden_layers, text_embedding_dim, max_length
    ["tiny-bert-4l-en", 4, 312, 512],
    ["bert-base-uncased", 12, 768, 512],
    ["deberta-v3-large", 24, 1024, 512],
    ["gpt2", 12, 768, 1024],
    ["Llama-2-7b-hf", 32, 4096, 4096],
]

text_encoder_model_name, layer_num, text_embedding_dim, max_length = text_encoder_models[0]

text_encoder_model = os.path.join("../pretrained_models/", text_encoder_model_name)
text_tokenizer = os.path.join("../pretrained_models/", text_encoder_model_name)

ladder_block = ["wo_block", "w_lora", "w_self_attention", "w_transformer_block"]
ladder_block = ladder_block[3]
r = 4
num_heads = 2
narrowed_ratio = 0.25

use_peft = True
pretrained = False
load_path = os.path.join("saved_models", str(dataset), "multi_domain", "llm_based", text_encoder_model_name,
                         ladder_block, "epoch60.pt")
save_path = os.path.join("saved_models", str(dataset), "multi_domain", "traditional")

epochs = 5
dropout = 0.2

mixed_precision = True
clip_grad = False
clip_value = 1.0

use_special_token = False

# 多场景传统模型参数
focal_loss = True
batch_size = 1024
