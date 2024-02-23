import os
import torch

# LLM
seed = 2012

dataset = "amazon"
dataset_mode = "w_clicked_product_title"
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

scenarios = [0, 1, 2, 3, 4]
# device_ids = [0]
# device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")

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
lr = 8e-5
max_lr = 5e-4

text_encoder_models = [
    # Name, num_hidden_layers, text_embedding_dim, max_length
    ["tiny-bert-4l-en", 4, 312, 512],
    ["bert-base-uncased", 12, 768, 512],
    ["deberta-v3-base", 12, 768, 512],
    ["deberta-v3-large", 24, 1024, 512],
    ["gpt2", 12, 768, 1024],
    ["Llama-2-7b-hf", 32, 4096, 4096],
    ["chatglm2-6b-32k", 28, 4096, 2],
]

text_encoder_model_name, layer_num, text_embedding_dim, max_length = text_encoder_models[0]

if text_encoder_model_name == "tiny-bert-4l-en":
    nlp_finetune_batch_size = 50 * torch.cuda.device_count()
    ladder_frequency = 2
elif text_encoder_model_name == "deberta-v3-base":
    nlp_finetune_batch_size = 60 * torch.cuda.device_count()
    ladder_frequency = 6
else:
    nlp_finetune_batch_size = 1
    ladder_frequency = 28

text_encoder_model = os.path.join("./pretrained_models/", text_encoder_model_name)
text_tokenizer = os.path.join("./pretrained_models/", text_encoder_model_name)

ladder_block = ["wo_block", "w_lora", "w_self_attention", "w_transformer_block"]
ladder_block = ladder_block[3]
r = 4
num_heads = 2
narrowed_ratio = 0.25

use_peft = True
pretrained = False
load_path = "/root/multi-domain/saved_models/amazon/multi_domain/llm_based/deberta-v3-base/w_transformer_block/w_clicked_product_title/v2/epoch0.pt"
save_path = os.path.join("./saved_models", str(dataset), "multi_domain", "llm_based", text_encoder_model_name,
                         ladder_block, dataset_mode)
save_every_n_epoch = 1

max_epochs = 10
lr_step = "epoch"
dropout = 0.2

mixed_precision = True
clip_grad = False
clip_value = 1.0

use_special_token = False

# Early stopping parameters
patience = 1
# early_stop = False