import sys

import torch
import torch.nn as nn
import numpy as np
from layers.utils import MLP
from tqdm import tqdm


class STAR(nn.Module):

    def __init__(self, feature_columns, sequence_feature_columns, input_dims, star_output_dim=1024):
        super().__init__()

        # 阳哥的代码
        self.sparse_feature_columns = feature_columns
        self.sequence_feature_columns = sequence_feature_columns[0]

        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_columns)
        })
        self.sequence_embed_layers = nn.Embedding(
            self.sequence_feature_columns['feat_num'],
            self.sequence_feature_columns['embed_dim']
        )


        # self.embedding = nn.Embedding(sum(feature_dims), embedding_dim=embed_dim)
        # torch.nn.init.xavier_uniform_(self.embedding.weight.data)

        # self.embedding_output_dims = len(sparse_cols) * embed_dim + len(dense_cols)
        self.embedding_output_dims = input_dims
        self.auxiliary = MLP(self.embedding_output_dims, True, [512, 256, 128], 0.3)

        self.shared_weight = nn.Parameter(torch.empty(self.embedding_output_dims, star_output_dim))
        self.shared_bias = nn.Parameter(torch.zeros(star_output_dim))

        self.slot1_weight = nn.Parameter(torch.empty(self.embedding_output_dims, star_output_dim))
        self.slot1_bias = nn.Parameter(torch.zeros(star_output_dim))
        self.slot2_weight = nn.Parameter(torch.empty(self.embedding_output_dims, star_output_dim))
        self.slot2_bias = nn.Parameter(torch.zeros(star_output_dim))
        self.slot3_weight = nn.Parameter(torch.empty(self.embedding_output_dims, star_output_dim))
        self.slot3_bias = nn.Parameter(torch.zeros(star_output_dim))
        self.slot4_weight = nn.Parameter(torch.empty(self.embedding_output_dims, star_output_dim))
        self.slot4_bias = nn.Parameter(torch.zeros(star_output_dim))

        self.mlp = MLP(star_output_dim, True, [512, 256, 128], 0.3)



        for m in [self.shared_weight, self.slot1_weight, self.slot2_weight, self.slot3_weight]:
            torch.nn.init.xavier_uniform_(m.data)

    def forward(self, sparse_inputs, sequence_inputs, scenario_id):

        sparse_embed = torch.cat(
            [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(len(self.sparse_feature_columns))],
            dim=-1,
        )
        sequence_embed = self.sequence_embed_layers(sequence_inputs)

        # Pooling
        pool_x = torch.mean(sequence_embed, dim=1)

        # Concatenate with sparce features
        x = torch.cat([sparse_embed, pool_x], dim=-1)

        sa = self.auxiliary(x)

        slot1_mask = (scenario_id == 0)
        slot2_mask = (scenario_id == 1)
        slot3_mask = (scenario_id == 2)
        slot4_mask = (scenario_id == 3)

        slot1_output = torch.matmul(x, torch.multiply(self.slot1_weight, self.shared_weight)) + self.slot1_bias + self.shared_bias
        slot2_output = torch.matmul(x, torch.multiply(self.slot2_weight, self.shared_weight)) + self.slot2_bias + self.shared_bias
        slot3_output = torch.matmul(x, torch.multiply(self.slot3_weight, self.shared_weight)) + self.slot3_bias + self.shared_bias
        slot4_output = torch.matmul(x, torch.multiply(self.slot4_weight, self.shared_weight)) + self.slot4_bias + self.shared_bias

        output = torch.zeros_like(slot1_output)
        output = torch.where(slot1_mask, slot1_output, output)
        output = torch.where(slot2_mask, slot2_output, output)
        output = torch.where(slot3_mask, slot3_output, output)
        output = torch.where(slot4_mask, slot4_output, output)

        output = self.mlp(output)

        ctr = torch.sigmoid(sa + output)

        return ctr, ctr
