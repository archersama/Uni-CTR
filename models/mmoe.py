import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import configs.config_multi_domain as cfg
import numpy as np

from layers import Expert
from layers.utils import MLP
from tqdm import tqdm



class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.2):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class Gate(nn.Module):
    def __init__(self, input_size, output_size):
        super(Gate, self).__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.softmax(self.layer(x), dim=1)


class MMOE(nn.Module):
    def __init__(self, feature_columns, sequence_feature_columns, input_dims, tower_input_dims, tower_hidden_dims,
                 scenarios, num_experts=3, hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []

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

        self.dropout = dropout
        self.num_experts = num_experts
        self.input_dims = input_dims
        self.tower_input_dims = tower_input_dims
        self.tower_hidden_dims = tower_hidden_dims
        self.scenarios = scenarios
        self.num_tasks = len(scenarios)
        self.num_scenario = len(scenarios)
        self.hidden_dims = hidden_dims

        # self.embedding = nn.Embedding(sum(feature_dims), embedding_dim=embed_dim)
        # torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        # self.offsets = np.array((0, *np.cumsum(feature_dims)[:-1]))
        # # self.embedding_output_dims = len(sparse_cols) * embed_dim + len(dense_cols)
        # self.embedding_output_dims = len(sparse_cols) * embed_dim
        #
        # self.input_size = self.embedding_output_dims
        # self.num_experts = 8
        # self.experts_out = 32
        # self.experts_hidden = 32
        # self.towers_hidden = 32

        self.experts = nn.ModuleList(
            [Expert(self.input_dims, self.tower_input_dims, self.hidden_dims) for _ in range(self.num_experts)])
        self.gates = nn.ParameterList(
            [Gate(input_dims, self.num_experts) for _ in range(self.num_tasks)])
        self.towers = nn.ModuleList([Tower(self.tower_input_dims, 1, self.tower_hidden_dims, dropout) for _ in range(self.num_tasks)])


    def forward(self, sparse_inputs, sequence_inputs, scenario_id):
        outputs = []
        final_outputs = []

        sparse_embed = torch.cat(
            [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(len(self.sparse_feature_columns))],
            dim=-1,
        )
        sequence_embed = self.sequence_embed_layers(sequence_inputs)

        # Pooling
        pool_x = torch.mean(sequence_embed, dim=1)

        # Concatenate with sparce features
        x = torch.cat([sparse_embed, pool_x], dim=-1)

        for i in range(self.num_tasks):
            gate_outputs = self.gates[i](x)
            if i == 0:
                for j in range(self.num_experts):
                    e = self.experts[j](x)
                    if j == 0:
                        expert_outputs = e * gate_outputs[:, j].view(-1, 1)
                    else:
                        expert_outputs += e * gate_outputs[:, j].view(-1, 1)
                outputs.append(expert_outputs)
            else:
                expert_outputs = outputs[0]
            if cfg.mixed_precision:
                y = self.towers[i](expert_outputs)
            else:
                y = F.sigmoid(self.towers[i](expert_outputs))
            final_outputs.append(y)

        scenario_mask = [(scenario_id == i) for i in self.scenarios]

        out = torch.zeros_like(final_outputs[0])
        for i in range(self.num_scenario):
            out = torch.where(scenario_mask[i], final_outputs[i], out)

        return out, final_outputs
