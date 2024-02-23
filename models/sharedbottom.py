import sys

sys.path.append("../")

import torch
import torch.nn as nn
import numpy as np
from layers.utils import MLP
from tqdm import tqdm


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        return out


class SharedBottom(nn.Module):
    def __init__(self, feature_dims, dense_cols, sparse_cols, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(sum(feature_dims), embedding_dim=embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        self.offsets = np.array((0, *np.cumsum(feature_dims)[:-1]))
        # self.embedding_output_dims = len(sparse_cols) * embed_dim + len(dense_cols)
        self.embedding_output_dims = len(sparse_cols) * embed_dim

        self.input_size = self.embedding_output_dims
        self.experts_out = 32
        self.experts_hidden = 32
        self.towers_hidden = 32

        self.experts = MLP(self.input_size, False, [self.experts_hidden, self.experts_out], 0)

        self.towers = nn.ModuleList([Tower(self.experts_out, 16, self.towers_hidden) for _ in range(3)])

        self.mlp = MLP(16, True, [32, 16], 0)

    def forward(self, sparse):
        b = sparse.shape[0]
        slot_id = sparse[:, 18].clone().detach()  # sencario_id
        sparse = sparse + sparse.new_tensor(self.offsets).unsqueeze(0)
        sparse = self.embedding(sparse).reshape(b, -1)

        slot1_mask = (slot_id == 1)
        slot2_mask = (slot_id == 2)
        slot3_mask = (slot_id == 3)

        experts_o = self.experts(sparse)  # [b, experts_out]

        final_output = [t(ti) for t, ti in zip(self.towers, experts_o)]  # [3, b, 16]

        output = torch.zeros(b, 16).to(sparse.device)
        output = torch.where(slot1_mask.unsqueeze(1), final_output[0], output)
        output = torch.where(slot2_mask.unsqueeze(1), final_output[1], output)
        output = torch.where(slot3_mask.unsqueeze(1), final_output[2], output)

        ctr = torch.sigmoid(self.mlp(output))

        return ctr