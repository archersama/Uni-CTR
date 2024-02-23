import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.activation import activation_layer


class LocalActivationUnit(nn.Module):
    """The LocalActivationUnit used in DIN with which the representation of
        user interests varies adaptively given different candidate items.

    Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

    Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

    Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_units=(64, 32), embedding_dim=4, activation='sigmoid', dropout_rate=0, dice_dim=3,
                 l2_reg=0, use_bn=False):
        super(LocalActivationUnit, self).__init__()

        self.dnn = DNN(inputs_dim=4 * embedding_dim,
                       hidden_units=hidden_units,
                       activation=activation,
                       l2_reg=l2_reg,
                       dropout_rate=dropout_rate,
                       dice_dim=dice_dim,
                       use_bn=use_bn)

        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        user_behavior_len = user_behavior.size(1)

        queries = query.expand(-1, user_behavior_len, -1)

        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior],
                                    dim=-1)  # as the source code, subtraction simulates verctors' difference
        attention_output = self.dnn(attention_input)

        attention_score = self.dense(attention_output)  # [B, T, 1]

        return attention_score


class DNN(nn.Module):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class PredictionLayer(nn.Module):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        return output


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation,
            groups, bias)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        oh = math.ceil(ih / self.stride[0])
        ow = math.ceil(iw / self.stride[1])
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        out = F.conv2d(x, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.key = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states):
        """
        hidden_states: [batch_size, seq_len, hidden_dim]
        """
        attn_scores = torch.matmul(F.tanh(self.key(hidden_states)), self.query)
        atte_weights = F.softmax(attn_scores, dim=1)  # shape: [batch_size, seq_len]
        weighted_sum = torch.matmul(atte_weights.unsqueeze(1), hidden_states).squeeze(
            1)  # shape: [batch_size, hidden_dim]

        return weighted_sum


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=[], dropout_rate=0.3, activation=nn.ReLU, use_bn=True):
        super(Expert, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation()

        prev_size = input_dim
        for size in hidden_size:
            self.layers.append(nn.Linear(prev_size, size))
            if use_bn:
                self.layers.append(nn.BatchNorm1d(size))
            self.layers.append(self.activation)
            self.layers.append(self.dropout)
            prev_size = size
        self.layers.append(nn.Linear(prev_size, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Gate(nn.Module):
    def __int__(self, input_dim, output_dim):
        super(Gate, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.layer(x), dim=1)


class AvgMeter:
    def __init__(self, name='Metric'):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0]*3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f'{self.name}: {self.avg:.4f}'
        return text


class Lora(nn.Module):
    def __init__(self, input_dim, output_dim, r, use_dnn, dnn_hidden_states):
        super(Lora, self).__init__()
        self.A = nn.Linear(input_dim, r)
        torch.nn.init.normal_(self.A.weight, mean=0, std=0.01)
        torch.nn.init.constant_(self.A.bias, 0)
        self.use_dnn = use_dnn
        self.B = nn.Linear(r, output_dim)
        torch.nn.init.zeros_(self.B.weight)
        torch.nn.init.zeros_(self.B.bias)

        self.dnn_layer = DNN(inputs_dim=r, hidden_units=dnn_hidden_states+[r])

    def forward(self, x):
        x = self.A(x)
        x = torch.sigmoid(x)
        if self.use_dnn:
            x = self.dnn_layer(x)
        x = self.B(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.2, narrowed_ratio=1):
        super(TransformerLayer, self).__init__()
        self.d_model = dim
        self.num_heads = num_heads
        self.narrowed_ratio = narrowed_ratio
        if self.narrowed_ratio < 1:
            self.down_sample_forward = nn.Sequential(
                nn.Linear(dim, int(dim * narrowed_ratio)),
                nn.ReLU(),
                nn.Linear(int(dim * narrowed_ratio), int(dim * narrowed_ratio))
            )
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(int(dim * narrowed_ratio), num_heads, dropout=dropout)
        self.up_sample_forward = nn.Sequential(
            nn.Linear(int(dim * narrowed_ratio), dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.layer_norm1 = nn.LayerNorm(int(dim * narrowed_ratio))
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, hidden_states):
        if self.narrowed_ratio < 1:
            # Down sample
            ff_output = self.down_sample_forward(hidden_states)
            ff_output = self.dropout(ff_output)
        else:
            ff_output = hidden_states

        # Self attention
        attention_output, _ = self.attention(ff_output, ff_output, ff_output)
        attention_output = self.dropout(attention_output)

        # Add & Norm
        attention_output = attention_output + ff_output
        attention_output = self.layer_norm1(attention_output)

        # Up sample
        output = self.up_sample_forward(attention_output)
        output = self.dropout(output)

        # Add & Norm
        hidden_states = hidden_states + output
        output = self.layer_norm2(hidden_states)

        return output
