# Based on https://github.com/tkipf/pygcn/blob/master/pygcn/

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCNSynthetic(nn.Module):
    """
    3-layer GCN used in GNN Explainer synthetic tasks
    """

    def __init__(self, nfeat: int, hid_list: list, nclass: int, dropout: int):
        super(GCNSynthetic, self).__init__()

        self.layers: nn.ModuleList = nn.ModuleList()
        self.use_dropout = dropout > 0.0

        current_dim = nfeat
        for hids in hid_list:
            self.layers.append(
                GraphConvolution(in_features=current_dim, out_features=hids)
            )

            current_dim = hids

        self.layers.append(nn.Linear(sum(hid_list), nclass))
        self.dropout = dropout

    def forward(self, x, adj):
        cat_list = []
        # Apply a ReLU activation function and dropout (if used) to each hidden layer
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            if isinstance(layer, GraphConvolution):
                x = F.relu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)
            cat_list.append(x)
        # No activation function for the output layer (assuming classification task)
        x = self.layers[-1](torch.cat(cat_list, dim=1))

        # x = self.lin(torch.cat((x1, x2, x3), dim=1)) da vedere
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)
