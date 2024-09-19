import math

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.dense import DenseGATConv, DenseGCNConv


class GAT(nn.Module):
    def __init__(
        self, nfeat, hid_list_att, hid_list_conv, nclass, dropout, alpha, nheads
    ):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.nclass = nclass
        self.layers_conv: nn.ModuleList = nn.ModuleList()
        self.use_dropout = dropout > 0.0
        current_dim = nfeat
        i=0
        # hid_list_att = hid_list_att[1:]
        for hids in hid_list_att:
            
            if i==0:
                self.layers_conv.append(
                    DenseGATConv(in_channels=nfeat, out_channels= hids, heads=nheads, dropout=dropout, negative_slope= alpha, concat=True)
                )
            
            else:
                self.layers_conv.append(
                    DenseGATConv(in_channels=current_dim * nheads, out_channels= hids, heads=nheads, dropout=dropout, negative_slope= alpha, concat=True)
                )
            current_dim = hids
            i+=1


        current_dim = hid_list_att[-1] * nheads
        # hid_list_conv = hid_list_conv[1:]
        for hids in hid_list_conv:
            self.layers_conv.append(
                DenseGCNConv(in_channels=current_dim, out_channels=hids)
            )

            current_dim = hids
        
        if self.nclass <= 1:
            self.layers_conv.append(nn.Linear(current_dim, 1))
        # multiclass
        else:
            # self.lin = nn.Linear(current_dim, self.nclass)
            self.layers_conv.append(nn.Linear(current_dim, self.nclass))
        

    def forward(self, x, adj):
        # dynamic conv
        for layer in self.layers_conv:
            
            if isinstance(layer, DenseGCNConv):
                x = layer(x, adj)
                x = F.relu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)

            # enter if it is type GATConv
            elif isinstance(layer, DenseGATConv):
                x = layer(x, adj)
                x = F.elu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)

            else:
                x = layer(x)
            # cat_list.append(x)
        # No activation function for the output layer (assuming classification task)
        # x = self.layers_conv[-1](torch.cat(cat_list, dim=1))

        if self.nclass <= 1:
            return F.sigmoid(x)
        # multiclass
        else:
            return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        if self.nclass <= 1:
            return F.binary_cross_entropy(pred, label)
        # multiclass
        else:
            return F.nll_loss(pred, label)
        # return F.nll_loss(pred, label)



'''
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, int(out_features)))
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


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # print(out_features)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(
            h, self.W
        )  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GAT(nn.Module):
    def __init__(
        self, nfeat, hid_list_att, hid_list_conv, nclass, dropout, alpha, nheads
    ):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nclass = nclass
        
        # Dynamic attentions
        self.attentions = [
            GraphAttentionLayer(
                nfeat, hid_list_att[i], dropout=dropout, alpha=alpha, concat=True
            )
            for i in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        self.out_att = GraphAttentionLayer(
            hid_list_att[nheads - 1] * nheads,
            hid_list_conv[0],
            dropout=dropout,
            alpha=alpha,
            concat=False,
        )

        # dynamic layers conv
        self.layers_conv: nn.ModuleList = nn.ModuleList()
        self.use_dropout = dropout > 0.0

        current_dim = hid_list_conv[0]
        hid_list = hid_list_conv[1:]
        for hids in hid_list:
            self.layers_conv.append(
                GraphConvolution(in_features=current_dim, out_features=hids)
            )

            current_dim = hids

        if self.nclass <= 1:
            # self.layers_conv.append(nn.Linear(sum(hid_list), 1))
            self.layers_conv.append(nn.Linear(current_dim, 1))
        # multiclass
        else:
            # self.layers_conv.append(nn.Linear(sum(hid_list), self.nclass))
            self.layers_conv.append(nn.Linear(current_dim, self.nclass))
        self.dropout = dropout

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        # cat_list = []
        # dynamic conv
        for layer in self.layers_conv:
            if isinstance(layer, GraphConvolution):
                x = layer(x, adj)
                x = F.relu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = layer(x)
            # cat_list.append(x)
        # No activation function for the output layer (assuming classification task)
        # x = self.layers_conv[-1](torch.cat(cat_list, dim=1))

        if self.nclass <= 1:
            return F.sigmoid(x)
        # multiclass
        else:
            return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        if self.nclass <= 1:
            return F.binary_cross_entropy(pred, label)
        # multiclass
        else:
            return F.nll_loss(pred, label)
        # return F.nll_loss(pred, label)
'''

