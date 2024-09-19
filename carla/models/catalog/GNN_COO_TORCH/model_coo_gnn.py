import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, GCNConv, BatchNorm


class GNN_COO(torch.nn.Module):
    def __init__(
        self, nfeat, hid_list_conv, nclass, dropout
    ):
        super(GNN_COO, self).__init__()
        self.nclass = nclass
        self.layers_conv: nn.ModuleList = nn.ModuleList()
        self.use_dropout = dropout > 0.0
        current_dim = nfeat
        # hid_list_conv = hid_list_conv[1:]
        for hids in hid_list_conv:
            self.layers_conv.append(
                GCNConv(in_channels=current_dim, out_channels=hids)
            )

            current_dim = hids
        
        if self.nclass <= 1:
            self.layers_conv.append(nn.Linear(current_dim, 1))
        # multiclass
        else:
            # self.lin = nn.Linear(current_dim, self.nclass)
            self.layers_conv.append(nn.Linear(current_dim, self.nclass))
        
    def forward(self, x, edge_index, edge_attr):

        # cat_list = []
        # dynamic conv
        for layer in self.layers_conv:
            
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
                x = F.relu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = layer(x)
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
            