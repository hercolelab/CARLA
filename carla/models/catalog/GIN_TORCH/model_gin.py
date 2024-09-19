import math

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.dense import DenseGINConv, DenseGCNConv

class GIN(nn.Module):
    def __init__(
        self, nfeat, hid_list_gin, hid_list_conv, nclass, dropout
    ):
        """Dense version of GAT."""
        super(GIN, self).__init__()
        self.nclass = nclass
        self.layers_conv: nn.ModuleList = nn.ModuleList()
        self.use_dropout = dropout > 0.0
        current_dim = nfeat
        i=0
        # hid_list_att = hid_list_att[1:]
        for hids in hid_list_gin:
            
            if i==0:
                self.layers_conv.append(
                    DenseGINConv(nn.Sequential(nn.Linear(nfeat, hids), 
                                               nn.ReLU(), 
                                               nn.Linear(hids, hids)
                ))
                )
            
            else:
                self.layers_conv.append(
                    DenseGINConv(nn.Sequential(nn.Linear(hids, hids), 
                                               nn.ReLU(), 
                                               nn.Linear(hids, hids)
                ))
                )
            current_dim = hids
            i+=1


        current_dim = hid_list_gin[-1]
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
            elif isinstance(layer, DenseGINConv):
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
