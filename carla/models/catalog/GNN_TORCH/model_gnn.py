import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.dense import DenseGCNConv
# from typing import *

'''

# from carla.models.api import MLModel
class GCNSynthetic(torch.nn.Module):
    def __init__(
        self, nfeat, hid_list_conv, nclass, dropout
    ):
        super(GCNSynthetic, self).__init__()
        self.nclass = nclass
        self.layers_conv: nn.ModuleList = nn.ModuleList()
        self.use_dropout = dropout > 0.0
        current_dim = nfeat
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

        # cat_list = []
        # dynamic conv
        for layer in self.layers_conv:
            
            if isinstance(layer, DenseGCNConv):
                x = layer(x, adj)
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
            


'''

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

    def __init__(self, nfeat: int, hid_list_conv: list, nclass: int, dropout: float):
        super(GCNSynthetic, self).__init__()

        self.layers: nn.ModuleList = nn.ModuleList()
        self.use_dropout = dropout > 0.0
        self.nclass = nclass

        current_dim = nfeat
        for hids in hid_list_conv:
            self.layers.append(
                GraphConvolution(in_features=current_dim, out_features=hids)
            )

            current_dim = hids
        
        if self.nclass <= 1:
            # self.layers.append(nn.Linear(sum(hid_list_conv), 1))
            self.layers.append(nn.Linear(current_dim, 1))
        # multiclass
        else:
            # self.layers.append(nn.Linear(sum(hid_list_conv), self.nclass))
            self.layers.append(nn.Linear(current_dim, self.nclass))
        self.dropout = dropout

    def forward(self, x, adj):
        # cat_list = []
        # Apply a ReLU activation function and dropout (if used) to each hidden layer
        for layer in self.layers:
            if isinstance(layer, GraphConvolution):
                x = layer(x, adj)
                x = F.relu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = layer(x)
            # cat_list.append(x)
        # No activation function for the output layer (assuming classification task)
        # x = self.layers[-1](torch.cat(cat_list, dim=1))

        # x = self.lin(torch.cat((x1, x2, x3), dim=1)) da vedere
        # return F.log_softmax(x, dim=1)
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
            return F.cross_entropy(pred, label)
        
        

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x, adj):
        return self._model(x, adj)

    def predict(self, x, adj):
        """
        One-dimensional prediction of ml model for an output interval of [0, 1].

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array or pd.DataFrame
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        iterable object
            Ml model prediction for interval [0, 1] with shape N x 1
        """

        return torch.argmax(self._model(x, adj), dim=1)

