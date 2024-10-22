# from .gat import GraphAttentionLayer
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .utils import (
    create_symm_matrix_from_vec,
    create_vec_from_symm_matrix,
    get_degree_matrix,
)

from torch_geometric.nn.dense import DenseGATConv, DenseGCNConv


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


class GraphConvolutionPerturb(nn.Module):
    """
    Similar to GraphConvolution except includes P_hat
    It is referred in the original paper as g(A_v, X_v, W; P) = $softmax [\sqrt(D^{line}_v]) (P*A_v + I) \sqrt(D^{line}_v]) X_v * W$
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionPerturb, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Build a random weight tensor
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # If bias then build a random tensor for bias
        if bias is not None:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        args:
                input: the features of the graph

                adj: the adjacency matrix
        """

        support = torch.mm(input, self.weight)

        output = torch.spmm(adj, support)

        return output if self.bias is None else output + self.bias

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


class GATSyntheticPerturb(nn.Module):
    def __init__(
        self,
        nfeat: int,
        hid_list_att: list,
        hid_list_conv: list,
        nclass: int,
        adj,
        dropout,
        beta,
        alpha,
        nheads,
        edge_additions=False,
    ):
        super(GATSyntheticPerturb, self).__init__()

        self.adj = adj
        self.nclass = nclass
        self.alpha = alpha
        self.num_nodes = self.adj.shape[0]
        self.edge_additions = (
            edge_additions  # are edge additions included in perturbed matrix
        )
        self.dropout = dropout
        self.beta = beta
        self.nheads = nheads

        # P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        self.P_vec_size = (
            int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes
        )

        # Perturbation matrix initialization
        if self.edge_additions:
            self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        else:
            self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

        # self.reset_parameters()
        self.dropout = dropout

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
        i = 0
        for hids in hid_list:
            if i == len(hid_list):  # last layer --> Convolution Layer
                self.layers_conv.append(
                    GraphConvolution(in_features=current_dim, out_features=hids)
                )
            else:  # before last layer --> Convolution Perturb Layer
                self.layers_conv.append(
                    GraphConvolutionPerturb(in_features=current_dim, out_features=hids)
                )
            current_dim = hids
            i += 1

        if self.nclass <= 1:
            self.layers_conv.append(nn.Linear(sum(hid_list), 1))
        # multiclass
        else:
            self.layers_conv.append(nn.Linear(sum(hid_list), self.nclass))
        self.dropout = dropout
        """

        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        self.out_att = GraphAttentionLayer(
            nhid * nheads, 50, dropout=dropout, alpha=alpha, concat=False
        )

        self.gcn = GraphConvolution(50, self.nclass)
        """

    def reset_parameters(self, eps=10**-4):
        # Think more about how to initialize this
        with torch.no_grad():
            if self.edge_additions:
                adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).numpy()
                for i in range(len(adj_vec)):
                    if i < 1:
                        adj_vec[i] = adj_vec[i] - eps
                    else:
                        adj_vec[i] = adj_vec[i] + eps
                torch.add(
                    self.P_vec, torch.FloatTensor(adj_vec)
                )  # self.P_vec is all 0s
            else:
                torch.sub(self.P_vec, eps)

    def forward(self, x, sub_adj):
        """ """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sub_adj = sub_adj

        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(
            self.P_vec, self.num_nodes
        )  # Ensure symmetry

        # aggiunta

        # Initizlize the adjacency matrix with self-loops
        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True
        if self.edge_additions:  # Learn new adj matrix directly
            A_tilde = F.sigmoid(self.P_hat_symm) + torch.eye(
                self.num_nodes, device=device
            )  # Use sigmoid to bound P_hat in [0,1]
        else:  # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            # print(self.P_hat_symm.get_device())
            # print(self.sub_adj.get_device())
            A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(
                self.num_nodes, device=device
            )  # Use sigmoid to bound P_hat in [0,1]

        # D_tilde is the degree matrix
        D_tilde = get_degree_matrix(A_tilde).detach()  # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, norm_adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, norm_adj))
        # Convolution Layers
        cat_list = []
        # dynamic conv
        for layer in self.layers_conv[:-1]:
            x = layer(x, norm_adj)
            if isinstance(layer, GraphConvolution):
                x = F.relu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)
            cat_list.append(x)
        # No activation function for the output layer (assuming classification task)
        x = self.layers_conv[-1](torch.cat(cat_list, dim=1))

        if self.nclass <= 1:
            return F.sigmoid(x)
        # multiclass
        else:
            return F.log_softmax(x, dim=1)

    def forward_prediction(self, x):
        """ """
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()  # threshold P_hat

        if self.edge_additions:
            A_tilde = self.P + torch.eye(self.num_nodes)
        else:
            A_tilde = self.P * self.adj.to(device) + torch.eye(
                self.num_nodes, device=device
            )

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, norm_adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, norm_adj))
        # Convolution Layers
        cat_list = []
        # dynamic conv
        for layer in self.layers_conv[:-1]:
            x = layer(x, norm_adj)
            if isinstance(layer, GraphConvolution):
                x = F.relu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)
            cat_list.append(x)
        # No activation function for the output layer (assuming classification task)
        x = self.layers_conv[-1](torch.cat(cat_list, dim=1))

        if self.nclass <= 1:
            return F.sigmoid(x), self.P
        # multiclass
        else:
            return F.log_softmax(x, dim=1), self.P
        # return F.log_softmax(x, dim=1), self.P

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        """ """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pred_same = (y_pred_new_actual == y_pred_orig).float()


        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        if self.edge_additions:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj.to(device)
        cf_adj.requires_grad = (
            True  # Need to change this otherwise loss_graph_dist has no gradient
        )

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        if self.nclass <= 1:
            loss_pred = -F.binary_cross_entropy(output.float(), y_pred_orig.float())
        else:
            loss_pred = -F.nll_loss(output, y_pred_orig)
        loss_graph_dist = (
            sum(sum(abs(cf_adj - self.adj.to(device)))) / 2
        )  # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, cf_adj
    
'''

class GATSyntheticPerturb(nn.Module):
    def __init__(
        self,
        nfeat: int,
        hid_list_att: list,
        hid_list_conv: list,
        nclass: int,
        adj,
        dropout,
        beta,
        alpha,
        nheads,
        edge_additions=False,
    ):
        super(GATSyntheticPerturb, self).__init__()

        self.adj = adj
        self.nclass = nclass
        self.alpha = alpha
        self.num_nodes = self.adj.shape[0]
        self.edge_additions = (
            edge_additions  # are edge additions included in perturbed matrix
        )
        self.dropout = dropout
        self.beta = beta
        self.nheads = nheads

        # P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        self.P_vec_size = (
            int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes
        )

        # Perturbation matrix initialization
        if self.edge_additions:
            self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        else:
            self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))

        # self.reset_parameters()
        self.dropout = dropout
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
        

    def reset_parameters(self, eps=10**-4):
        # Think more about how to initialize this
        with torch.no_grad():
            if self.edge_additions:
                adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).numpy()
                for i in range(len(adj_vec)):
                    if i < 1:
                        adj_vec[i] = adj_vec[i] - eps
                    else:
                        adj_vec[i] = adj_vec[i] + eps
                torch.add(
                    self.P_vec, torch.FloatTensor(adj_vec)
                )  # self.P_vec is all 0s
            else:
                torch.sub(self.P_vec, eps)

    def forward(self, x, sub_adj):
        """ """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sub_adj = sub_adj

        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(
            self.P_vec, self.num_nodes
        )  # Ensure symmetry

        # aggiunta

        # Initizlize the adjacency matrix with self-loops
        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True
        if self.edge_additions:  # Learn new adj matrix directly
            A_tilde = F.sigmoid(self.P_hat_symm) + torch.eye(
                self.num_nodes, device=device
            )  # Use sigmoid to bound P_hat in [0,1]
        else:  # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            # print(self.P_hat_symm.get_device())
            # print(self.sub_adj.get_device())
            A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(
                self.num_nodes, device=device
            )  # Use sigmoid to bound P_hat in [0,1]

        # D_tilde is the degree matrix
        D_tilde = get_degree_matrix(A_tilde).detach()  # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        for layer in self.layers_conv:
            
            if isinstance(layer, DenseGCNConv):
                x = layer(x, norm_adj)
                x = F.relu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)

            # enter if it is type GATConv
            elif isinstance(layer, DenseGATConv):
                x = layer(x, norm_adj)
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

    def forward_prediction(self, x):
        """ """
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()  # threshold P_hat

        if self.edge_additions:
            A_tilde = self.P + torch.eye(self.num_nodes)
        else:
            A_tilde = self.P * self.adj.to(device) + torch.eye(
                self.num_nodes, device=device
            )

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)


        for layer in self.layers_conv:
            
            if isinstance(layer, DenseGCNConv):
                x = layer(x, norm_adj)
                x = F.relu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)

            # enter if it is type GATConv
            elif isinstance(layer, DenseGATConv):
                x = layer(x, norm_adj)
                x = F.elu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)

            else:
                x = layer(x)

        if self.nclass <= 1:
            return F.sigmoid(x), self.P
        # multiclass
        else:
            return F.log_softmax(x, dim=1), self.P
        # return F.log_softmax(x, dim=1), self.P

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        """ """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pred_same = (y_pred_new_actual == y_pred_orig).float()
        if pred_same == 1.0:
            print('sono entrato')

        # Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        if self.edge_additions:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj.to(device)
        cf_adj.requires_grad = (
            True  # Need to change this otherwise loss_graph_dist has no gradient
        )

        # Want negative in front to maximize loss instead of minimizing it to find CFs
        if self.nclass <= 1:
            loss_pred = -F.binary_cross_entropy(output.float(), y_pred_orig.float())
        else:
            loss_pred = -F.nll_loss(output, y_pred_orig)
        loss_graph_dist = (
            sum(sum(abs(cf_adj - self.adj.to(device)))) / 2
        )  # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, cf_adj
'''