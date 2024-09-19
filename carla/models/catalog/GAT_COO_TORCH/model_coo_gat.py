import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, GCNConv, BatchNorm


class GAT_COO(torch.nn.Module):
    def __init__(
        self, nfeat, hid_list_att, hid_list_conv, nclass, dropout, alpha, nheads
    ):
        super(GAT_COO, self).__init__()
        self.nclass = nclass
        self.layers_conv: nn.ModuleList = nn.ModuleList()
        self.use_dropout = dropout > 0.0
        current_dim = nfeat
        i=0
        # hid_list_att = hid_list_att[1:]
        for hids in hid_list_att:
            
            if i==0:
                self.layers_conv.append(
                    GATConv(in_channels=nfeat, out_channels= hids, heads=nheads, dropout=dropout, negative_slope= alpha, concat=True)
                )
            
            else:
                self.layers_conv.append(
                    GATConv(in_channels=current_dim * nheads, out_channels= hids, heads=nheads, dropout=dropout, negative_slope= alpha, concat=True)
                )
            current_dim = hids
            i+=1


        current_dim = hid_list_att[-1] * nheads
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


        # dynamic conv
        for layer in self.layers_conv:
            
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
                x = F.relu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)

            # enter if it is type GATConv
            elif isinstance(layer, GATConv):
                x = layer(x, edge_index, edge_attr)
                x = F.elu(x)
                if self.use_dropout:
                    x = F.dropout(x, self.dropout, training=self.training)

            else:
                x = layer(x)
        # No activation function for the output layer (assuming classification task)
        # x = self.layers_conv[-1](torch.cat(cat_list, dim=1))
        # x = self.lin(x)
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

class GAT_COO(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, n_heads=4, edge_updates=False, edge_dim=None, dropout=0.0, final_dropout=0.5):
        super(GAT_COO, self).__init__()
        # GAT specific code
        tmp_out = n_hidden // n_heads
        n_hidden = tmp_out * n_heads

        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.nclass = n_classes
        # First Layer: Linear
        # self.node_emb = nn.Linear(num_features, n_hidden)
        # self.edge_emb = nn.Linear(edge_dim, n_hidden)
        
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        # self.batch_norms = nn.ModuleList()
        conv = GATConv(num_features, self.n_hidden, self.n_heads, concat = True, dropout = self.dropout, add_self_loops = True)
        self.convs.append(conv)
        for _ in range(self.num_gnn_layers):
            conv = GATConv(self.n_hidden, tmp_out, self.n_heads, concat = True, dropout = self.dropout, add_self_loops = True)
            
            if self.edge_updates: 
                self.emlps.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden),
                                                nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden)))
                
            self.convs.append(conv)
            # self.batch_norms.append(BatchNorm(n_hidden))
                
        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),
                                 Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                                 Linear(25, n_classes))
                
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        
        # x = self.node_emb(x)
        # edge_attr = self.edge_emb(edge_attr)
        
        for i in range(self.num_gnn_layers):
            x = (F.relu(self.convs[i](x, edge_index, edge_attr))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                    
        # logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        # logging.debug(f"x.shape = {x.shape}")
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        # logging.debug(f"x.shape = {x.shape}")
        out = x
        out = self.mlp(out)
        
        if self.nclass <= 1:
            return F.sigmoid(out)
        # multiclass
        else:
            return F.log_softmax(out, dim=1)

    def loss(self, pred, label):
        if self.nclass <= 1:
            return F.binary_cross_entropy(pred, label)
        # multiclass
        else:
            return F.nll_loss(pred, label)
        # return F.nll_loss(pred, label)
        
'''