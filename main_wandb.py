import wandb
#wandb.login(key='61aac9b0b0cd9f6b611dda5a9137dd62870e5cdf')

import os
import random

import numpy as np
import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch_geometric.loader import NeighborLoader
from carla.data.catalog import AMLtoGraph

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.dense import DenseGCNConv
from torch.nn.utils import clip_grad_norm_
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, BinaryF1Score, BinaryAccuracy
from carla.models.catalog.parse_gnn import normalize_adj


# Import Model
from carla.models.catalog.GAT_TORCH import GAT as gat_torch
from carla.models.catalog.GAT_COO_TORCH import GAT_COO as gat_coo_torch

from carla.models.catalog.GNN_TORCH import GCNSynthetic as gnn_torch
from carla.models.catalog.GNN_COO_TORCH import GNN_COO as gnn_coo_torch

from carla.models.catalog.GIN_TORCH import GIN as gin_torch
from carla.models.catalog.GIN_COO_TORCH import GIN_COO as gin_coo_torch

wandb.login()
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


sweep_config = {
    'method': 'bayes',
    
    'metric':{
        'goal': 'maximize',
        'name': 'F1'
    },
    
    'parameters':{
        'hidden_list_conv': {
            'values': [[31]*i for i in range(8)] + [[50]*i for i in range(8)] + [[100]*i for i in range(8)]
        },
        
        'hidden_list_att':{
            'values': [[31]*i for i in range(8)] + [[50]*i for i in range(8)] + [[100]*i for i in range(8)]
        },
        
        'alpha':{
            'max': 0.4,
            'min': 0.1,
            'distribution': 'uniform'
        },
        
        'nheads':{
            'max': 8,
            'min': 1,
            'distribution': 'int_uniform'   
        },
        
        'dropout':{
            'values': [0.0]
        },
        
        
        'epochs':{
            'max': 2000,
            'min': 500,
            'distribution': 'int_uniform'   
        },
        
        'classes':{
            'values': [2]
        },
        
        'batch_size':{
            'max': 2048,
            'min': 512,
            'distribution': 'int_uniform'   
        },
        
        'num_neig':{
            'values': [[31]*i for i in range(4)] + [[50]*i for i in range(4)] + [[100]*i for i in range(4)]
        },
        
        'learning_rate':{
            'values': [0.01, 0.02, 0.001, 0.002, 0.0001, 0.0002]
        },

        'clip':{
            'max': 4.0,
            'min': 1.0,
            'distribution': 'int_uniform'   
        },
        
        'dataset':{
            'values': ['AML']
        },

        'architecture':{
            'values': ['gat'],
            'distribution': 'categorical'
        }
        
    }

}


def make(config):
    # Make the data
    datagraph = get_data(path = "/home/hl-turing/VSCodeProjects/Flavio/CARLA/data/AML_Laund_Clean.csv")
    train_loader = make_loader(datagraph, datagraph.train_mask, batch_size=config.batch_size, num_neig=config.num_neig)
    test_loader = make_loader(datagraph, datagraph.test_mask, batch_size=config.batch_size, num_neig=config.num_neig)

    # Make the model
    if config.architecture == 'gnn':
        model = gnn_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_conv=config.hidden_list_conv,
                nclass=config.classes,
                dropout=config.dropout,
        ).to(device)
    elif config.architecture == "gat":
        #hidden_list_att = [50 for _ in range(3)]
        model = gat_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_att=config.hidden_list_att,  # da parametrizzare
                hid_list_conv=config.hidden_list_conv,
                nclass=config.classes,
                dropout=config.dropout,
                alpha=config.alpha,
                nheads=len(config.hidden_list_att),
            ).to(device)
    elif config.architecture == "gin":
            # forse definire hidden_list_att DA PARAMETRIZZARE
        #hidden_list_att = [50 for _ in range(7)]
        model = gin_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_gin=config.hidden_list_att,  # da parametrizzare
                hid_list_conv=config.hidden_list_conv,
                nclass=config.classes,
                dropout=config.dropout,
                alpha=config.alpha,
                nheads=config.nheads
            ).to(device)
    elif config.architecture == 'gnn_coo':
        model = gnn_coo_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_conv=config.hidden_list_conv,
                nclass=config.classes,
                dropout=config.dropout,
        ).to(device)
        
    elif config.architecture == "gat_coo":
            # forse definire hidden_list_att DA PARAMETRIZZARE
        #hidden_list_att = [31 for _ in range(4)]
        model = gat_coo_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_att=config.hidden_list_att,  # da parametrizzare
                hid_list_conv=config.hidden_list_conv,
                nclass=config.classes,
                dropout=config.dropout,
                alpha=config.alpha,
                nheads=len(config.hidden_list_att),
            ).to(device)
    elif config.architecture == "gin_coo":
            # forse definire hidden_list_att DA PARAMETRIZZARE
        #hidden_list_att = [31 for _ in range(4)]
        model = gin_coo_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_gin=config.hidden_list_att,  # da parametrizzare
                hid_list_conv=config.hidden_list_conv,
                edge_dim = datagraph.edge_attr.shape[1],
                nclass=config.classes,
                dropout=config.dropout
            ).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer

def get_data(path):
    data = AMLtoGraph(path)
    datagraph = data.construct_GraphData()
    return datagraph

def make_loader(datagraph, mask, batch_size, num_neig):
    loader = NeighborLoader(
        datagraph,
        num_neighbors=num_neig,
        batch_size=batch_size,
        input_nodes=mask
    )
    return loader


def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    metric = BinaryAccuracy()
    # f1score = MulticlassF1Score(num_classes=2, average="macro")
    f1score = BinaryF1Score()
    t_total = time.time()
    # Run training and track with wandb

    for epoch in tqdm(range(config.epochs)):
        t = time.time()
        average_epoch_loss = 0.0
        model.train()
        for data in loader:
            data.to(device)

            adj = _create_adj_mat(data)
            norm_adj = normalize_adj(adj).to(device)
            features = torch.tensor(data.x).squeeze().to(device)
            labels = torch.tensor(data.y).squeeze().long().to(device)

            node_idx = [i for i in range(0, len(data.y))]
            idx_train = torch.masked_select(torch.Tensor(node_idx, device="cpu"), data.train_mask.cpu())
            # idx_test = torch.masked_select(torch.Tensor(node_idx, device="cpu"), data.test_mask.cpu())
            idx_train = idx_train.type(torch.int64)
            optimizer.zero_grad()
            if config.architecture == "gin":
                features, norm_adj = features.to(device), norm_adj.to(device)
                output = model(features, norm_adj).squeeze()
            
            # Nostri layer
            if config.architecture == "gnn" or config.architecture == "gat":
                features, norm_adj = features.to(device), norm_adj.to(device)
                output = model(features, norm_adj)
            ## coo list 
            elif config.architecture == "gnn_coo" or config.architecture == "gat_coo" or config.architecture == "gin_coo":
                features =  features.to(device)
                edge_index, edge_attr = data.edge_index.to(device), data.edge_attr.to(device)
                output = model(features, edge_index, edge_attr)
            elif config.architecture == "gUnet_coo":
                features =  features.to(device)
                edge_index, edge_attr = data.edge_index.to(device), data.edge_attr.to(device)
                output = model(features, edge_index)

            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])            
            y_pred = torch.argmax(output, dim=1)

            accuracy = metric.update(y_pred[idx_train], labels[idx_train])
            f1 = f1score.update(y_pred[idx_train], labels[idx_train])
            average_epoch_loss += loss_train.item()
            
            loss_train.backward()
            
            clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
        
        avg_loss = average_epoch_loss/len(loader)
        # Report metrics 
        train_log(avg_loss, epoch, accuracy, f1, t)
        
        
        
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    

def train_log(avg_loss, epoch, accuracy, f1, t):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": avg_loss, "accuracy":accuracy.compute(), "F1": f1.compute() })
    print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(avg_loss),
            "acc_train: {:.4f}".format(accuracy.compute()),
            "f1_train: {:.4f}".format(f1.compute()),
            "time: {:.4f}s".format(time.time() - t),
        )
    
    
    
def _create_adj_mat(data_graph):
    edges_index = data_graph.edge_index
    row_indices = edges_index[0]
    col_indices = edges_index[1]
    values = torch.ones(len(edges_index[0]))  # valori tutti a uno
    size = torch.Size([len(data_graph.x), len(data_graph.x)])
    sparse_matrix = torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices]), values, size=size, device="cuda"
    )
    adj_matrix = sparse_matrix.to_dense()
    return adj_matrix

def train_sweep(hyperparameters=None):

    # tell wandb to get started
    with wandb.init(config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config)

      # and test its final performance
      # test(model, test_loader)
      
    # Save the model checkpoint

      # torch.onnx.export(model, "model.onnx")
      #wandb.save("model.onnx")
      
sweep_id = wandb.sweep(sweep_config, project="pytorch-gnn")
wandb.agent(sweep_id, train_sweep, count=75)
