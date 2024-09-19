import time
from typing import Union

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics import F1Score
import xgboost
from sklearn.ensemble import RandomForestClassifier
from torch import nn
import matplotlib.pyplot as plt

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import NeighborLoader

# from torch_geometric.utils import dense_to_sparse
# from torch_geometric.utils.metric import accuracy
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, BinaryF1Score, BinaryAccuracy

from torch_geometric.nn.models import GraphUNet as gUnet_coo_torch

from carla.data.catalog.graph_catalog import AMLtoGraph, PlanetoidGraph
from carla.models.catalog.ANN_TF import AnnModel
from carla.models.catalog.ANN_TF import AnnModel as ann_tf
from carla.models.catalog.ANN_TORCH import AnnModel as ann_torch

from carla.models.catalog.GAT_TORCH import GAT as gat_torch
from carla.models.catalog.GAT_COO_TORCH import GAT_COO as gat_coo_torch

from carla.models.catalog.GNN_TORCH import GCNSynthetic as gnn_torch
from carla.models.catalog.GNN_COO_TORCH import GNN_COO as gnn_coo_torch

from carla.models.catalog.GIN_TORCH import GIN as gin_torch
from carla.models.catalog.GIN_COO_TORCH import GIN_COO as gin_coo_torch

from carla.models.catalog.Linear_TF import LinearModel
from carla.models.catalog.Linear_TF import LinearModel as linear_tf
from carla.models.catalog.Linear_TORCH import LinearModel as linear_torch

# new import
from carla.models.catalog.parse_gnn import normalize_adj


def train_model(
    catalog_model,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    hidden_size: list,
    n_estimators: int,
    max_depth: int,
) -> Union[LinearModel, AnnModel, RandomForestClassifier, xgboost.XGBClassifier]:
    """

    Parameters
    ----------
    catalog_model: MLModelCatalog
        API for classifier
    x_train: pd.DataFrame
        training features
    y_train: pd.DataFrame
        training labels
    x_test: pd.DataFrame
        test features
    y_test: pd.DataFrame
        test labels
    learning_rate: float
        Learning rate for the training.
    epochs: int
        Number of epochs to train on.
    batch_size: int
        Size of each batch
    hidden_size: list[int]
        hidden_size[i] contains the number of nodes in layer [i].
    n_estimators: int
        Number of trees in forest
    max_depth: int
        Max depth of trees in forest

    Returns
    -------
    Union[LinearModel, AnnModel, RandomForestClassifier, xgboost.XGBClassifier]
    """
    print(f"balance on test set {y_train.mean()}, balance on test set {y_test.mean()}")

    # NON VEDERE QUESTO BRANCH
    if catalog_model.backend == "tensorflow":
        if catalog_model.model_type == "linear":
            model = linear_tf(
                dim_input=x_train.shape[1],
                num_of_classes=len(pd.unique(y_train)),
                data_name=catalog_model.data.name,
            )  # type: Union[linear_tf, ann_tf]
        elif catalog_model.model_type == "ann":
            model = ann_tf(
                dim_input=x_train.shape[1],
                dim_hidden_layers=hidden_size,
                num_of_classes=len(pd.unique(y_train)),
                data_name=catalog_model.data.name,
            )
        else:
            raise ValueError(
                f"model type not recognized for backend {catalog_model.backend}"
            )
        model.build_train_save_model(
            x_train,
            y_train,
            x_test,
            y_test,
            epochs,
            batch_size,
            model_name=catalog_model.model_type,
        )
        return model.model

    # PYTORCH
    elif catalog_model.backend == "pytorch":
        train_dataset = DataFrameDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = DataFrameDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # aggiungere GNN
        if catalog_model.model_type == "linear":
            model = linear_torch(
                dim_input=x_train.shape[1], num_of_classes=len(pd.unique(y_train))
            )
        elif catalog_model.model_type == "ann":
            model = ann_torch(
                input_layer=x_train.shape[1],
                hidden_layers=hidden_size,
                num_of_classes=len(pd.unique(y_train)),
            )

        else:
            raise ValueError(
                f"model type not recognized for backend {catalog_model.backend}"
            )

        _training_torch(
            model,
            train_loader,
            test_loader,
            learning_rate,
            epochs,
        )

        return model

    # NON VEDERE QUESTO
    elif catalog_model.backend == "sklearn":
        if catalog_model.model_type == "forest":
            random_forest_model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth
            )
            random_forest_model.fit(X=x_train, y=y_train)
            train_score = random_forest_model.score(X=x_train, y=y_train)
            test_score = random_forest_model.score(X=x_test, y=y_test)
            print(
                "model fitted with training score {} and test score {}".format(
                    train_score, test_score
                )
            )
            return random_forest_model
        else:
            raise ValueError(
                f"model type not recognized for backend {catalog_model.backend}"
            )
    # NON VEDERE QUESTO
    elif catalog_model.backend == "xgboost":
        if catalog_model.model_type == "forest":
            param = {
                "max_depth": max_depth,
                "objective": "binary:logistic",
                "n_estimators": n_estimators,
            }
            xgboost_model = xgboost.XGBClassifier(**param)
            xgboost_model.fit(
                x_train,
                y_train,
                eval_set=[(x_train, y_train), (x_test, y_test)],
                eval_metric="logloss",
                verbose=True,
            )
            return xgboost_model
        else:
            raise ValueError(
                f"model type not recognized for backend {catalog_model.backend}"
            )
    else:
        raise ValueError("model backend not recognized")


# %%-------------------------------------------------------


class DataFrameDataset(Dataset):
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        # PyTorch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.X_train = torch.tensor(x.to_numpy(), dtype=torch.float32).to(device)
        self.Y_train = torch.tensor(y.to_numpy()).to(device)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


# %%-------------------------------------------------------


def _training_torch(
    model,
    train_loader,
    test_loader,
    learning_rate,
    epochs,
):
    loaders = {"train": train_loader, "test": test_loader}

    # Use GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # define the loss
    criterion = nn.BCELoss()

    # declaring optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    # training
    for e in range(epochs):
        print("Epoch {}/{}".format(e, epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:

            running_loss = 0.0
            running_corrects = 0.0

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode

            for i, (inputs, labels) in enumerate(loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device).type(torch.int64)
                labels = torch.nn.functional.one_hot(labels, num_classes=2)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs.float())
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(
                    torch.argmax(outputs, axis=1)
                    == torch.argmax(labels, axis=1).float()
                )

            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(loaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            print()


# %%-------------------------------------------------------
# -------------------------------------------------------  TRAINING GNN


def train_model_gnn(
    catalog_model,
    data: pd.DataFrame,
    lr: float,
    weight_decay: float,
    epochs: int,
    clip: float,
    hidden_list: list,
    hidden_list_conv: list,
    alpha: float,
    nheads: int,
    batch: int,
    neig: list
):
    if catalog_model.backend == "pytorch":
        # initialize dataframe
        # data = AMLtoGraph(data_table=data)
        # create datagraph with type Data
        if isinstance(data, AMLtoGraph):
            datagraph = data.construct_GraphData()
        elif isinstance(data, PlanetoidGraph):
            datagraph = data.getDataGraph()

        nclass = torch.max(datagraph.y.long()).item() + 1
        # create adj matrix by COO
        # adj_matrix = data.create_adj_matrix(datagraph).squeeze()
        # initialize the model
        if catalog_model.model_type == "gnn":
            model = gnn_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_conv=hidden_list,
                nclass=nclass,
                dropout=0.0,
            )
        # per ora è la GAT
        elif catalog_model.model_type == "gat":
            # forse definire hidden_list_att DA PARAMETRIZZARE
            # hidden_list_att = [31 for _ in range(2)]
            model = gat_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_att=hidden_list,  # da parametrizzare
                hid_list_conv=hidden_list_conv,
                nclass=nclass,
                dropout=0.0,
                alpha=alpha,
                nheads=nheads,
            )
        elif catalog_model.model_type == "gin":
            # forse definire hidden_list_att DA PARAMETRIZZARE
            hidden_list_att = [100 for _ in range(1)]
            model = gin_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_gin=hidden_list,  # da parametrizzare
                hid_list_conv=hidden_list_conv,
                nclass=nclass,
                dropout=0.0
            )
        elif catalog_model.model_type == "gnn_coo":
            model = gnn_coo_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_conv=hidden_list,
                nclass=nclass,
                dropout=0.0,
            )
            # TODO: rendi dinamico nclass (2 per AML) 7 per Cora
        elif catalog_model.model_type == "gat_coo":
            # forse definire hidden_list_att DA PARAMETRIZZARE
            # hidden_list_att = [31 for _ in range(4)]
            model = gat_coo_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_att=hidden_list,  # da parametrizzare
                hid_list_conv=hidden_list_conv,
                nclass=nclass,
                dropout=0.0,
                alpha=alpha,
                nheads=nheads,
            )
        elif catalog_model.model_type == "gin_coo":
            # forse definire hidden_list_att DA PARAMETRIZZARE
            # hidden_list_att = [50 for _ in range(4)]
            model = gin_coo_torch(
                nfeat=len(datagraph.x[0]),
                hid_list_gin=hidden_list,  # da parametrizzare
                hid_list_conv=hidden_list_conv,
                edge_dim = datagraph.edge_attr.shape[1],
                nclass=nclass,
                dropout=0.0
            )
        elif catalog_model.model_type == "gUnet_coo":
            model = gUnet_coo_torch(in_channels= len(datagraph.x[0]),
                                    hidden_channels= 31,
                                    out_channels= 2,
                                    depth = 3)
        # training gnn
        # _training_gnn_torch(
        #     model=model,
        #     data_graph=datagraph,
        #     adj=adj_matrix,
        #     learn_rate=lr,
        #     weight_decay=weight_decay,
        #     epochs=epochs,
        #     clip=clip,
        # )
        
        _training_gnn_torch_batch(model= model,
                                  data_graph= datagraph,
                                  learn_rate= lr,
                                  weight_decay=weight_decay,
                                  epochs=epochs,
                                  clip=clip,
                                  type_model = catalog_model.model_type,
                                  batch_size = batch,
                                  num_neig = neig)

        return model

    else:
        raise ValueError("model backend not recognized")


def _training_gnn_torch(model, data_graph, adj, learn_rate, weight_decay, epochs, clip):
    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    norm_adj = normalize_adj(adj).to(device)
    features = torch.tensor(data_graph.x).squeeze().to(device)
    labels = torch.tensor(data_graph.y, dtype=torch.long).squeeze().to(device)
    
    node_idx = [i for i in range(0, len(data_graph.y))]
    idx_train = torch.masked_select(torch.Tensor(node_idx), data_graph.train_mask)
    idx_test = torch.masked_select(torch.Tensor(node_idx), data_graph.test_mask)
    idx_train = idx_train.type(torch.int64)
    idx_test = idx_test.type(torch.int64)

    metric = BinaryAccuracy()
    f1score = MulticlassF1Score(num_classes=2, average="macro")

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learn_rate, weight_decay=weight_decay
    )

    t_total = time.time()
    for epoch in range(epochs):
        t = time.time()
        model.train()
        features, norm_adj = features.to(device), norm_adj.to(device)
        optimizer.zero_grad()
        output = model(features, norm_adj)

        loss_train = model.loss(output[idx_train], labels[idx_train])

        y_pred = torch.argmax(output, dim=1)

        acc_train = metric.update(y_pred[idx_train], labels[idx_train])
        f1_train = f1score.update(y_pred[idx_train], labels[idx_train])

        loss_train.backward()

        # clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(loss_train.item()),
            "acc_train: {:.4f}".format(acc_train.compute()),
            "f1_train: {:.4f}".format(f1_train.compute()),
            "time: {:.4f}s".format(time.time() - t),
        )

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # torch.save(model.state_dict(), "../models/gcn_3layer_{}".format(args.dataset) + ".pt")
    y_pred = _test(model, features, labels, norm_adj, idx_test)
    print(
        "y_true counts: {}".format(np.unique(labels.cpu().numpy(), return_counts=True))
    )
    print(
        "y_pred_orig counts: {}".format(
            np.unique(y_pred.cpu().numpy(), return_counts=True)
        )
    )
    print("Finished training!")


def _test(model, features, labels, norm_adj, idx_test):
    # modalità di evaluation
    model.eval()
    metric = MulticlassAccuracy()
    f1score = MulticlassF1Score(num_classes=2, average="macro")

    output = model(features, norm_adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    y_pred = torch.argmax(output, dim=1)
    acc_test = metric.update(y_pred[idx_test], labels[idx_test])
    f1_test = f1score.update(y_pred[idx_test], labels[idx_test])
    print(
        "Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.compute()),
        "f1-score= {:.4f}".format(f1_test.compute()),
    )
    return y_pred


# %%



def _training_gnn_torch_batch(
    model, data_graph, learn_rate, weight_decay, epochs, clip, type_model, batch_size ,num_neig
):
    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = NeighborLoader(
        data_graph,
        num_neighbors= num_neig,
        batch_size=batch_size,
        input_nodes=data_graph.train_mask,
    )

    test_loader = NeighborLoader(
        data_graph,
        num_neighbors=num_neig,
        batch_size=batch_size,
        input_nodes=data_graph.test_mask,
    )

    labels = torch.tensor(data_graph.y, dtype=torch.long).squeeze().to(device)
    
    metric = MulticlassAccuracy(num_classes=2)
    f1score = MulticlassF1Score(num_classes=2, average='macro')
    #metric = BinaryAccuracy()
    #f1score = BinaryF1Score()
    # define optimizer

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learn_rate
    )
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    train_f1 = []
    test_f1 = []
    t_total = time.time()
    for epoch in range(epochs):
        t = time.time()
        average_epoch_loss = 0.0
        model.train()
        #epoch_acc = 0.0
        #epoch_f1 = 0.0

        for data in train_loader:
            data.to(device)

            adj = _create_adj_mat(data)
            norm_adj = normalize_adj(adj).to(device)
            features = torch.tensor(data.x).squeeze().to(device)
            labels = torch.tensor(data.y).squeeze().long().to(device)

            node_idx = [i for i in range(0, len(data.y))]
            idx_train = torch.masked_select(torch.Tensor(node_idx, device="cpu"), data.train_mask.cpu())
            # idx_test = torch.masked_select(torch.Tensor(node_idx, device="cpu"), data.test_mask.cpu())
            idx_train = idx_train.type(torch.int64)
            # idx_test = idx_test.type(torch.int64)
            optimizer.zero_grad()
            if type_model == "gat" or type_model == "gin":
                features, norm_adj = features.to(device), norm_adj.to(device)
                output = model(features, norm_adj).squeeze()
            # layer nostri 
            if type_model == "gnn":
                features, norm_adj = features.to(device), norm_adj.to(device)
                output = model(features, norm_adj)
            ## coo list 
            elif type_model == "gnn_coo" or type_model == "gat_coo" or type_model == "gin_coo":
                features =  features.to(device)
                edge_index, edge_attr = data.edge_index.to(device), data.edge_attr.to(device)
                output = model(features, edge_index, edge_attr)
            elif type_model == "gUnet_coo":
                features =  features.to(device)
                edge_index, edge_attr = data.edge_index.to(device), data.edge_attr.to(device)
                output = model(features, edge_index)
            #loss_train = model.loss(output[idx_train], labels[idx_train])
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            # loss_train = model.loss(output[idx_train].squeeze(), labels[idx_train])
            # loss_train = F.binary_cross_entropy(output[idx_train].squeeze(), labels[idx_train])
            
            y_pred = torch.argmax(output, dim=1)
            # acc_train = metric.update(y_pred[idx_train], labels[idx_train])
            #acc_train = metric.update(y_pred, labels)
            
            #correct = (y_pred[idx_train] == labels[idx_train]).sum().item()
            #accuracy = correct/labels[idx_train].size(0)
            # f1_train = f1score.update(y_pred[idx_train], labels[idx_train])
            #correct_f1 = _binary_f1_score(y_pred[idx_train], labels[idx_train])
            # f1 = correct_f1/labels[idx_train].size(0)
            accuracy = metric.update(y_pred[idx_train], labels[idx_train])
            f1 = f1score.update(y_pred[idx_train], labels[idx_train])
            # epoch_acc += accuracy
            average_epoch_loss += loss_train.item()
            
            loss_train.backward()
            
            clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(average_epoch_loss/len(train_loader)),
            "acc_train: {:.4f}".format(accuracy.compute()),
            "f1_train: {:.4f}".format(f1.compute()),
            "time: {:.4f}s".format(time.time() - t),
        )
        train_losses.append(average_epoch_loss/len(train_loader))
        train_accuracies.append(accuracy.compute())
        train_f1.append(f1.compute())
        # inserimento del test 
        
        num_classes = len(labels.unique())
        # modalità di evaluation
        model.eval()
        if num_classes <= 1:
            metric_test = BinaryAccuracy()
            f1score_test = BinaryF1Score()
        else:
            metric_test = MulticlassAccuracy(num_classes=2)
            f1score_test = MulticlassF1Score(num_classes=2, average='macro')
            #metric_test = BinaryAccuracy()
            #f1score_test = BinaryF1Score()
        # f1score = MulticlassF1Score(num_classes=2, average="macro")
        average_epoch_loss_test = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                test_data.to(device)

                adj = _create_adj_mat(test_data)
                norm_adj = normalize_adj(adj).to(device)
                features = torch.tensor(test_data.x).squeeze().to(device)
                labels = torch.tensor(test_data.y).squeeze().long().to(device)

                node_idx = [i for i in range(0, len(test_data.y))]
                # idx_train = torch.masked_select(torch.Tensor(node_idx), test_data.train_mask)
                idx_test = torch.masked_select(torch.Tensor(node_idx, device = "cpu"), test_data.test_mask.cpu())
                # idx_train = idx_train.type(torch.int64)
                idx_test = idx_test.type(torch.int64)
                
                if type_model == "gin" or type_model == "gat":
                    features, norm_adj = features.to(device), norm_adj.to(device)
                    output = model(features, norm_adj).squeeze()
                
                if type_model == "gnn":
                    features, norm_adj = features.to(device), norm_adj.to(device)
                    output = model(features, norm_adj)
                    ## coo list 
                elif type_model == "gat_coo" or type_model == "gnn_coo" or type_model == "gin_coo":
                    features =  features.to(device)
                    edge_index, edge_attr = test_data.edge_index.to(device), test_data.edge_attr.to(device)
                    output = model(features, edge_index, edge_attr)
                elif type_model == "gUnet_coo":
                    features =  features.to(device)
                    edge_index, edge_attr = test_data.edge_index.to(device), test_data.edge_attr.to(device)
                    output = model(features, edge_index)
                    
                    
                if num_classes <=1:
                    loss_test = F.binary_cross_entropy(output[idx_test], labels[idx_test])
                else:
                    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
                    
                y_pred = torch.argmax(output, dim=1)
                acc_test = metric_test.update(y_pred[idx_test], labels[idx_test])
                f1_test = f1score_test.update(y_pred[idx_test], labels[idx_test])
                average_epoch_loss_test += loss_test.item()
            print(
                    "Test set results:",
                    "loss= {:.4f}".format(average_epoch_loss_test/len(test_loader)),
                    "accuracy= {:.4f}".format(acc_test.compute()),
                    "f1-score= {:.4f}".format(f1_test.compute()),
                )
            test_losses.append(average_epoch_loss_test/len(test_loader))
            test_accuracies.append(acc_test.compute())
            test_f1.append(f1_test.compute())
            
        
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    plotting(epochs, train_losses, train_accuracies, train_f1 ,test_losses, test_accuracies, test_f1, type_model)
    print(
            "Train set results:",
            "loss= {:.4f}".format(train_losses[-1]),
            "accuracy= {:.4f}".format(train_accuracies[-1]),
            "f1-score= {:.4f}".format(train_f1[-1]),
        )
    # torch.save(model.state_dict(), "../models/gcn_3layer_{}".format(args.dataset) + ".pt")
    y_pred = _test_batch(model, test_loader, labels, type_model, device)
    print(
        "y_true counts: {}".format(np.unique(labels.cpu().numpy(), return_counts=True))
    )
    print(
        "y_pred_orig counts: {}".format(
            np.unique(y_pred.cpu().numpy(), return_counts=True)
        )
    )
    
    print("Finished training!")



def _test_batch(model, test_model, labels, type_model, device):
    num_classes = len(labels.unique())
    # modalità di evaluation
    model.eval()
    if num_classes <= 1:
        metric = BinaryAccuracy()
        f1score = BinaryF1Score()
    else:
        metric = MulticlassAccuracy(num_classes=num_classes)
        f1score = MulticlassF1Score(num_classes=num_classes)
    # f1score = MulticlassF1Score(num_classes=2, average="macro")
    average_epoch_loss = 0.0
    
    for test_data in test_model:
        test_data.to(device)

        adj = _create_adj_mat(test_data)
        norm_adj = normalize_adj(adj).to(device)
        features = torch.tensor(test_data.x).squeeze().to(device)
        labels = torch.tensor(test_data.y).squeeze().long().to(device)

        node_idx = [i for i in range(0, len(test_data.y))]
        # idx_train = torch.masked_select(torch.Tensor(node_idx), test_data.train_mask)
        idx_test = torch.masked_select(torch.Tensor(node_idx, device = "cpu"), test_data.test_mask.cpu())
        # idx_train = idx_train.type(torch.int64)
        idx_test = idx_test.type(torch.int64)
        
        if type_model == "gin" or type_model == "gat":
            features, norm_adj = features.to(device), norm_adj.to(device)
            output = model(features, norm_adj).squeeze()
        
        if type_model == "gnn" :
            features, norm_adj = features.to(device), norm_adj.to(device)
            output = model(features, norm_adj)
            ## coo list 
        elif type_model == "gat_coo" or type_model == "gnn_coo" or type_model == "gin_coo":
            features =  features.to(device)
            edge_index, edge_attr = test_data.edge_index.to(device), test_data.edge_attr.to(device)
            output = model(features, edge_index, edge_attr)
        elif type_model == "gUnet_coo":
            features =  features.to(device)
            edge_index, edge_attr = test_data.edge_index.to(device), test_data.edge_attr.to(device)
            output = model(features, edge_index)
            
            
        if num_classes <=1:
            loss_test = F.binary_cross_entropy(output[idx_test], labels[idx_test])
        else:
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            
        y_pred = torch.argmax(output, dim=1)
        acc_test = metric.update(y_pred[idx_test], labels[idx_test])
        f1_test = f1score.update(y_pred[idx_test], labels[idx_test])
        average_epoch_loss += loss_test.item()
    print(
            "Test set results:",
            "loss= {:.4f}".format(average_epoch_loss/len(test_model)),
            "accuracy= {:.4f}".format(acc_test.compute()),
            "f1-score= {:.4f}".format(f1_test.compute()),
        )

    print(
            "y_true counts: {}".format(
                np.unique(labels.cpu().numpy(), return_counts=True)
            )
        )
    print(
            "y_pred_orig counts: {}".format(
                np.unique(y_pred.cpu().numpy(), return_counts=True)
            )
        )
    return y_pred


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


def _binary_f1_score(predictions, targets):
    # Convertire le probabilità in previsioni binarie (0 o 1)
    predicted = (predictions > 0.5).float()
    
    # Calcolare True Positives (TP), False Positives (FP), e False Negatives (FN)
    tp = (predicted * targets).sum().item()
    fp = ((predicted == 1) & (targets == 0)).sum().item()
    fn = ((predicted == 0) & (targets == 1)).sum().item()
    
    # Calcolare precision e recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    # Calcolare F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


def plotting(num_epochs, train_losses, train_accuracies, train_f1 ,test_losses, test_accuracies, test_f1,type_model):
    # Creare una figura con 3 sottografici
    plt.figure(figsize=(15, 10))

    # Plot della loss
    plt.subplot(3, 1, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()

    # Plot dell'accuracy
    plt.subplot(3, 1, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Test Accuracy')
    plt.legend()

    # Plot dell'F1-score
    plt.subplot(3, 1, 3)
    plt.plot(range(1, num_epochs+1), train_f1, label='Train F1')
    plt.plot(range(1, num_epochs+1), test_f1, label='Test F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Train vs Test F1 Score')
    plt.legend()

    # Salvare l'immagine
    plt.tight_layout()
    path = '/home/hl-turing/VSCodeProjects/Flavio/CARLA/carla/images_plotting/training_results_with_' + type_model +'.png'
    plt.savefig(path)  # Salva l'immagine
    plt.show()


# %%
