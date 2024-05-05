import time
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader, Dataset

# from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils.metric import accuracy

from carla.data.catalog.graph_catalog import AMLtoGraph
from carla.models.catalog.ANN_TF import AnnModel
from carla.models.catalog.ANN_TF import AnnModel as ann_tf
from carla.models.catalog.ANN_TORCH import AnnModel as ann_torch
from carla.models.catalog.GNN_TORCH import GCNSynthetic as gnn_torch
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
    hidden_size: int,
):
    if catalog_model.backend == "pytorch":
        # initialize dataframe
        df = AMLtoGraph(data_table=data)
        # create datagraph with type Data
        datagraph = df.construct_GraphData()

        # create adj matrix by COO
        adj_matrix = df.create_adj_matrix(datagraph).squeeze()
        # initialize the model
        model = gnn_torch(
            nfeat=len(datagraph.x[0]),
            nhid=hidden_size,  # da parametrizzare
            nout=hidden_size,  # da parametrizzare
            nclass=len(datagraph.y[0]),
            dropout=0.0,
        )

        # training gnn
        _training_gnn_torch(
            model=model,
            data_graph=datagraph,
            adj=adj_matrix,
            learn_rate=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            clip=clip,
        )

        return model

    else:
        raise ValueError("model backend not recognized")


def _training_gnn_torch(model, data_graph, adj, learn_rate, weight_decay, epochs, clip):
    # use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    norm_adj = normalize_adj(adj)
    features = torch.tensor(data_graph.x).squeeze()
    labels = torch.tensor(data_graph.y).squeeze()

    node_idx = [i for i in range(0, len(data_graph.y))]
    idx_train = torch.masked_select(torch.Tensor(node_idx), data_graph.train_mask)
    idx_test = torch.masked_select(torch.Tensor(node_idx), data_graph.test_mask)
    idx_train = idx_train.type(torch.int64)
    idx_test = idx_test.type(torch.int64)

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learn_rate, weight_decay=weight_decay
    )

    t_total = time.time()
    for epoch in range(epochs):
        t = time.time()
        model.train()

        optimizer.zero_grad()
        output = model(features, norm_adj)

        loss_train = model.loss(output[idx_train], labels[idx_train])

        y_pred = torch.argmax(output, dim=1)

        acc_train = accuracy(y_pred[idx_train], labels[idx_train])

        loss_train.backward()

        clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(loss_train.item()),
            "acc_train: {:.4f}".format(acc_train),
            "time: {:.4f}s".format(time.time() - t),
        )

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # torch.save(model.state_dict(), "../models/gcn_3layer_{}".format(args.dataset) + ".pt")
    y_pred = _test(model, features, labels, norm_adj, idx_test)
    print("y_true counts: {}".format(np.unique(labels.numpy(), return_counts=True)))
    print(
        "y_pred_orig counts: {}".format(np.unique(y_pred.numpy(), return_counts=True))
    )
    print("Finished training!")


def _test(model, features, labels, norm_adj, idx_test):
    # modalità di evaluation
    model.eval()
    output = model(features, norm_adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    y_pred = torch.argmax(output, dim=1)
    acc_test = accuracy(y_pred[idx_test], labels[idx_test])
    print(
        "Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test),
    )
    return y_pred


# %%
