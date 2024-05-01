from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch_geometric.utils import dense_to_sparse

# from carla.data.api import Data
from carla.data.catalog.online_catalog import DataCatalog

# from carla.models.api import MLModel
from carla.models.catalog.GNN_TORCH.model_gnn import GCNSynthetic
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import merge_default_parameters

from .library.data_conversion import (
    add_identifier,
    construct_GraphData,
    create_adj_matrix,
)
from .library.gcn_perturb import GCNSyntheticPerturb
from .library.utils import get_degree_matrix, get_neighbourhood, normalize_adj


class CFExplainer(RecourseMethod):
    """
    Code adapted from: CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks
    Link ArXiv: https://arxiv.org/abs/2102.03322

    CF Explainer class, returns counterfactual subgraph

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    data: carla.data.Data
        Dataset to perform on
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

    """

    _DEFAULT_HYPERPARAMS = {
        "cf_optimizer": "Adadelta",
        "lr": 0.05,
        "num_epochs": 100,
        "n_hid": 3,
        "dropout": 0.0,
        "beta": 0.5,
        "num_classes": 2,
        "n_layers": 3,
        "n_momentum": 0,
        "verbose": True,
        "device": "cpu",
    }

    def __init__(
        self, mlmodel: GCNSynthetic, data: DataCatalog, hyperparams: Dict = None
    ):

        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super(CFExplainer, self).__init__(mlmodel=mlmodel)
        self.data = data
        self.mlmodel = mlmodel
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)

        self.cf_optimizer = self._params["cf_optimizer"]
        self.lr = self._params["lr"]
        self.num_epochs = self._params["num_epochs"]
        self.n_hid = self._params["n_hid"]
        self.dropout = self._params["n_search_samples"]
        self.beta = self._params["beta"]
        self.num_classes = self._params["num_classes"]
        self.n_layers = self._params["n_layers"]
        self.n_momentum = self._params["n_momentum"]
        self.verbose = self._params["verbose"]
        self.device = self._params["cpu"]

    def explain(
        self,
        cf_optimizer: str,
        node_idx: int,
        new_idx: int,
        lr: float,
        n_momentum: float,
        num_epochs: int,
        verbose: bool = True,
    ):
        r"""Explain a factual instance:

        Args:
                cf_optimizer (str): choose the optimizer between (Adadelta, SDG)
                node_idx (bool): if true shows more infos about the training phase
                new_idx (int)
                lr (float)
                n_momentum (float): only applied with SDG, it is the Nestor Momentum
                num_epochs (int): Epoch numbers
                verbose (bool)

        """
        # Save new and old index
        self.node_idx = torch.tensor(node_idx)
        self.new_idx = new_idx

        # Save the nodes features (just the ones in the subgraph)
        self.x = self.sub_feat

        # Save the sub adjacency matrix and compute the degree matrix
        self.A_x = self.sub_adj
        self.D_x = get_degree_matrix(self.A_x)

        # choose the optimizer
        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(
                self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum
            )
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)

        best_cf_example = []
        best_loss = np.inf
        num_cf_examples = 0

        for epoch in range(num_epochs):
            new_example, loss_total = self.train(epoch)

            if new_example != [] and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1

        if verbose:
            print(f"{num_cf_examples} CF examples for node_idx = {self.node_idx}\n")

        return best_cf_example

    def train(self, epoch: int, verbose: bool = True) -> Tuple[List, float]:
        r"""Train the counterfactual model:

        Args:
                epoch (int): The epoch number
                verbose (bool): if true shows more infos about the training phase

        """

        # Set the cf model in training mode
        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        output = self.cf_model.forward(self.x, self.A_x)
        output_actual, self.P = self.cf_model.forward_prediction(self.x)

        # Need to use new_idx from now on since sub_adj is reindexed
        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])

        # compute the loss function and perform optim step
        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(
            output[self.new_idx], self.y_pred_orig, y_pred_new_actual
        )
        loss_total.backward()
        clip_grad_norm(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        if verbose:

            print(
                f"Node idx: {self.node_idx}",
                f"New idx: {self.new_idx}",
                "Epoch: {:04d}".format(epoch + 1),
                "loss: {:.4f}".format(loss_total.item()),
                "pred loss: {:.4f}".format(loss_pred.item()),
                "graph loss: {:.4f}".format(loss_graph_dist.item()),
            )

            print(
                f"Output: {output[self.new_idx].data}\n",
                f"Output nondiff: {output_actual[self.new_idx].data}\n",
                f"orig pred: {self.y_pred_orig}, new pred: {y_pred_new}, new pred nondiff: {y_pred_new_actual}\n",
            )

        cf_stats = []

        # if a cf example has been found then add it to the cf_stats list
        if y_pred_new_actual != self.y_pred_orig:

            cf_stats = [
                self.node_idx.item(),
                self.new_idx,
                cf_adj.detach().numpy(),
                self.sub_adj.detach().numpy(),
                self.y_pred_orig.item(),
                y_pred_new.item(),
                y_pred_new_actual.item(),
                self.sub_labels[self.new_idx].numpy(),
                self.sub_adj.shape[0],
                loss_total.item(),
                loss_pred.item(),
                loss_graph_dist.item(),
            ]

        return (cf_stats, loss_total.item())

    def get_counterfactuals(self, factuals: pd.DataFrame):
        # factuals = predict_negative_instances(model, data.df) DA VEDERE
        # get preprocessed data of test set
        df_test = self._mlmodel.data.df_test
        x_test = df_test[list(set(df_test.columns) - {self.data.target})]
        y_test = df_test[self.data.target]
        x_test = self._mlmodel.get_ordered_features(x_test)

        # get graph data
        x_testID = add_identifier(x_test)
        y_testID = add_identifier(y_test)
        merged_df = pd.merge(x_testID, y_testID, on="ID")

        # get a list of features and labels
        list_feat = x_test.columns.tolist()
        list_lab = y_test.columns.tolist()
        conn = ""  # da definire come parametro
        # diz_conn da vedere
        data_graph, values_edges, diz_conn = construct_GraphData(
            merged_df, list_feat, list_lab, conn
        )

        # define
        adj = create_adj_matrix(
            data_graph, values_edges
        ).squeeze()  # Does not include self loops
        features = torch.tensor(data_graph.x).squeeze()
        labels = torch.tensor([list(elem).index(1) for elem in data_graph.y]).squeeze()
        # idx_train  # non dovrebbe servire

        node_idx = [i for i in range(0, len(data_graph.y))]
        idx_test = torch.masked_select(torch.Tensor(node_idx), data_graph.test_mask)
        idx_test = idx_test.type(torch.int64)

        norm_edge_index = dense_to_sparse(adj)  # Needed for pytorch-geo functions
        norm_adj = normalize_adj(adj)  # According to reparam trick from GCN paper

        # output of GCN model
        output = self.mlmodel.predict_proba(features, norm_adj)
        y_pred_orig = torch.argmax(output, dim=1)

        # Get CF examples in test set
        test_cf_examples = []
        # start = time.time()
        for i in idx_test[:]:
            # funzione get_neighbourhood da vedere su utils.py
            sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(
                int(i), norm_edge_index, self.n_layers + 1, features, labels
            )
            new_idx = node_dict[int(i)]

            self.sub_adj = sub_adj
            self.sub_feat = sub_feat
            self.sub_labels = sub_labels
            self.y_pred_orig = y_pred_orig[new_idx]

            # Instantiate CF model class, load weights from original model
            # The syntentic model load the weights from the model to explain then freeze them
            # and train the perturbation matrix to change the prediction
            self.cf_model = GCNSyntheticPerturb(
                nfeat=self.sub_feat.shape[1],
                nhid=self.n_hid,
                nout=self.n_hid,
                nclass=self.num_classes,
                adj=self.sub_adj,
                dropout=self.dropout,
                beta=self.beta,
            )

            self.cf_model.load_state_dict(
                self.mlmodel.raw_model.state_dict(), strict=False
            )

            # Freeze weights from original model in cf_model
            for name, param in self.cf_model.named_parameters():
                if name.endswith("weight") or name.endswith("bias"):
                    param.requires_grad = False

            if self.verbose:

                # Check the gradient for each parameter
                # .named_parameters(): DA CAPIRE
                for name, param in self.mlmodel.raw_model.named_parameters():
                    print("orig model requires_grad: ", name, param.requires_grad)
                for name, param in self.cf_model.named_parameters():
                    print("cf model requires_grad: ", name, param.requires_grad)

                print(f"y_true counts: {np.unique(labels.numpy(), return_counts=True)}")
                print(
                    f"y_pred_orig counts: {np.unique(y_pred_orig.numpy(), return_counts=True)}"
                )  # Confirm model is actually doing something

                # Check that original model gives same prediction on full graph and subgraph
                with torch.no_grad():
                    print(f"Output original model, full adj: {output[i]}")
                    print(
                        f"Output original model, sub adj: {self.mlmodel.predict_proba(sub_feat, normalize_adj(sub_adj))[new_idx]}"
                    )

            # If cuda is avaialble move the computation on GPU
            if self.device == "cuda":
                # self.mlmodel.cuda()
                self.cf_model.cuda()
                adj = adj.cuda()
                norm_adj = norm_adj.cuda()
                features = features.cuda()
                labels = labels.cuda()
                # idx_train = idx_train.cuda()
                idx_test = idx_test.cuda()

            # node to explain i, node_dict maps the old node_idx into the new node_idx
            # because of the subgraph
            cf_example = self.explain(
                node_idx=i,
                cf_optimizer=self.cf_optimizer,
                new_idx=new_idx,
                lr=self.lr,
                n_momentum=self.n_momentum,
                num_epochs=self.num_epochs,
                verbose=self.verbose,
            )

            # da trasformare cf_example (DataGraph) in DataFrame (utilizzando forse diz_conn)

            test_cf_examples.append(cf_example)

        return test_cf_examples
