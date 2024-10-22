import os
from typing import Dict, List, Tuple, Union


import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from torch import nn
from abc import ABC, abstractmethod
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor

from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Dataset

from carla.data.catalog.graph_catalog import AMLtoGraph, PlanetoidGraph

# from carla.data.api import Data
from carla.data.catalog.online_catalog import DataCatalog

# from carla.models.api import MLModel
# commento
from carla.models.catalog.GIN_TORCH.model_gin import GIN
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import merge_default_parameters

from .library.gin_perturb import GINSyntheticPerturb
from .library.utils import get_degree_matrix, get_neighbourhood, normalize_adj


class CFGINExplainer(RecourseMethod):
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
        "hid_attr_list": [31, 31, 31, 31, 31, 31, 31, 31],
        "hid_list": [50, 50, 50],
        "dropout": 0.0,
        "alpha": 0.2,
        "beta": 0.5,
        "nheads": 8,
        "num_classes": 2,
        "n_layers": 3,
        "n_momentum": 0,
        "verbose": False,
        "device": "cpu",
    }

    def __init__(self, mlmodel: GIN, data: DataCatalog, hyperparams: Dict = None):

        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super(CFGINExplainer, self).__init__(mlmodel=mlmodel)
        self.data = data
        self.mlmodel = mlmodel
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)

        self.cf_optimizer = self._params["cf_optimizer"]
        self.lr = self._params["lr"]
        self.num_epochs = self._params["num_epochs"]
        self.hid_attr_list = self._params["hid_attr_list"]
        self.hid_list = self._params["hid_list"]
        self.dropout = self._params["dropout"]
        self.beta = self._params["beta"]
        self.alpha = self._params["alpha"]
        self.num_classes = self._params["num_classes"]
        self.n_layers = self._params["n_layers"]
        self.n_momentum = self._params["n_momentum"]
        self.verbose = self._params["verbose"]
        self.device = self._params["device"]
        self.nheads = self._params["nheads"]

    def explain(
        self,
        cf_optimizer: str,
        node_idx: int,
        new_idx: int,
        lr: float,
        n_momentum: float,
        num_epochs: int,
        verbose: bool = False,
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
                if (np.sum(new_example[2]!=new_example[3]) != 0):
                    
                    best_cf_example.append(new_example)
                    best_loss = loss_total
                    num_cf_examples += 1

        if verbose:
            print(f"{num_cf_examples} CF examples for node_idx = {self.node_idx}\n")

        return best_cf_example

    def train(self, epoch: int, verbose: bool = False) -> Tuple[List, float]:
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
        output = self.cf_model.forward(self.x, self.A_x.to(self.device))
        output_actual, self.P = self.cf_model.forward_prediction(self.x)

        # Need to use new_idx from now on since sub_adj is reindexed
        y_pred_new = torch.argmax(output.squeeze()[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual.squeeze()[self.new_idx])

        # compute the loss function and perform optim step
        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(
            output.squeeze()[self.new_idx], self.y_pred_orig, y_pred_new_actual
        )
        loss_total.backward()
        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()

        if verbose:

            print(
                f"Node idx: {self.node_idx}",
                f"New idx: {self.new_idx}",
                "Epoch: {:04d}".format(epoch + 1),
                "loss: {}".format(loss_total.item()),
                "pred loss: {:.4f}".format(loss_pred.item()),
                "graph loss: {:.4f}".format(loss_graph_dist.item()),
            )

            print(
                f"Output: {output.squeeze()[self.new_idx].data}\n",
                f"Output nondiff: {output_actual.squeeze()[self.new_idx].data}\n",
                f"orig pred: {self.y_pred_orig}, new pred: {y_pred_new}, new pred nondiff: {y_pred_new_actual}\n",
            )

        cf_stats = []

        # if a cf example has been found then add it to the cf_stats list
        if y_pred_new_actual != self.y_pred_orig:
            
            cf_stats = [
                self.node_idx.item(),
                self.new_idx,
                cf_adj.detach().cpu().numpy(),
                self.sub_adj.detach().numpy(),
                self.y_pred_orig.item(),
                y_pred_new.item(),
                y_pred_new_actual.item(),
                self.sub_labels[self.new_idx].cpu().numpy(),
                self.sub_adj.shape[0],
                loss_total.item(),
                loss_pred.item(),
                loss_graph_dist.item(),
            ]

        return (cf_stats, loss_total.item())

    def get_counterfactuals(self, factuals: Union[str, pd.DataFrame]):

        df_test_plat = None
        plat = ["Cora", "CiteSeer", "PubMed"]
        if isinstance(factuals, str) and factuals in plat:
            df_test_plat = PlanetoidGraph(factuals)
        else:
            # Construct df_test by factuals
            df_test_AML = AMLtoGraph(factuals)

        if isinstance(df_test_plat, PlanetoidGraph):
            df_test = df_test_plat
            data_graph = df_test.getDataGraph()
        elif isinstance(df_test_AML, AMLtoGraph):
            df_test = df_test_AML
            data_graph = df_test.construct_GraphData()

        adj = df_test.create_adj_matrix(
            data_graph
        ).squeeze()  # Does not include self loops
        features = torch.tensor(data_graph.x).squeeze().to(self.device)
        labels = torch.tensor(data_graph.y).squeeze().long().to(self.device)
        # idx_train  # non dovrebbe servire

        node_idx = [i for i in range(0, len(data_graph.y))]
        idx_test = torch.masked_select(torch.Tensor(node_idx), data_graph.test_mask)
        idx_test = idx_test.type(torch.int64)

        norm_edge_index = dense_to_sparse(adj)  # Needed for pytorch-geo functions
        norm_adj = normalize_adj(adj).to(
            self.device
        )  # According to reparam trick from GCN paper

        # output of GCN Syntethic model
        output = self.mlmodel.predict_proba_gnn(features, norm_adj).squeeze()
        y_pred_orig = torch.argmax(output, dim=1)
        # print(torch.max(y_pred_orig))
        # Get CF examples in test set
        test_cf_examples = []
        test_cf_sts = []
        
        df_cf_examples = pd.DataFrame()
        num_cf = 0
        numerator_sparsity = 0.0
        numerator_fidelity = 0.0
        num_graphs=0
        total_nodes = 0
        
        # start = time.time()
        for i in idx_test[:]:
            # funzione get_neighbourhood da vedere su utils.py
            # vedere se modificare get_neighbourhood per norm_edge_index
            sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(
                int(i), norm_edge_index, self.n_layers + 1, features, labels
            )
            new_idx = node_dict[int(i)]
            sub_index = list(node_dict.keys())
            
            if len(sub_adj.shape) < 1 or len(sub_adj.shape) > 1500 or all(
                [True if i == 0 else False for i in sub_adj.shape]
            ):
                continue

            self.sub_adj = sub_adj
            self.sub_feat = sub_feat.to(self.device)
            self.sub_labels = sub_labels
            self.y_pred_orig = y_pred_orig[i]
            
            factual= Data( x = self.sub_feat,
                          adj = self.sub_adj,
                          y = y_pred_orig[sub_index], # sub_y
                          y_ground = sub_labels,
                          new_idx = new_idx,
                          node_dict = node_dict               
            )

            # Instantiate CF model class, load weights from original model
            # The syntentic model load the weights from the model to explain then freeze them
            # and train the perturbation matrix to change the prediction
            self.cf_model = GINSyntheticPerturb(
                nfeat=self.sub_feat.shape[1],
                hid_list_gin=self.hid_attr_list,
                hid_list_conv=self.hid_list,
                nclass=self.num_classes,
                adj=self.sub_adj,
                dropout=self.dropout,
                beta=self.beta,
                alpha=self.alpha,
                nheads=self.nheads,
            )

            self.cf_model.load_state_dict(
                self.mlmodel.raw_model.state_dict(), strict=False
            )

            # Freeze weights from original model in cf_model
            for name, param in self.cf_model.named_parameters():
                if (
                    name.endswith("weight")
                    or name.endswith("bias")
                    or name.endswith("bias")
                    or ("attention" in name)
                    or ("out_att" in name)
                ):
                    param.requires_grad = False

            if self.verbose:

                # Check the gradient for each parameter
                # .named_parameters(): DA CAPIRE
                for name, param in self.mlmodel.raw_model.named_parameters():
                    print("orig model requires_grad: ", name, param.requires_grad)
                for name, param in self.cf_model.named_parameters():
                    print("cf model requires_grad: ", name, param.requires_grad)

                print(
                    f"y_true counts: {np.unique(labels.cpu().numpy(), return_counts=True)}"
                )
                print(
                    f"y_pred_orig counts: {np.unique(y_pred_orig.detach().cpu().numpy(), return_counts=True)}"
                )  # Confirm model is actually doing something

                # Check that original model gives same prediction on full graph and subgraph
                with torch.no_grad():
                    print(f"Output original model, full adj: {y_pred_orig[i]}")
                    print(
                        f"Output original model, sub adj: {self.mlmodel.predict_proba_gnn(sub_feat, normalize_adj(sub_adj).to(self.device)).squeeze()[new_idx]}"
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
                
            num_graphs+=1

            # node to explain i, node_dict maps the old node_idx into the new node_idx
            # because of the subgraph
            cf_example = self.explain(
                node_idx=i,
                cf_optimizer="Adadelta",
                new_idx=new_idx,
                lr=self.lr,
                n_momentum=self.n_momentum,
                num_epochs=self.num_epochs,
                verbose=self.verbose,
            )
            
            if len(cf_example) == 0:
                continue
            else:
                num_cf+=1
                one_cf_example = cf_example[-1]
                
                counterfactual= Data(x = self.sub_feat,
                                     adj =  one_cf_example[2],
                                     y = one_cf_example[6], 
                                     new_idx = one_cf_example[1]              
                )
                numerator_fidelity+=fidelity(factual, counterfactual)
                numerator_sparsity+=sparsity(factual, counterfactual)
            '''
            try:
                pd_cf = df_test.reconstruct_Tabular(factuals, cf_adj, node_dict)
                test_cf_examples.append(pd_cf)

            except AttributeError:

                UserWarning(f"Dataset {factuals} cannot converted into a csv file!")

        df_cf_example = pd.DataFrame(
            test_cf_sts,
            columns=[
                "node_idx",
                "new_idx",
                "cf_adj",
                "sub_adj",
                "y_pred_orig",
                "y_pred_new",
                "y_pred_new_actual",
                "sub_labels",
                "sub_adj",
                "loss_total",
                "loss_pred",
                "loss_graph_dist",
            ],
        )

        path: str = "test/saved_sts_cf/"

        file_name: str = "results_1"

        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception:
                raise Exception

        df_cf_example.to_csv(path + f"{file_name}.csv")
        '''
        validity_acc = num_cf/num_graphs
        sparsity_acc = numerator_sparsity/num_graphs
        fidelity_acc = numerator_fidelity/num_graphs
        
        #print('printo total_nodes:')
        #print(total_nodes)
        #print('printo num_graphs:')
        #print(num_graphs)   
        return df_cf_examples, num_cf, validity_acc, sparsity_acc, fidelity_acc


def fidelity(factual: Data, counterfactual: Data):
    # print('sono entrato')

    #TODO: Check Fidelity
    factual_index = factual.new_idx
    # cfactual_index = counterfactual.new_idx
    phi_G = factual.y[factual_index]
    y = factual.y_ground[factual_index]
    phi_G_i = counterfactual.y
    
    prediction_fidelity = 1 if phi_G == y else 0
    
    counterfactual_fidelity = 1 if phi_G_i == y else 0
    
    result = prediction_fidelity - counterfactual_fidelity
    
    return result

def sparsity(factual: Data, counterfactual: Data):

    modified_edges = (torch.sum((factual.adj != torch.tensor(counterfactual.adj)))//2)
    return ((modified_edges) / (factual.adj.numel()//2)).item()