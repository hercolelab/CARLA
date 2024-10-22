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
from carla.models.catalog.GAT_TORCH.model_gat import GAT
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import merge_default_parameters
from .edgepertuber import EdgePerturber

from .library.utils import get_degree_matrix, get_neighbourhood, normalize_adj, get_neighbourhood_v2


    

class CFEdgeExplainer(RecourseMethod):
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
        "verbose": True,
        "model_type": "gnn",
        "device": "cpu",
        
    }

    def __init__(self, mlmodel: GAT, data: DataCatalog, hyperparams: Dict = None):

        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super(CFEdgeExplainer, self).__init__(mlmodel=mlmodel)
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
        self.model_type = self._params["model_type"]

    def explain(
        self,
        graph: Data,
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
        # self.x = self.sub_feat

        # Save the sub adjacency matrix and compute the degree matrix
        # self.A_x = self.sub_adj
        # self.D_x = get_degree_matrix(self.A_x)



        best_cf_example = None
        self.best_loss = np.inf
        self.cf_model = EdgePerturber(num_classes=self.num_classes, 
                                      adj=graph.adj, 
                                      model=self.mlmodel, 
                                      nfeat=graph.edge_attr.shape[1], 
                                      edge_number=graph.edge_attr.shape[0]).to(self.device)
        
        # choose the optimizer
        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(
                self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum
            )
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)
        # num_cf_examples = 0

        for epoch in range(num_epochs):
            new_example = self.train(epoch, graph, self.mlmodel)
            '''
            print(
                f"new example: {new_example}",
                f"epoch: {epoch}"
            )
            '''
            
            if new_example is not None:
                best_cf_example = new_example

        if verbose and (best_cf_example != None):
            print(f"CF example for node_idx = {self.node_idx}\n")

        return best_cf_example

    def train(self, epoch: int, graph: Data, oracle):

        self.cf_optimizer.zero_grad()
        
        edge_ind = dense_to_sparse(graph.adj)
        
        differentiable_output = self.cf_model.forward(graph.x, edge_ind, graph.edge_attr) 
        model_out, V_pert, P_x = self.cf_model.forward_prediction(graph.x, edge_ind, graph.edge_attr) 
        
        y_pred_new_actual = torch.argmax(model_out, dim=1)

        loss, results, cf_adj = self.cf_model.loss(graph, differentiable_output, y_pred_new_actual[graph.new_idx])
        loss.backward()

        clip_grad_norm_(self.cf_model.parameters(), 1.0)
        self.cf_optimizer.step()

        counterfactual = None
        '''
        losses = {"feature":results["loss_feat"], "prediction": results["loss_pred"], "graph":results["loss_graph_dist"]}
            # embedding_repr = torch.mean(oracle.get_embedding_repr(V_pert, cf_adj), dim=0)
        print(losses)
        '''
        if y_pred_new_actual[graph.new_idx] == graph.targets[graph.new_idx] and results["loss_total"] < self.best_loss:
            
            losses = {"feature":results["loss_feat"], "prediction": results["loss_pred"], "graph":results["loss_graph_dist"]}
            # embedding_repr = torch.mean(oracle.get_embedding_repr(V_pert, cf_adj), dim=0)
            # print(losses)

            counterfactual = Data(x=graph.x, 
                                  adj=graph.adj, 
                                  y=y_pred_new_actual,
                                  sub_index=graph.new_idx,
                                  loss=losses,
                                  edge_attr=V_pert,
                                  edge_index= graph.edge_index)
                                  #x_projection=embedding_repr

            self.best_loss = results["loss_total"]

        return counterfactual

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
        labels = torch.tensor(data_graph.y).squeeze().to(self.device)
        attributes = torch.tensor(data_graph.edge_attr).squeeze().to(self.device)
        # idx_train  # non dovrebbe servire

        node_idx = [i for i in range(0, len(data_graph.y))]
        idx_test = torch.masked_select(torch.Tensor(node_idx), data_graph.test_mask)
        idx_test = idx_test.type(torch.int64)

        norm_edge_index = dense_to_sparse(adj)  # Needed for pytorch-geo functions
        norm_adj = normalize_adj(adj).to(
            self.device
        )  # According to reparam trick from GCN paper

        # output of GCN Syntethic model
        y_pred_orig = self.mlmodel.predict_gnn_coo(features, data_graph.edge_index, attributes)
        targets =  (1 + y_pred_orig)% self.num_classes
        # y_pred_orig = torch.argmax(output, dim=1)
        # print(torch.max(y_pred_orig))
        # Get CF examples in test set
        df_cf_examples = pd.DataFrame()
        df_factual_examples = pd.DataFrame()
        num_cf = 0
        numerator_sparsity = 0.0
        numerator_fidelity = 0.0
        num_graphs=0
        total_nodes = 0
        # start = time.time()
        for i in idx_test[:]:
            # funzione get_neighbourhood da vedere su utils.py
            # vedere se modificare get_neighbourhood per norm_edge_index
            sub_adj, sub_feat, sub_labels, sub_edge_attr, node_dict, sub_edge_index = get_neighbourhood_v2(
                int(i), norm_edge_index, attributes, self.n_layers + 1, features, labels
            )
            new_idx = node_dict[int(i)]
            sub_index = list(node_dict.keys())

            if len(sub_adj.shape) < 1 or len(sub_adj.shape) > 1500 or all(
                [True if i == 0 else False for i in sub_adj.shape]
            ):
                continue

            # self.sub_adj = sub_adj
            # self.sub_feat = sub_feat.to(self.device)
            # self.sub_labels = sub_labels
            # self.y_pred_orig = y_pred_orig[i]
            sub_y = y_pred_orig[sub_index]
            sub_targets = targets[sub_index]
            
            factual = Data(x = sub_feat, 
                           adj = sub_adj,
                           y = sub_y,
                           y_ground = sub_labels,
                           new_idx = new_idx,
                           targets = sub_targets,
                           node_dict = node_dict,
                           edge_attr= sub_edge_attr,
                           edge_index= sub_edge_index
                           )
            
            total_nodes += int(factual.y.shape[0])
            factual = factual.to(self.device)
            # Instantiate CF model class, load weights from original model
            # The syntentic model load the weights from the model to explain then freeze them
            # and train the perturbation matrix to change the prediction
            

            # If cuda is avaialble move the computation on GPU
            if self.device == "cuda":
                # self.mlmodel.cuda()
                # self.cf_model.cuda()
                adj = adj.cuda()
                norm_adj = norm_adj.cuda()
                features = features.cuda()
                labels = labels.cuda()
                # idx_train = idx_train.cuda()
                idx_test = idx_test.cuda()
                
            # print(f"factual: {factual}")
            # node to explain i, node_dict maps the old node_idx into the new node_idx
            # because of the subgraph
            cf_example = self.explain(
                graph = factual,
                node_idx=i,
                cf_optimizer="Adadelta",
                new_idx=new_idx,
                lr=self.lr,
                n_momentum=self.n_momentum,
                num_epochs=self.num_epochs,
                verbose=self.verbose,
            )
            if cf_example is None:
                num_graphs+=1
                # numerator_fidelity_prob+= single_fid_prob(self.mlmodel.predict_proba_gnn(factual.x, factual.adj), predict_proba_cf)
                continue
            else:
                num_graphs+=1
                num_cf+=1
                numerator_fidelity+=fidelity(factual, cf_example)
                numerator_sparsity+=edge_sparsity(factual, cf_example)
                
                

                # break
            # one_cf_example = cf_example[0]
        '''
            try:
                pd_factual= df_test.reconstruct_Tabular(factual, factual.adj, node_dict, int(i))
                pd_cf = df_test.reconstruct_Tabular(cf_example, factual.adj, node_dict, int(i))
                #print(pd_factual)
                #print(pd_cf)
                df_cf_examples = pd.concat([df_cf_examples, pd_cf], ignore_index=True)
                df_factual_examples = pd.concat([df_factual_examples, pd_factual], ignore_index=True)
                
            except AttributeError:

                UserWarning(f"Dataset {factuals} cannot converted into a csv file!")
            
        path_cf: str = "test/saved_cf/"

        file_name_cf: str = "results_cf_"+ self.model_type

        if not os.path.exists(path_cf):
            try:
                os.makedirs(path_cf)
            except Exception:
                raise Exception

        df_cf_examples.to_csv(path_cf + f"{file_name_cf}.csv")
        
        path_factual: str = "test/saved_factual/"

        file_name_factual: str = "results_factual_"+ self.model_type

        if not os.path.exists(path_factual):
            try:
                os.makedirs(path_factual)
            except Exception:
                raise Exception
        

        df_factual_examples.to_csv(path_factual + f"{file_name_factual}.csv")
            # cf_example = [ [cf_example0], [cf_example1], [cf_example2], etc...]
        '''
            # da trasformare cf_example (DataGraph) in DataFrame (utilizzando forse diz_conn)
        # num_graphs = int(idx_test.shape[0])
        validity_acc = num_cf/num_graphs
        sparsity_acc = numerator_sparsity/num_graphs
        fidelity_acc = numerator_fidelity/num_graphs
        
        #print('printo total_nodes:')
        #print(total_nodes)
        #print('printo num_graphs:')
        #print(num_graphs)   
        return df_cf_examples, num_cf, validity_acc, sparsity_acc, fidelity_acc


def fidelity(factual: Data, counterfactual: Data):
    print('sono entrato')

    #TODO: Check Fidelity
    factual_index = factual.new_idx
    cfactual_index = counterfactual.sub_index
    phi_G = factual.y[factual_index]
    y = factual.y_ground[factual_index]
    phi_G_i = counterfactual.y[cfactual_index]
    
    prediction_fidelity = 1 if phi_G == y else 0
    
    counterfactual_fidelity = 1 if phi_G_i == y else 0
    
    result = prediction_fidelity - counterfactual_fidelity
    
    return result

def edge_sparsity(factual: Data, counterfactual: Data):

    modified_edges = (torch.sum((factual.edge_attr != counterfactual.edge_attr))//2)
    return ((modified_edges) / (factual.edge_attr.numel()//2)).item()

def node_sparsity(factual: Data, counterfactual: Data):

    modified_attributes =  torch.sum(factual.x  != counterfactual.x)
    return ((modified_attributes) / (factual.x.numel())).item()