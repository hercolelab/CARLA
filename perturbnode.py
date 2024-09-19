# NODE EXPLAINER PY

from functools import partial
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import numpy as np
from src.abstract.explainer import Explainer
from src.utils.explainer import get_node_explainer
from ....abstract.wrapper import Wrapper
from torch_geometric.utils import to_dense_adj
from ...utils.utils import TimeOutException, get_neighbourhood, normalize_adj
from ...evaluation.evaluate import compute_metrics
from src.datasets.dataset import DataInfo
from torch.nn import Module
import torch.multiprocessing as mp
import main_wandb
import concurrent.futures
import copy
    

class NodesExplainerWrapper(Wrapper):
    
    queue = None
    results_queue = None
    
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)

        torch.manual_seed(cfg.general.seed)
        torch.cuda.manual_seed(cfg.general.seed)
        torch.cuda.manual_seed_all(cfg.general.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        np.random.seed(cfg.general.seed)

    def check_graphs(self, sub_adj) -> bool:
        """
        Check if the input adjacency matrix represents an empty or trivial graph.

        A graph is considered empty if there are no edges between any nodes. A graph
        is trivial if it consists of a single node without any self-loops. This function
        checks for these conditions by inspecting the shape and the number of elements
        in the adjacency matrix.

        Parameters:
        - sub_adj (Tensor): The adjacency matrix of a graph.

        Returns:
        - bool: True if the graph is empty or trivial, False otherwise.
        """
        return len(sub_adj.shape) == 1 or sub_adj.numel() <= 1


    def explain(self, data: Dataset, datainfo: DataInfo, explainer: str, oracle: Module)->dict:
        """
        Explain the node predictions of a graph dataset.

        This method applies the provided explainer to each test instance in the dataset to generate
        counterfactual explanations. It computes and saves the explanation metrics.

        Parameters:
        - data (Dataset): The graph dataset containing features, edges, and test masks.
        - datainfo (DataInfo): Object containing dataset metadata and other relevant information.
        - explainer (Explainer): The explainer algorithm to generate explanations.

        Returns:
        - dict: A dictionary containing the results of the explanation process and metrics.
        """
        print(f"{explainer=}")
        device = "cuda" if torch.cuda.is_available() and self.cfg.device == "cuda" else "cpu"
        
        self.current_explainer_name = explainer
        self.current_datainfo = datainfo
        adj = to_dense_adj(data.edge_index).squeeze()    
        norm_adj = normalize_adj(adj).to(device)   

        x = data.x.to(device)
        adj = norm_adj.to(device)

        output = oracle(x, adj).detach()
        y_pred_orig = torch.argmax(output, dim=1)
        targets = (1 + y_pred_orig)%datainfo.num_classes
        metric_list = []

        embedding_repr = oracle.get_embedding_repr(x, adj).detach()
        datainfo.distribution_mean_projection = torch.mean(embedding_repr, dim=0).cpu()
        covariance =  np.cov(embedding_repr.detach().cpu().numpy(), rowvar=False)
        datainfo.inv_covariance_matrix = torch.from_numpy(np.linalg.inv(covariance)).float().cpu()

        mp.set_start_method('spawn', force=True)
        oracle.share_memory()
        
        try:
        
            with mp.Manager() as manager:
                
                queue = manager.Queue(self.cfg.workers)
                results_queue = manager.Queue()
                worker_func = partial(self.worker_process, queue, results_queue)

                workers = []
                for _ in range(self.cfg.workers):
                    p = mp.Process(target=worker_func)
                    p.start()
                    workers.append(p)
                    
                
                pid = 0
                
                for mask_index in tqdm(data.test_mask):
                    pid+=1

                    sub_adj, sub_feat, sub_labels, node_dict = get_neighbourhood(node_idx=int(mask_index), 
                                                                                    edge_index=data.edge_index.cpu(),
                                                                                    n_hops=self.cfg.model.n_layers+1, 
                                                                                    features=data.x.cpu(), 
                                                                                    labels=data.y.cpu())
                            
                    if self.check_graphs(sub_adj=sub_adj):
                        continue

                    new_idx = node_dict[int(mask_index)]
                    sub_index = list(node_dict.keys())
                    
                    # Pass everything on cpu because of Queue
                    args = (sub_index, 
                            new_idx, 
                            y_pred_orig.cpu(), 
                            targets.cpu(), 
                            device, 
                            oracle.cpu(), 
                            sub_labels.cpu(), 
                            sub_feat.cpu(), 
                            explainer, 
                            datainfo, 
                            node_dict, 
                            self.cfg, 
                            sub_adj.cpu(), 
                            pid)
                    
                    queue.put(args)
                
                # Signal the end of tasks
                for _ in range(self.cfg.workers):
                    queue.put(None)
                    
                for worker in workers:
                    worker.join()

                # Collect the results
                while not results_queue.empty():
                    result = results_queue.get()
                    if result is not None:
                        metric_list.append(result)
                
                dataframe = pd.DataFrame.from_dict(metric_list)
                        
                main_wandb.log(dataframe.mean().to_dict())
            
        except Exception as e:
            
            print(f"{e}")


    @staticmethod
    def worker_process(queue, results_queue):
        while True:
            args = queue.get()
            if args is None:
                break
            result = process(*args)
            results_queue.put(result)

def process(sub_index, new_idx, y_pred_orig, targets, device, oracle, sub_labels, sub_feat, explainer_name, datainfo, node_dict, cfg, sub_adj, pid):
    
    import copy 
    model = copy.deepcopy(oracle).to(device)
    explainer = get_node_explainer(explainer_name)
    explainer = explainer(cfg)
    explainer.num_classes = datainfo.num_classes
    sub_y = y_pred_orig[sub_index]
    sub_targets = targets[sub_index]
    sub_feat = sub_feat.to(device)
    sub_adj = sub_adj.to(device)
    repr = model.get_embedding_repr(sub_feat, sub_adj)
    embedding_repr = torch.mean(repr, dim=0).to(device) 
    factual = Data(x=sub_feat, adj=sub_adj, y=sub_y, y_ground=sub_labels, new_idx=new_idx, targets=sub_targets, node_dict=node_dict, x_projection=embedding_repr)  
    factual = factual.to(device)
    counterfactual = explainer.explain_node(graph=factual, oracle=model.to(device))
    metrics = compute_metrics(factual, counterfactual, device=device, data_info=datainfo)
    print(f"Terminated {pid=}")
    return metrics


#%%
# CF_EXPLAINER.PY

# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py
import time
from torch_geometric.data import Data
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from src.node_level_explainer.oracles.perturber.pertuber import NodePerturber
from omegaconf import DictConfig
from ...utils.utils import print_info, get_optimizer
from ....abstract.explainer import Explainer  
from tqdm import tqdm 

class CFExplainerFeatures(Explainer):
    """
    CF Explainer class, returns counterfactual subgraph
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)

        self.set_reproducibility()


    def explain_node(self, graph: Data, oracle):

        self.best_loss = np.inf
        self.cf_model = NodePerturber(cfg=self.cfg, 
                                      num_classes=self.num_classes, 
                                      adj=graph.adj, 
                                      model=oracle, 
                                      nfeat=graph.x.shape[1], 
                                      nodes_number=graph.x.shape[0]).to(self.device)
        
        self.optimizer = get_optimizer(self.cfg, self.cf_model)
        best_cf_example = None
        
        start = time.time()

        for epoch in range(self.cfg.optimizer.num_epochs):
            
            new_sample = self.train(epoch, graph, oracle)
            
            if time.time() - start > self.cfg.timeout:
                
                return best_cf_example

            if new_sample is not None:
                best_cf_example = new_sample

        return best_cf_example
    

    def train(self, epoch: int, graph: Data, oracle):

        self.optimizer.zero_grad()

        differentiable_output = self.cf_model.forward(graph.x, graph.adj) 
        model_out, V_pert, P_x = self.cf_model.forward_prediction(graph.x) 
        
        y_pred_new_actual = torch.argmax(model_out, dim=1)

        loss, results, cf_adj = self.cf_model.loss(graph, differentiable_output, y_pred_new_actual[graph.new_idx])
        loss.backward()

        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.optimizer.step()

        counterfactual = None

        if y_pred_new_actual[graph.new_idx] == graph.targets[graph.new_idx] and results["loss_total"] < self.best_loss:
            
            losses = {"feature":results["loss_feat"], "prediction": results["loss_pred"], "graph":results["loss_graph_dist"]}
            embedding_repr = torch.mean(oracle.get_embedding_repr(V_pert, cf_adj), dim=0)

            counterfactual = Data(x=V_pert, 
                                  adj=graph.adj, 
                                  y=y_pred_new_actual,
                                  sub_index=graph.new_idx,
                                  loss=losses,
                                  x_projection=embedding_repr)

            self.best_loss = results["loss_total"]

        return counterfactual
    
    @property
    def name(self):
        
        return "CF-GNNExplainer Features" 
    
    
# PERTURBER.PY

from omegaconf import DictConfig
import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from abc import abstractmethod, ABC
from typing import Tuple
from src.node_level_explainer.utils.utils import normalize_adj

class Perturber(nn.Module, ABC):

    def __init__(self, model ) -> None:
        super().__init__()

        # self.cfg = cfg
        self.model = model
        self.deactivate_model()
        # self.set_reproducibility()
        
    def deactivate_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def set_reproducibility(self)->None:
        torch.manual_seed(self.cfg.general.seed)
        torch.cuda.manual_seed(self.cfg.general.seed)
        torch.cuda.manual_seed_all(self.cfg.general.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def forward_prediction(self):
        pass

class NodePerturberMultiple(Perturber):

    def __init__(self, 
                 cfg: DictConfig, 
                 num_classes: int, 
                 adj, 
                 model: nn.Module, 
                 nfeat: int):
        super().__init__(cfg=cfg, model=model)

        self.adj = normalize_adj(adj)
        self.nclass = num_classes
        self.num_nodes = self.adj.shape[0]
        self.features_add = False
        self.perturb_1 = Parameter(torch.FloatTensor(torch.zeros(self.num_nodes, 14)))
        self.perturb_2 = Parameter(torch.FloatTensor(torch.zeros(self.num_nodes, 15)))


    def forward(self, x, adj):

        if not self.features_add:
            x = torch.cat((F.sigmoid(self.perturb_1), self.perturb_2)) * x
        else:
            x = F.sigmoid(self.x_vec)

        return self.model(x, self.adj)

    
    def forward_prediction(self, x):

        self.perturb_1_thr = (F.sigmoid(self.perturb_1) >= 0.5).float()
        perturbation =  torch.cat((self.perturb_1_thr, self.perturb_2))

        if not self.features_add:
            x = perturbation * x
        else:
            x = self.features_t

        out = self.model(x, self.adj)
        return out, x, perturbation
    
    def loss(self, graph, output, y_node_non_differentiable):

        node_to_explain = graph.new_idx
        y_node_predicted = output[node_to_explain].unsqueeze(0)
        y_target = graph.targets[node_to_explain].unsqueeze(0)
        constant = ((y_target != torch.argmax(y_node_predicted)) or (y_target != y_node_non_differentiable)).float()
        loss_pred =  F.cross_entropy(y_node_predicted, y_target)
        loss_feat = F.l1_loss(graph.x, F.sigmoid(self.x_vec) * graph.x)
	
        loss_total =  constant * loss_pred + loss_feat

        results = {
			"loss_total":  loss_total.item(),
			"loss_pred": loss_pred.item(),
            "loss_graph_dist": 0.0,
            "loss_feat": loss_feat.item()
		}
        
        return loss_total, results, self.adj


class NodePerturber(Perturber):

    def __init__(self, 
                 num_classes: int, 
                 adj, 
                 model, 
                 nfeat: int,
                 nodes_number: int,
                 device: str = "cpu"):
        super().__init__(model=model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nclass = num_classes
        self.num_nodes = nodes_number
        self.features_add = False
        self.P_x = Parameter(torch.FloatTensor(torch.ones(self.num_nodes, nfeat))).to(device)
        self.adj = normalize_adj(adj).to(self.device)

          
    def gat_forward(self, V_x, edge_index, edge_attr):


        V_pert = F.sigmoid(self.P_x) * V_x

        return self.model(V_pert, edge_index, edge_attr)
    
    def forward(self, V_x, adj):

        if not self.features_add:
            V_pert = F.sigmoid(self.P_x) * V_x
        else:
            V_pert = F.sigmoid(self.P_x)

        return self.model(V_pert, self.adj)
    
    def gat_forward_prediction(self, V_x, edge_index, edge_attr)->Tuple[Tensor, Tensor, Tensor]:

        #pert = (F.sigmoid(self.P_x) >= 0.5).float()

        V_pert = F.sigmoid(self.P_x) * V_x

        out = self.model(V_pert, edge_index, edge_attr)
        return out, V_pert, self.P_x
    
    def forward_prediction(self, V_x)->Tuple[Tensor, Tensor, Tensor]:

        pert = (F.sigmoid(self.P_x) >= 0.5).float()

        if not self.features_add:
            V_pert = pert * V_x

        else:
            V_pert = pert 

        out = self.model(V_pert, self.adj)
        return out, V_pert, self.P_x
    
    
    def loss(self, graph, output, y_node_non_differentiable):

        node_to_explain = graph.new_idx
        y_node_predicted = output[node_to_explain].unsqueeze(0)
        y_target = graph.targets[node_to_explain].unsqueeze(0)
        constant = ((y_target != torch.argmax(y_node_predicted)) or (y_target != y_node_non_differentiable)).float()
        loss_pred =  F.cross_entropy(y_node_predicted, y_target) 
        loss_feat = F.l1_loss(graph.x, F.sigmoid(self.P_x) * graph.x)
	
        loss_total =  constant * loss_pred + loss_feat

        results = {
			"loss_total":  loss_total.item(),
			"loss_pred": loss_pred.item(),
            "loss_graph_dist": 0.0,
            "loss_feat": loss_feat.item()
		}
        
        return loss_total, results, self.adj


#%%

import os
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# from torch.nn.utils import clip_grad_norm
from torch_geometric.utils import dense_to_sparse

from carla.data.catalog.graph_catalog import AMLtoGraph, PlanetoidGraph

# from carla.data.api import Data
from carla.data.catalog.online_catalog import DataCatalog

# from carla.models.api import MLModel
# commento
from carla.models.catalog.GAT_TORCH.model_gat import GAT
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import merge_default_parameters

from .library.gat_perturb import GATSyntheticPerturb
from .library.utils import get_degree_matrix, get_neighbourhood, normalize_adj


class CFGATExplainer(RecourseMethod):
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
        "device": "cpu",
    }

    def __init__(self, mlmodel: GAT, data: DataCatalog, hyperparams: Dict = None):

        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super(CFGATExplainer, self).__init__(mlmodel=mlmodel)
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

        best_cf_example = None
        self.best_loss = np.inf
        self.cf_model = NodePerturber(num_classes=self.num_classes, 
                                      adj=graph.adj, 
                                      model=self.mlmodel, 
                                      nfeat=graph.x.shape[1], 
                                      nodes_number=graph.x.shape[0]).to(self.device)
        # num_cf_examples = 0

        for epoch in range(num_epochs):
            new_example = self.train(epoch, graph, self.mlmodel)
            
            if new_example is not None:
                best_cf_example = new_example

        if verbose:
            print(f"CF example for node_idx = {self.node_idx}\n")

        return best_cf_example

    def train(self, epoch: int, graph: Data, oracle):

        self.optimizer.zero_grad()

        differentiable_output = self.cf_model.forward(graph.x, graph.adj) 
        model_out, V_pert, P_x = self.cf_model.forward_prediction(graph.x) 
        
        y_pred_new_actual = torch.argmax(model_out, dim=1)

        loss, results, cf_adj = self.cf_model.loss(graph, differentiable_output, y_pred_new_actual[graph.new_idx])
        loss.backward()

        clip_grad_norm_(self.cf_model.parameters(), 2.0)
        self.optimizer.step()

        counterfactual = None

        if y_pred_new_actual[graph.new_idx] == graph.targets[graph.new_idx] and results["loss_total"] < self.best_loss:
            
            losses = {"feature":results["loss_feat"], "prediction": results["loss_pred"], "graph":results["loss_graph_dist"]}
            # embedding_repr = torch.mean(oracle.get_embedding_repr(V_pert, cf_adj), dim=0)

            counterfactual = Data(x=V_pert, 
                                  adj=graph.adj, 
                                  y=y_pred_new_actual,
                                  sub_index=graph.new_idx,
                                  loss=losses)
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
        # idx_train  # non dovrebbe servire

        node_idx = [i for i in range(0, len(data_graph.y))]
        idx_test = torch.masked_select(torch.Tensor(node_idx), data_graph.test_mask)
        idx_test = idx_test.type(torch.int64)

        norm_edge_index = dense_to_sparse(adj)  # Needed for pytorch-geo functions
        norm_adj = normalize_adj(adj).to(
            self.device
        )  # According to reparam trick from GCN paper

        # output of GCN Syntethic model
        y_pred_orig = self.mlmodel.predict_gnn(features, norm_adj)
        targets =  (1 + y_pred_orig)% self.num_classes
        # y_pred_orig = torch.argmax(output, dim=1)
        # print(torch.max(y_pred_orig))
        # Get CF examples in test set
        test_cf_examples = []
        test_cf_sts = []
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

            # self.sub_adj = sub_adj
            # self.sub_feat = sub_feat.to(self.device)
            # self.sub_labels = sub_labels
            # self.y_pred_orig = y_pred_orig[i]
            sub_y = y_pred_orig[sub_index]
            sub_targets = targets[sub_index]
            
            factual = Data( x = sub_feat, 
                           adj = sub_adj,
                           y = sub_y,
                           y_ground = sub_labels,
                           new_idx = new_idx,
                           targets = sub_targets,
                           node_dict = node_dict)
            
            factual = factual.to(self.device)
            # Instantiate CF model class, load weights from original model
            # The syntentic model load the weights from the model to explain then freeze them
            # and train the perturbation matrix to change the prediction
            
            '''
            self.cf_model = GATSyntheticPerturb(
                nfeat=self.sub_feat.shape[1],
                hid_list_att=self.hid_attr_list,
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
            '''

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
                        f"Output original model, sub adj: {self.mlmodel.predict_proba_gnn(sub_feat, normalize_adj(sub_adj).to(self.device))[new_idx]}"
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
                graph = factual,
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
            one_cf_example = cf_example[0]
            test_cf_sts.append(one_cf_example)

            # cf_example = [ [cf_example0], [cf_example1], [cf_example2], etc...]

            # da trasformare cf_example (DataGraph) in DataFrame (utilizzando forse diz_conn)

            # prendo da cf_example cf_adj
            cf_adj = cf_example[0][2]  # dovrei iterare (?)

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
        return test_cf_examples

