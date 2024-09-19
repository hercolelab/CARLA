import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import Tensor
from .library.utils import get_degree_matrix, get_neighbourhood, normalize_adj, dense_to_sparse

class Perturber(nn.Module, ABC):

    def __init__(self, model ) -> None:
        super(Perturber, self).__init__()

        # self.cfg = cfg
        self.model = model
        self.deactivate_model()
        # self.set_reproducibility()
        
    def deactivate_model(self):
        for name, param in self.model.raw_model.named_parameters(): 
            if (
                    name.endswith("weight")
                    or name.endswith("bias")
                    or name.endswith("bias")
                    or ("attention" in name)
                    or ("out_att" in name)
            ):
                param.requires_grad = False
            

        for name, param in self.model.raw_model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)


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


class EdgePerturber(Perturber):

    def __init__(self, 
                 num_classes: int, 
                 adj, 
                 model, 
                 nfeat: int,
                 edge_number: int,
                 device: str = "cpu"):
        super().__init__(model=model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nclass = num_classes
        self.num_edges = edge_number
        self.features_add = False
        self.P_x = Parameter(torch.FloatTensor(torch.zeros(self.num_edges, nfeat))).to(device)
        #self.P_x = Parameter(torch.FloatTensor(torch.ones(self.num_edges, nfeat))).to(device)

        self.adj = normalize_adj(adj).to(self.device)
        self.edge_ind = dense_to_sparse(self.adj)

    
    def forward(self, x, edge_index, V_edge_attr):
        
        if not self.features_add:
            V_pert = F.sigmoid(self.P_x) + V_edge_attr
        else:
            V_pert = F.sigmoid(self.P_x)

        return self.model.predict_proba_gnn_coo(x, edge_index[0], V_pert)

    
    def forward_prediction(self, x, edge_index, V_edge_attr)->Tuple[Tensor, Tensor, Tensor]:
        
        pert = (F.sigmoid(self.P_x) >= 0.5).float()
        
        if not self.features_add:
            V_pert = pert + V_edge_attr
        else:
            V_pert = pert
        
        out = self.model.predict_proba_gnn_coo(x, edge_index[0], V_pert)
        return out, V_pert, self.P_x
    
    
    def loss(self, graph, output, y_node_non_differentiable):

        edge_to_explain = graph.new_idx
        y_node_predicted = output[edge_to_explain].unsqueeze(0)
        y_target = graph.targets[edge_to_explain].unsqueeze(0)
        constant = ((y_target != torch.argmax(y_node_predicted)) or (y_target != y_node_non_differentiable)).float()
        loss_pred =  F.cross_entropy(y_node_predicted, y_target) 
        # loss_feat = F.l1_loss(graph.x, F.sigmoid(self.P_x) * graph.x)
        loss_feat = F.mse_loss(graph.edge_attr, F.sigmoid(self.P_x) * graph.edge_attr)
	
        loss_total =  constant * loss_pred + loss_feat

        results = {
			"loss_total":  loss_total.item(),
			"loss_pred": loss_pred.item(),
            "loss_graph_dist": 0.0,
            "loss_feat": loss_feat.item()
		}
        
        return loss_total, results, self.adj
