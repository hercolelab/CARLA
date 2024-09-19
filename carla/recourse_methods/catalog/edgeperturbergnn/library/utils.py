import errno
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import (
    dense_to_sparse,
    k_hop_subgraph,
    subgraph,
    to_dense_adj,
)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def safe_open(path, w):
    """Open "path" for writing, creating any parent directories as needed."""
    mkdir_p(os.path.dirname(path))
    return open(path, w)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_degree_matrix(adj):
    return torch.diag(sum(adj))


def normalize_adj(adj):
   
    # Normalize adjacancy matrix according to reparam trick in GCN paper
    A_tilde = adj.cpu() + torch.eye(adj.shape[0] , device="cpu")
    D_tilde = get_degree_matrix(A_tilde)
    # Raise to power -1/2, set all infs to 0s
    D_tilde_exp = D_tilde ** (-1 / 2)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

    # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
    norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
    return norm_adj


def get_neighbourhood(node_idx: int, edge_index, n_hops, features, labels):
    """
    Get the subgraph induced by a node "node_idx" along with all the features
    """
    edge_subset = k_hop_subgraph(
        node_idx, n_hops, edge_index[0]
    )  # Get all nodes involved
    edge_subset_relabel = subgraph(
        edge_subset[0], edge_index[0], relabel_nodes=True
    )  # Get relabelled subset of edges
    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    sub_feat = features[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    new_index = np.array([i for i in range(len(edge_subset[0]))])
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))  # Maps orig labels to new
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    return sub_adj, sub_feat, sub_labels, node_dict

def get_neighbourhood_v2(node_idx: int, edge_index, edge_attr, n_hops, features, labels):
    """
    Get the subgraph induced by a node "node_idx" along with all the features and edge attributes.
    
    Parameters:
    - node_idx (int): Index of the node.
    - edge_index (Tensor): The edge indices of the graph.
    - edge_attr (Tensor): The edge attributes of the graph.
    - n_hops (int): The number of hops to consider for the subgraph.
    - features (Tensor): Node feature matrix.
    - labels (Tensor): Node labels.
    
    Returns:
    - sub_adj (Tensor): The adjacency matrix of the subgraph.
    - sub_feat (Tensor): The node features of the subgraph.
    - sub_labels (Tensor): The labels of the nodes in the subgraph.
    - sub_edge_attr (Tensor): The edge attributes of the subgraph.
    - node_dict (dict): Mapping from original node indices to new indices in the subgraph.
    """
    # Get all nodes involved in the k-hop neighborhood
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index[0])
    
    # Get relabelled subset of edges
    edge_subset_relabel = subgraph(edge_subset[0], edge_index[0], relabel_nodes=True)
    
    # Extract the subset of edge attributes
    sub_edge_attr = find_sub_edge_attr(edge_index[0] ,edge_subset[1], edge_attr)
    
    # Create adjacency matrix for the subgraph
    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    
    # Extract node features and labels for the subgraph
    sub_feat = features[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    
    # Create a mapping from original node indices to new indices in the subgraph
    new_index = np.array([i for i in range(len(edge_subset[0]))])
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))
    
    return sub_adj, sub_feat, sub_labels, sub_edge_attr, node_dict, edge_subset[1]


def create_symm_matrix_from_vec(vector, n_rows):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix = torch.zeros(n_rows, n_rows, device=device)
    idx = torch.tril_indices(n_rows, n_rows, device=device)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def get_S_values(pickled_results, header):
    df_prep = []
    for example in pickled_results:
        if example != []:
            df_prep.append(example[0])
    return pd.DataFrame(df_prep, columns=header)


def redo_dataset_pgexplainer_format(dataset, train_idx, test_idx):

    dataset.data.train_mask = index_to_mask(train_idx, size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(
        test_idx[len(test_idx)], size=dataset.data.num_nodes
    )


def get_S_values(pickled_results, header):
    df_prep = []
    for example in pickled_results:
        if example != []:
            df_prep.append(example[0])
    return pd.DataFrame(df_prep, columns=header)


def redo_dataset_pgexplainer_format(dataset, train_idx, test_idx):

    dataset.data.train_mask = index_to_mask(train_idx, size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(
        test_idx[len(test_idx)], size=dataset.data.num_nodes
    )


def find_sub_edge_attr(edge_index ,edge_index_subset, edge_attr):
    indexes = []
    for i in range(len(edge_index_subset[0])):
        find = edge_index_subset[:, i]
        for j in range(len(edge_index[0])):
            if list(edge_index[:, j]) == list(find):
                indexes.append(j)
                break
    
    sub_edge_attr = torch.tensor([list(edge_attr[ind]) for ind in indexes])
    return sub_edge_attr