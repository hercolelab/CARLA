import itertools

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Aggiunta dell'ID


def add_identifier(dataframe):
    new_dataframe = dataframe.copy()
    new_col = pd.Series([i for i in range(len(new_dataframe))])
    new_dataframe["ID"] = new_col
    return new_dataframe


# Convert non-numeric columns using One-Hot Encoding
def convert_non_numeric_col(node_feat, col):
    pd.set_option("mode.chained_assignment", None)
    pos = node_feat[col].str.split(",", expand=True)
    node_feat["first_position"] = pos[0]
    # One-hot encoding
    node_feat = pd.concat(
        [node_feat, pd.get_dummies(node_feat["first_position"])], axis=1, join="inner"
    )
    node_feat.drop([col, "first_position"], axis=1, inplace=True)
    # trasform boolean value with 0 and 1
    node_feat = node_feat.astype(int)
    return node_feat


# ident = 'ID'
def construct_array_tab(dataframe, list_feat):
    sorted_df = dataframe.sort_values(by="ID")
    node_feat = sorted_df[list_feat]
    # Convert non-numeric columns using One-Hot Encoding
    for feature in list_feat:
        if isinstance(node_feat[feature][0], str):
            node_feat = convert_non_numeric_col(node_feat, feature)
    # Convert in numpy
    arr = node_feat.to_numpy()
    return arr


def construct_edge_index(dataframe, connection):
    edges = dataframe[connection].unique()
    """
    Array con due liste:
    - colonne
    -righe
    """
    diz = {elem: i for i, elem in enumerate(set(dataframe[connection]))}
    values = []

    all_edges = np.array([], dtype=np.int32).reshape((0, 2))
    for edg in edges:
        # creo un dataframe education dove ho solo l'education selezionato
        edg_df = dataframe[dataframe[connection] == edg]
        # prendo gli id
        ident_edg = edg_df["ID"].values
        permutations = list(itertools.combinations(ident_edg, 2))
        for _ in range(len(permutations)):
            values.append(diz[edg])

        edges_source = [e[0] for e in permutations]
        edges_target = [e[1] for e in permutations]
        team_edges = np.column_stack([edges_source, edges_target])
        all_edges = np.vstack([all_edges, team_edges])

    # Convert to Pytorch Geometric format
    edge_index = all_edges.transpose()
    return edge_index, diz, np.array(values)


def change_false(arr, idx_start, idx_stop):
    mask = arr.copy()
    for i in range(idx_start, idx_stop):
        mask[i] = True
    return mask


def create_mask(num_nodes):
    # split 80/20
    idx_split = int(num_nodes * 80 / 100)
    # list with all false
    all_false = [False for _ in range(0, num_nodes)]
    train_mask = torch.tensor(change_false(all_false, 0, idx_split), dtype=torch.long)
    test_mask = torch.tensor(
        change_false(all_false, idx_split, len(all_false)), dtype=torch.long
    )
    return train_mask, test_mask


def construct_GraphData(dataframe, list_feat, list_labels, connection):
    dataframeID = add_identifier(dataframe)
    x = torch.tensor(construct_array_tab(dataframeID, list_feat), dtype=torch.long)
    y = torch.tensor(construct_array_tab(dataframeID, list_labels), dtype=torch.long)
    train_mask, test_mask = create_mask(len(x))
    edge_index, diz_conn, arr_values = construct_edge_index(dataframeID, connection)

    e = torch.tensor(edge_index, dtype=torch.long)
    # edge_index1 = e.t().clone().detach()
    data_graph = Data(
        x=x,
        edge_index=e.to().contiguous(),
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
    )
    values_edges = torch.tensor(arr_values, dtype=torch.long)

    return data_graph, values_edges, diz_conn


def create_adj_matrix(data_graph, values_edges):
    edges_index = data_graph.edge_index
    row_indices = edges_index[0]
    col_indices = edges_index[1]
    # values = torch.ones(len(edges_index[0]))    # valori tutti a uno
    size = torch.Size([len(data_graph.x), len(data_graph.x)])
    sparse_matrix = torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices]), values_edges, size=size
    )
    adj_matrix = sparse_matrix.to_dense()
    return adj_matrix


# %%-----------------------


def get_degree_matrix(adj):
    return torch.diag(sum(adj))


def normalize_adj(adj):

    # Normalize adjacancy matrix according to reparam trick in GCN paper
    A_tilde = adj + torch.eye(adj.shape[0])
    D_tilde = get_degree_matrix(A_tilde)
    # Raise to power -1/2, set all infs to 0s
    D_tilde_exp = D_tilde ** (-1 / 2)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

    # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
    norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
    return norm_adj
