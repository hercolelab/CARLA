# from abc import ABC
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from sklearn import preprocessing
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from carla.data.load_catalog import load

from .catalog import DataCatalog

# from torch_geometric.data import InMemoryDataset

"""

class DataGraph(ABC):
    def __init__(self, data_table: Union[str, pd.DataFrame]):

        self._data_table = data_table

    def change_false(self, arr, idx_start, idx_stop):
        mask = arr.copy()
        for i in range(idx_start, idx_stop):
            mask[i] = True
        return mask

    def create_mask(self, num_nodes):
        # split 80/20
        idx_split = int(num_nodes * 80 / 100)
        # list with all false
        all_false = [False for _ in range(0, num_nodes)]
        train_mask = torch.tensor(self.change_false(all_false, 0, idx_split))
        test_mask = torch.tensor(
            self.change_false(all_false, idx_split, len(all_false))
        )
        return train_mask, test_mask

    def get_edge_df(self):
        pass

    def get_node_attr(self):
        pass

    def construct_GraphData(self):
        pass

    def reconstruct_Tabular(self):
        pass
"""


class AMLtoGraph(DataCatalog):
    def __init__(self, data_table: Union[str, pd.DataFrame]):
        self._data_table = data_table
        self.name = 'AML'

        # preso da Online Catalog
        # aggiungere in data_catalog.yaml AML
        catalog_content = ["continuous", "categorical", "immutable", "target"]
        self.catalog: Dict[str, Any] = load(  # type: ignore
            "data_catalog.yaml", self.name, catalog_content
        )

        for key in ["continuous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []
                
        

    @property
    def categorical(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continuous(self) -> List[str]:
        return self.catalog["continuous"]

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        return self.catalog["target"]

    # trasform the values with label from 0 to num_columns
    def df_label_encoder(self, df, columns):
        le = preprocessing.LabelEncoder()
        for i in columns:
            df[i] = le.fit_transform(df[i].astype(str))
        return df

    """
    - Modify the original dataframe
    - Create a dataframe for receiving Account with feauture: Account | Amount Received | Receving Currency
    - Create a dataframe for paying Account with feauture: Account | Amount Paid | Payment Currency
    """

    def preprocess(self, df):
        df = self.df_label_encoder(
            df, ["Payment Format", "Payment Currency", "Receiving Currency"]
        )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Timestamp"] = df["Timestamp"].apply(lambda x: x.value)
        df["Timestamp"] = (df["Timestamp"] - df["Timestamp"].min()) / (
            df["Timestamp"].max() - df["Timestamp"].min()
        )

        df["Account"] = df["From Bank"].astype(str) + "_" + df["Account"]
        df["Account.1"] = df["To Bank"].astype(str) + "_" + df["Account.1"]
        df = df.sort_values(by=["Account"])
        receiving_df = df[["Account.1", "Amount Received", "Receiving Currency"]]
        paying_df = df[["Account", "Amount Paid", "Payment Currency"]]
        receiving_df = receiving_df.rename({"Account.1": "Account"}, axis=1)
        currency_ls = sorted(df["Receiving Currency"].unique())

        return df, receiving_df, paying_df, currency_ls

    # Get a dataframe with the following columns: Account | Bank | Is Laundering
    def get_all_account(self, df):
        ldf = df[["Account", "From Bank"]]
        rdf = df[["Account.1", "To Bank"]]
        suspicious = df[df["Is Laundering"] == 1]
        s1 = suspicious[["Account", "Is Laundering"]]
        s2 = suspicious[["Account.1", "Is Laundering"]]
        s2 = s2.rename({"Account.1": "Account"}, axis=1)
        suspicious = pd.concat([s1, s2], join="outer")
        suspicious = suspicious.drop_duplicates()

        ldf = ldf.rename({"From Bank": "Bank"}, axis=1)
        rdf = rdf.rename({"Account.1": "Account", "To Bank": "Bank"}, axis=1)
        df = pd.concat([ldf, rdf], join="outer")
        df = df.drop_duplicates()

        df["Is Laundering"] = 0
        df.set_index("Account", inplace=True)
        df.update(suspicious.set_index("Account"))
        df = df.reset_index()
        return df

    def paid_currency_aggregate(self, currency_ls, paying_df, accounts):
        for i in currency_ls:
            temp = paying_df[paying_df["Payment Currency"] == i]
            accounts["avg paid " + str(i)] = (
                temp["Amount Paid"].groupby(temp["Account"]).transform("mean")
            )
        return accounts

    def received_currency_aggregate(self, currency_ls, receiving_df, accounts):
        for i in currency_ls:
            temp = receiving_df[receiving_df["Receiving Currency"] == i]
            accounts["avg received " + str(i)] = (
                temp["Amount Received"].groupby(temp["Account"]).transform("mean")
            )
        accounts = accounts.fillna(0)
        return accounts

    def get_edge_df(self, accounts, df):
        accounts = accounts.reset_index(drop=True)
        accounts["ID"] = accounts.index
        mapping_dict = dict(zip(accounts["Account"], accounts["ID"]))
        df["From"] = df["Account"].map(mapping_dict)
        df["To"] = df["Account.1"].map(mapping_dict)
        df = df.drop(["Account", "Account.1", "From Bank", "To Bank"], axis=1)

        edge_index = torch.stack(
            [torch.from_numpy(df["From"].values), torch.from_numpy(df["To"].values)],
            dim=0,
        )

        # df = df.drop(["Is Laundering", "From", "To"], axis=1)
        df = df.drop(["Is Laundering"], axis=1)

        edge_attr = torch.from_numpy(df.values).to(torch.float)
        return edge_attr, edge_index

    def get_node_attr(self, currency_ls, paying_df, receiving_df, accounts):
        node_df = self.paid_currency_aggregate(currency_ls, paying_df, accounts)
        node_df = self.received_currency_aggregate(currency_ls, receiving_df, node_df)
        node_label = torch.from_numpy(node_df["Is Laundering"].values).to(torch.float)
        node_df = node_df.drop(["Account", "Is Laundering"], axis=1)
        node_df = self.df_label_encoder(node_df, ["Bank"])
        node_df = torch.from_numpy(node_df.values).to(torch.float)
        return node_df, node_label

    def change_false(self, arr, idx_start, idx_stop):
        mask = arr.copy()
        for i in range(idx_start, idx_stop):
            mask[i] = True
        return mask

    def create_mask(self, num_nodes):
        # split 80/20
        idx_split = int(num_nodes * 80 / 100)
        # list with all false
        all_false = [False for _ in range(0, num_nodes)]
        train_mask = torch.tensor(
            self.change_false(all_false, 0, idx_split), dtype=torch.bool
        )
        test_mask = torch.tensor(
            self.change_false(all_false, idx_split, len(all_false)), dtype=torch.bool
        )
        return train_mask, test_mask

    def construct_GraphData(self):
        if isinstance(self._data_table, str):
            # csv path
            df = pd.read_csv(self._data_table)
            df = df.iloc[:int(len(df)*0.001)]
        else:
            # dataframe
            df = self._data_table

        df, receiving_df, paying_df, currency_ls = self.preprocess(df)
        accounts = self.get_all_account(df)
        node_attr, node_label = self.get_node_attr(
            currency_ls, paying_df, receiving_df, accounts
        )
        edge_attr, edge_index = self.get_edge_df(accounts, df)
        train_mask, test_mask = self.create_mask(len(node_attr))
        

        data = Data(
            x=node_attr,
            edge_index=edge_index,
            y=node_label,
            edge_attr=edge_attr,
            train_mask=train_mask,
            test_mask=test_mask,
        )
        return data

    def create_adj_matrix(self, data_graph):
        edges_index = data_graph.edge_index
        row_indices = edges_index[0]
        col_indices = edges_index[1]
        values = torch.ones(len(edges_index[0]))  # valori tutti a uno
        size = torch.Size([len(data_graph.x), len(data_graph.x)])
        sparse_matrix = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]), values, size=size
        )
        adj_matrix = sparse_matrix.to_dense()
        return adj_matrix

    def get_feat_node(dataGraph, norm_edge_index, node_dict):
        done = []
        array_data = []
        for i, arr in enumerate(norm_edge_index[0]):
            for j, elem in enumerate(arr):
                if int(elem) not in done:
                    for key, value in node_dict.items():
                        if value == int(elem):
                            array_data.append([key] + list(dataGraph.x[key].numpy()))
                            done.append(int(elem))
                            break
        return array_data

    def get_feat_edges(dataGraph, norm_edge_index, node_dict):
        # create a new edge index with original coordinate
        new_edge_index = []
        for i, arr in enumerate(norm_edge_index[0]):
            col = []
            for j, elem in enumerate(arr):
                for key, value in node_dict.items():
                    if value == int(elem):
                        col.append(key)
                        break
            new_edge_index.append(col)
        tensor_new_edge_index = torch.tensor(new_edge_index)
        # print(tensor_new_edge_index[0])

        # get edge_index and edge_attr
        orig_edge_index = dataGraph.edge_index  # tensor
        orig_edge_attr = dataGraph.edge_attr  # numpy array

        # create list with indexes to find edge_attr
        index_corr = []
        for i in range(len(tensor_new_edge_index[0])):
            # print(tensor[:,i])
            for j in range(len(orig_edge_index[0])):
                if list(tensor_new_edge_index[:, i]) == list(orig_edge_index[:, j]):
                    index_corr.append(j)
                    break

        array_attr_edge = []
        for i in index_corr:
            array_attr_edge.append(list(orig_edge_attr[i].numpy()))

        return array_attr_edge

    def reconstruct_Tabular(self, dataGraph, adj_matrix, node_dict):
        columns_name_node = [
            "Account",
            "Bank",
            "avg paid 0",
            "avg paid 1",
            "avg paid 2",
            "avg paid 3",
            "avg paid 4",
            "avg paid 5",
            "avg paid 6",
            "avg paid 7",
            "avg paid 8",
            "avg paid 9",
            "avg paid 10",
            "avg paid 11",
            "avg paid 12",
            "avg paid 13",
            "avg paid 14",
            "avg received 0",
            "avg received 1",
            "avg received 2",
            "avg received 3",
            "avg received 4",
            "avg received 5",
            "avg received 6",
            "avg received 7",
            "avg received 8",
            "avg received 9",
            "avg received 10",
            "avg received 11",
            "avg received 12",
            "avg received 13",
            "avg received 14",
        ]
        columns_name_edge = [
            "Timestamp",
            "Amount Received",
            "Receiving Currency",
            "Amount Paid",
            "Payment Currency",
            "Payment Format",
            "Account",
            "Account 1",
        ]
        norm_edge_index = dense_to_sparse(adj_matrix)

        array_feat_node = self.get_feat_node(dataGraph, norm_edge_index, node_dict)
        df1 = pd.DataFrame(array_feat_node, columns=columns_name_node)

        array_feat_edge = self.get_feat_edges(dataGraph, norm_edge_index, node_dict)
        df2 = pd.DataFrame(array_feat_edge, columns=columns_name_edge)
        # df = pd.DataFrame(array_data, columns=columns_name)

        result = pd.merge(df1, df2, left_on="Account", right_on="Account")
        return result
