# from abc import ABC
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from sklearn import preprocessing
from torch_geometric.data import Data

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

        # preso da Online Catalog
        # aggiungere in data_catalog.yaml AML
        catalog_content = ["continuous", "categorical", "immutable", "target"]
        self.catalog: Dict[str, Any] = load(  # type: ignore
            "data_catalog.yaml", "AML", catalog_content
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

        df = df.drop(["Is Laundering", "From", "To"], axis=1)

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
            self.change_false(all_false, 0, idx_split), dtype=torch.long
        )
        test_mask = torch.tensor(
            self.change_false(all_false, idx_split, len(all_false)), dtype=torch.long
        )
        return train_mask, test_mask

    def construct_GraphData(self):
        if isinstance(self._data_table, str):
            # csv path
            df = pd.read_csv(self._data_table)
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
