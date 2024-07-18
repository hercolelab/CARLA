import pandas as pd

import carla.recourse_methods.catalog as recourse_catalog
from carla.data.catalog import AMLtoGraph
from carla.models.catalog import MLModelCatalog

path_file = "data/AML_Laund_Clean.csv"
# Data
dataset = AMLtoGraph(path_file)

# Model

ml_model = MLModelCatalog(
    data=dataset, model_type="gnn", load_online=False, backend="pytorch"
)

training_params = {
    "lr": 0.002,
    "epochs": 1000,
    "batch_size": 1024,
    "hidden_size": [31, 31, 31],
}

ml_model.train(
    learning_rate=training_params["lr"],
    epochs=training_params["epochs"],
    batch_size=training_params["batch_size"],
    hidden_size=training_params["hidden_size"],
)


def data_testing(path):

    dataset = pd.read_csv(path)

    idx = [i for i in range(int(len(dataset) * 0.2))]
    test_set = dataset.iloc[idx]
    # test_set = dataset.sample(frac=percentuale_training, random_state=42)
    return test_set


# Recourse Method

test_factual = data_testing(path_file)
# hyper = {
#     "cf_optimizer": "Adadelta",
#     "lr": 0.05,
#     "num_epochs": 500,
#     "n_hid": 31,
#     "dropout": 0.0,
#     "beta": 0.5,
#     "num_classes": 2,
#     "n_layers": 3,
#     "n_momentum": 0,
#     "verbose": True,
#     "device": "cuda",
# }

hyper_gnn = {
    "cf_optimizer": "Adadelta",
    "lr": 0.005,
    "num_epochs": 1000,
    "hid_list": [31, 31, 31],
    "dropout": 0.0,
    "beta": 0.5,
    "num_classes": 2,
    "n_layers": 3,
    "n_momentum": 0,
    "verbose": True,
    "device": "cuda",
}

hyper_gat = {
    "cf_optimizer": "Adadelta",
    "lr": 0.5,
    "num_epochs": 1000,
    "hid_attr_list": [31, 31, 31, 31, 31, 31, 31, 31],
    "hid_list": [31, 31, 31],
    "dropout": 0.0,
    "alpha": 0.2,
    "beta": 0.5,
    "nheads": 8,
    "num_classes": 2,
    "n_layers": 3,
    "n_momentum": 0,
    "verbose": True,
    "device": "cuda",
}

recourse_method = recourse_catalog.CFExplainer(
    mlmodel=ml_model, data=dataset, hyperparams=hyper_gnn
)

# recourse_method = recourse_catalog.CFGATExplainer(
#     mlmodel=ml_model, data=dataset, hyperparams=hyper
# )

df_cfs = recourse_method.get_counterfactuals(test_factual)


print(f"{df_cfs=}")
