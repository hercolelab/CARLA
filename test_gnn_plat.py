import pandas as pd
import torch
import carla.recourse_methods.catalog as recourse_catalog
from carla.data.catalog import PlanetoidGraph
from carla.models.catalog import MLModelCatalog

path_file = "Cora"
# Data
dataset = PlanetoidGraph(path_file)

# Model

ml_model = MLModelCatalog(
    data=dataset, model_type="gat", load_online=False, backend="pytorch"
)

training_params = {
    "lr": 0.002,
    "epochs": 100,
    "batch_size": 1024,
    "hidden_size": [18, 9, 3],
}

ml_model.train(
    learning_rate=training_params["lr"],
    epochs=training_params["epochs"],
    batch_size=training_params["batch_size"],
    hidden_size=training_params["hidden_size"],
)


def data_testing(path):

    dataset = pd.read_csv(path)

    # percentuale_training = 0.005
    idx = [i for i in range(int(len(dataset) * 0.001))]
    test_set = dataset.iloc[idx]
    # test_set = dataset.sample(frac=percentuale_training, random_state=42)
    return test_set


# Recourse Method

#test_factual = data_testing(path_file)
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


hyper = {
    "cf_optimizer": "Adadelta",
    "lr": 0.5,
    "num_epochs": 500,
    "n_hid": 31,
    "dropout": 0.0,
    "alpha": 0.2,
    "beta": 0.5,
    "nheads": 8,
    "num_classes": 6,
    "n_layers": 3,
    "n_momentum": 0,
    "verbose": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

recourse_method = recourse_catalog.CFGATExplainer(
    mlmodel=ml_model, data=dataset, hyperparams=hyper
)

df_cfs = recourse_method.get_counterfactuals("Cora")


print(f"{df_cfs=}")
