import pandas as pd

import carla.recourse_methods.catalog as recourse_catalog
from carla.data.catalog import AMLtoGraph
from carla.models.catalog import MLModelCatalog

path_file = "/home/hl-turing/VSCodeProjects/Flavio/CARLA/data/AML_Laund_Clean.csv"
# Data
dataset = AMLtoGraph(path_file)

# Model

#model_type = "gin"
#hidden_size = [50,50,]

model_type = 'gin'
hidden_size = [100]

ml_model = MLModelCatalog(
    data=dataset, model_type=model_type, load_online=False, backend="pytorch"
)

training_params = {
    "lr": 0.001,
    "epochs": 250,
    "batch_size": 732,
    "hidden_size": hidden_size,
    "hidden_size_conv": [100,100],
    "clip": 2.0,
    
    "alpha": None,
    "nheads": 3,
    "neigh": [31,31,31]
    
    
}

'''
model_type = 'gnn'
hidden_size = [100,100,100,100]

ml_model = MLModelCatalog(
    data=dataset, model_type=model_type, load_online=False, backend="pytorch"
)

training_params = {
    "lr": 0.001,
    "epochs": 1500,
    "batch_size": 878,
    "hidden_size": hidden_size,
    "hidden_size_conv": None,
    "clip": 1.0,
    
    "alpha": None,
    "nheads": 8,
    "neigh": [100]
    
    
}
'''

ml_model.train(
    learning_rate=training_params["lr"],
    epochs=training_params["epochs"],
    batch_size=training_params["batch_size"],
    hidden_size=training_params["hidden_size"],
    hidden_conv= training_params["hidden_size_conv"],
    clip=training_params["clip"],
    alpha=training_params["alpha"],
    nheads=training_params["nheads"],
    neig=training_params["neigh"]
    
)


def data_testing(path):

    dataset = pd.read_csv(path)

    idx = [i for i in range(int(len(dataset) * 0.3))]
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
    "lr": 0.5,
    "num_epochs": 1000,
    "hid_list": hidden_size,
    "dropout": 0.0,
    "beta": 0.5,
    "num_classes": 2,
    "n_layers": 4,
    "n_momentum": 0,
    "verbose": True,
    "device": "cuda",
}

hyper_gat = {
    "cf_optimizer": "Adadelta",
    "lr": 0.5,
    "num_epochs": 1000,
    "hid_attr_list": [31, 31, 31, 31, 31, 31, 31, 31],
    "hid_list": hidden_size,
    "dropout": 0.0,
    "alpha": 0.2,
    "beta": 0.5,
    "nheads": 8,
    "num_classes": 2,
    "n_layers": 4,
    "n_momentum": 0,
    "verbose": True,
    "device": "cuda",
}

hyper_gin = {
    "cf_optimizer": "Adadelta",
    "lr": 0.5,
    "num_epochs": 1500,
    "hid_attr_list": hidden_size,
    "hid_list": [100, 100],
    "dropout": 0.0,
    "alpha": 0.2,
    "beta": 0.5,
    "nheads": 3,
    "num_classes": 2,
    "n_layers": 2,
    "n_momentum": 0,
    "verbose": True,
    "device": "cuda",
}

if model_type == "gnn":
    recourse_method = recourse_catalog.CFExplainer(
    mlmodel=ml_model, data=dataset, hyperparams=hyper_gnn)

elif model_type == "gin":
    recourse_method = recourse_catalog.CFGINExplainer(
    mlmodel=ml_model, data=dataset, hyperparams=hyper_gin)
else:
    recourse_method = recourse_catalog.CFGATExplainer(
     mlmodel=ml_model, data=dataset, hyperparams=hyper_gat)

df_cfs = recourse_method.get_counterfactuals(test_factual)


print(f"{df_cfs=}")
