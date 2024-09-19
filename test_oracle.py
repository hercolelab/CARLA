import pandas as pd

import carla.recourse_methods.catalog as recourse_catalog
from carla.data.catalog import AMLtoGraph
from carla.models.catalog import MLModelCatalog

path_file = "/home/hl-turing/VSCodeProjects/Flavio/CARLA/data/AML_Laund_Clean.csv"
# Data
dataset = AMLtoGraph(path_file)

'''
# GIN COO E GIN
model_type = 'gin_coo'
hidden_size = [50,50,50,50]
hidden_list_conv = [100]

ml_model = MLModelCatalog(
    data=dataset, model_type=model_type, load_online=False, backend="pytorch"
)

training_params = {
    "lr": 0.001,
    "epochs": 300,
    "batch_size": 1435,
    "hidden_size": hidden_size,
    "hidden_size_conv": hidden_list_conv,
    "clip": 1.0,
    
    "alpha": None,
    "nheads": 6,
    "neigh": [100,100,100]
}
'''


'''
# GAT E GAT COO
model_type = 'gat_coo'
hidden_size = [100,100]
hidden_list_conv = [50,50]

ml_model = MLModelCatalog(
    data=dataset, model_type=model_type, load_online=False, backend="pytorch"
)

training_params = {
    "lr": 0.002,
    "epochs": 300,
    "batch_size": 1028,
    "hidden_size": hidden_size,
    "hidden_size_conv": hidden_list_conv,
    "clip": 1.0,
    
    "alpha": 0.2,
    "nheads": 4,
    "neigh": [100,100,100]
}
'''


# GNN COO e GNN
model_type = 'gnn_coo'
hidden_size = [50,50,50,50]
hidden_list_conv = None

ml_model = MLModelCatalog(
    data=dataset, model_type=model_type, load_online=False, backend="pytorch"
)

training_params = {
    "lr": 0.001,
    "epochs": 300,
    "batch_size": 647,
    "hidden_size": hidden_size,
    "hidden_size_conv": hidden_list_conv,
    "clip": 1.0,
    
    "alpha": None,
    "nheads": None,
    "neigh": [50,50,50]
}




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

