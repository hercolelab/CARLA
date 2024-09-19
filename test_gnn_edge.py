import pandas as pd

import carla.recourse_methods.catalog as recourse_catalog
from carla.data.catalog import AMLtoGraph
from carla.models.catalog import MLModelCatalog

path_file = "/home/hl-turing/VSCodeProjects/Flavio/CARLA/data/AML_Laund_Clean.csv"
# Data
dataset = AMLtoGraph(path_file)

# Model
#model_type = "gat_coo"
# model_type = "gUnet_coo"

'''

model_type = "gnn"
hidden_size =[100, 100, 100, 100]
'''

'''

model_type = 'gat'
hidden_size = []
'''

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


def get_random_sample(csv_file, fraction=0.15, random_state=None):
    """
    Carica un CSV e ritorna un DataFrame contenente un campione casuale del file.
    
    :param csv_file: Percorso al file CSV.
    :param fraction: La frazione di elementi da restituire (default 15%).
    :param random_state: (Opzionale) Valore per fissare il seed della randomizzazione.
    :return: Un DataFrame contenente il campione casuale.
    """
    # Leggi il file CSV in un DataFrame
    df = pd.read_csv(csv_file)
    
    # Prendi un campione casuale del 15% del DataFrame
    sample_df = df.sample(frac=fraction, random_state=random_state)
    
    return sample_df


def data_testing(path):

    dataset = pd.read_csv(path)

    idx = [i for i in range(int(len(dataset) * 0.15))]
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

hyper_edgeperturb = {
    "cf_optimizer": "Adadelta",
    "lr": 0.5,
    "num_epochs": 1000,
    "dropout": 0.0,
    "nheads": 8,
    "num_classes": 2,
    "n_layers": 3,
    "n_momentum": 0,
    "verbose": True,
    "device": "cuda",
}

if model_type == "gnn":
    recourse_method = recourse_catalog.CFEdgeExplainer(
        mlmodel=ml_model, data=dataset, hyperparams=hyper_edgeperturb
        )

    df_cfs_gnn, num_cf_gnn, validity_gnn, sparsity_gnn, fidelity_acc_gnn, fidelity_prob_gnn = recourse_method.get_counterfactuals(test_factual)


    print(f"{df_cfs_gnn=}")
    print(f"{num_cf_gnn=}")
    print(f"{sparsity_gnn=}")
    print(f"{validity_gnn=}")
    print(f"{fidelity_acc_gnn=}")
    print(f"{fidelity_prob_gnn=}")

else:
    recourse_method = recourse_catalog.CFEdgeExplainer(
        mlmodel=ml_model, data=dataset, hyperparams=hyper_edgeperturb
        )

    df_cfs, num_cf, validity, sparsity, fidelity = recourse_method.get_counterfactuals(test_factual)


    print(f"{df_cfs=}")
    print(f"{num_cf=}")
    print(f"{sparsity=}")
    print(f"{validity=}")
    print(f"{fidelity=}")