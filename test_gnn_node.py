import pandas as pd
import numpy as np
import carla.recourse_methods.catalog as recourse_catalog
from carla.data.catalog import AMLtoGraph
from carla.models.catalog import MLModelCatalog

path_file = "/home/hl-turing/VSCodeProjects/Flavio/CARLA/data/AML_Laund_Clean.csv"
# Data
dataset = AMLtoGraph(path_file)

# Model
'''
model_type = "gnn"
hidden_size = [31, 31, 31]
'''

'''
model_type = 'gnn'
hidden_size = [50, 50, 50, 50, 50]
'''

'''
model_type = 'gin'
hidden_size = [100,100]


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


model_type = 'gnn_coo'
hidden_size = [50,50,50,50]
hidden_list_conv = [100]

ml_model = MLModelCatalog(
    data=dataset, model_type=model_type, load_online=False, backend="pytorch"
)

training_params = {
    "lr": 0.001,
    "epochs": 250,
    "batch_size": 1435,
    "hidden_size": hidden_size,
    "hidden_size_conv": hidden_list_conv,
    "clip": 3.0,
    
    "alpha": None,
    "nheads": 6,
    "neigh": [100,100,100]
}

'''

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

def data_testing(csv_file, fraction=0.15, random_state=None):
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
    "model_type": "gnn"
}

hyper_gat = {
    "cf_optimizer": "Adadelta",
    "lr": 0.5,
    "num_epochs": 2000,
    "hid_attr_list": [30, 30, 30, 30, 30, 30, 30, 30],
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
    "model_type": model_type
}

hyper_gin = {
    "cf_optimizer": "Adadelta",
    "lr": 0.5,
    "num_epochs": 1500,
    "hid_attr_list": [100],
    "hid_list": hidden_size,
    "dropout": 0.0,
    "alpha": 0.2,
    "beta": 0.5,
    "nheads": 3,
    "num_classes": 2,
    "n_layers": 2,
    "n_momentum": 0,
    "verbose": True,
    "device": "cuda",
    "model_type": model_type
}

fidelity_list = []
sparsity_list = []
validity_list = []

num_campione = 5

# Recourse Method
for _ in range(num_campione):
    test_factual = data_testing(path_file)
    
    if model_type == "gnn":
        recourse_method = recourse_catalog.CFNodeExplainer(
            mlmodel=ml_model, data=dataset, hyperparams=hyper_gnn
            )

        df_cfs_gnn, num_cf_gnn, validity_gnn, sparsity_gnn, fidelity_acc_gnn, fidelity_prob_gnn = recourse_method.get_counterfactuals(test_factual)


        print(f"{df_cfs_gnn=}")
        print(f"{num_cf_gnn=}")
        print(f"{sparsity_gnn=}")
        print(f"{validity_gnn=}")
        print(f"{fidelity_acc_gnn=}")
        print(f"{fidelity_prob_gnn=}")

    else:
        recourse_method = recourse_catalog.CFNodeExplainer(
            mlmodel=ml_model, data=dataset, hyperparams=hyper_gin
            )

        df_cfs, num_cf, validity, sparsity, fidelity = recourse_method.get_counterfactuals(test_factual)
        
    validity_list.append(validity)
    sparsity_list.append(sparsity)
    fidelity_list.append(fidelity)

mean_validity = np.mean(validity_list)
mean_sparsity = np.mean(sparsity_list)
mean_fidelity = np.mean(fidelity_list)

std_dev_validity = np.std(validity_list, ddof=1)
std_dev_sparsity = np.std(sparsity_list, ddof=1)
std_dev_fidelity = np.std(fidelity_list, ddof=1)

print(f"{mean_validity=}" )
print(f"{mean_sparsity=}" )
print(f"{mean_fidelity=}" )

print(f"{std_dev_validity=}" )
print(f"{std_dev_sparsity=}" )
print(f"{std_dev_fidelity=}" )

        # print(f"{df_cfs=}")
        # print(f"{num_cf=}")
        # print(f"{sparsity=}")
        # print(f"{validity=}")
        # print(f"{fidelity=}")

