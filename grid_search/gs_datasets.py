import numpy as np
from dataset_parsing import simulations_dataset as ds

def load_all_data():
    datasets = []
    for simulation_number in range(1, 2):
        if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
            continue
        datasets.append((f"Sim{simulation_number}", ds.get_dataset_simulation(simNr=simulation_number)))

    return datasets


def data_normalisation(X, norm_type=""):
    """Apply different normalization techniques to the data."""
    if norm_type == "":
        return X
    elif norm_type == "minmax":
        from sklearn import preprocessing
        scaler = preprocessing.MinMaxScaler().fit(X)
        X = scaler.transform(X)
        X = np.clip(X, 0, 1)
        return X
    elif norm_type == "standard":
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        return X
    elif norm_type == "robust":
        from sklearn import preprocessing
        scaler = preprocessing.RobustScaler().fit(X)
        X = scaler.transform(X)
        return X
    elif norm_type == "quantile":
        from sklearn import preprocessing
        scaler = preprocessing.QuantileTransformer(output_distribution='normal').fit(X)
        X = scaler.transform(X)
        return X
    elif norm_type == "power":
        from sklearn import preprocessing
        scaler = preprocessing.PowerTransformer().fit(X)
        X = scaler.transform(X)
        return X
    else:
        return None

