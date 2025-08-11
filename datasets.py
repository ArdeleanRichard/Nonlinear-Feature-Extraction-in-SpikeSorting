import numpy as np
from dataset_parsing import simulations_dataset as ds
from dataset_parsing.read_kampff import read_kampff_c37, read_kampff_c28

def load_all_data():
    datasets = []
    for simulation_number in range(1,96):
        if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
            continue
        datasets.append((f"Sim{simulation_number}", ds.get_dataset_simulation(simNr=simulation_number)))

    return datasets


def load_real_data():
    datasets = []

    X, y, y_true = read_kampff_c28()
    datasets.append((f"kampff_c28", (np.array(X[0]), np.array(y[0]), np.array(y_true))))
    X, y, y_true = read_kampff_c37()
    datasets.append((f"kampff_c37", (np.array(X[0]), np.array(y[0]), np.array(y_true))))

    return datasets

if __name__ == "__main__":
    load_real_data()