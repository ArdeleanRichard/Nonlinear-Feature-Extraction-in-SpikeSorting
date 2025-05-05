import numpy as np
from dataset_parsing import simulations_dataset as ds

def load_all_data():
    datasets = []
    # for simulation_number in range(1,96):
    for simulation_number in [65]:
        if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
            continue
        datasets.append((f"Sim{simulation_number}", ds.get_dataset_simulation(simNr=simulation_number)))

    # datasets = [
    #     ("Sim1", ds.get_dataset_simulation(simNr=1)),
    #     # ("Sim2", ds.get_dataset_simulation(simNr=2)),
    #     # ("Sim4", ds.get_dataset_simulation(simNr=4)),
    #     # ("Sim8", ds.get_dataset_simulation(simNr=8)),
    # ]

    return datasets



