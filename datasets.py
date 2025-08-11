import numpy as np
from dataset_parsing import simulations_dataset as ds
from dataset_parsing.read_kampff import read_kampff_c37, read_kampff_c28
from dataset_parsing.realdata_parsing import read_timestamps, read_waveforms
from dataset_parsing.realdata_ssd import find_ssd_files, separate_by_unit, units_by_channel
from dataset_parsing.realdata_ssd_1electrode import parse_ssd_file


def load_all_data():
    datasets = []
    for simulation_number in range(1,96):
        if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
            continue
        datasets.append((f"Sim{simulation_number}", ds.get_dataset_simulation(simNr=simulation_number)))

    return datasets

def read_m045():
    DATASET_PATH = "../DATA/TINS/M045_0005/"
    spikes_per_unit, unit_electrode = parse_ssd_file(DATASET_PATH)
    WAVEFORM_LENGTH = 58

    timestamp_file, waveform_file, _, _ = find_ssd_files(DATASET_PATH)

    timestamps = read_timestamps(timestamp_file)
    timestamps_by_unit = separate_by_unit(spikes_per_unit, timestamps, 1)

    waveforms = read_waveforms(waveform_file)
    waveforms_by_unit = separate_by_unit(spikes_per_unit, waveforms, WAVEFORM_LENGTH)

    units_in_channels, labels = units_by_channel(unit_electrode, waveforms_by_unit, data_length=WAVEFORM_LENGTH, number_of_channels=33)

    return units_in_channels, labels

def load_real_data():
    datasets = []

    X, y, y_true = read_kampff_c28()
    datasets.append((f"kampff_c28", (np.array(X[0]), np.array(y[0]), np.array(y_true))))
    X, y, y_true = read_kampff_c37()
    datasets.append((f"kampff_c37", (np.array(X[0]), np.array(y[0]), np.array(y_true))))

    return datasets

if __name__ == "__main__":
    for channel in read_m045():
        channel = np.array(channel)
        print(channel.shape)