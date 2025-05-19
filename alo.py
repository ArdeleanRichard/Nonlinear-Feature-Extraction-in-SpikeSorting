import numpy as np
from matplotlib import pyplot as plt

path = "../DATA/Real/ec012ec.187/ec012ec.11/ec012ec.187/ec012ec.187.spk.1"

data = np.fromfile(path, dtype=np.int16, count=-1, sep='', offset=0)
print(data.shape)

data = data.reshape(89148, -1)
print(data.shape)

waveforms = []
for x in data:
    if np.any(x < -1000):
        continue
    waveforms.append(x)
waveforms = np.array(waveforms)



def plot_spikes(title, spikes, limit=False, mean=False, color='blue', amp_thr=False, AMP_THR=None, save=False, FIG_PATH=None):
    plt.title(title)
    for spike in spikes:
        if mean == True:
            plt.plot(spike, 'gray')
        else:
            plt.plot(spike)
    if mean == True:
        plt.plot(np.mean(spikes, axis=0), color)
    if limit == True:
        plt.ylim(-220, 120)
    if amp_thr == True:
        plt.axhline(y=AMP_THR, color='r', label='')
    plt.ylabel("Voltage [mV]")
    plt.xlabel("Time [samples]")
    if save == True:
        plt.savefig(FIG_PATH + f"{title}.png")
        plt.savefig(FIG_PATH + f"{title}.svg")
    plt.show()

plot_spikes("test", waveforms)

np.save('../DATA/Real/waveforms_ec012ec.npy', waveforms)