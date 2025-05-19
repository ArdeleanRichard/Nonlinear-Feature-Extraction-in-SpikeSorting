# import numpy as np
# from matplotlib import pyplot as plt
#
# threshold = 10
# juxta_channel = 254
# min_peak = -89.5428
# channel = 128
# padding = 1867
#
# offset = 1867  # Replace with the actual offset value from the .txt file
# data = np.memmap('../DATA/Real/20160426_patch3/patch_3.raw', dtype='uint16', offset=offset, mode='r')
# data = data.reshape(-1, 256)  # Adjust the number of channels as needed
#
# one_channel = data[:, 0].astype('float32')
#
# #If we want to center data around 0
# one_channel -= 2**15 - 1
#
# #And if we want to display data in micro volt, we must use the gain factor of 0.1042 provided in the header
# one_channel *= 0.1042
# print(one_channel.shape)
# print(one_channel)
#
# triggers = np.load("../DATA/Real/20160426_patch3/patch_3.triggers.npy")
# print(triggers.shape)
# print(triggers)
#


import numpy as np
from matplotlib import pyplot as plt

threshold = 10
juxta_channel = 254
min_peak = -89.5428
channel = 128
offset = 1867


# === User parameters ===
raw_fname      = '../DATA/Real/20160426_patch3/patch_3.raw'         # extracellular data
juxta_fname    = '../DATA/Real/20160426_patch3/patch_3.juxta.raw'   # juxtacellular data
spike_times_fn = '../DATA/Real/20160426_patch3/patch_3.triggers.npy'       # ground-truth spike indices
n_channels     = 256                     # as per Zenodo record
dtype_raw      = np.uint16               # data type in .raw
dtype_juxta    = np.float32              # data type in .juxta.raw
fs             = 30000                   # sampling rate (Hz)

# === 1) Load spike times ===
spike_times = np.load(spike_times_fn)     # array of sample indices :contentReference[oaicite:6]{index=6}

# === 2) Memory‑map extracellular data ===
#    This avoids loading the full file into RAM at once :contentReference[oaicite:7]{index=7}
extracell = np.memmap(raw_fname, dtype=dtype_raw, offset=offset, mode='r')
n_samples = extracell.size // n_channels
extracell = extracell.reshape(n_samples, n_channels)

# === 3) Memory‑map juxtacellular data ===
juxtacell = np.memmap(juxta_fname, dtype=dtype_juxta, mode='r')

# === 4) Plot snippet around the first ground‑truth spike ===
spk_idx = spike_times[100]
t_win   = np.arange(spk_idx-50, spk_idx+150)  # 200-sample window (~6.7s)
fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

# Juxtacellular (ground truth)
axs[0].plot(t_win / fs, juxtacell[t_win], lw=1)
axs[0].set_ylabel('Juxta (mV)')
axs[0].set_title(f'Ground‑truth spike at {spk_idx/fs:.3f}s')

# Extracellular (first channel as example)
axs[1].plot(t_win / fs, extracell[t_win, 100], lw=1)
axs[1].set_ylabel('Extracell Ch1 (uint16)')
axs[1].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()

waveforms = []

WAVEFORM_ALIGNMENT = 20
WAVEFORM_LENGTH = 60
waveforms = []
for ts in spike_times:
    waveform = juxtacell[ts - WAVEFORM_ALIGNMENT:ts + (WAVEFORM_LENGTH - WAVEFORM_ALIGNMENT)]
    if np.any(waveform > 1000):
        continue
    waveforms.append(waveform)
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

np.save('../DATA/Real/waveforms_trial.npy', waveforms)