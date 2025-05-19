import numpy as np
from scipy.signal import butter, filtfilt

# meta = np.load("../DATA/KAMPFF/c16/c16_expt_meta.npy", allow_pickle=True)
# print(meta)
#
# timestamps = np.load("../DATA/KAMPFF/c16/c16_extracellular_spikes.npy")
# print(timestamps.shape)
# print(timestamps[:10])
#
#
# n_channels, n_samples = 384, 20250849
# signal = np.fromfile("../DATA/KAMPFF/c16/c16_npx_raw.bin", dtype=np.int16)
# print(signal.shape)

cell_value=28
npx_path = f"../DATA/KAMPFF/c{cell_value}/c{cell_value}_npx_raw.bin"
npx_recording = np.memmap( npx_path, mode = 'r', dtype=np.int16, order = 'C')

npx_channels = 384
npx_samples =int(len(npx_recording)/npx_channels)
print(npx_samples)

npx_recording = npx_recording.reshape((npx_channels, npx_samples), order = 'F')
print(npx_recording.shape)

# timestamps_intra = np.load("../DATA/KAMPFF/c16/c16_extracellular_spikes.npy")


def bandpass_filter(data, fs, lowcut=300, highcut=3000, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def estimate_noise_sigma(filtered):
    mad = np.median(np.abs(filtered))
    return mad / 0.6745

def detect_spikes(filtered, sigma, k=4, fs=30000, refractory_ms=1.0):
    thr = -k * sigma
    crossings = np.where(filtered < thr)[0]
    # enforce refractory period (in samples)
    refractory_samples = int(refractory_ms * 1e-3 * fs)
    spike_times = []
    last_spike = -np.inf
    for idx in crossings:
        if idx - last_spike > refractory_samples:
            spike_times.append(idx)
            last_spike = idx
    return np.array(spike_times)

for channel_nr in range(npx_channels):
    channel = npx_recording[channel_nr]
    fs=30000
    filtered = bandpass_filter(channel, fs)
    sigma = estimate_noise_sigma(filtered)
    spike_idx = detect_spikes(filtered, sigma, k=4, fs=fs)
    print(channel_nr, len(spike_idx))

    waveforms = []
    for ts in spike_idx[:-1]:
        waveforms.append(channel[ts-20:ts+40])
    waveforms = np.array(waveforms)

    import matplotlib.pyplot as plt

    for wf in waveforms:
        plt.plot(wf)

    plt.savefig(f"./figures/kampff/kampff_c{cell_value}_channel{channel_nr}.png")
    plt.close()

    import matplotlib.pyplot as plt

    rand_idx = np.random.randint(0, len(spike_idx), 20)
    for wf in waveforms[rand_idx]:
        plt.plot(wf)

    plt.savefig(f"./figures/kampff/kampff_c{cell_value}_channel{channel_nr}_randoms.png")


    # save to 'matrix.npy'
    np.save(f'../DATA/KAMPFF/kampff_c{cell_value}_channel{channel_nr}.npy', waveforms)