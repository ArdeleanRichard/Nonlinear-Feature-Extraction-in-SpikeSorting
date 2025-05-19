import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import spikeinterface.full as si
import probeinterface as pi

import warnings
warnings.simplefilter("ignore")

base_folder = Path('../DATA/Real/')
recording_file = 'cambridge_data.bin'
oe_folder = base_folder / recording_file


num_channels = 64
sampling_frequency = 20000
gain_to_uV = 0.195
offset_to_uV = 0
dtype="int16"
time_axis = 1

recording = si.read_binary(oe_folder, num_channels=num_channels, sampling_frequency=sampling_frequency,
                           dtype=dtype, gain_to_uV=gain_to_uV, offset_to_uV=offset_to_uV,
                           time_axis=time_axis)

print(recording)
print(recording.channel_ids)
print(si.recording_extractor_full_dict)

fs = recording.get_sampling_frequency()
trace_snippet = recording.get_traces(start_frame=int(fs*0), end_frame=int(fs*2))

print('Traces shape:', trace_snippet.shape)

manufacturer = 'cambridgeneurotech'
probe_name = 'ASSY-156-P-1'

probe = pi.get_probe(manufacturer, probe_name)
probe.wiring_to_device('ASSY-156>RHD2164')

raw_rec = recording.set_probe(probe)


recording_f = si.bandpass_filter(raw_rec, freq_min=300, freq_max=9000)
bad_channel_ids, channel_labels = si.detect_bad_channels(recording_f, method='coherence+psd')
print('bad_channel_ids', bad_channel_ids)
print('channel_labels', channel_labels)

recording_good_channels_f = recording_f.remove_channels(bad_channel_ids)
recording_good_channels = si.common_reference(recording_good_channels_f, reference='global', operator='median')

print(recording_good_channels)
print(recording_good_channels.channel_ids)

print("\n\n\n")
trace_snippet = recording_f.get_traces(start_frame=int(fs*0), end_frame=int(fs*300))
print('Traces shape:', trace_snippet.shape)

trace_channel = trace_snippet[:, 1]


def split_consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def threshold_signal_by_std_dev(data, multiplier=4):
    AMP_THR = -1 * multiplier * np.std(data)
    all_timestamps = np.where(data < AMP_THR)[0]

    consecutive_timestamp_groups = split_consecutive(all_timestamps)

    timestamps = []
    for consecutive_ts in consecutive_timestamp_groups:
        timestamps.append(consecutive_ts[0] + np.argmin(data[consecutive_ts]))

    return np.array(timestamps)

timestamps = threshold_signal_by_std_dev(trace_channel, multiplier=5)
print(timestamps.shape)
print(timestamps)

WAVEFORM_ALIGNMENT=20
WAVEFORM_LENGTH=60
waveforms = []
for ts in timestamps:
    waveform = trace_channel[ts - WAVEFORM_ALIGNMENT:ts + (WAVEFORM_LENGTH - WAVEFORM_ALIGNMENT)]
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

np.save('../DATA/Real/cambridge_data_waveforms.npy', waveforms)