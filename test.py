# import json
# import spikeforest as sf
#
#
# def main():
#     hybrid_janelia_uri = 'sha1://43298d72b2d0860ae45fc9b0864137a976cb76e8?hybrid-janelia-spikeforest-recordings.json'
#     synth_monotrode_uri = 'sha1://3b265eced5640c146d24a3d39719409cceccc45b?synth-monotrode-spikeforest-recordings.json'
#     paired_boyden_uri = 'sha1://849e53560c9241c1206a82cfb8718880fc1c6038?paired-boyden-spikeforest-recordings.json'
#     paired_kampff_uri = 'sha1://b8b571d001f9a531040e79165e8f492d758ec5e0?paired-kampff-spikeforest-recordings.json'
#     paired_english_uri = 'sha1://dfb1fd134bfc209ece21fd5f8eefa992f49e8962?paired-english-spikeforest-recordings.json'
#
#     # the default URI includes the PAIRED_BOYDEN, PAIRED_CRCNS_HC1,
#     # PAIRED_ENGLISH, PAIRED_KAMPFF, and PAIRED_MEA64C_YGER recordings.
#     all_recordings = sf.load_spikeforest_recordings(hybrid_janelia_uri)
#
#     # Other recording sets are being migrated to the new data distribution protocol as needed.
#     # E.G. to load the Hybrid Janelia data set, use the following:
#     # all_recordings = sf.load_spikeforest_recordings(hybrid_janelia_uri)
#
#     for R in all_recordings:
#         print('=========================================================')
#         print(f'{R.study_set_name}/{R.study_name}/{R.recording_name}')
#         print(f'Num. channels: {R.num_channels}')
#         print(f'Duration (sec): {R.duration_sec}')
#         print(f'Sampling frequency (Hz): {R.sampling_frequency}')
#         print(f'Num. true units: {R.num_true_units}')
#         print(f'Sorting true object: {json.dumps(R.sorting_true_object)}')
#         print('')
#
# if __name__ == '__main__':
#     main()



# import spikeforest as sf
#
#
# def main():
#     R = sf.load_spikeforest_recording(study_name='paired_boyden32c', recording_name='1103_1_1', uri=None)
#     print(f'{R.study_set_name}/{R.study_name}/{R.recording_name}')
#     print(f'Num. channels: {R.num_channels}')
#     print(f'Duration (sec): {R.duration_sec}')
#     print(f'Sampling frequency (Hz): {R.sampling_frequency}')
#     print(f'Num. true units: {R.num_true_units}')
#     print('')
#
#     recording = R.get_recording_extractor()
#     sorting_true = R.get_sorting_true_extractor()
#
#     print(f'Recording extractor info: {recording.get_num_channels()} channels, {recording.get_sampling_frequency()} Hz, {recording.get_total_duration()} sec')
#     print(f'Sorting extractor info: unit ids = {sorting_true.get_unit_ids()}, {sorting_true.get_sampling_frequency()} Hz')
#     print('')
#     for unit_id in sorting_true.get_unit_ids():
#         st = sorting_true.get_unit_spike_train(unit_id=unit_id)
#         print(f'Unit {unit_id}: {len(st)} events')
#     print('')
#     print('Channel locations:')
#     print('X:', recording.get_channel_locations()[:, 0].T)
#     print('Y:', recording.get_channel_locations()[:, 1].T)
#
#
#
#     X = sf.load_spikeforest_sorting_output(study_name='paired_boyden32c', recording_name='1103_1_1', sorter_name='SpykingCircus')
#     print('=========================================================')
#     print(f'{X.study_name}/{X.recording_name}/{X.sorter_name}')
#     print(f'CPU time (sec): {X.cpu_time_sec}')
#     print(f'Return code: {X.return_code}')
#     print(f'Timed out: {X.timed_out}')
#     print(f'Sorting true object: {json.dumps(X.sorting_object)}')
#     print('')
#
#     sorting = X.get_sorting_extractor()
#
#     print(f'Sorting extractor info: unit ids = {sorting.get_unit_ids()}, {sorting.get_sampling_frequency()} Hz')
#     print('')
#     for unit_id in sorting.get_unit_ids():
#         st = sorting.get_unit_spike_train(unit_id=unit_id)
#         print(f'Unit {unit_id}: {len(st)} events')
#     print('')
#
# if __name__ == '__main__':
#     main()


# # load_and_extract.py
# import spikeforest as sf                               # SpikeForest loader :contentReference[oaicite:5]{index=5}
# from spikeinterface import extract_waveforms           # Waveform extraction :contentReference[oaicite:6]{index=6}
#
#
#
# def main():
#     # 1. Load ground‑truth recording
#     R = sf.load_spikeforest_recording(
#         study_name='paired_boyden32c',   # replace as desired
#         recording_name='1103_1_1',
#         uri=None
#     )
#     rec = R.get_recording_extractor()
#     sort = R.get_sorting_true_extractor()
#
#     # # 2. Extract waveforms
#     # we = extract_waveforms(
#     #     recording=rec,
#     #     sorting=sort,
#     #     folder='wf_cache',
#     #     ms_before=1.0,
#     #     ms_after=2.0,
#     #     max_spikes_per_unit=500,
#     #     load_if_exists=None
#     # )
#     # print("Waveforms extracted and cached in:", we.folder)
#
#     from spikeinterface.core import load_waveforms
#
#     # Rebuild a MockWaveformExtractor from your cache folder
#     we = load_waveforms(folder='wf_cache')
#
#     unit_ids = we.sorting.unit_ids  # list of unit IDs
#     channel_ids = we.recording.channel_ids  # list of channel indices
#     print("Units:", unit_ids)
#     print("Channels:", channel_ids)
#
#     for unit_id in unit_ids:
#         wfs = we.get_waveforms(unit_id)
#
#         # Compute the average template
#         template = we.get_template(unit_id, mode='average')
#
#         print(f"Unit {unit_id}: {wfs.shape[0]} spikes, each {wfs.shape[2]} samples long")
#
# if __name__ == '__main__':
#     main()


import spikeinterface.extractors as se
import kachery_cloud as kcl

# Define the URI for the dataset
uri = 'sha1://b5b3e1e2c3f4d5a6b7c8d9e0f1a2b3c4d5e6f7g8/1103_1_1.json'  # Replace with the actual URI

import spikeforest as sf
recording = sf.load_spikeforest_recording(
    study_name='paired_boyden32c',   # replace as desired
    recording_name='1103_1_1',
    uri=None
)

# Bandpass filter the recording
recording_f = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)

# Detect peaks
peaks = spd.detect_peaks(
    recording=recording_f,
    method='by_channel',
    peak_sign='neg',
    detect_threshold=5,
    n_shifts=2,
    chunk_size=10000,
    chunk_memory="200MB",
    verbose=True
)

# Convert peaks to a Sorting object
sorting = si.sortingcomponents.from_peaks(
    peaks=peaks,
    sampling_frequency=recording.get_sampling_frequency()
)

# Extract waveforms
we_folder = "unsorted_waveforms"
we = extract_waveforms(
    recording=recording,
    sorting=sorting,
    folder=we_folder,
    ms_before=1.0,
    ms_after=2.0,
    max_spikes_per_unit=None,
    load_if_exists=False
)
