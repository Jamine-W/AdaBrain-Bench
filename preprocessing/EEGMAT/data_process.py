import os
import mne
import numpy as np
import pickle
from scipy.signal import iirnotch, filtfilt
import sys

data_root = sys.argv[1]  
print(f"Data root: {data_root}")


raw_data_path = os.path.join(data_root,'EEGMAT/raw_data')
processed_data_path = os.path.join(data_root,'EEGMAT/processed_data')
os.makedirs(processed_data_path, exist_ok=True)

retain_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
segment_duration = 4
low_freq, high_freq = 0.1, 75.0
notch_freq = 50.0



def apply_filters(raw, sfreq):
    """Define a bandpass filter (0.1Hz - 75Hz) & a notch filter (50Hz)"""
    raw.filter(low_freq, high_freq, fir_design='firwin', verbose=False)
    notch_freq_ratio = notch_freq / (sfreq / 2)
    b, a = iirnotch(notch_freq_ratio, Q=30)
    raw._data = filtfilt(b, a, raw.get_data(), axis=1)
    return raw


def process_and_save(file_path, subject_id, file_type, label):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    channel_mapping = {ch: ch.replace('EEG ', '') for ch in raw.ch_names}
    raw.rename_channels(channel_mapping)
    raw.pick_channels(retain_channels)
    raw = apply_filters(raw, raw.info['sfreq'])

    # Target signal range
    sfreq = raw.info['sfreq']
    if file_type == "_1":  # The last 60s
        start_idx = int((raw.n_times / sfreq - 60) * sfreq)
        data = raw.get_data(start=start_idx)
    elif file_type == "_2":  # The first 60s
        data = raw.get_data(stop=int(60 * sfreq))

    # Split
    n_points = int(segment_duration * sfreq)  # 4s
    segments = [data[:, i * n_points:(i + 1) * n_points] for i in range(data.shape[1] // n_points)]

    subject_folder = os.path.join(processed_data_path, f"Subject{subject_id:02d}")
    os.makedirs(subject_folder, exist_ok=True)
    for idx, segment in enumerate(segments):
        output_file = os.path.join(subject_folder, f"Subject{subject_id:02d}{file_type}_{idx + 1}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump({"X": segment, "Y": label}, f)


for subject_id in range(36):
    for file_type, label in [("_1", 0), ("_2", 1)]:
        file_name = f"Subject{subject_id:02d}{file_type}.edf"
        file_path = os.path.join(raw_data_path, file_name)
        if os.path.exists(file_path):
            process_and_save(file_path, subject_id, file_type, label)
