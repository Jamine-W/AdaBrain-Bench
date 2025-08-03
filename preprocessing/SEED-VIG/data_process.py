import h5py
import scipy
from scipy import signal
import os
import lmdb
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, resample, filtfilt, iirnotch
import scipy.io
from collections import defaultdict
import sys


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
raw_data_path = os.path.join(data_root,'SEED-VIG/raw_data/Raw_Data')
processed_data_path = os.path.join(data_root,'SEED-VIG/processed_data')
label_path = os.path.join(data_root,'SEED-VIG/raw_data/perclos_labels')
os.makedirs(processed_data_path, exist_ok=True)

# data_dir = './Preprocessing/SEED-VIG/raw_data/Raw_Data'
# labels_dir = './Preprocessing/SEED-VIG/raw_data/perclos_labels'
# output_root = "./Preprocessing/SEED-VIG/processed_data"

def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch_filter(freq=50.0, fs=200, Q=30.0):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = signal.iirnotch(w0, Q)
    return b, a

def get_number(filename):
    # Capture digits prior to the first '_' character
    num_str = filename.split('_')[0]
    return int(num_str)


files = [file for file in os.listdir(raw_data_path)]
files = sorted(files, key=get_number)
print(files)


def process_and_save_subject_data(files, raw_data_path, label_path, processed_data_path):
    """Process all subject data and merge multiple files from the same subject."""
    from collections import defaultdict

    # Group the files by subject_id
    subject_files = defaultdict(list)
    for file in files:
        subject_id = file.split('_')[0]
        subject_files[subject_id].append(file)
    print(subject_files)

    # Process each subject's files.
    for subject_id, files in subject_files.items():
        print(subject_id)
        all_eeg = []
        all_labels = []

        for file in files:
            eeg = scipy.io.loadmat(os.path.join(raw_data_path, file))['EEG'][0][0][0]  # (1416000,17)
            labels = scipy.io.loadmat(os.path.join(label_path, file))['perclos'][:, 0]  # (885,)

            b, a = butter_bandpass(0.1, 75, 200)
            eeg = filtfilt(b, a, eeg, axis=0)
            notch_b, notch_a = notch_filter(freq=50.0, fs=200, Q=30)
            eeg = filtfilt(notch_b, notch_a, eeg, axis=0)

            eeg = eeg.reshape(885, 1600, 17).transpose(0, 2, 1)  # (885,17,1600)
            print(eeg.shape, labels.shape)
            all_eeg.append(eeg)
            all_labels.append(labels)

        subject_eeg = np.concatenate(all_eeg, axis=0)  # (total_trials,17,1600)
        subject_labels = np.concatenate(all_labels, axis=0)  # (total_trials,)
        print(subject_eeg.shape, subject_labels.shape)

        save_subject_eeg(subject_eeg, subject_labels, subject_id, processed_data_path)


def save_subject_eeg(eeg_data, labels, subject_id, output_dir):
    subject_dir = os.path.join(output_dir, f"subject_{subject_id}")
    os.makedirs(subject_dir, exist_ok=True)

    for i in range(eeg_data.shape[0]):
        trial_data = {
            'X': eeg_data[i],  # (17,1600)
            'Y': float(labels[i]),
        }

        save_path = os.path.join(subject_dir, f"{subject_id}_{i + 1}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(trial_data, f)


process_and_save_subject_data(files, raw_data_path, label_path, processed_data_path)
