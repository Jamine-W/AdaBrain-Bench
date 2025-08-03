from pathlib import Path
import mne
import glob
import scipy.io as sio
import pdb
import pickle
import numpy as np
import os
import sys


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
raw_data_path = os.path.join(data_root,'SEED/raw_data/Preprocessed_EEG')
processed_data_path = os.path.join(data_root,'SEED/processed_data')
os.makedirs(processed_data_path, exist_ok=True)



# data_folder = './Preprocessing/SEED/raw_data/Preprocessed_EEG'
# savePath = "./Preprocessing/SEED/processed_data"
# os.makedirs(savePath, exist_ok=True)


rawDataPath = Path(raw_data_path)
group = rawDataPath.glob('*.mat')
group = [path for path in group if 'label' not in path.name]
sorted_group = sorted(group, key=lambda x: x.name)


labelpath = f"{raw_data_path}/label.mat"
labels = sio.loadmat(labelpath)
labels = labels["label"]
labels = labels.flatten().tolist()
# [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

l_freq = 0.1
h_freq = 75.0
rsfreq = 200

num_files = 0

for n, cntFile in enumerate(sorted_group):
    print(f'processing file {n + 1}/{len(sorted_group)}')
    data = sio.loadmat(cntFile)
    keys = [key for key in data.keys() if "eeg" in key]
    for i, key in enumerate(keys):
        print(f'  processing trial {i + 1}/{len(keys)}')
        raw = data[key]
        raw = mne.filter.filter_data(raw, 200.0, l_freq, h_freq)
        raw = mne.filter.notch_filter(raw, Fs=200.0, freqs=50.0)
        # raw = mne.filter.resample(raw, up=200, n_jobs=5)
        per_label = labels[i]

        os.makedirs(f"{processed_data_path}/{int(n // 3 + 1)}", exist_ok=True)

        for j in range((raw.shape[-1] // (1 * 200)) - 1):
            per_data = raw[:, j * 1 * 200: (j + 1) * 1 * 200]
            per_path = f"{processed_data_path}/{int(n // 3 + 1)}/S{int(n // 3 + 1)}_{int(n % 3 + 1)}_{i + 1}_{j + 1}.pkl"
            pickle.dump(
                {"X": per_data, "Y" : int(per_label)},
                open(per_path, "wb"),
            )
            num_files += 1
            print(per_path, " saved")

