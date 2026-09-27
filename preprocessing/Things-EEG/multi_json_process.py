import json
import os
import pickle
from natsort import natsorted
import numpy as np
import random
import sys


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
processed_data_path = os.path.join(data_root,'Things-EEG/processed_data/subjects_data')
data_split_path = './preprocessing/Things-EEG/multi_subject_json'
os.makedirs(data_split_path, exist_ok=True)
save_train_path = os.path.join(data_split_path, 'train.json')
save_test_path = os.path.join(data_split_path, 'test.json')

sampling_rate = 250
ch_names = [
        'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
        'O1', 'Oz', 'O2',
    ]
num_channels = len(ch_names)
total_mean = np.zeros(num_channels)
total_std = np.zeros(num_channels)
num_all = 0
max_value = -1
min_value = 1e6

tuples_list_train = []
tuples_list_test = []
error_list = []

data_folder_train = os.path.join(processed_data_path, 'train')
data_folder_test = os.path.join(processed_data_path, 'test')
train_folders = [os.path.join(data_folder_train, f) for f in os.listdir(data_folder_train)]
train_folders = natsorted(train_folders)
test_folders = [os.path.join(data_folder_test, f) for f in os.listdir(data_folder_test)]
test_folders = natsorted(test_folders)

# training set
for folder_id, per_folder in enumerate(train_folders):
    pkl_files = [os.path.join(per_folder, f) for f in os.listdir(per_folder) if f.endswith('.pkl')]
    pkl_files = natsorted(pkl_files)

    for pkl_id, pkl_file in enumerate(pkl_files):
        print(f"Processing train_sub {folder_id + 1}/{len(train_folders)}, pkl_file {pkl_id + 1} / {len(pkl_files)}...")
        try:
            eeg_data = pickle.load(open(pkl_file, "rb"))
            eeg = eeg_data['X']
            label = int(eeg_data['Y'])
        except Exception as e:
            print(f"Error loading file {pkl_file}: {e}")
            error_list.append(pkl_file)
            continue

        data = {
            "subject_id": folder_id + 1,
            "EEG": pkl_file,
            "label": label
        }

        per_max_value = max(eeg.reshape(-1))
        if per_max_value > max_value:
            max_value = per_max_value
        per_min_value = min(eeg.reshape(-1))
        if per_min_value < min_value:
            min_value = per_min_value
        for j in range(num_channels):
            total_mean[j] += eeg[j].mean()
            total_std[j] += eeg[j].std()
        num_all += 1
        tuples_list_train.append(data)

data_mean = (total_mean / num_all).tolist()
data_std = (total_std / num_all).tolist()

train_dataset = {
    "subject_data": tuples_list_train,
    "dataset_info": {
        "sampling_rate": sampling_rate,
        "ch_names": ch_names,
        "min": min_value,
        "max": max_value,
        "mean": data_mean,
        "std": data_std
    }
}
formatted_json_train = json.dumps(train_dataset, indent=2)
with open(save_train_path, 'w') as f:
    f.write(formatted_json_train)

# test set
for folder_id, per_folder in enumerate(test_folders):
    print(f"Processing test {folder_id + 1}/{len(test_folders)}...")
    pkl_files = [os.path.join(per_folder, f) for f in os.listdir(per_folder) if f.endswith('.pkl')]
    pkl_files = natsorted(pkl_files)

    for pkl_id, pkl_file in enumerate(pkl_files):
        print(f"Processing test_sub {folder_id + 1}/{len(test_folders)}, pkl_file {pkl_id + 1} / {len(pkl_files)}...")
        try:
            eeg_data = pickle.load(open(pkl_file, "rb"))
            label = int(eeg_data['Y'])
        except Exception as e:
            print(f"Error loading file {pkl_file}: {e}")
            error_list.append(pkl_file)
            # pdb.set_trace()
            continue

        data = {
            "subject_id": folder_id + 1,
            "EEG": pkl_file,
            "label": label
        }
        tuples_list_test.append(data)

test_dataset = {
    "subject_data": tuples_list_test,
    "dataset_info": {
        "sampling_rate": sampling_rate,
        "ch_names": ch_names,
        "min": min_value,
        "max": max_value,
        "mean": data_mean,
        "std": data_std
    }
}

formatted_json_test = json.dumps(test_dataset, indent=2)
with open(save_test_path, 'w') as f:
    f.write(formatted_json_test)

print("Wrong file list: ", error_list)
