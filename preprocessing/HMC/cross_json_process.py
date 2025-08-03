import json
import os
import pickle
from natsort import natsorted
import numpy as np
import random
import sys


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
processed_data_path = os.path.join(data_root,'HMC/processed_data')
data_split_path = './preprocessing/HMC/cross_subject_json'
os.makedirs(data_split_path, exist_ok=True)
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')

# data_folder = "./Preprocessing/HMC/processed"
# os.makedirs(os.path.dirname('./Preprocessing/HMC/cross_subject_json'), exist_ok=True)
# save_folder_train = './Preprocessing/HMC/cross_subject_json/train.json'
# save_folder_test = './Preprocessing/HMC/cross_subject_json/test.json'
# save_folder_val = './Preprocessing/HMC/cross_subject_json/val.json'

random.seed(42)
sampling_rate = 256
ch_names = ["F4", "C4", "O2", "C3"]
num_channels = len(ch_names)
total_mean = np.zeros(num_channels)
total_std = np.zeros(num_channels)
num_all = 0
max_value = -1
min_value = 1e6

tuples_list_train = []
tuples_list_test = []
tuples_list_val = []
error_list = []

subject_folder = [os.path.join(processed_data_path, f) for f in os.listdir(processed_data_path)]
subject_folder = natsorted(subject_folder)
random.shuffle(subject_folder)

# Divide into 80% training, 10% validation, and 10% test sets
train_size = int(0.8 * len(subject_folder))
val_size = int(0.1 * len(subject_folder))
test_size = len(subject_folder) - train_size - val_size
train_folders = subject_folder[:train_size]
val_folders = subject_folder[train_size:train_size + val_size]
test_folders = subject_folder[train_size + val_size:]

for folder_id, per_folder in enumerate(subject_folder):
    print("Processing ", folder_id, '/', len(subject_folder) - 1)
    pkl_files = [os.path.join(per_folder, f) for f in os.listdir(per_folder) if f.endswith('.pkl')]
    pkl_files = natsorted(pkl_files)

    for pkl_id, pkl_file in enumerate(pkl_files):
        subject_name = pkl_file.split("/")[-1].split(".")[0].split("_")[0]

        try:
            eeg_data = pickle.load(open(pkl_file, "rb"))
            eeg = eeg_data['X']
            label = eeg_data['Y']
        except Exception as e:
            print(f"Failed to load the pickle file: {pkl_file}: {e}")
            error_list.append(pkl_file)
            continue
            
        data = {
            "subject_id": int(per_folder.split("/")[-1]),
            "subject_name": subject_name,
            "file": pkl_file,
            "label": label
        }

        if per_folder in train_folders:
            if folder_id < len(train_folders):
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
        elif per_folder in val_folders:
            tuples_list_val.append(data)
        elif per_folder in test_folders:
            tuples_list_test.append(data)

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

val_dataset = {
    "subject_data": tuples_list_val,
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

formatted_json_test = json.dumps(test_dataset, indent=2)
with open(save_test_path, 'w') as f:
    f.write(formatted_json_test)

formatted_json_val = json.dumps(val_dataset, indent=2)
with open(save_val_path, 'w') as f:
    f.write(formatted_json_val)

print("Wrong file list: ", error_list)
