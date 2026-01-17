import json
import os
import pickle
import numpy as np
from natsort import natsorted
import sys

data_root = sys.argv[1]
print(f"Data root: {data_root}")
processed_data_path = os.path.join(data_root, 'TUEV/processed_data/')
data_split_path = './preprocessing/TUEV/cross_subject_json'
os.makedirs(data_split_path, exist_ok=True)
train_folder = os.path.join(processed_data_path, "train")
val_folder = os.path.join(processed_data_path, "eval")
eval_folder = os.path.join(processed_data_path, "test")
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')

sampling_rate = 250
ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
num_channels = len(ch_names)

total_mean = np.zeros(num_channels)
total_std = np.zeros(num_channels)
num_all = 0
max_value = -np.inf
min_value = np.inf


def get_subject_name(file_path):
    base = os.path.basename(file_path)
    if "_" in base:
        return base.split("_")[0]
    return base[:8]


def load_eeg(file_path):
    try:
        with open(file_path, "rb") as f:
            eeg_data = pickle.load(f)
        eeg = eeg_data['X']
        if eeg.shape[0] != num_channels:
            return None
        return eeg_data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


train_files = natsorted([os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.pkl')])
tuples_list_train = []
subject_id_counter = 0
subject_id_map = {}

for file in train_files:
    eeg_data = load_eeg(file)
    if eeg_data is None:
        continue
    label = eeg_data['Y']
    eeg = eeg_data['X']
    subject_name = get_subject_name(file)
    if subject_name not in subject_id_map:
        subject_id_map[subject_name] = subject_id_counter
        subject_id_counter += 1

    total_mean += eeg.mean(axis=1)
    total_std += eeg.std(axis=1)
    num_all += 1
    max_value = max(max_value, eeg.max())
    min_value = min(min_value, eeg.min())

    data = {
        "subject_id": subject_id_map[subject_name],
        "subject_name": subject_name,
        "file": file,
        "label": label
    }
    tuples_list_train.append(data)

if num_all == 0:
    data_mean = total_mean.tolist()
    data_std = total_std.tolist()
else:
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
with open(save_train_path, 'w') as f:
    json.dump(train_dataset, f, indent=2)

val_files = natsorted([os.path.join(val_folder, f) for f in os.listdir(val_folder) if f.endswith('.pkl')])
tuples_list_val = []

for file in val_files:
    eeg_data = load_eeg(file)
    if eeg_data is None:
        continue
    label = eeg_data['Y']
    subject_name = get_subject_name(file)
    if subject_name not in subject_id_map:
        subject_id_map[subject_name] = subject_id_counter
        subject_id_counter += 1
    data = {
        "subject_id": subject_id_map[subject_name],
        "subject_name": subject_name,
        "file": file,
        "label": label
    }
    tuples_list_val.append(data)

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
with open(save_val_path, 'w') as f:
    json.dump(val_dataset, f, indent=2)

eval_files = natsorted([os.path.join(eval_folder, f) for f in os.listdir(eval_folder) if f.endswith('.pkl')])
tuples_list_test = []
error_list = []

for file in eval_files:
    eeg_data = load_eeg(file)
    if eeg_data is None:
        continue
    try:
        label = eeg_data['Y']
        subject_name = get_subject_name(file)
        if subject_name not in subject_id_map:
            subject_id_map[subject_name] = subject_id_counter
            subject_id_counter += 1
        subject_id = subject_id_map[subject_name]
        data = {
            "subject_id": subject_id,
            "subject_name": subject_name,
            "file": file,
            "label": label
        }
        tuples_list_test.append(data)
    except Exception as e:
        print(f"Error loading file {file}: {e}")
        error_list.append(file)

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
with open(save_test_path, 'w') as f:
    json.dump(test_dataset, f, indent=2)

print("error list: ", error_list)
