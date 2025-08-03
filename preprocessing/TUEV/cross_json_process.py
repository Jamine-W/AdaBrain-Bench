import json
import os
import random
import pickle
import numpy as np
from natsort import natsorted
import sys

data_root = sys.argv[1]  
print(f"Data root: {data_root}")
processed_data_path = os.path.join(data_root,'TUEV/processed_data/')
data_split_path = './preprocessing/TUEV/cross_subject_json'
os.makedirs(data_split_path, exist_ok=True)
train_folder = os.path.join(processed_data_path, "train")
val_folder = os.path.join(processed_data_path, "eval")
eval_folder = os.path.join(processed_data_path, "test")
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')

# base_folder = ".Preprocessing/TUEV/processed_data/final_data"
# train_folder = os.path.join(base_folder, "train")
# val_folder = os.path.join(base_folder, "eval")
# eval_folder = os.path.join(base_folder, "test")
# cross_subject_json_folder = ".Preprocessing/TUEV/cross_subject_json"
# os.makedirs(os.path.dirname(cross_subject_json_folder), exist_ok=True)

# save_folder_test = os.path.join(cross_subject_split_folder, 'test.json')
# save_folder_train = os.path.join(cross_subject_split_folder, 'train.json')
# save_folder_val = os.path.join(cross_subject_split_folder, 'val.json')


sampling_rate = 250
ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
num_channels = len(ch_names)
total_mean = np.zeros(num_channels)
total_std = np.zeros(num_channels)
num_all = 0
max_value = -1
min_value = 1e6

def is_23_channels(pkl_file):
    try:
        eeg_data = pickle.load(open(pkl_file, "rb"))
        eeg = eeg_data['X']
        return eeg.shape[0] == 23
    except Exception as e:
        print(f"Error loading file {pkl_file}: {e}")
        return False

train_files = natsorted([os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.pkl')])
tuples_list_train = []
subject_id_counter = 0
subject_id_map = {}

for file in train_files:
    if not is_23_channels(file):
        continue
    try:
        eeg_data = pickle.load(open(file, "rb"))
        label = eeg_data['Y']
        eeg = eeg_data['X']
        subject_folder = os.path.basename(file)[:8]
        if subject_folder not in subject_id_map:
            subject_id_map[subject_folder] = subject_id_counter
            subject_id_counter += 1
        # Calculate normalization parameters.
        for j in range(num_channels):
            total_mean[j] += eeg[j].mean()
            total_std[j] += eeg[j].std()
        num_all += 1
        per_max_value = max(eeg.reshape(-1))
        per_min_value = min(eeg.reshape(-1))
        if per_max_value > max_value:
            max_value = per_max_value
        if per_min_value < min_value:
            min_value = per_min_value
        data = {
            "subject_id": subject_id_map[subject_folder],
            "subject_name": subject_folder,
            "file": file,
            "label": label
        }
        tuples_list_train.append(data)
    except Exception as e:
        print(f"Error loading file {file}: {e}")

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

val_files = natsorted([os.path.join(val_folder, f) for f in os.listdir(val_folder) if f.endswith('.pkl')])
tuples_list_val = []

for file in val_files:
    if not is_23_channels(file):
        continue
    try:
        eeg_data = pickle.load(open(file, "rb"))
        label = eeg_data['Y']
        eeg = eeg_data['X']
        subject_folder = os.path.basename(file)[:8]
        if subject_folder not in subject_id_map:
            subject_id_map[subject_folder] = subject_id_counter
            subject_id_counter += 1
        data = {
            "subject_id": subject_id_map[subject_folder],
            "subject_name": subject_folder,
            "file": file,
            "label": label
        }
        tuples_list_val.append(data)
    except Exception as e:
        print(f"Error loading file {file}: {e}")

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
formatted_json_val = json.dumps(val_dataset, indent=2)
with open(save_val_path, 'w') as f:
    f.write(formatted_json_val)

eval_files = natsorted([os.path.join(eval_folder, f) for f in os.listdir(eval_folder) if f.endswith('.pkl')])
tuples_list_test = []
error_list = []

for file in eval_files:
    if not is_23_channels(file):
        continue
    try:
        data_name = os.path.basename(file).split('_')[1][:3]
        if data_name not in subject_id_map:
            subject_id_map[data_name] = subject_id_counter
            subject_id_counter += 1
        subject_id = subject_id_map[data_name]
        eeg_data = pickle.load(open(file, "rb"))
        label = eeg_data['Y']
        eeg = eeg_data['X']
        data = {
            "subject_id": subject_id,
            "subject_name": data_name,
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
formatted_json_test = json.dumps(test_dataset, indent=2)
with open(save_test_path, 'w') as f:
    f.write(formatted_json_test)

print("error list: ", error_list)
