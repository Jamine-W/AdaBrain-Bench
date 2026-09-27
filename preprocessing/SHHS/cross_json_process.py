import json
import os
import pickle
import numpy as np
from natsort import natsorted
import random
import sys

data_root = sys.argv[1]  
print(f"Data root: {data_root}")
processed_data_path = os.path.join(data_root,'SHHS/processed_data')
data_split_path = './preprocessing/SHHS/cross_subject_json'
os.makedirs(data_split_path, exist_ok=True)
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')

# data_folder = "./Preprocessing/SHHS/processed_data"
# os.makedirs(os.path.dirname('./Preprocessing/SHHS/cross_subject_json'), exist_ok=True)
# save_folder_train = './Preprocessing/SHHS/cross_subject_json/train.json'
# save_folder_val = './Preprocessing/SHHS/cross_subject_json/val.json'
# save_folder_test = './Preprocessing/SHHS/cross_subject_json/test.json'

random.seed(42)
sampling_rate = 125
ch_names = ['C4']
num_channels = len(ch_names)
max_value = -1
min_value = 1e6

subject_folders = [os.path.join(processed_data_path, f) for f in os.listdir(processed_data_path) if f.startswith("shhs")]
subject_folders = natsorted(subject_folders)
random.shuffle(subject_folders)

# Divide into 80% training, 10% validation, and 10% test sets
total_subjects = len(subject_folders)
num_train = int(total_subjects * 0.8)
num_val = int(total_subjects * 0.1)
num_test = total_subjects - num_train - num_val
train_subjects = subject_folders[:num_train]
val_subjects = subject_folders[num_train:num_train + num_val]
test_subjects = subject_folders[num_train + num_val:]

tuples_list_train = []
tuples_list_val = []
tuples_list_test = []
error_list = []

num_samples = 0
data_mean = np.zeros(1)
data_std = np.zeros(1)
for subject_folder in train_subjects:
    eeg_files = [os.path.join(subject_folder, f) for f in os.listdir(subject_folder) if f.endswith('.pkl')]
    eeg_files = natsorted(eeg_files)

    for file in eeg_files:
        eeg_data = pickle.load(open(file, "rb"))
        eeg = eeg_data['X']
        label = eeg_data['Y']
        eeg_expanded = np.expand_dims(eeg, axis=0)  # Expand dimensions by prepending a new axis to axis 0

        per_max_value = max(eeg.reshape(-1))
        if per_max_value > max_value:
            max_value = per_max_value
        per_min_value = min(eeg.reshape(-1))
        if per_min_value < min_value:
            min_value = per_min_value

        data_mean += eeg_expanded[0].mean()
        data_std += eeg_expanded[0].std()
        num_samples += 1

data_mean /= num_samples
data_std /= num_samples

for subject_folder in train_subjects:
    subject_name = os.path.basename(subject_folder)
    eeg_files = [os.path.join(subject_folder, f) for f in os.listdir(subject_folder) if f.endswith('.pkl')]
    eeg_files = natsorted(eeg_files)

    for file in eeg_files:
        eeg_data = pickle.load(open(file, "rb"))
        eeg = eeg_data['X']
        label = eeg_data['Y']
        data = {
            "subject_id": subject_folders.index(subject_folder),
            "subject_name": subject_name,
            "file": file,
            "label": label
        }
        tuples_list_train.append(data)

for subject_folder in val_subjects:
    subject_name = os.path.basename(subject_folder)
    eeg_files = [os.path.join(subject_folder, f) for f in os.listdir(subject_folder) if f.endswith('.pkl')]
    eeg_files = natsorted(eeg_files)

    for file in eeg_files:
        eeg_data = pickle.load(open(file, "rb"))
        eeg = eeg_data['X']
        label = eeg_data['Y']
        data = {
            "subject_id": subject_folders.index(subject_folder),
            "subject_name": subject_name,
            "file": file,
            "label": label
        }
        tuples_list_val.append(data)

for subject_folder in test_subjects:
    subject_name = os.path.basename(subject_folder)
    eeg_files = [os.path.join(subject_folder, f) for f in os.listdir(subject_folder) if f.endswith('.pkl')]
    eeg_files = natsorted(eeg_files)

    for file in eeg_files:
        eeg_data = pickle.load(open(file, "rb"))
        eeg = eeg_data['X']
        label = eeg_data['Y']
        data = {
            "subject_id": subject_folders.index(subject_folder),
            "subject_name": subject_name,
            "file": file,
            "label": label
        }
        tuples_list_test.append(data)

def build_dataset(data_list):
    return {
        "subject_data": data_list,
        "dataset_info": {
            "sampling_rate": sampling_rate,
            "ch_names": ch_names,
            "min": min_value,
            "max": max_value,
            "mean": data_mean,
            "std": data_std
        }
    }


train_dataset = build_dataset(tuples_list_train)
val_dataset = build_dataset(tuples_list_val)
test_dataset = build_dataset(tuples_list_test)


def convert_data_types(data):
    if isinstance(data, dict):
        return {k: convert_data_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_data_types(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.int32):
        return int(data)
    elif isinstance(data, np.float32):
        return float(data)
    else:
        return data


def save_dataset(dataset, file_path):
    dataset = convert_data_types(dataset)
    formatted_json = json.dumps(dataset, indent=2)
    with open(file_path, 'w') as f:
        f.write(formatted_json)


save_dataset(train_dataset, save_train_path)
save_dataset(val_dataset, save_val_path)
save_dataset(test_dataset, save_test_path)

print("Error list: ", error_list)
print(f"Total subjects: {total_subjects}")
print(f"Train subjects: {len(train_subjects)}")
print(f"Validation subjects: {len(val_subjects)}")
print(f"Test subjects: {len(test_subjects)}")
