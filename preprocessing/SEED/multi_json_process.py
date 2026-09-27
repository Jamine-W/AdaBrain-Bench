import json
import pdb
import os
import pickle
from natsort import natsorted
import random
import shutil
import numpy as np
import sys


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
processed_data_path = os.path.join(data_root,'SEED/processed_data')
data_split_path = './preprocessing/SEED/multi_subject_json'
os.makedirs(data_split_path, exist_ok=True)
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')

# data_folder = "./Preprocessing/SEED/processed_data"
# save_folder = "./Preprocessing/SEED/multi_subject_json"
# os.makedirs(save_folder, exist_ok=True)
# save_folder_train = save_folder + '/train.json'
# save_folder_test = save_folder + '/test.json'
# save_folder_val = save_folder + '/val.json'

sampling_rate = 200
ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
num_channels = len(ch_names)
num_labels = 3
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

for folder_id, per_folder in enumerate(subject_folder):
    print("processing ", folder_id, '/', len(subject_folder)-1)
    pkl_files = [os.path.join(per_folder, f) for f in os.listdir(per_folder) if f.endswith('.pkl')]
    pkl_files = natsorted(pkl_files)
    for pkl_id, pkl_file in enumerate(pkl_files):
        subject_name = str(pkl_file.split("/")[-1].split(".")[0].split("_")[0][1:])
        trial_num = int(pkl_file.split("/")[-1].split(".")[0].split("_")[2])
        try:
            eeg_data = pickle.load(open(pkl_file, "rb"))
            eeg = eeg_data['X']
            label = eeg_data['Y']
        except Exception as e:
            print(f"Error loading file {pkl_file}: {e}")
            error_list.append(pkl_file)

        data={
            "subject_id": int(per_folder.split("/")[-1]),
            "subject_name": subject_name,
            "file": pkl_file,
            "label": label
        }
        
        if trial_num < 10:
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

        if trial_num < 10:
            tuples_list_train.append(data)
        elif trial_num < 13:
            tuples_list_val.append(data)
        elif trial_num < 16:
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
        "std": data_std,
        "num_labels": num_labels
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
        "std": data_std,
        "num_labels": num_labels
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
        "std": data_std,
        "num_labels": num_labels
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

print("error list: ", error_list)
