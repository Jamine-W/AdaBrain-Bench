import json
import os
import pickle
import numpy as np
from natsort import natsorted
from collections import defaultdict
import random
import sys


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
processed_data_path = os.path.join(data_root,'SEED/processed_data')
data_split_path = './preprocessing/SEED/cross_subject_json'
os.makedirs(data_split_path, exist_ok=True)
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')

# data_folder = "./Preprocessing/SEED/processed_data"
# os.makedirs('./Preprocessing/SEED/cross_subject_json', exist_ok=True)
# save_folder_train = './Preprocessing/SEED/cross_subject_json/train.json'
# save_folder_val = './Preprocessing/SEED/cross_subject_json/val.json'
# save_folder_test = './Preprocessing/SEED/cross_subject_json/test.json'

sampling_rate = 200
ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
num_channels = len(ch_names)
num_labels = 3
random.seed(42)


def load_subject_data(subject_folder):
    """Load all data from a subject."""
    subject_data = []
    subject_num = int(subject_folder.split("/")[-1])
    subject_name = str(subject_num + 1)

    for file in natsorted(f for f in os.listdir(subject_folder) if f.endswith('.pkl')):
        try:
            with open(os.path.join(subject_folder, file), 'rb') as f:
                eeg_data = pickle.load(f)
            subject_data.append({
                "subject_id": subject_num,
                "subject_name": subject_name,
                "file": os.path.join(subject_folder, file),
                "label": eeg_data['Y'],
                "eeg_data": eeg_data['X']
            })
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    return subject_data


def split_subject_data(subject_data, val_ratio=0.2):
    """Create a class-balanced validation set from the data of a subject."""
    label_to_data = defaultdict(list)
    for data in subject_data:
        label_to_data[data["label"]].append(data)

    train_data, val_data = [], []
    for label, data_list in label_to_data.items():
        random.shuffle(data_list)
        split_idx = int(len(data_list) * (1 - val_ratio))
        train_data.extend(data_list[:split_idx])
        val_data.extend(data_list[split_idx:])

    return train_data, val_data


def compute_normalization_params(data_list):
    """Compute normalization parameters"""
    total_mean = np.zeros(num_channels)
    total_std = np.zeros(num_channels)
    max_val, min_val = -np.inf, np.inf

    for data in data_list:
        eeg = data["eeg_data"]
        max_val = max(max_val, eeg.max())
        min_val = min(min_val, eeg.min())
        for j in range(num_channels):
            total_mean[j] += eeg[j].mean()
            total_std[j] += eeg[j].std()

    mean = (total_mean / len(data_list)).tolist()
    std = (total_std / len(data_list)).tolist()
    return mean, std, max_val, min_val


def save_dataset(data_list, save_path, norm_params=None):
    if norm_params is None:
        mean, std, max_val, min_val = compute_normalization_params(data_list)
    else:
        mean, std, max_val, min_val = norm_params

    dataset = {
        "subject_data": [{k: v for k, v in d.items() if k != "eeg_data"} for d in data_list],
        "dataset_info": {
            "sampling_rate": sampling_rate,
            "ch_names": ch_names,
            "min": min_val,
            "max": max_val,
            "mean": mean,
            "std": std,
            "num_labels": num_labels
        }
    }

    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)


def main():
    subject_folders = natsorted([os.path.join(processed_data_path, f) for f in os.listdir(processed_data_path)])
    train_subjects = [s for s in subject_folders if int(s.split('/')[-1]) <= 12]
    test_subjects = [s for s in subject_folders if int(s.split('/')[-1]) > 12]

    all_train_data, all_val_data = [], []
    for sub_id, subject in enumerate(train_subjects):
        print(f"Processing training subject {sub_id}...")
        subject_data = load_subject_data(subject)
        train_data, val_data = split_subject_data(subject_data)
        all_train_data.extend(train_data)
        all_val_data.extend(val_data)

    all_test_data = []
    for sub_id, subject in enumerate(test_subjects):
        print(f"Processing testing subject {sub_id}...")
        all_test_data.extend(load_subject_data(subject))

    print("Getting norm params...")
    norm_params = compute_normalization_params(all_train_data)

    save_dataset(all_train_data, save_train_path, norm_params)
    save_dataset(all_val_data, save_val_path, norm_params)
    save_dataset(all_test_data, save_test_path, norm_params)


if __name__ == "__main__":
    main()
