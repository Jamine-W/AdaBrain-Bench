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
processed_data_path = os.path.join(data_root,'EEGMAT/processed_data')
data_split_path = './preprocessing/EEGMAT/cross_subject_json'
os.makedirs(data_split_path, exist_ok=True)
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')




# data_folder = "./Preprocessing/EDMAT/processed_data"
# os.makedirs('./Preprocessing/EDMAT/cross_subject_json', exist_ok=True)
# save_folder_train = './Preprocessing/EDMAT/cross_subject_json/train.json'
# save_folder_val = './Preprocessing/EDMAT/cross_subject_json/val.json'
# save_folder_test = './Preprocessing/EDMAT/cross_subject_json/test.json'

sampling_rate = 500
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4',
            'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
num_channels = len(ch_names)
random.seed(42)


def load_subject_data(subject_folder):
    """Load all data from a subject."""
    subject_data = []
    subject_num = int(os.path.basename(subject_folder)[7:])
    subject_name = f"Subject{subject_num:02d}"

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
            "std": std
        }
    }

    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)


def main():
    subject_folders = natsorted(os.path.join(processed_data_path, f) for f in os.listdir(processed_data_path) if f.startswith("Subject"))
    train_subjects = [s for s in subject_folders if int(os.path.basename(s)[7:]) <= 31]
    test_subjects = [s for s in subject_folders if int(os.path.basename(s)[7:]) > 31]

    all_train_data, all_val_data = [], []
    for subject in train_subjects:
        subject_data = load_subject_data(subject)
        train_data, val_data = split_subject_data(subject_data)
        all_train_data.extend(train_data)
        all_val_data.extend(val_data)

    all_test_data = []
    for subject in test_subjects:
        all_test_data.extend(load_subject_data(subject))

    norm_params = compute_normalization_params(all_train_data)

    save_dataset(all_train_data, save_train_path)
    save_dataset(all_val_data, save_val_path, norm_params)
    save_dataset(all_test_data, save_test_path, norm_params)


if __name__ == "__main__":
    main()
