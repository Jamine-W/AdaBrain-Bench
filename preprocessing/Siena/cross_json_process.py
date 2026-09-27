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
processed_data_path = os.path.join(data_root,'Siena/processed_data')
data_split_path = './preprocessing/Siena/cross_subject_json'
os.makedirs(data_split_path, exist_ok=True)
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')

# data_folder = "./Preprocessing/Siena/raw_data"
# os.makedirs('./Preprocessing/Siena/cross_subject_json', exist_ok=True)
# save_folder_train = './Preprocessing/Siena/cross_subject_json/train.json'
# save_folder_val = './Preprocessing/Siena/cross_subject_json/val.json'
# save_folder_test = './Preprocessing/Siena/cross_subject_json/test.json'

sampling_rate = 512
ch_names = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fc1', 'Fc5', 'Cp1', 'Cp5', 'F9', 'Fz', 'Cz', 'Pz', 'Fp2',
            'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'Fc2', 'Fc6', 'Cp2', 'Cp6', 'F10']
num_channels = len(ch_names)
random.seed(42)


def load_subject_metadata(subject_folder):
    subject_data = []
    folder_name = os.path.basename(subject_folder)
    subject_num = int(folder_name[2:])
    subject_name = f"PN{subject_num:02d}"

    for file in natsorted(f for f in os.listdir(subject_folder) if f.endswith('.pkl')):
        file_path = os.path.join(subject_folder, file)
        try:
            with open(file_path, 'rb') as f:
                eeg_data = pickle.load(f)
            subject_data.append({
                "subject_id": subject_num,
                "subject_name": subject_name,
                "file": file_path,
                "label": eeg_data['Y']
            })
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    return subject_data


def compute_normalization_params(data_list):
    """Calculate normalization parameters."""
    total_mean = np.zeros(num_channels)
    total_std = np.zeros(num_channels)
    max_val, min_val = -np.inf, np.inf
    count = 0

    for data in data_list:
        with open(data["file"], 'rb') as f:
            eeg = pickle.load(f)['X']

        max_val = max(max_val, eeg.max())
        min_val = min(min_val, eeg.min())
        total_mean += eeg.mean(axis=1)
        total_std += eeg.std(axis=1)
        count += 1

    return (total_mean / count).tolist(), (total_std / count).tolist(), max_val, min_val


def split_subject_data(subject_data, val_ratio=0.2):
    """Split the data of a single subject into a validation set with class-balanced partitioning."""
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


def save_dataset(data_list, save_path, norm_params=None):
    if norm_params is None:
        print("Computing normalization parameters...")
        mean, std, max_val, min_val = compute_normalization_params(data_list)
    else:
        mean, std, max_val, min_val = norm_params

    dataset = {
        "subject_data": data_list,  # 只包含元数据
        "dataset_info": {
            "sampling_rate": sampling_rate,
            "ch_names": ch_names,
            "min": min_val,
            "max": max_val,
            "mean": mean,
            "std": std
        }
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved to {save_path}")


def main():
    subject_folders = natsorted(
        os.path.join(processed_data_path, f)
        for f in os.listdir(processed_data_path)
        if f.startswith("PN") and os.path.isdir(os.path.join(processed_data_path, f))
    )

    train_subjects = [s for s in subject_folders if int(os.path.basename(s)[2:]) <= 14]
    test_subjects = [s for s in subject_folders if int(os.path.basename(s)[2:]) > 14]  # The last two subjects are PN16 and PN17.

    all_train_data, all_val_data = [], []
    for subject in train_subjects:
        subject_data = load_subject_metadata(subject)
        train_data, val_data = split_subject_data(subject_data)
        all_train_data.extend(train_data)
        all_val_data.extend(val_data)

    all_test_data = []
    for subject in test_subjects:
        all_test_data.extend(load_subject_metadata(subject))

    print(f"\nData counts:")
    print(f"Train: {len(all_train_data)}, Val: {len(all_val_data)}, Test: {len(all_test_data)}")

    print("\nComputing normalization...")
    norm_params = compute_normalization_params(all_train_data)

    print("\nSaving datasets...")
    save_dataset(all_train_data, save_train_path, norm_params)
    save_dataset(all_val_data, save_val_path, norm_params)
    save_dataset(all_test_data, save_test_path, norm_params)


if __name__ == "__main__":
    main()
