import json
import os
import pickle
import numpy as np
import random
from collections import defaultdict
import sys

data_root = sys.argv[1]  
print(f"Data root: {data_root}")
processed_data_path = os.path.join(data_root,'BCI-IV-2A/processed_data')
data_split_path = './preprocessing/BCI-IV-2A/cross_subject_json'
os.makedirs(data_split_path, exist_ok=True)
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"File has been saved to {filename}")


def calculate_dataset_stats(data_list):
    max_value = -float('inf')
    min_value = float('inf')
    all_means = []
    all_stds = []

    for file_data in data_list:
        file_path = file_data['file']
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            continue

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        X = data['X']

        current_max = np.max(X)
        current_min = np.min(X)
        if current_max > max_value:
            max_value = current_max
        if current_min < min_value:
            min_value = current_min

        channel_means = np.mean(X, axis=-1)
        channel_stds = np.std(X, axis=-1)
        all_means.append(channel_means)
        all_stds.append(channel_stds)

    mean_values = np.mean(np.array(all_means), axis=0)
    std_values = np.mean(np.array(all_stds), axis=0)

    return max_value, min_value, mean_values, std_values


def split_train_val(all_train_data, val_ratio=0.2):
    subject_label_dict = defaultdict(lambda: defaultdict(list))
    for item in all_train_data:
        subject_id = item["subject_id"]
        label = item["label"]
        subject_label_dict[subject_id][label].append(item)

    train_data = []
    val_data = []
    for subject_id, label_dict in subject_label_dict.items():
        for label, items in label_dict.items():
            random.shuffle(items)
            split_idx = int(len(items) * (1 - val_ratio))

            train_data.extend(items[:split_idx])  # The first 80% of the data is the training set.
            val_data.extend(items[split_idx:])  # The last 20% of the data is the validation set.

    return train_data, val_data


def main():
    # Split by subject
    train_range = (1, 7)  # train/val
    test_range = (8, 9)  # test
    all_train_data = []  # Store all data from subjects 1-7
    test_data = []

    for i in range(9):
        subject_id = i + 1
        subj_path = 'A0' + str(subject_id) + '/'
        data_folder = os.path.join(processed_data_path, subj_path)

        for trial in range(1, 577):
            file_name = f"{subject_id}_{trial}.pkl"
            file_path = os.path.join(data_folder, file_name)

            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                continue

            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            label = data['Y'].tolist()

            file_data = {
                "subject_id": i,
                "subject_name": "A0" + str(subject_id),
                "file": file_path,
                "label": label
            }

            if train_range[0] <= subject_id <= train_range[1]:
                all_train_data.append(file_data)
            elif test_range[0] <= subject_id <= test_range[1]:
                test_data.append(file_data)

    # Split all data from subjects 1-7 into training and validation sets with an 8:2 ratio
    train_data, val_data = split_train_val(all_train_data, val_ratio=0.2)
    # Compute normalization parameters
    train_max, train_min, train_mean, train_std = calculate_dataset_stats(train_data)

    dataset_info = {
        "sampling_rate": 250,
        "ch_names": ["Fz", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP3", "CP1",
                     "CPZ", "CP2", "CP4", "P1", "PZ", "P2", "POZ"],
        "min": train_min,
        "max": train_max,
        "mean": train_mean.tolist(),
        "std": train_std.tolist()
    }

    final_train_data = {
        "dataset_info": dataset_info,
        "subject_data": train_data
    }
    final_val_data = {
        "dataset_info": dataset_info,
        "subject_data": val_data
    }
    final_test_data = {
        "dataset_info": dataset_info,
        "subject_data": test_data
    }

    save_to_json(final_train_data, save_train_path)
    save_to_json(final_val_data, save_val_path)
    save_to_json(final_test_data, save_test_path)

    print("Cross-subject splitting completed")
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")
    print(f"Number of test samples: {len(test_data)}")


if __name__ == "__main__":
    main()
