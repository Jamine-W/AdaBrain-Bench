import json
import os
import pickle
import numpy as np
import random
from collections import defaultdict
from natsort import natsorted
import sys

data_root = sys.argv[1]  
print(f"Data root: {data_root}")
processed_data_path = os.path.join(data_root,'SEED-VIG/processed_data')
data_split_path = './preprocessing/SEED-VIG/cross_subject_json'
os.makedirs(data_split_path, exist_ok=True)
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')



# path1 = './Preprocessing/SEED-VIG/processed_data/'
# path2 = './Preprocessing/SEED-VIG/cross_subject_json'
# os.makedirs(path2, exist_ok=True)


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


def main():
    # Split by subject
    train_range = (1, 13)
    val_range = (14, 17)
    test_range = (18, 21)

    train_data = []
    val_data = []
    test_data = []

    for i in range(21):
        subject_id = i + 1
        print(subject_id)
        subject_path = 'subject_' + str(subject_id) + '/'
        data_folder = os.path.join(processed_data_path, subject_path)

        trial_files = natsorted([f for f in os.listdir(data_folder) if f.endswith(".pkl")])

        for trial in trial_files:
            file_path = os.path.join(data_folder, trial)

            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                continue

            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            label = data['Y']

            file_data = {
                "subject_id": i+1,
                "subject_name": i+1,
                "file": file_path,
                "label": label
            }

            if train_range[0] <= subject_id <= train_range[1]:
                train_data.append(file_data)
            elif val_range[0] <= subject_id <= val_range[1]:
                val_data.append(file_data)
            elif test_range[0] <= subject_id <= test_range[1]:
                test_data.append(file_data)

    # Compute normalization parameters
    train_max, train_min, train_mean, train_std = calculate_dataset_stats(train_data)

    dataset_info = {
        "sampling_rate": 200,
        "ch_names": ["FT7", "FT8", "T7", "T8", "TP7", "TP8", "CP1", "CP2", "P1", "PZ", "P2", "PO3", "POZ", "PO4", "O1",
                     "OZ", "O2"],
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


if __name__ == "__main__":
    main()
