import json
import os
import pickle
import numpy as np
import random
from collections import defaultdict
import sys

data_root = sys.argv[1]  
print(f"Data root: {data_root}")
processed_data_path = os.path.join(data_root,'SHU/processed_data')
data_split_path = './preprocessing/SHU/cross_subject_json'
os.makedirs(data_split_path, exist_ok=True)
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')

# path1 = './Preprocessing/SHU/processed_data/'
# output_dir = './Preprocessing/SHU/cross_subject_json/'
# os.makedirs(output_dir, exist_ok=True)

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"File has been saved to {filename}")


def calculate_dataset_stats(data_list):
    max_value = -float('inf')
    min_value = float('inf')
    channel_means = 0
    channel_stds = 0
    i = 0

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

        channel_means += np.mean(X, axis=-1)
        channel_stds += np.std(X, axis=-1)
        i += 1

    mean_values = channel_means / i
    std_values = channel_stds / i

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


dataset_info_template = {
    "sampling_rate": 250,
    "ch_names": ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1",
                 "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T3", "T4",
                 "A1", "A2", "CP1", "CP2", "CP5", "CP6", "Pz", "P3",
                 "P4", "T5", "T6", "PO3", "PO4", "Oz", "O1", "O2"],
    "min": None,
    "max": None,
    "mean": None,
    "std": None
}


# Split by subject
all_train_data = []
all_val_data = []
all_test_data = []

for subject_id in range(1, 26):
    subject_path = str(subject_id) + '/'
    data_folder = os.path.join(processed_data_path, subject_path)

    if not os.path.exists(data_folder):
        print(f"Folder does not exist: {data_folder}")
        continue

    subject_files = []
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".pkl"):
            parts = file_name.split('_')
            if len(parts) == 3:
                subject = int(parts[0])
                session = int(parts[1])
                trial = int(parts[2].split('.')[0])
                file_path = os.path.join(data_folder, file_name)

                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)

                    file_data = {
                        "subject_id": subject - 1,
                        "subject_name": f"{subject:03d}",
                        "file": file_path,
                        "label": data['Y'].tolist()
                    }
                    subject_files.append(file_data)
                except Exception as e:
                    print(f"Error loading file {file_path}: {str(e)}")

    if 1 <= subject_id <= 22:
        train, val = split_train_val(subject_files)
        all_train_data.extend(train)
        all_val_data.extend(val)
    elif 23 <= subject_id <= 25:
        all_test_data.extend(subject_files)

print(f"train_set: {len(all_train_data)}, val_set: {len(all_val_data)}, test_set: {len(all_test_data)}")

# Compute normalization parameters
train_max, train_min, train_mean, train_std = calculate_dataset_stats(all_train_data)

dataset_info = dataset_info_template.copy()
dataset_info.update({
    "min": train_max,
    "max": train_min,
    "mean": train_mean.tolist(),
    "std": train_std.tolist()
})

final_train_data = {
    "dataset_info": dataset_info,
    "subject_data": all_train_data
}
final_val_data = {
    "dataset_info": dataset_info,
    "subject_data": all_val_data
}
final_test_data = {
    "dataset_info": dataset_info,
    "subject_data": all_test_data
}

save_to_json(final_train_data, save_train_path)
save_to_json(final_val_data, save_val_path)
save_to_json(final_test_data, save_test_path)
print("Cross-subject splitting completed")
