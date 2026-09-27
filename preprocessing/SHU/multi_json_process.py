import json
import os
import pickle
import numpy as np
from collections import defaultdict
import sys


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
processed_data_path = os.path.join(data_root,'SHU/processed_data')
data_split_path = './preprocessing/SHU/multi_subject_json'
os.makedirs(data_split_path, exist_ok=True)
save_train_path = os.path.join(data_split_path, 'train.json')
save_val_path = os.path.join(data_split_path, 'val.json')
save_test_path = os.path.join(data_split_path, 'test.json')

# path1 = './Preprocessing/SHU/processed_data/'
# output_dir = './Preprocessing/SHU/multi_subject_json/'
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


# Split by session
train_data = []
val_data = []
test_data = []

for i in range(1,26):
    subject_id = i
    subject_path = str(subject_id) + '/'
    data_folder = os.path.join(processed_data_path, subject_path)

    if not os.path.exists(data_folder):
        print(f"Folder does not exist: {data_folder}")
        continue

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

                    label = data['Y'].tolist()
                    file_data = {
                        "subject_id": subject - 1,
                        "subject_name": f"{subject:03d}",
                        "file": file_path,
                        "label": label
                    }

                    if session in [1, 2, 3]:
                        train_data.append(file_data)
                    elif session == 4:
                        val_data.append(file_data)
                    elif session == 5:
                        test_data.append(file_data)

                except Exception as e:
                    print(f"Error loading file {file_path}: {str(e)}")

print(f"train_set: {len(train_data)}, val_set: {len(val_data)}, test_set: {len(test_data)}")

# Compute normalization parameters
train_max, train_min, train_mean, train_std = calculate_dataset_stats(train_data)

dataset_info = dataset_info_template.copy()
dataset_info.update({
    "min": train_max,
    "max": train_min,
    "mean": train_mean.tolist(),
    "std": train_std.tolist()
})

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
print("Multi-subject splitting completed")
