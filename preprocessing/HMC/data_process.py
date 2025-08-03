import os
import pandas as pd
import mne
import math
import pickle
import pdb
import sys


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
raw_data_path = os.path.join(data_root,'HMC/raw_data/1.1/recordings')
processed_data_path = os.path.join(data_root,'HMC/processed_data')
os.makedirs(processed_data_path, exist_ok=True)


# edf_path = './Preprocessing/HMC/raw_data/recordings'
# output_path = './Preprocessing/HMC/processed'
# os.makedirs(os.path.dirname(output_path), exist_ok=True)

edf_files = [f for f in os.listdir(raw_data_path) if ((f.endswith('.edf')) and ("sleepscoring" not in f))]
edf_files.sort()
fs = 256

# label_list = ['Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R', 'Lights off']
used_label = ['Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R']
label_to_number = {label: index for index, label in enumerate(used_label)}

file_num = 0

name_list = []
for idx, file in enumerate(edf_files):
    # /physionet.org/files/hmc-sleep-staging/1.1/recordings/SN001.edf
    file_name = file.split("/")[-1].split(".")[0]
    if file_name not in name_list:
        name_list.append(file_name)
name_to_number = {label: index for index, label in enumerate(name_list)}
num_subjects = len(name_list)
for m in range(num_subjects):
    os.makedirs(f"{processed_data_path}/{m}/", exist_ok=True)

for idx, file in enumerate(edf_files):
    print("processing ", idx, "/", len(edf_files), '...')
    # /physionet.org/files/hmc-sleep-staging/1.1/recordings/SN001.edf
    file_name = file.split("/")[-1].split(".")[0]
    file_path = os.path.join(raw_data_path, file)
    txt_path = f"{raw_data_path}/{file_name}_sleepscoring.txt"
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True)
    except Exception as e:
        print(f"Wrong with {raw_data_path}: {e}")
    
    droped_channels = [ch for ch in raw.info['ch_names'] if 'EEG' not in ch]
    raw.drop_channels(droped_channels)
    # keeped_channels = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2']
    # channels = [ch.replace("EEG ", "")[:-3] for ch in raw.info["ch_names"] if 'EEG' in ch]

    # 0.1~75.0Hz
    raw = raw.filter(l_freq=0.1, h_freq=75)
    # 50Hz
    raw = raw.notch_filter(50.0)
    # uV
    data = raw.get_data(units='uV')

    if data.shape[0] != 4:
        pdb.set_trace()

    annotation_df = pd.read_csv(txt_path, skipinitialspace=True)
    annotation_df = annotation_df[annotation_df['Annotation'].isin(used_label)].reset_index(drop=True)

    idxx = 1
    for _, row in annotation_df.iterrows():
        start_time = row['Recording onset']
        duration = row['Duration']
        label = row['Annotation']

        if label in used_label:
            start_point = math.ceil(start_time * fs)
            per_data = data[:, start_point : int(start_point + duration * fs)]
            per_label = label_to_number.get(label)

            folder_id = name_to_number.get(file_name)
            save_file_path = f"{processed_data_path}/{folder_id}/{file_name}_{idxx}.pkl"
            pickle.dump(
                    {"X": per_data, "Y":per_label},
                    open(save_file_path, "wb"),
                )
            print(save_file_path, " saved")
            idxx = idxx + 1
            file_num = file_num + 1

print("file_num = ", file_num)
