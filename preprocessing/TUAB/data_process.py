import os
import pickle
from multiprocessing import Pool
import numpy as np
import mne
import sys

data_root = sys.argv[1]  
print(f"Data root: {data_root}")
raw_data_path = os.path.join(data_root,'TUAB/raw_data/v3.0.1/edf')
processed_data_path = os.path.join(data_root,'TUAB/processed_data')
os.makedirs(processed_data_path, exist_ok=True)

# root = ".Preprocessing/TUAB/raw_data/v3.0.1/edf"
# dump_root = ".Preprocessing/TUAB/processed_data"


channel_std = "01_tcp_ar"
drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']


standard_channels = [
    "EEG FP1-REF",
    "EEG F7-REF",
    "EEG T3-REF",
    "EEG T5-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F8-REF",
    "EEG T4-REF",
    "EEG T6-REF",
    "EEG O2-REF",
    "EEG FP1-REF",
    "EEG F3-REF",
    "EEG C3-REF",
    "EEG P3-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F4-REF",
    "EEG C4-REF",
    "EEG P4-REF",
    "EEG O2-REF",
]


def split_and_dump(params):
    fetch_folder, sub, dump_folder, label = params
    for file in os.listdir(fetch_folder):
        if not file.endswith(".edf"):
            continue
        if sub in file:
            print("process:", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            try:
                if drop_channels is not None:
                    useless_chs = []
                    for ch in drop_channels:
                        if ch in raw.ch_names:
                            useless_chs.append(ch)
                    raw.drop_channels(useless_chs)
                if chOrder_standard is not None and len(chOrder_standard) == len(raw.ch_names):
                    raw.reorder_channels(chOrder_standard)
                if raw.ch_names != chOrder_standard:
                    raise Exception("channel order is wrong!")

                raw.filter(l_freq=0.1, h_freq=75.0)
                raw.notch_filter(60.0)
                raw.resample(250, n_jobs=5)
                
                segment_len = 10 * 250

                ch_name = raw.ch_names
                raw_data = raw.get_data(units='uV')
                channeled_data = raw_data.copy()
                total_len = channeled_data.shape[1]

                for i in range(total_len // segment_len):
                    segment = channeled_data[:, i * segment_len:(i + 1) * segment_len]
                    dump_path = os.path.join(dump_folder, file.split(".")[0] + f"_{i}.pkl")
                    with open(dump_path, "wb") as f:
                        pickle.dump({"X": segment, "Y": label}, f)
            except:
                with open("tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue


if __name__ == "__main__":

    train_val_abnormal = os.path.join(raw_data_path, "train", "abnormal", channel_std)
    train_val_a_sub = list(set([item.split("_")[0] for item in os.listdir(train_val_abnormal)]))
    np.random.shuffle(train_val_a_sub)
    train_a_sub = train_val_a_sub[: int(len(train_val_a_sub) * 0.8)]
    val_a_sub = train_val_a_sub[int(len(train_val_a_sub) * 0.8):]

    train_val_normal = os.path.join(raw_data_path, "train", "normal", channel_std)
    train_val_n_sub = list(set([item.split("_")[0] for item in os.listdir(train_val_normal)]))
    np.random.shuffle(train_val_n_sub)
    train_n_sub = train_val_n_sub[: int(len(train_val_n_sub) * 0.8)]
    val_n_sub = train_val_n_sub[int(len(train_val_n_sub) * 0.8):]

    test_abnormal = os.path.join(raw_data_path, "eval", "abnormal", channel_std)
    test_a_sub = list(set([item.split("_")[0] for item in os.listdir(test_abnormal)]))

    test_normal = os.path.join(raw_data_path, "eval", "normal", channel_std)
    test_n_sub = list(set([item.split("_")[0] for item in os.listdir(test_normal)]))

    os.makedirs(os.path.join(processed_data_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(processed_data_path, "val"), exist_ok=True)
    os.makedirs(os.path.join(processed_data_path, "test"), exist_ok=True)

    parameters = []
    for train_sub in train_a_sub:
        parameters.append([train_val_abnormal, train_sub, os.path.join(processed_data_path, "train"), 1])
    for train_sub in train_n_sub:
        parameters.append([train_val_normal, train_sub, os.path.join(processed_data_path, "train"), 0])
    for val_sub in val_a_sub:
        parameters.append([train_val_abnormal, val_sub, os.path.join(processed_data_path, "val"), 1])
    for val_sub in val_n_sub:
        parameters.append([train_val_normal, val_sub, os.path.join(processed_data_path, "val"), 0])
    for test_sub in test_a_sub:
        parameters.append([test_abnormal, test_sub, os.path.join(processed_data_path, "test"), 1])
    for test_sub in test_n_sub:
        parameters.append([test_normal, test_sub, os.path.join(processed_data_path, "test"), 0])

    with Pool(processes=24) as pool:
        pool.map(split_and_dump, parameters)
