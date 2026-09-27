from pathlib import Path
import os
import mne
import glob
import scipy.io as sio
import dhedfreader
from datetime import datetime
import pdb
import numpy as np
import pickle
import sys

data_root = sys.argv[1]  
print(f"Data root: {data_root}")
raw_data_path = os.path.join(data_root,'Sleep-EDF/raw_data/sleep-cassette')
processed_data_path = os.path.join(data_root,'Sleep-EDF/processed_data')
os.makedirs(processed_data_path, exist_ok=True)

# savePath = './Preprocessing/Sleepedf/processed'
# os.makedirs(savePath, exist_ok=True)

data_path = glob.glob(os.path.join(raw_data_path, '*PSG.edf'))
data_path.sort()
data_path = np.asarray(data_path)
label_path = glob.glob(os.path.join(raw_data_path, '*Hypnogram.edf'))
label_path.sort()
label_path = np.asarray(label_path)


name_list = []
for i in range(len(data_path)):
    subject_name = data_path[i].split("/")[-1].split(".")[0].split("-")[0][3:5]
    if subject_name not in name_list:
        name_list.append(subject_name)
print(name_list)
print(len(name_list))
label_to_number = {label: index for index, label in enumerate(name_list)}

num_subjects = len(name_list)
for m in range(num_subjects):
    os.makedirs(f"{processed_data_path}/{m}/", exist_ok=True)

sum_all = 0
fs = 100
for i in range(len(data_path)):
    print("processing ", i, " / ", len(data_path))
    raw = mne.io.read_raw_edf(data_path[i], preload=True, stim_channel=None)    #7950000
    data_name = data_path[i].split("/")[-1].split(".")[0].split("-")[0][3:5]
    # raw = raw.resample(fs, n_jobs=5)
    raw = raw.filter(l_freq=0.1, h_freq=49.9)
    # raw = raw.notch_filter(50.0)
    raw_ch_df_1 = raw.to_data_frame()["EEG Fpz-Cz"]
    raw_ch_df_1 = raw_ch_df_1.to_frame()
    raw_ch_df_1.set_index(np.arange(len(raw_ch_df_1)))
    # raw_ch_1 = raw_ch_df_1.values

    raw_ch_df_2 = raw.to_data_frame()["EEG Pz-Oz"]
    raw_ch_df_2 = raw_ch_df_2.to_frame()
    raw_ch_df_2.set_index(np.arange(len(raw_ch_df_2)))
    # raw_ch_2 = raw_ch_df_2.values

    raw_ch = np.vstack((raw_ch_df_1.values.transpose(-1, -2), raw_ch_df_2.values.transpose(-1, -2)))    #2, n

    if data_name[:-2] not in label_path[i]:
        print("label file error!")
        pdb.set_trace()
    f = open(label_path[i], 'r', errors='ignore')
    reader_ann = dhedfreader.BaseEDFReader(f)
    reader_ann.read_header()
    h_ann = reader_ann.header
    _, _, ann = zip(*reader_ann.records())
    f.close()
    ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")
    print(ann_start_dt)

    for index, a in enumerate(ann[0]):
        onset_sec, duration_sec, ann_char = a
        ann_char = "".join(ann_char)
        # print(onset_sec, "   ", duration_sec, "   ", ann_char)
        ann_char = ann_char.replace("Sleep stage W", "0")
        ann_char = ann_char.replace("Sleep stage 1", "1")
        ann_char = ann_char.replace("Sleep stage 2", "2")
        ann_char = ann_char.replace("Sleep stage 3", "3")
        ann_char = ann_char.replace("Sleep stage 4", "3")
        ann_char = ann_char.replace("Sleep stage R", "4")
        ann_char = ann_char.replace("Sleep stage ?", "6")
        ann_char = ann_char.replace("Movement time", "7")
        ann_char = ann_char.replace("b'", "")
        ann_char = ann_char.replace("'", "")

        if int(ann_char) < 6:
            start = 0.
            while(start < int(duration_sec)):
                # print(onset_sec + start, "   ", 30.0, "   ", int(ann_char))

                per_data = raw_ch[:, int((onset_sec + start) * fs) : int((onset_sec + start + 30) * fs)]
                label = ann_char

                if per_data.shape[-1] == 30 * fs:
                    folder_id = label_to_number.get(data_name)
                    save_file_path = f"{processed_data_path}/{folder_id}/E_{folder_id}_{index + 1}_{int(start // 30) + 1}.pkl"
                    pickle.dump(
                        {"X": per_data, "Y":label},
                        open(save_file_path, "wb"),
                    )
                    sum_all += 1
                    print(save_file_path, " saved")
                start += 30.0

print("file_nums=", sum_all)
