import os
import pickle
from glob import glob
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import mne


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
raw_data_path = os.path.join(data_root,'SEED-IV/raw_data/Preprocessed_EEG')
processed_data_path = os.path.join(data_root,'SEED-IV/processed_data')
os.makedirs(processed_data_path, exist_ok=True)



SAMPLING_RATE = 200              # Hz
SEG_LEN = 1 * SAMPLING_RATE      # 200 samples
SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
}
SUBJECT_NAMES = [f"S{i}" for i in range(1, 16)]   # S1 … S15

# ================= 2. 工具函数 =================
def extract_trials(mat_path):
    data = loadmat(mat_path)
    trials = {}
    for k in data.keys():
        if k.startswith("__"):
            continue
        try:
            trial_id = int(k.split("eeg")[-1])
            trials[trial_id] = data[k].astype(np.float32)   # shape (62, T)
        except ValueError:
            continue
    return trials


def slice_trial(eeg, label):
    T = eeg.shape[1]
    slices = []
    for start in range(0, T - SEG_LEN + 1, SEG_LEN):
        seg = eeg[:, start:start + SEG_LEN]
        slices.append((seg, label))
    return slices


def filter_eeg(raw_array, sfreq=200, l_freq=0.1, h_freq=75, notch=50):
    info = mne.create_info(
        ch_names=[f"C{i}" for i in range(raw_array.shape[0])],
        sfreq=sfreq,
        ch_types="eeg"
    )
    raw = mne.io.RawArray(raw_array, info, verbose=False)


    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    raw.notch_filter(freqs=notch, notch_widths=2, fir_design='firwin', verbose=False)

    return raw.get_data()


os.makedirs(processed_data_path, exist_ok=True)

for sub_idx, sub_name in enumerate(tqdm(SUBJECT_NAMES, desc="Subject")):
    sub_dir = os.path.join(processed_data_path, str(sub_idx))
    os.makedirs(sub_dir, exist_ok=True)

    for ses in [1, 2, 3]:
        ses_dir = os.path.join(data_root, str(ses))
        sub_num = sub_name.lstrip("S")
        mat_files = glob(os.path.join(ses_dir, f"{sub_num}_*.mat"))
        print(mat_files)
        if not mat_files:
            print(f"[WARN] Missing {sub_num} in session {ses}")
            continue
        mat_path = mat_files[0]

        trials = extract_trials(mat_path)      # dict{trial_id: (62, T)}
        labels = SESSION_LABELS[ses]

        for trial_id in range(1, 25):          # 1..24
            if trial_id not in trials:
                print(f"[WARN] Missing trial {trial_id} in {mat_path}")
                continue
            eeg = trials[trial_id]
            eeg = filter_eeg(eeg, sfreq=SAMPLING_RATE)
            y = labels[trial_id - 1]           
            slices = slice_trial(eeg, y)

            for seg_idx, (X, Y) in enumerate(slices, 1):
                pkl_name = f"{sub_name}_{ses}_{trial_id}_{seg_idx}.pkl"
                pkl_path = os.path.join(sub_dir, pkl_name)
                with open(pkl_path, "wb") as f:
                    pickle.dump({"X": X, "Y": Y}, f)