import scipy.io as scio
import numpy as np
from scipy import signal
import pickle
import os
import sys


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
raw_data_path = os.path.join(data_root,'SHU/raw_data')
processed_data_path = os.path.join(data_root,'SHU/processed_data')
os.makedirs(processed_data_path, exist_ok=True)

# file_path = './Preprocessing/SHU/raw_data/'
# save_path = './Preprocessing/SHU/processed_data/'

# Define a bandpass filter (0.1Hz - 75Hz)
def bandpass_filter(data, lowcut=0.1, highcut=75.0, fs=250, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


# Define a notch filter (50Hz)
def notch_filter(data, freq=50.0, fs=250, Q=30.0):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = signal.iirnotch(w0, Q)
    return signal.filtfilt(b, a, data, axis=-1)


# Resampling
def resample_data(data, old_rate=250, new_rate=256):
    number_of_samples = int(data.shape[-1] * new_rate / old_rate)
    return signal.resample(data, number_of_samples, axis=-1)


# Save to .pkl files
def save_event_to_pkl(X, Y, save_dir, people, session):
    event_count = 1
    for i in range(X.shape[0]):
        event_data = X[i, :, :]
        label = Y[i]
        file_name = os.path.join(save_dir, str(people) + '_' + str(session) + '_' + f'{event_count}.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump({'X': event_data, 'Y': label}, f)
        event_count += 1

for i in range(25):
    subject = i + 1
    for j in range(5):
        session = j + 1
        eeg = 'sub-0' + f"{subject:02d}" + '_ses-' + f"{session:02d}" + '_task_motorimagery_eeg.mat'
        data = scio.loadmat(os.path.join(raw_data_path, eeg))
        data1 = data['data']
        label = np.squeeze(data['labels']) - 1
        print(data1.shape)
        print(label.shape)
        for trial in range(len(label)):
            eeg_trial = data1[trial, :, :]
            eeg_trial = bandpass_filter(eeg_trial, lowcut=0.1, highcut=75.0, fs=250)
            eeg_trial = notch_filter(eeg_trial, freq=50.0, fs=250)
            save_dir = processed_data_path + str(subject) + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = os.path.join(save_dir,  str(subject) + '_' + str(session) + '_' + str(trial+1) + '.pkl')
            with open(file_name, 'wb') as f:
                pickle.dump({'X': eeg_trial, 'Y': label[trial]}, f)
