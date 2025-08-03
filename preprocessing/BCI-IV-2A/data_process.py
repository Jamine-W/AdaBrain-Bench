import scipy.io as scio
import numpy as np
from scipy import signal
import pickle
import os
import sys


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
raw_data_path = os.path.join(data_root,'BCI-4-2A/raw_data')
processed_data_path = os.path.join(data_root,'BCI-4-2A/processed_data')
os.makedirs(processed_data_path, exist_ok=True)

# Sampling rate and time split range
sampling_rate = 250  # 250Hz
min_time = 2  # 2s
max_time = 6  # 6s

# Calculate data point count from time split range
min_samples = min_time * sampling_rate
max_samples = max_time * sampling_rate

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
def save_event_to_pkl(X, Y, save_dir, subject):
    os.makedirs(save_dir, exist_ok=True)
    event_count = 1
    for i in range(X.shape[0]):
        event_data = X[i, :, :]
        label = Y[i]
        file_name = os.path.join(save_dir, str(subject) + '_' + f'{event_count}.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump({'X': event_data, 'Y': label}, f)
        event_count += 1

##!
for i in range(1):
    subject = i + 1
    print(subject)
    path1 = raw_data_path + '/A0' + str(subject) + 'T.mat'
    path2 = raw_data_path + '/A0' + str(subject) + 'E.mat'
    data1 = scio.loadmat(path1)
    data2 = scio.loadmat(path2)
    data1, data2 = data1['data'], data2['data']

    train_event = []
    train_label = []
    for j in range(len(data1[0]) - 6, len(data1[0])):
        struct = data1[0, j]
        X = (struct['X'])[0, 0]
        X = X[:, :-3].T
        trial = (struct['trial'])[0, 0]
        y = (struct['y'])[0, 0]
        trial = np.squeeze(trial)
        y = np.squeeze(y)

        if len(train_label) == 0:
            train_label = y
        else:
            train_label = np.concatenate((train_label, y), axis=0)
        for start in trial:
            end_min = start + min_samples
            end_max = start + max_samples
            if end_max <= X.shape[1]:
                event_data = X[:, end_min:end_max]
                event_data = bandpass_filter(event_data, lowcut=0.1, highcut=75.0, fs=250)
                event_data = notch_filter(event_data, freq=50.0, fs=250)
                train_event.append(event_data)
            else:
                print(f"Event at {start} exceeds data length, skipping this event.")
    train_event = np.array(train_event)
    print(train_event.shape)
    print(train_label.shape)

    validation_event = []
    validation_label = []
    for j in range(len(data2[0]) - 6, len(data2[0]) - 3):
        struct = data2[0, j]
        X = (struct['X'])[0, 0]
        X = X[:, :-3].T
        trial = (struct['trial'])[0, 0]
        y = (struct['y'])[0, 0]
        trial = np.squeeze(trial)
        y = np.squeeze(y)

        if len(validation_label) == 0:
            validation_label = y
        else:
            validation_label = np.concatenate((validation_label, y), axis=0)
        for start in trial:
            end_min = start + min_samples
            end_max = start + max_samples
            if end_max <= X.shape[1]:
                event_data = X[:, end_min:end_max]
                event_data = bandpass_filter(event_data, lowcut=0.1, highcut=75.0, fs=250)
                event_data = notch_filter(event_data, freq=50.0, fs=250)
                validation_event.append(event_data)
            else:
                print(f"Event at {start} exceeds data length, skipping this event.")
    validation_event = np.array(validation_event)
    print(validation_event.shape)
    print(validation_label.shape)

    test_event = []
    test_label = []
    for j in range(len(data2[0]) - 3, len(data2[0])):
        struct = data2[0, j]
        X = (struct['X'])[0, 0]
        X = X[:, :-3].T
        trial = (struct['trial'])[0, 0]
        y = (struct['y'])[0, 0]
        trial = np.squeeze(trial)
        y = np.squeeze(y)

        if len(test_label) == 0:
            test_label = y
        else:
            test_label = np.concatenate((test_label, y), axis=0)
        for start in trial:
            end_min = start + min_samples
            end_max = start + max_samples
            if end_max <= X.shape[1]:
                event_data = X[:, end_min:end_max]
                event_data = bandpass_filter(event_data, lowcut=0.1, highcut=75.0, fs=250)
                event_data = notch_filter(event_data, freq=50.0, fs=250)
                test_event.append(event_data)
            else:
                print(f"Event at {start} exceeds data length, skipping this event.")
    test_event = np.array(test_event)
    print(test_event.shape)
    print(test_label.shape)

    X = np.concatenate((train_event, validation_event, test_event), axis=0)
    Y = np.concatenate((train_label, validation_label, test_label), axis=0) - 1
    print(X.shape)
    print(Y.shape)
    save_event_to_pkl(X, Y, save_dir=processed_data_path + '/A0' + str(subject) + '/', subject=subject)