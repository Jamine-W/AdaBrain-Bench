import os
import numpy as np
import scipy.signal as signal
import pickle
import sys

data_root = sys.argv[1]  
print(f"Data root: {data_root}")
raw_data_path = os.path.join(data_root,'SHHS/raw_data')
processed_data_path = os.path.join(data_root,'SHHS/processed_data')
os.makedirs(processed_data_path, exist_ok=True)

# npz_path = '.Preprocessing/SHHS/raw_data'
# output_path = './Preprocessing/SHHS/processed_data'

def bandstop_filter(data, fs, lowcut=59, highcut=61):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='bandstop')
    return signal.filtfilt(b, a, data)


def highpass_filter(data, fs, cutoff=0.1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='highpass')
    return signal.filtfilt(b, a, data)


def process_npz_files(raw_data_path, processed_data_path):
    npz_files = [f for f in os.listdir(raw_data_path) if f.endswith('.npz')]

    for npz_file in npz_files:
        # Load npz file
        npz_data = np.load(os.path.join(raw_data_path, npz_file))
        x = npz_data['x']  # Shape (a, 3750)
        y = npz_data['y']  # Shape (a,)
        fs = npz_data['fs']  # 125 Hz

        # Create output folder
        folder_name = npz_file.replace('.npz', '')
        output_folder = os.path.join(processed_data_path, folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Process each row of x and save as pkl
        for i in range(x.shape[0]):
            filtered_data = highpass_filter(x[i], fs)
            filtered_data = bandstop_filter(filtered_data, fs)
            pkl_data = {
                'X': filtered_data,
                'Y': y[i]
            }
            pkl_file = os.path.join(output_folder, f'{folder_name}-{i}.pkl')
            with open(pkl_file, 'wb') as f:
                pickle.dump(pkl_data, f)



process_npz_files(raw_data_path, processed_data_path)
