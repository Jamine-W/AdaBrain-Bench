import os
import mne
import sys

data_root = sys.argv[1]  
print(f"Data root: {data_root}")
raw_data_path = os.path.join(data_root,'Siena/raw_data')
# base_dir = "./Preprocessing/Siena/raw_data"
patient_id = "PN17"
edf_dir = os.path.join(raw_data_path, patient_id)

def list_edf_channels(edf_dir):
    edf_files = [f for f in os.listdir(edf_dir) if f.endswith('.edf') and f.startswith(patient_id)]

    if not edf_files:
        print(f"No EDF files found in the directory {edf_dir}.")
        return

    for edf_file in edf_files:
        edf_path = os.path.join(edf_dir, edf_file)
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False)
            channels = raw.ch_names
            n_channels = len(channels)
            print(f"File: {edf_file}")
            print(f"Number of channel: {n_channels}")
            print(f"Sample rate: {raw.info['sfreq']}")
            '''
            print("Channel list:")
            for i, ch in enumerate(channels, 1):
                print(f"  {i:2d}: {ch}")
            '''
            print("-" * 50)

        except Exception as e:
            print(f"Failed to process {edf_file} - Reason: {str(e)}")
            continue


if __name__ == "__main__":
    list_edf_channels(edf_dir)
