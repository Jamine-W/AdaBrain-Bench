import mne
import numpy as np
import os
import pickle
from tqdm import tqdm
import sys

"""
https://github.com/Abhishaike/EEG_Event_Classification
"""


data_root = sys.argv[1]  
print(f"Data root: {data_root}")
raw_data_path = os.path.join(data_root,'TUEV/raw_data/v2.0.1')
processed_data_path = os.path.join(data_root,'TUEV/processed_data')
os.makedirs(processed_data_path, exist_ok=True)
train_out_dir = os.path.join(processed_data_path, "train_out_dir")
eval_out_dir = os.path.join(processed_data_path, "eval_out_dir")
if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)

  
final_data = os.path.join(processed_data_path, "final_data")
final_train_dir = os.path.join(final_data, "train")
final_eval_dir = os.path.join(final_data, "eval")
final_test_dir = os.path.join(final_data, "test")
os.makedirs(final_train_dir, exist_ok=True)
os.makedirs(final_eval_dir, exist_ok=True)
os.makedirs(final_test_dir, exist_ok=True)

# raw_data = ".Preprocessing/TUEV/raw_data/v2.0.1"
# processed_data = ".Preprocessing/TUEV/processed_data"

drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']


def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    fs = 250.0
    [numChan, numPoints] = signals.shape
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])  # channel that had the detected thing
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  # for each event
        chan = int(EventData[i, 0])  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        # print (offset + start - 2 * int(fs), offset + end + 2 * int(fs), signals.shape)
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    if chOrder_standard is not None and len(chOrder_standard) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(chOrder_standard)
    if Rawdata.ch_names != chOrder_standard:
        raise ValueError

    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(60.0)
    Rawdata.resample(250, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


def load_up_objects(BaseDir, Features, OffendingChannels, Labels, OutDir):
    for dirName, subdirList, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname[-4:] == ".edf":
                print("\t%s" % fname)
                try:
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + "/" + fname
                    )  # event is the .rec file in the form of an array
                except (ValueError, KeyError):
                    print("something funky happened in " + dirName + "/" + fname)
                    continue
                signals, offending_channels, labels = BuildEvents(signals, times, event)

                for idx, (signal, offending_channel, label) in enumerate(
                    zip(signals, offending_channels, labels)
                ):
                    sample = {
                        "X": signal,
                        "offending_channel": offending_channel,
                        "Y": int(label[0]) - 1,
                    }
                    save_pickle(
                        sample,
                        os.path.join(
                            OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"
                        ),
                    )

    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


# Step 1: Process the raw training data
BaseDirTrain = os.path.join(raw_data_path, "edf", "train")
fs = 250
TrainFeatures = np.empty((0, 23, fs))
TrainLabels = np.empty([0, 1])
TrainOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel, train_out_dir
)

# Step 2: Process the raw eval data
BaseDirEval = os.path.join(raw_data_path, "edf", "eval")
fs = 250
EvalFeatures = np.empty((0, 23, fs))
EvalLabels = np.empty([0, 1])
EvalOffendingChannel = np.empty([0, 1])
load_up_objects(
    BaseDirEval, EvalFeatures, EvalLabels, EvalOffendingChannel, eval_out_dir
)

# Step 3: Split the data into train/eval/test sets by subject.
seed = 4523
np.random.seed(seed)

train_files = os.listdir(train_out_dir)
train_sub = list(set([f.split("_")[0] for f in train_files]))
print("train sub", len(train_sub))

test_files = os.listdir(eval_out_dir)

val_sub = np.random.choice(train_sub, size=int(len(train_sub) * 0.2), replace=False)
train_sub = list(set(train_sub) - set(val_sub))

val_files = [f for f in train_files if f.split("_")[0] in val_sub]
train_files = [f for f in train_files if f.split("_")[0] in train_sub]

for file in train_files:
    os.system(f"cp {os.path.join(train_out_dir, file)} {os.path.join(final_train_dir, file)}")
for file in val_files:
    os.system(f"cp {os.path.join(train_out_dir, file)} {os.path.join(final_eval_dir, file)}")
for file in test_files:
    os.system(f"cp {os.path.join(eval_out_dir, file)} {os.path.join(final_test_dir, file)}")
