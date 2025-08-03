import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from scipy.signal import resample
from scipy import signal
import pickle
import json
import pdb

proxy = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
cuda_device_count = torch.cuda.device_count()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class EEGDataset():
    """
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """
    def __init__(self, dataset, train=True, subject_mod='single', subject_id=1, sampling_rate=200, norm_method='z_score', factor=100):
        self.train = train
        self.sampling_rate = sampling_rate
        self.subject_mod = subject_mod
        self.subject_id = subject_id
        self.normalize_method = norm_method

        # Load the configuration from the JSON file
        config_path = "./dataset_config/Retrieval.json"
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        dataset_info = config.get(dataset)

        train_root = f"{dataset_info['root']['multi']}/train.json"
        test_root = f"{dataset_info['root']['multi']}/test.json"

        dataset_root = train_root if self.train else test_root
        all_json_data = json.load(open(dataset_root, "r"))
        data_info = all_json_data['dataset_info']
        self.default_rate = data_info['sampling_rate']
        self.ch_names = data_info['ch_names']
        self.min_value = data_info['min']
        self.max_value = data_info['max']
        self.mean_value = data_info['mean']
        self.std_value = data_info['std']
        self.factor = factor
        
        all_files = all_json_data['subject_data']
        if self.subject_mod == 'single':
            self.files = [item for item in all_files if int(item.get("subject_id")) == self.subject_id]
        elif self.subject_mod == 'multi':
            self.files = all_files
        elif self.subject_mod == 'loso':
            if self.train:
                self.files = [item for item in all_files if int(item.get("subject_id")) != self.subject_id]
            else:
                self.files = [item for item in all_files if int(item.get("subject_id")) == self.subject_id]
        else:
            print("Unknown subject_mod! ")
            exit(0)
        
        if not self.train:
            self.files = self._process_test_data(self.files)
        
        # Try to load the saved features if they exist
        features_filename = dataset_info['features']['train_features'] if self.train else dataset_info['features']['test_features']
        if os.path.exists(features_filename) :
            saved_features = torch.load(features_filename)
            self.img_features = saved_features['img_features']
        else:
            print("No image features found. ")
            exit(0)
    
    def _process_test_data(self, files):
        label_groups = {}
        for item in files:
            label = item["label"]
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)
        processed_test_files = []
        for label, items in label_groups.items():
            eeg_paths = [item["EEG"] for item in items]
            eeg_features = []
            for eeg_path in eeg_paths:
                sample = pickle.load(open(eeg_path, "rb"))['X']
                if self.sampling_rate != self.default_rate:
                    sample = self.resample_data(sample)
                sample = self.normalize(sample)
                eeg_features.append(sample) 
            eeg_features = np.mean(eeg_features, axis=0)
            new_item = {
                "id": len(processed_test_files),  
                "label": label,
                "eeg_feature": torch.tensor(eeg_features)
            }
            processed_test_files.append(new_item)
        return processed_test_files
    
    def normalize(self, X):
        if self.normalize_method == 'z_score':
            mean_value, std_value = np.array(self.mean_value), np.array(self.std_value)
            mu, sigma = np.expand_dims(mean_value, axis=1), np.expand_dims(std_value, axis=1)
            X = (X - mu) / (sigma + 1e-8)
        elif self.normalize_method == 'min_max':
            X = (X - self.min_value) / (self.max_value - self.min_value)
        elif self.normalize_method == 'ems':
            X = self.exponential_moving_standardize(X)
        elif self.normalize_method == '0.1mv':
            X = X / self.factor
        elif self.normalize_method == '95':
            X = self.percentile95(X)
        
        return X
    
    def percentile95(self, X):
        X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        return X
        #normalized_signal = np.zeros_like(X)
        #for i in range(X.shape[0]):
        #    percentile_95 = np.percentile(np.abs(X[i]), 95)
        #    if percentile_95 != 0:
        #        normalized_signal[i] = X[i] / percentile_95
        #    else:
        #        normalized_signal[i] = X[i]
        #return normalized_signal

    def exponential_moving_standardize(self, X, eps=1e-4):  # From braindecode.preprocessing.exponential_moving_standardize
        X = X.T
        df = pd.DataFrame(X)
        meaned = df.ewm(alpha=self.ems_factor).mean()
        demeaned = df - meaned
        squared = demeaned * demeaned
        square_ewmed = squared.ewm(alpha=self.ems_factor).mean()
        standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
        standardized = np.array(standardized)
        return standardized.T
    
    def resample_data(self, data):
        number_of_samples = int(data.shape[-1] * self.sampling_rate / self.default_rate)
        return signal.resample(data, number_of_samples, axis=-1)
    
    def get_ch_names(self):
        return self.ch_names
    
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        eeg_sources = self.files[index]
        if self.train:
            datapath = eeg_sources['EEG']
            x = pickle.load(open(datapath, "rb"))['X']
            if self.sampling_rate != self.default_rate:
                x = self.resample_data(x)
            x = self.normalize(x)
            x_index = int(datapath.split('/')[-1].split('.')[0].split('_')[-2]) - 1
        else:
            x = eeg_sources['eeg_feature']
            x_index = int(eeg_sources['label'] - 1)
        label = int(eeg_sources['label'] - 1)

        img_features = self.img_features[x_index]

        return x, label, img_features
