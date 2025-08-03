#!/bin/bash
data_root="path/to/data/root"
# dataset_choices = [SEED SEED-IV SEED-VIG BCI-IV-2A SHU EEGMAT HMC SHHS Sleep-EDF Siena TUAB TUEV Things-EEG]
dataset="SEED"  


data_process_script="preprocessing/$dataset/data_process.py"
python "$data_process_script" "$data_root"
