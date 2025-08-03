#!/bin/bash
data_root="path/to/data/root"
# dataset_choices = [SEED SEED-IV SEED-VIG BCI-IV-2A SHU EEGMAT HMC SHHS Sleep-EDF Siena TUAB TUEV Things-EEG]
dataset="SEED"  


cross_json_process_script="preprocessing/$dataset/cross_json_process.py"
python "$cross_json_process_script" "$data_root"
# multi_json_process_script="preprocessing/$dataset/multi_json_process.py"
# python "$multi_json_process_script" "$data_root"
