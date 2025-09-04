# How to prepare datasets

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– SEED/
|–– SEED-IV/
|–– EEGMAT/
|–– SEED-VIG/
```

Datasets list:
- [SEED](#seed)
- [SEED-IV](#seed-iv)
- [EEGMAT](#eegmat)
- [SEED-VIG](#seed-vig) 
- [BCI-IV-2A](#bci-iv-2a) 
- [SHU](#shu) 
- [Things-EEG](#things-eeg)
- [TUEV](#tuev)
- [TUAB](#tuab)
- [Siena](#siena)
- [HMC](#hmc)
- [SHHS](#shhs)
- [Sleep-EDF](#sleep-edf)

## Dataset Download

### SEED
              
- Create a folder named `/SEED/raw_data` under `$DATA`.
- Download the dataset from the [official website](https://bcmi.sjtu.edu.cn/~seed/seed.html) and put the data to `$DATA/SEED/raw_data`. The directory structure should look like
```
SEED/raw_data
|–– Preprocessed_EEG
```
                          
### SEED-IV

- Create a folder named `/SEED-IV` under `$DATA`.
- Download the dataset from the [official website](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html) and put the data to `$DATA/SEED-IV/raw_data`. The directory structure should look like
```
SEED-IV/raw_data
|–– 1
|–– 2
|–– 3
```


### EEGMAT

- Create a folder named `/EEGMAT/raw_data` under `$DATA`.
- Download the dataset from the [official website](https://physionet.org/content/eegmat/1.0.0/) and put the data to `$DATA/EEGMAT/raw_data`. The directory structure should look like

```
EEGMAT/raw_data
|–– *SubjectXX_1.edf*
|–– *SubjectXX_2.edf*
```
XX denotes the subject ID (ranging from 00 to 35).

### SEED-VIG

- Create a folder named `/SEED-VIG/raw_data` under `$DATA`.
- Download the dataset from the [official website](https://bcmi.sjtu.edu.cn/~seed/seed-vig.html) and put the data to `$DATA/SEED-VIG/raw_data`. The directory structure should look like

```
SEED-VIG/raw_data
|–– perclos_labels
|–– Raw_Data
```

### BCI-IV-2A

- Create a folder named `/BCI-IV-2A/raw_data` under `$DATA`.
- Download the dataset from the [official website](http://bnci-horizon-2020.eu/database/data-sets) and put the data to `$DATA/BCI-IV-2A/raw_data`. The directory structure should look like

```
BCI-IV-2A/raw_data
|–– *A0XT.mat*
|–– *A0XE.mat*
```

X denotes the subject ID (ranging from 1 to 9).


### SHU

- Create a folder named `/SHU/raw_data` under `$DATA`.
- Download the dataset from the [official website](https://figshare.com/articles/code/shu_dataset/19228725/3) and put the data to `$DATA/SHU/raw_data`. The directory structure should look like

```
SHU/raw_data
|–– *sub-0XX_ses-0Y_task-motorimagery_eeg.mat*
```

XX denotes the subject ID (ranging from 01 to 25) and Y indicates the session number (ranging from 1 to 5).

### Things-EEG

- Create a folder named `/Things-EEG/raw_data` under `$DATA`.
- Download the dataset from the [official website](https://drive.google.com/drive/folders/1KnOcV38RthPcpZR2vtiSm0jtZ6p63RNt?usp=share_link) and put the data to `$DATA/Things-EEG/raw_data`. The directory structure should look like

```
Things-EEG/raw_data
|–– sub-XX
```
XX denotes the subject ID (ranging from 01 to 10).

### TUEV

- Create a folder named `/TUEV/raw_data` under `$DATA`.
- Download the dataset from the [official website](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) and put the data to `$DATA/TUEV/raw_data`. The directory structure should look like

```
TUEV/raw_data
|–– v2.0.1
```


### TUAB
- Create a folder named `/TUAB/raw_data` under `$DATA`.
- Download the dataset from the [official website](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) and put the data to `$DATA/TUAB/raw_data`. The directory structure should look like
```
TUAB/raw_data
|–– v3.0.1
```


### Siena

- Create a folder named `/Siena/raw_data` under `$DATA`.
- Download the dataset from the [official website](https://physionet.org/content/siena-scalp-eeg/1.0.0/) and put the data to `$DATA/Siena/raw_data`. The directory structure should look like
```
Siena/raw_data
|–– PNXX
```
XX denotes the subject ID (ranging from 00 to 17).

### HMC

- Create a folder named `/HMC/raw_data` under `$DATA`.
- Download the dataset from the [official website](https://physionet.org/content/hmc-sleep-staging/1.1/) and put the data to `$DATA/HMC/raw_data`. The directory structure should look like
```
HMC/raw_data
|–– 1.1
```


### SHHS

- Create a folder named `/SHHS/raw_data` under `$DATA`.
- Download the dataset from the [official website](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/EAMYFO) and put the data to `$DATA/SHHS/raw_data`. The directory structure should look like
```
SHHS/raw_data
|–– *shhs1-20XXXX*
```


### Sleep-EDF

- Create a folder named `/Sleep-EDF/raw_data` under `$DATA`.
- Download the dataset from the [official website](https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/#files-panel) and put the data to `$DATA/Sleep-EDF/raw_data`. The directory structure should look like
```
Sleep-EDF/raw_data
|–– sleep-cassette
```

## Data Preprocess            
Subsequently, to complete the data preprocessing pipeline, navigate to `AdaBrain-Bench` folder and execute `data_preprocess.sh` by specifying the data root and dataset name to run `data_process.py` for each dataset:
```
bash .preprocessing/data_preprocess.sh
```
The preprocessed data are stored in folder `processed_data` folder under data root $DATA. The directory structure should look like
```
$DATA/
|–– SEED/
|   |–– raw_data
|   |–– processed_data
```

Then, run the data splitting script to generate the data split JSON files in various settings: `cross_json_generate.py` for the cross-subejct transfer setting, `multi_json_generate.py` for the multi-subejct adaptation setting and few-shot transfer setting.
```
bash .preprocessing/json_process.sh
```
