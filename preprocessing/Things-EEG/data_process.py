"""
Refer to the code of Things-EEG2 but with a few differences. 
Many thanks!
https://www.sciencedirect.com/science/article/pii/S1053811922008758
"""

"""Preprocess the  raw EEG data: channel selection, epoching, frequency
downsampling, baseline correction, multivariate noise normalization (MVNN),
sorting of the data image conditions and reshaping the data to:
Image conditions × EEG repetitions × EEG channels × EEG time points.
Then, the data of both test and training EEG partitions is saved.

Parameters
----------
sub : int
	Used subject.
n_ses : int
	Number of EEG sessions.
sfreq : int
	Downsampling frequency.
mvnn_dim : str
	Whether to compute the MVNN covariace matrices for each time point
	('time') or for each epoch/repetition ('epochs').

"""
import numpy as np
from sklearn.utils import shuffle
import os
import pickle
import sys
import pdb




# raw_data_dir = './Preprocessing/Things-EEG/raw_data'
# preprocessed_data_250Hz_dir = './Preprocessing/Things-EEG/preprocessed_data_250Hz'


n_ses = 4
sfreq = 250
mvnn_dim = 'epochs'
seed = 20200220


def epoching(n_ses, sub, raw_data_path, sfreq, data_part, seed):
	"""This function first converts the EEG data to MNE raw format, and
	performs channel selection, epoching, baseline correction and frequency
	downsampling. Then, it sorts the EEG data of each session according to the
	image conditions.

	Returns
	-------
	epoched_data : list of float
		Epoched EEG data.
	img_conditions : list of int
		Unique image conditions of the epoched and sorted EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.

	"""

	import os
	import mne
	import numpy as np
	from sklearn.utils import shuffle

	chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
				  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
				  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
				  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
				  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
				  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
				  'O1', 'Oz', 'O2']

	### Loop across data collection sessions ###
	epoched_data = []
	img_conditions = []
	for s in range(n_ses):

		### Load the EEG data and convert it to MNE raw format ###
		eeg_dir = os.path.join('sub-'+ format(sub,'02'), 'ses-'+format(s+1,'02'), 
				       'raw_eeg_' + data_part + '.npy')
		eeg_data = np.load(os.path.join(raw_data_path, eeg_dir), allow_pickle=True).item()
		ch_names = eeg_data['ch_names']
		sfreq = eeg_data['sfreq']
		ch_types = eeg_data['ch_types']
		eeg_data = eeg_data['raw_eeg_data']
		# Convert to MNE raw format
		info = mne.create_info(ch_names, sfreq, ch_types)
		raw = mne.io.RawArray(eeg_data, info)
		del eeg_data

		### Get events, drop unused channels and reject target trials ###
		events = mne.find_events(raw, stim_channel='stim')
		# # Select only occipital (O) and posterior (P) channels
		# chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],
		# 	'^O *|^P *'))
		# new_chans = [raw.info['ch_names'][c] for c in chan_idx]
		# raw.pick_channels(new_chans)
		# * chose all channels
		raw.pick_channels(chan_order, ordered=True)
		# Reject the target trials (event 99999)
		idx_target = np.where(events[:,2] == 99999)[0]
		events = np.delete(events, idx_target, 0)
		### Epoching, baseline correction and resampling ###
		# * [0, 1.0]
		epochs = mne.Epochs(raw, events, tmin=-.2, tmax=1.0, baseline=(None,0),
			preload=True)
		# epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, baseline=(None,0),
		# 	preload=True)
		del raw
		# Resampling
		if sfreq < 1000:
			epochs.resample(sfreq)
		ch_names = epochs.info['ch_names']
		times = epochs.times

		### Sort the data ###
		data = epochs.get_data()
		events = epochs.events[:,2]
		img_cond = np.unique(events)
		del epochs
		# Select only a maximum number of EEG repetitions
		if data_part == 'test':
			max_rep = 20
		else:
			max_rep = 2
		# Sorted data matrix of shape:
		# Image conditions × EEG repetitions × EEG channels × EEG time points
		sorted_data = np.zeros((len(img_cond),max_rep,data.shape[1],
			data.shape[2]))
		for i in range(len(img_cond)):
			# Find the indices of the selected image condition
			idx = np.where(events == img_cond[i])[0]
			# Randomly select only the max number of EEG repetitions
			idx = shuffle(idx, random_state=seed, n_samples=max_rep)
			sorted_data[i] = data[idx]
		del data
		epoched_data.append(sorted_data[:, :, :, 50:])
		img_conditions.append(img_cond)
		del sorted_data

	### Output ###
	return epoched_data, img_conditions, ch_names, times


def mvnn(n_ses, mvnn_dim, epoched_test, epoched_train):
	"""Compute the covariance matrices of the EEG data (calculated for each
	time-point or epoch/repetitions of each image condition), and then average
	them across image conditions and data partitions. The inverse of the
	resulting averaged covariance matrix is used to whiten the EEG data
	(independently for each session).
	
	zero-score standardization also has well performance

	Parameters
	----------
	epoched_test : list of floats
		Epoched test EEG data.
	epoched_train : list of floats
		Epoched training EEG data.

	Returns
	-------
	whitened_test : list of float
		Whitened test EEG data.
	whitened_train : list of float
		Whitened training EEG data.

	"""

	import numpy as np
	from tqdm import tqdm
	from sklearn.discriminant_analysis import _cov
	import scipy

	### Loop across data collection sessions ###
	whitened_test = []
	whitened_train = []
	for s in range(n_ses):
		session_data = [epoched_test[s], epoched_train[s]]

		### Compute the covariance matrices ###
		# Data partitions covariance matrix of shape:
		# Data partitions × EEG channels × EEG channels
		sigma_part = np.empty((len(session_data),session_data[0].shape[2],
			session_data[0].shape[2]))
		for p in range(sigma_part.shape[0]):
			# Image conditions covariance matrix of shape:
			# Image conditions × EEG channels × EEG channels
			sigma_cond = np.empty((session_data[p].shape[0],
				session_data[0].shape[2],session_data[0].shape[2]))
			for i in tqdm(range(session_data[p].shape[0])):
				cond_data = session_data[p][i]
				# Compute covariace matrices at each time point, and then
				# average across time points
				if mvnn_dim == "time":
					sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
						shrinkage='auto') for t in range(cond_data.shape[2])],
						axis=0)
				# Compute covariace matrices at each epoch (EEG repetition),
				# and then average across epochs/repetitions
				elif mvnn_dim == "epochs":
					sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]),
						shrinkage='auto') for e in range(cond_data.shape[0])],
						axis=0)
			# Average the covariance matrices across image conditions
			sigma_part[p] = sigma_cond.mean(axis=0)
		# # Average the covariance matrices across image partitions
		# sigma_tot = sigma_part.mean(axis=0)
		# ? It seems not fair to use test data for mvnn, so we change to just use training data
		sigma_tot = sigma_part[1]
		# Compute the inverse of the covariance matrix
		sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

		### Whiten the data ###
		whitened_test.append(np.reshape((np.reshape(session_data[0], (-1,
			session_data[0].shape[2],session_data[0].shape[3])).swapaxes(1, 2)
			@ sigma_inv).swapaxes(1, 2), session_data[0].shape))
		whitened_train.append(np.reshape((np.reshape(session_data[1], (-1,
			session_data[1].shape[2],session_data[1].shape[3])).swapaxes(1, 2)
				@ sigma_inv).swapaxes(1, 2), session_data[1].shape))

	### Output ###
	return whitened_test, whitened_train


def save_prepr(n_ses, processed_data_path, sub, whitened_test, whitened_train, img_conditions_train,
	ch_names, times, seed):
	"""Merge the EEG data of all sessions together, shuffle the EEG repetitions
	across sessions and reshaping the data to the format:
	Image conditions × EGG repetitions × EEG channels × EEG time points.
	Then, the data of both test and training EEG partitions is saved.

	Parameters
	----------
		Input arguments.
	whitened_test : list of float
		Whitened test EEG data.
	whitened_train : list of float
		Whitened training EEG data.
	img_conditions_train : list of int
		Unique image conditions of the epoched and sorted train EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.
	seed : int
		Random seed.

	"""

	### Merge and save the test data ###
	for s in range(n_ses):
		if s == 0:
			merged_test = whitened_test[s]
		else:
			merged_test = np.append(merged_test, whitened_test[s], 1)
	del whitened_test
	# Shuffle the repetitions of different sessions
	idx = shuffle(np.arange(0, merged_test.shape[1]), random_state=seed)
	merged_test = merged_test[:,idx]
	# Insert the data into a dictionary
	test_dict = {
		'preprocessed_eeg_data': merged_test,
		'ch_names': ch_names,
		'times': times
	}
	del merged_test
	# Saving directories
	save_dir = os.path.join(processed_data_path, 'sub-'+format(sub,'02'))
	file_name_test = 'preprocessed_eeg_test.npy'
	file_name_train = 'preprocessed_eeg_training.npy'
	# Create the directory if not existing and save the data
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	# np.save(os.path.join(save_dir, file_name_test), test_dict)
	save_pic = open(os.path.join(save_dir, file_name_test), 'wb')
	pickle.dump(test_dict, save_pic, protocol=4)
	save_pic.close()
	del test_dict

	### Merge and save the training data ###
	for s in range(n_ses):
		if s == 0:
			white_data = whitened_train[s]
			img_cond = img_conditions_train[s]
		else:
			white_data = np.append(white_data, whitened_train[s], 0)
			img_cond = np.append(img_cond, img_conditions_train[s], 0)
	del whitened_train, img_conditions_train
	# Data matrix of shape:
	# Image conditions × EGG repetitions × EEG channels × EEG time points
	merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1]*2,
		white_data.shape[2],white_data.shape[3]))
	for i in range(len(np.unique(img_cond))):
		# Find the indices of the selected category
		idx = np.where(img_cond == i+1)[0]
		for r in range(len(idx)):
			if r == 0:
				ordered_data = white_data[idx[r]]
			else:
				ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
		merged_train[i] = ordered_data
	# Shuffle the repetitions of different sessions
	idx = shuffle(np.arange(0, merged_train.shape[1]), random_state=seed)
	merged_train = merged_train[:,idx]
	# Insert the data into a dictionary
	train_dict = {
		'preprocessed_eeg_data': merged_train,
		'ch_names': ch_names,
		'times': times
	}
	del merged_train
	# Create the directory if not existing and save the data
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)
	# np.save(os.path.join(save_dir, file_name_train),
	# 	train_dict)
	save_pic = open(os.path.join(save_dir, file_name_train), 'wb')
	pickle.dump(train_dict, save_pic, protocol=4)
	save_pic.close()
	del train_dict




def train_test_save(processed_data_path):
    save_folder=os.path.join(processed_data_path, 'subjects_data')
    #### train data
    for sub_id in range(1, 11):
        train_file_path = f"{processed_data_path}/sub-{sub_id:02d}/preprocessed_eeg_training.npy"
        data = np.load(train_file_path, allow_pickle=True)
        data = data['preprocessed_eeg_data']    # 16540, 4, 63, 250
        save_folder_path = f"{save_folder}/train/sub-{sub_id:02d}"
        os.makedirs(save_folder_path, exist_ok=True)
        for per_img in range(data.shape[0]):
            print(f"Processing sub {sub_id}/10, img {per_img + 1}/{data.shape[0]}...")
            for per_trial in range(data.shape[1]):
                print(f"    Processing trial {per_trial + 1}/{data.shape[1]}...")
                per_data = data[per_img][per_trial]     # 63, 250
                if per_data.shape[0] != 63 and per_data.shape[1] != 250:
                    pdb.set_trace()
                per_label = per_img // 10 + 1
                save_file_path = f"{save_folder_path}/S{sub_id}_{per_img + 1}_{per_trial + 1}.pkl"
                pickle.dump(
                    {"X": per_data, "Y": per_label},
                    open(save_file_path, "wb"),
                )



    ##### test data
    for sub_id in range(1, 11):
        train_file_path = f"{processed_data_path}/sub-{sub_id:02d}/preprocessed_eeg_test.npy"
        data = np.load(train_file_path, allow_pickle=True)
        data = data['preprocessed_eeg_data']    # 200, 80, 63, 250
        save_folder_path = f"{save_folder}/test/sub-{sub_id:02d}"
        os.makedirs(save_folder_path, exist_ok=True)
        for per_img in range(data.shape[0]):    # 200
            print(f"Processing sub {sub_id}/10, img {per_img + 1}/{data.shape[0]}...")
            for per_trial in range(data.shape[1]):
                print(f"    Processing trial {per_trial + 1}/{data.shape[1]}...")
                per_data = data[per_img][per_trial]     # 63, 250
                if per_data.shape[0] != 63 and per_data.shape[1] != 250:
                    pdb.set_trace()
                per_label = per_img + 1
                save_file_path = f"{save_folder_path}/S{sub_id}_{per_img + 1}_{per_trial + 1}.pkl"
                pickle.dump(
                    {"X": per_data, "Y": per_label},
                    open(save_file_path, "wb"),
                )

if __name__ == "__main__":
	data_root = sys.argv[1]  
	print(f"Data root: {data_root}")
	raw_data_path = os.path.join(data_root,'Things-EEG/raw_data')
	processed_data_path = os.path.join(data_root,'Things-EEG/processed_data')
	os.makedirs(processed_data_path, exist_ok=True)
    
	for sub in range(1, 11):
		# =============================================================================
		# Epoch and sort the data
		# =============================================================================
		# Channel selection, epoching, baseline correction and frequency downsampling of
		# the test and training data partitions.
		# Then, the conditions are sorted and the EEG data is reshaped to:
		# Image conditions × EGG repetitions × EEG channels × EEG time points
		# This step is applied independently to the data of each partition and session.
		epoched_test, _, ch_names, times = epoching(n_ses, sub, raw_data_path, sfreq, 'test', seed)
		epoched_train, img_conditions_train, _, _ = epoching(n_ses, sub, raw_data_path, sfreq, 'training', seed)


		# =============================================================================
		# Multivariate Noise Normalization
		# =============================================================================
		# MVNN is applied independently to the data of each session.
		whitened_test, whitened_train = mvnn(n_ses, mvnn_dim, epoched_test, epoched_train)
		del epoched_test, epoched_train
		# =============================================================================
		# Merge and save the preprocessed data
		# =============================================================================
		# In this step the data of all sessions is merged into the shape:
		# Image conditions × EGG repetitions × EEG channels × EEG time points
		# Then, the preprocessed data of the test and training data partitions is saved.
		save_prepr(n_ses, processed_data_path, sub, whitened_test, whitened_train, img_conditions_train, ch_names,
			times, seed)

	train_test_save(processed_data_path)
