"""Streamingly preprocess Things-EEG and save each trial as a pickle file.

The raw epoch spans -0.2 s to 1 s. The first 50 samples are discarded after
resampling to 250 Hz, so every saved trial keeps the original 0 s to 1 s
selection used by ``[..., 50:]``.
"""

import gc
import os
import pickle
import sys

import numpy as np
from sklearn.utils import shuffle


n_ses = 4
sfreq = 250
mvnn_dim = 'epochs'
seed = 20200220


def epoching_session(session, sub, raw_data_path, sfreq, data_part, seed):
    """Epoch and sort one session, returning data from 0 s to 1 s."""
    import mne

    chan_order = [
        'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
        'O1', 'Oz', 'O2',
    ]

    eeg_dir = os.path.join(
        f'sub-{sub:02d}',
        f'ses-{session + 1:02d}',
        f'raw_eeg_{data_part}.npy',
    )
    eeg_dict = np.load(
        os.path.join(raw_data_path, eeg_dir), allow_pickle=True
    ).item()
    ch_names = eeg_dict['ch_names']
    raw_sfreq = eeg_dict['sfreq']
    ch_types = eeg_dict['ch_types']
    eeg_data = eeg_dict['raw_eeg_data']
    del eeg_dict

    info = mne.create_info(ch_names, raw_sfreq, ch_types)
    raw = mne.io.RawArray(eeg_data, info)
    del eeg_data

    events = mne.find_events(raw, stim_channel='stim')
    raw.pick_channels(chan_order, ordered=True)
    events = events[events[:, 2] != 99999]

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.2,
        tmax=1.0,
        baseline=(None, 0),
        preload=True,
    )
    del raw

    if sfreq < 1000:
        epochs.resample(sfreq)

    # MNE 1.10 supports copy=False. Keep the epoch buffer owned by ``epochs``
    data = epochs.get_data(copy=False)
    event_ids = epochs.events[:, 2]
    img_conditions = np.unique(event_ids)
    max_rep = 20 if data_part == 'test' else 2

    if data.shape[2] <= 50:
        raise ValueError(
            f'Session {session + 1} has only {data.shape[2]} time samples; '
            'cannot discard the first 50 samples'
        )

    # Allocate only the 0 s to 1 s range. This preserves the exact meaning of
    # the old ``sorted_data[:, :, :, 50:]`` without retaining its full base.
    sorted_data = np.empty(
        (
            len(img_conditions),
            max_rep,
            data.shape[1],
            data.shape[2] - 50,
        )
    )
    for i, condition in enumerate(img_conditions):
        indices = np.flatnonzero(event_ids == condition)
        indices = shuffle(indices, random_state=seed, n_samples=max_rep)
        sorted_data[i] = data[indices, :, 50:]

    del data, epochs
    return sorted_data, img_conditions


def compute_mvnn_whitener(train_data, mvnn_dim):
    """Compute one session's MVNN matrix from training data only."""
    import scipy
    from sklearn.discriminant_analysis import _cov
    from tqdm import tqdm

    if mvnn_dim not in {'time', 'epochs'}:
        raise ValueError("mvnn_dim must be either 'time' or 'epochs'")

    n_channels = train_data.shape[2]
    sigma_total = np.zeros((n_channels, n_channels), dtype=np.float64)

    # Accumulate instead of allocating a covariance matrix for every image
    # condition. This is mathematically equivalent to the previous means.
    for i in tqdm(range(train_data.shape[0]), desc='Computing MVNN'):
        condition_data = train_data[i]
        condition_sigma = np.zeros_like(sigma_total)

        if mvnn_dim == 'time':
            for t in range(condition_data.shape[2]):
                condition_sigma += _cov(
                    condition_data[:, :, t], shrinkage='auto'
                )
            condition_sigma /= condition_data.shape[2]
        else:
            for repetition in range(condition_data.shape[0]):
                condition_sigma += _cov(
                    condition_data[repetition].T, shrinkage='auto'
                )
            condition_sigma /= condition_data.shape[0]

        sigma_total += condition_sigma

    sigma_total /= train_data.shape[0]
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma_total, -0.5)
    sigma_inv = np.real_if_close(sigma_inv)
    if np.iscomplexobj(sigma_inv):
        raise ValueError('MVNN inverse covariance matrix contains complex values')
    return np.asarray(sigma_inv)


def whiten_inplace(data, sigma_inv, chunk_size=64):
    """Whiten small condition chunks and write them back into ``data``."""
    for start in range(0, data.shape[0], chunk_size):
        stop = min(start + chunk_size, data.shape[0])
        whitened = np.einsum(
            '...ct,cd->...dt',
            data[start:stop],
            sigma_inv,
            optimize=True,
        )
        data[start:stop] = whitened


def inverse_shuffle(size, seed):
    """Map a source repetition index to its old post-shuffle output index."""
    permutation = shuffle(np.arange(size), random_state=seed)
    inverse = np.empty(size, dtype=np.int64)
    inverse[permutation] = np.arange(size)
    return inverse


def dump_trial(path, data, label):
    with open(path, 'wb') as save_file:
        pickle.dump({'X': data, 'Y': int(label)}, save_file, protocol=4)


def save_training_session(
    processed_data_path,
    sub,
    train_data,
    img_conditions,
    condition_occurrences,
    seed,
):
    """Save one training session while retaining the old trial numbering."""
    save_dir = os.path.join(
        processed_data_path, 'subjects_data', 'train', f'sub-{sub:02d}'
    )
    os.makedirs(save_dir, exist_ok=True)

    reps_per_session = train_data.shape[1]
    # In Things-EEG each training condition occurs in two sessions. The old
    # code concatenated those sessions and then applied this four-item shuffle.
    source_to_trial = inverse_shuffle(reps_per_session * 2, seed)

    for local_condition, condition in enumerate(img_conditions):
        condition = int(condition)
        occurrence = condition_occurrences.get(condition, 0)
        if occurrence >= 2:
            raise ValueError(
                f'Training condition {condition} occurs more than twice'
            )

        for repetition in range(reps_per_session):
            source_rep = occurrence * reps_per_session + repetition
            trial = int(source_to_trial[source_rep]) + 1
            label = (condition - 1) // 10 + 1
            save_path = os.path.join(
                save_dir, f'S{sub}_{condition}_{trial}.pkl'
            )
            dump_trial(
                save_path,
                train_data[local_condition, repetition],
                label,
            )

        condition_occurrences[condition] = occurrence + 1


def save_test_session(
    processed_data_path,
    sub,
    session,
    test_data,
    img_conditions,
    seed,
):
    """Save one test session while retaining the old 80-trial numbering."""
    save_dir = os.path.join(
        processed_data_path, 'subjects_data', 'test', f'sub-{sub:02d}'
    )
    os.makedirs(save_dir, exist_ok=True)

    reps_per_session = test_data.shape[1]
    source_to_trial = inverse_shuffle(reps_per_session * n_ses, seed)

    for local_condition, condition in enumerate(img_conditions):
        condition = int(condition)
        for repetition in range(reps_per_session):
            source_rep = session * reps_per_session + repetition
            trial = int(source_to_trial[source_rep]) + 1
            save_path = os.path.join(
                save_dir, f'S{sub}_{condition}_{trial}.pkl'
            )
            dump_trial(
                save_path,
                test_data[local_condition, repetition],
                condition,
            )


def validate_training_conditions(condition_occurrences, sub):
    invalid = {
        condition: count
        for condition, count in condition_occurrences.items()
        if count != 2
    }
    if invalid:
        preview = list(invalid.items())[:10]
        raise ValueError(
            f'Subject {sub} has training conditions that do not occur twice: '
            f'{preview}'
        )


def process_subject(sub, raw_data_path, processed_data_path):
    """Process and release one session at a time for one subject."""
    condition_occurrences = {}

    for session in range(n_ses):
        print(
            f'Processing subject {sub}/10, session {session + 1}/{n_ses}, '
            'training data...'
        )
        train_data, train_conditions = epoching_session(
            session,
            sub,
            raw_data_path,
            sfreq,
            'training',
            seed,
        )
        sigma_inv = compute_mvnn_whitener(train_data, mvnn_dim)
        whiten_inplace(train_data, sigma_inv)
        save_training_session(
            processed_data_path,
            sub,
            train_data,
            train_conditions,
            condition_occurrences,
            seed,
        )
        del train_data, train_conditions
        gc.collect()

        print(
            f'Processing subject {sub}/10, session {session + 1}/{n_ses}, '
            'test data...'
        )
        test_data, test_conditions = epoching_session(
            session,
            sub,
            raw_data_path,
            sfreq,
            'test',
            seed,
        )
        whiten_inplace(test_data, sigma_inv)
        save_test_session(
            processed_data_path,
            sub,
            session,
            test_data,
            test_conditions,
            seed,
        )
        del test_data, test_conditions, sigma_inv
        gc.collect()

    validate_training_conditions(condition_occurrences, sub)


if __name__ == '__main__':
    data_root = sys.argv[1]
    print(f'Data root: {data_root}')
    raw_data_path = os.path.join(data_root, 'Things-EEG', 'raw_data')
    processed_data_path = os.path.join(
        data_root, 'Things-EEG', 'processed_data'
    )
    os.makedirs(processed_data_path, exist_ok=True)

    for sub in range(1, 11):
        process_subject(sub, raw_data_path, processed_data_path)
