import numpy as np
import scipy
import neurokit2 as nk


def resample(signal, original_fs, target_fs):
    # Signal -> (12, sample) numpy array

    if original_fs == target_fs:
        return signal

    resampled_signal = scipy.signal.resample_poly(signal, up=target_fs, down=original_fs, axis=1)

    return resampled_signal



def normalise_ecg(signal, sampling_rate = 500):
    processed = np.zeros_like(signal, dtype=np.float32)

    for i, channel in enumerate(signal):
        cleaned = nk.ecg_clean(channel, sampling_rate=sampling_rate)

        if np.isclose(np.std(cleaned), 0):
            processed[i] = np.zeros_like(cleaned)
        else:
            processed[i] = scipy.stats.zscore(cleaned)

    return processed


def pad_truncate_ecg(signal, target_length=5000):
    # Check if the signal is 1D (single lead) or 2D (multi-lead)
    is_1d = signal.ndim == 1

    if is_1d:
        # Temporarily expand to 2D (1, samples) so the indexing works
        signal = np.expand_dims(signal, axis=0)

    current_length = signal.shape[1]

    if current_length < target_length:
        pad_amount = target_length - current_length
        signal = np.pad(signal, ((0, 0), (0, pad_amount)), mode='edge')

    elif current_length > target_length:
        start = (current_length - target_length) // 2
        end = start + target_length
        signal = signal[:, start:end]

    if is_1d:
        # Squeeze it back down to 1D before returning
        signal = signal.squeeze(axis=0)

    return signal
