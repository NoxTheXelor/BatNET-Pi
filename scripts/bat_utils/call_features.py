import numpy as np
from skimage.util.shape import view_as_windows

import bat_utils.spectrogram as sp

def compute_features_call(audio_samples, sampling_rate, params):
    """
    Computes the features related to the calls of the audio samples.

    Parameters
    -----------
    audio_samples : numpy array
        Data read from a wav file.
    sampling_rate : int
        Sample rate of a wav file.
    params : DataSetParams
        Parameters of the model.

    Returns
    -----------
    features_call : numpy array
        Array containing the 18 features call for each window of the audio file.
    """

    # load audio and create spectrogram
    spectrogram = sp.gen_spectrogram(audio_samples, sampling_rate, params.fft_win_length, params.fft_overlap,
                                     crop_spec=params.crop_spec, max_freq=params.max_freq, min_freq=params.min_freq)
    spectrogram = sp.process_spectrogram(spectrogram, denoise_spec=params.denoise, mean_log_mag=params.mean_log_mag, smooth_spec=params.smooth_spec)
    
    total_win_size = spectrogram.shape[1]
    spectrogram = view_as_windows(spectrogram, (spectrogram.shape[0], params.window_width))[0]
    spectrogram = spectrogram.sum(axis=2)
    spectrogram = spectrogram.T
    sorted_mag = np.sort(spectrogram, axis=0)
    sorted_mag_freq = np.argsort(spectrogram, axis=0)

    # max magnitude, peak frequency
    max_mag = sorted_mag[-1, :]
    max_mag_freq = sorted_mag_freq[-1, :]
    stacked = np.vstack((max_mag, max_mag_freq))
    # min magnitude
    min_mag = sorted_mag[0, :]
    min_mag_freq = sorted_mag_freq[0, :]
    stacked = np.vstack((stacked, min_mag, min_mag_freq))
    # center magnitude
    center_mag = (max_mag + min_mag) / 2
    center_mag_freq = (max_mag_freq + min_mag_freq) / 2
    stacked = np.vstack((stacked, center_mag, center_mag_freq))
    # mean magnitude
    mean_mag = np.mean(spectrogram, axis=0)
    stacked = np.vstack((stacked, mean_mag))

    # max frequency
    max_freq = spectrogram.shape[0] - 1 - np.argmax(np.flip(spectrogram!=0, axis=0), axis=0)
    stacked = np.vstack((stacked, max_freq))
    # min frequency
    min_freq = np.argmax(spectrogram!=0, axis=0)
    stacked = np.vstack((stacked, min_freq))
    # center frequency
    center_freq = (max_freq + min_freq) / 2
    stacked = np.vstack((stacked, center_freq))
    # mean frequence
    indices = np.arange(0,spectrogram.shape[0])
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_freq = np.true_divide(np.dot(indices, spectrogram), np.sum(spectrogram, axis=0))
        mean_freq[ ~ np.isfinite( mean_freq )] = 0
    stacked = np.vstack((stacked, mean_freq))
    
    # Bandwidth containing x% of call energy
    percentage = [0.9, 0.5]
    for p in percentage:
        limit_min = spectrogram.sum(axis=0) * (1 - p)/2
        limit_max = spectrogram.sum(axis=0) * (p + (1 - p)/2)
        band_min_freq = np.argmin((spectrogram.cumsum(axis=0) < limit_min), axis=0)
        band_max_freq = np.argmin((spectrogram.cumsum(axis=0) < limit_max), axis=0)
        bandwidth = band_max_freq - band_min_freq
        stacked = np.vstack((stacked, bandwidth))
    
    #  Frequency at x% of total call energy
    percentage = [0.05, 0.95, 0.5, 0.75, 0.25]
    for p in percentage:
        limit = spectrogram.sum(axis=0) * p
        freq_percentage = np.argmin((spectrogram.cumsum(axis=0) < limit), axis=0)
        stacked = np.vstack((stacked, freq_percentage))

    stacked = stacked.T
    
    # pad on extra features at the end as the sliding window will mean its a different size
    features = stacked.reshape((stacked.shape[0], np.prod(stacked.shape[1:])))
    features = np.vstack((features, np.tile(features[-1, :], (total_win_size - features.shape[0], 1))))
    features = features.astype(np.float32)

    # make the correct size for CNN
    features_padding = np.zeros((total_win_size, features.shape[1]), dtype=np.float32)
    features_padding[:features.shape[0], :] = features
    features_call = features_padding
    return features_call






