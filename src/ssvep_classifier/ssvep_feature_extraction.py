import numpy as np
from scipy.signal import welch


class SSVEPFeatureExtraction:
    def __init__(self, fs, data, freqs):
        """
        Initialize the SSVEPFeatureExtraction class.

        Args:
            fs (int): The sampling frequency.
            data (ndarray): The data matrix of shape (channels x time).
            freqs (list): The target frequencies for feature extraction.

        Attributes:
            fs (int): The sampling frequency.
            data (ndarray): The data matrix of shape (channels x time).
            freqs (list): The target frequencies for feature extraction.
        """
        self.fs = fs
        self.data = data
        if self.data.shape[0] > self.data.shape[1]:
            self.data = self.data.T
        self.freqs = freqs

    def spectral_signal(self, channel_data):
        """
        Compute the power spectral density for the given channel data using Welch's method.

        Parameters:
            channel_data (array-like): The input channel data.

        Returns:
            f (array): The frequencies corresponding to the power spectral density estimates.
            pxx (array): The power spectral density estimates.

        """
        f, pxx = welch(channel_data, fs=self.fs, nperseg=500, noverlap=250)
        return f, pxx

    def extract_features(self):
        """
        Extract features by finding the peak power spectral density in the specified frequency bands and their second harmonics.
        Returns:
            numpy.ndarray: A feature matrix of size (channels x (2*len(freqs))).
        """

        feature = np.zeros((self.data.shape[0], 2 * len(self.freqs)))

        for j, freq in enumerate(self.freqs):
            lowcut = freq - 0.5  # Lower bound of frequency range
            highcut = freq + 0.5  # Upper bound of frequency range

            harmonic_freq = 2 * freq
            harmonic_lowcut = harmonic_freq - 0.5
            harmonic_highcut = harmonic_freq + 0.5

            for i in range(self.data.shape[0]):
                f, pxx = self.spectral_signal(self.data[i, :])
                mask = (f >= lowcut) & (f <= highcut)
                harmonic_mask = (f >= harmonic_lowcut) & (f <= harmonic_highcut)
                if np.any(mask):
                    target_power = pxx[mask].max()
                    feature[i, j] = target_power

                if np.any(harmonic_mask):
                    harmonic_power = pxx[harmonic_mask].max()
                    feature[i, j + 4] = harmonic_power

        return feature
