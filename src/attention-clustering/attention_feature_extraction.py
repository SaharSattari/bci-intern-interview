import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import welch
from fooof import FOOOF
import seaborn as sns
from matplotlib import pyplot as plt


class AttentionFeatureExtraction:

    def __init__(self, eeg_data, sampling_rate):
        """
        Initialize the AttentionPrep object with EEG data and sampling rate.

        Args:
            eeg_data (numpy.ndarray): The EEG data as a 2D numpy array.
            sampling_rate (int): The sampling rate of the EEG data.

        Raises:
            TypeError: If eeg_data is not a numpy array or sampling_rate is not an integer.
            ValueError: If eeg_data is not a 2D array.

        """
        # Validate the input data
        if not isinstance(eeg_data, np.ndarray):
            raise TypeError("eeg_data must be a numpy array")
        if not isinstance(sampling_rate, int):
            raise TypeError("sampling_rate must be an integer")
        if eeg_data.ndim != 2:
            raise ValueError("eeg_data must be a 2D array")

        self.eeg_data = eeg_data
        self.sampling_rate = sampling_rate

        # Transpose data if necessary - ensures the channels are in the rows
        if self.eeg_data.shape[0] > self.eeg_data.shape[1]:
            self.eeg_data = self.eeg_data.T

    def filtering(self, lowcut, highcut):
        """
        Apply band-pass filtering to the EEG data.

        Parameters:
        - lowcut (float): The lower cutoff frequency for the band-pass filter (Hz)
        - highcut (float): The higher cutoff frequency (Hz)

        Returns:
        - filtered_data (ndarray): The filtered EEG data.

        """
        filtered_data = np.zeros(self.eeg_data.shape)

        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        # Designing band-pass butterworth filter with order 6
        b, a = butter(6, [low, high], btype="band")

        # Apply the filter to each channel of the EEG data
        for i in range(self.eeg_data.shape[0]):
            filtered_data[i, :] = filtfilt(
                b, a, self.eeg_data[i, :]
            )  ###### Here I am using filtfilt for zero phase shift but it can not be used in realtime processing

        return filtered_data

    def compute_power_spectral_density(
        self, use_filtered_data=True, lowcut=1, highcut=40, f_res=250
    ):
        """
        Compute the power spectral density (PSD) of the EEG data.

        Parameters:
        - use_filtered_data (bool): Flag indicating whether to use filtered data or raw data for computing PSD (default: True).
        - lowcut (float): Lower cutoff frequency for the filter (if use_filtered_data is True).
        - highcut (float): Upper cutoff frequency for the filter (if use_filtered_data is True).
        - f_res (int): Number of data points to use for each segment in the PSD calculation in welch method. (default: 250)

        Returns:
        - psd_result (list): List of power spectral density values for each channel of the EEG data.
        - f (array): Array of frequency values corresponding to the PSD.

        The PSD is computed using Welch's method, which divides the data into overlapping segments and applies a Fourier transform to each segment.
        The power values are then averaged across segments to obtain the final PSD.

        """

        # Use filtered data if requested, otherwise use raw data
        if use_filtered_data and lowcut and highcut:
            data = self.filtering(lowcut, highcut)
        else:
            data = self.eeg_data

        psd_result = []
        for i in range(data.shape[0]):
            # Compute power spectral density (PSD) using Welch's method
            f, Pxx = welch(
                data[i, :], fs=self.sampling_rate, nperseg=f_res, noverlap=f_res / 2
            )

            # Convert power to decibels (dB/Hz)
            # Pxx_dB = 10 * np.log10(Pxx)

            # Append the PSD values to the result list (if filtered data is used)
            if use_filtered_data and lowcut and highcut:
                psd_result.append(Pxx[(f >= lowcut) & (f <= highcut)])
                F = f[(f >= lowcut) & (f <= highcut)]
            # no filtering
            else:
                psd_result.append(Pxx)
                f = F
        return psd_result, F

    def fooof_fit(self, psd_result, f):
        """
        Fit a FOOOF model to the power spectral density data. Reference is provided in the readme file.

        Parameters:
        - psd_result (list): A list of power spectral density data for each channel.
        - f (numpy array): The frequency values corresponding to the power spectral density data.

        Returns:
        - peak_param (list): A list of peak parameters for each channel.

        """
        peak_param = []
        for ch in range(len(psd_result)):
            fm = FOOOF(
                peak_width_limits=[1, 8], max_n_peaks=8, min_peak_height=0.1
            )  # initialize FOOOF model
            fm.fit(
                f, psd_result[ch]
            )  # fit the model to the power spectral density data of each channel
            peak_param.append(
                fm.get_params("peak_params")
            )  # get the peak parameters (frequency, amplitude, and width)

        return peak_param

    def peak_power(self, peak_param, label, visual=False):
        """
        Providing and visualizing the peak amplitudes in the power spectral density data.

        Parameters:
        - peak_param (list): A list of peak parameters for each channel.
        - label (str): Data label for visualization. (e.g. "Cognitive Task")
        - visual (bool): Flag to visualize the output matrix (default: False).

        Returns:
        - peak_matrix (numpy.ndarray): A 4x4 matrix containing the peak amplitudes for each frequency band and channel.

        This method extracts the largest peak value within each frequency band (delta, theta, alpha and beta) for each channel and creates a heatmap to visualize the data.
        """

        # define frequency bands
        frequency_bands = {
            "delta": [1, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "beta": [13, 30],
        }

        # define channel names (based on the provided dataset)
        channel_names = ["O1", "O2", "T3", "T4"]
        num_channels = len(channel_names)

        # to store the peak values for each frequency band
        band_peak_values = {band: [] for band in frequency_bands}

        # extracting peak amplitude within each frequency band for each channel
        for ch in range(len(peak_param)):
            peaks = peak_param[ch]
            for band, (low, high) in frequency_bands.items():
                band_peaks = [
                    p for p in peaks if low <= p[0] <= high
                ]  # checking the frequency of the peaks
                if band_peaks:
                    largest_peak = max(
                        band_peaks, key=lambda p: p[1]
                    )  # store the amplitude of the largest peak
                    band_peak_values[band].append(
                        largest_peak[1]
                    )  # Append the peak amplitude
                else:
                    band_peak_values[band].append(0)  # No peak found in this band

        # creating output matrix (4 rows for frequency bands, 4 columns for channels)
        peak_matrix = np.array([band_peak_values[band] for band in frequency_bands])

        # Visualizing the output matrix
        if visual:
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                peak_matrix,
                annot=True,
                cmap="coolwarm",
                cbar=True,
                xticklabels=channel_names,
                yticklabels=[band.capitalize() for band in frequency_bands],
                vmin=0,
                vmax=1,
            )
            plt.title(f"Peak Amplitudes - Data: {label}")
            plt.xlabel("Channels")
            plt.ylabel("Frequency Bands")
            plt.show()

        return peak_matrix

    def relative_power(self, psd_result, f, label, visual=False):
        """
        Compute the relative power in each frequency band for each channel.

        Parameters:
        - psd_result (list): The power spectral density (PSD) result for each channel.
        - f (numpy.ndarray): The frequency values corresponding to the PSD result.
        - label (str): Data label for visualization. (e.g. "Cognitive Task").
        - visual (bool): Flag to visualize the output matrix (default: False).

        Returns:
        - relative power matrix (numpy.ndarray): A 4x4 matrix containing the relative power values for each frequency band and channel.

        this method computes the total power and the power in each frequency band and reports the relative power
        in each frequency band and for each channel.
        """

        frequency_bands = {
            "delta": [1, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "beta": [13, 30],
        }
        channel_names = ["O1", "O2", "T3", "T4"]
        num_channels = len(channel_names)

        band_power_values = {band: [] for band in frequency_bands}

        for ch in range(len(psd_result)):
            total_power = np.trapz(psd_result[ch], f)

            # power within each frequency band
            for band, (low, high) in frequency_bands.items():
                indices = np.where((f >= low) & (f <= high))[0]
                band_power = np.trapz(psd_result[ch][indices], f[indices])

                # relative power in the band
                relative_power = band_power / total_power
                band_power_values[band].append(relative_power)

        # 4*4 matrix containing the relative power in each frequency band for each channel
        power_matrix = np.array([band_power_values[band] for band in frequency_bands])

        # visualization of the ourput matrix
        if visual:
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                power_matrix,
                annot=True,
                cmap="coolwarm",
                cbar=True,
                xticklabels=channel_names,
                yticklabels=[band.capitalize() for band in frequency_bands],
                vmin=0,
                vmax=1,
            )
            plt.title(f"Relative Power - Data: {label}")
            plt.xlabel("Channels")
            plt.ylabel("Frequency Bands")
            # plt.show()
            plt.show()

        return power_matrix
