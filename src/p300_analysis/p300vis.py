import numpy as np
from matplotlib import pyplot as plt


class P300Vis:
    """
    A class for visualizing P300 event-related potentials.

    Attributes:
        epochs (np.ndarray): A 3D numpy array representing the epochs. (num_trials, num_channels, num_samples)
        sampling_rate (int): The sampling rate of the epochs.

    Methods:
        __init__(self, epochs, sampling_rate): Initializes the P300Vis object.
        Single_trial_vis(self, t_min, t_max, label): Visualizes single trial responses.
        Average_response_vis(self, t_min, t_max, label): Visualizes average responses.
        peak_detection(self, t_min, time_range=[0.2, 0.4]): Performs peak detection.

    """

    def __init__(self, epochs, sampling_rate):
        """
        Initializes the P300Vis object.

        Args:
            epochs (np.ndarray): A 3D numpy array representing the epochs.
            Important: epoch size should match: (num_trials, num_channels, num_samples)
            sampling_rate (int)
        """
        if not isinstance(sampling_rate, int):
            raise TypeError("sampling_rate must be an integer")

        if not isinstance(epochs, np.ndarray):
            raise TypeError("epochs must be a numpy array")
        if epochs.ndim != 3:
            raise ValueError("epochs must be a 3D numpy array")

        self.epoch = epochs
        self.sampling_rate = sampling_rate
        print("proper initialization")

    def Single_trial_vis(self, t_min, t_max, label):
        """
        Visualizes single trial responses.

        Args:
            t_min (float): The minimum time value for visualization. default -0.2 (200 ms before stimulus onset)
            t_max (float): The maximum time value for visualization. default 0.8 (800 ms after stimulus onset)
            label (str): The label for the visualization. (e.g., "standard" or "odd")

        """
        num_trials = self.epoch.shape[0]
        num_channels = self.epoch.shape[1]
        time = np.linspace(t_min, t_max, int((t_max - t_min) * self.sampling_rate))

        for ch in range(num_channels):
            plt.figure()
            for i in range(num_trials):
                plt.plot(time, self.epoch[i, ch, :])
            if num_channels > 1:
                plt.title(f"Channel {ch+1} for label {label}")
            else:
                plt.title(f"Wighted sum of channels for label {label}")
            plt.xticks(np.arange(t_min, t_max, 0.1))
            if t_min < 0:
                plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
            plt.axvspan(0.25, 0.35, color="pink", alpha=0.3)

            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.yticks(np.arange(-100e-6, 101e-6, 10e-6), np.arange(-100, 101, 10))
            plt.ylim(-100e-6, 100e-6)
            plt.grid()
            plt.show()

    def Average_response_vis(self, t_min, t_max, label):
        """
        Visualizes average responses.

        Args:
            t_min (float): The minimum time value for visualization. default -0.2 (200 ms before stimulus onset)
            t_max (float): The maximum time value for visualization. default 0.8 (800 ms after stimulus onset)
            label (str): The label for the visualization. (e.g., "standard" or "odd")

        """
        num_trials = self.epoch.shape[0]
        num_channels = self.epoch.shape[1]
        time = np.linspace(t_min, t_max, int((t_max - t_min) * self.sampling_rate))

        for ch in range(num_channels):
            plt.figure()
            average_response = np.zeros(int((t_max - t_min) * self.sampling_rate))
            all_responses = np.zeros(
                (num_trials, int((t_max - t_min) * self.sampling_rate))
            )
            for i in range(num_trials):
                all_responses[i, :] = self.epoch[i, ch, :]
                average_response += self.epoch[i, ch, :]
            average_response /= num_trials
            std_response = np.std(all_responses, axis=0)

            sem_response = std_response / np.sqrt(num_trials)
            confidence_interval = 1.96 * sem_response

            plt.plot(time, average_response, label="Average Response")
            plt.fill_between(
                time,
                average_response - confidence_interval,
                average_response + confidence_interval,
                color="gray",
                alpha=0.5,
                label="95% Confidence Interval",
            )
            plt.axvspan(0.25, 0.35, color="pink", alpha=0.3)

            if num_channels > 1:
                plt.title(f"Average response for channel {ch+1} for label {label}")
            else:
                plt.title(
                    f"Average response for weighted sum of channels for label {label}"
                )
            plt.xticks(np.arange(t_min, t_max, 0.1))
            if t_min < 0:
                plt.axvline(0, color="black", linewidth=0.5, linestyle="--")

            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.yticks(np.arange(-100e-6, 101e-6, 10e-6), np.arange(-100, 101, 10))
            plt.ylim(-30e-6, 30e-6)
            plt.legend()
            plt.grid()
            plt.show()

    def peak_detection(self, t_min, time_range=[0.25, 0.35]):
        """
        Performs peak detection.

        Args:
            t_min (float): The start time oe epoch as given to privious functions. default -0.2 (200 ms before stimulus onset)
            time_range (list, optional): The time range for peak detection. Defaults to [0.25, 0.35]. (50 ms before and after expected p300 location)

        Returns:
            average peak amplitudes, average latencies, individual peak amplitudes, and individual latencies.
        """
        time_range = (np.array(time_range) - t_min) * self.sampling_rate
        num_trials = self.epoch.shape[0]
        num_channels = self.epoch.shape[1]

        indiv_peak = np.zeros((num_channels, num_trials))
        indiv_latency = np.zeros((num_channels, num_trials))
        avg_peak = np.zeros(num_channels)
        avg_latency = np.zeros(num_channels)

        for ch in range(num_channels):
            avg_peak[ch] = np.max(
                np.abs(
                    np.mean(
                        self.epoch[:, ch, int(time_range[0]) : int(time_range[1])],
                        axis=0,
                    )
                )
            )
            avg_latency[ch] = np.argmax(
                np.abs(
                    np.mean(
                        self.epoch[:, ch, int(time_range[0]) : int(time_range[1])],
                        axis=0,
                    )
                )
            )
            for i in range(num_trials):
                indiv_peak[ch, i] = np.max(
                    np.abs(self.epoch[i, ch, int(time_range[0]) : int(time_range[1])])
                )
                indiv_latency[ch, i] = (
                    np.argmax(
                        np.abs(
                            self.epoch[i, ch, int(time_range[0]) : int(time_range[1])]
                        )
                    )
                    + int(time_range[0])
                    + int(t_min * self.sampling_rate)
                )

        return avg_peak, avg_latency, indiv_peak, indiv_latency
