# import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from p300vis import P300Vis
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from spatial_filter import SpatialFilter
from argparse import ArgumentParser


def preprocessing(
    folder_path,  # path to the data ('data/p300')
    subject_name,  # subject name ('0', '1', '2')
    epoch_start=-0.2,  # start of the epoch (in seconds, relative to the event onset)
    epoch_end=0.8,  # end of the epoch (in seconds, relative to the event onset)
    sampling_rate=250,
    lowcut=1,
    highcut=12,
):

    df = pd.read_parquet(folder_path + "/" + subject_name + ".parquet")

    # interpolate nans
    df["o1"] = df["o1"].interpolate(method="linear").bfill().ffill()
    df["o2"] = df["o2"].interpolate(method="linear").bfill().ffill()
    df["t3"] = df["t3"].interpolate(method="linear").bfill().ffill()
    df["t4"] = df["t4"].interpolate(method="linear").bfill().ffill()

    # re-reference to avg of channels
    df[["o1", "o2", "t3", "t4"]] -= np.mean(df[["o1", "o2", "t3", "t4"]], axis=0)

    # filtering
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(6, [low, high], btype="band")
    df["o1"] = filtfilt(b, a, df["o1"])
    df["o2"] = filtfilt(b, a, df["o2"])
    df["t3"] = filtfilt(b, a, df["t3"])
    df["t4"] = filtfilt(b, a, df["t4"])

    # standard vs odd
    standard_index = df[df["event_id"] == 1].index
    odd_index = df[df["event_id"] == 2].index

    # epoching
    epoch_standard = []
    for ind in standard_index:
        epoch_standard.append(
            df[["o1", "o2", "t3", "t4"]]
            .iloc[
                ind
                + int(epoch_start * sampling_rate) : ind
                + int(epoch_end * sampling_rate)
            ]
            .values
        )
    epoch_standard = np.array(epoch_standard)

    # remove bad epochs (amplitude > 100uV)
    epoch_standard = epoch_standard[
        np.all(np.abs(epoch_standard) < 100e-6, axis=(1, 2)), :, :
    ]
    # baseline correction
    epoch_standard -= np.mean(
        epoch_standard[:, : int(sampling_rate * abs(epoch_start)), :], axis=1
    )[:, None, :]

    epoch_odd = []
    for ind in odd_index:
        epoch_odd.append(
            df[["o1", "o2", "t3", "t4"]]
            .iloc[
                ind
                + int(epoch_start * sampling_rate) : ind
                + int(epoch_end * sampling_rate)
            ]
            .values
        )
    epoch_odd = np.array(epoch_odd)
    # remove bad epochs (amplitude > 100uV)
    epoch_odd = epoch_odd[np.all(np.abs(epoch_odd) < 100e-6, axis=(1, 2)), :, :]
    # baseline correction
    epoch_odd -= np.mean(
        epoch_odd[:, : int(sampling_rate * abs(epoch_start)), :], axis=1
    )[:, None, :]

    return epoch_standard.transpose(0, 2, 1), epoch_odd.transpose(0, 2, 1)


def spatial_filtering(epoch_standard, epoch_odd):
    """
    Apply spatial filtering to the data using contrastive PCA algorithm.

    Args:
        epoch_standard (np.array): standard epochs (num_trials x num_samples x num_channels)
        epoch_odd (np.array): odd epochs (num_trials x num_samples x num_channels)

    """

    # spatial filtering
    foreground = epoch_odd.reshape(-1, 4)
    background = epoch_standard.reshape(-1, 4)

    spatial_filter = SpatialFilter(foreground, background)
    weights = spatial_filter.calculate_spatial_filter()

    spatially_filtered_standard = []
    for ind in range(epoch_standard.shape[0]):
        spatially_filtered_standard.append(
            np.dot(epoch_standard[ind, :, :].T, weights.reshape(-1, 1))
        )

    spatially_filtered_odd = []
    for ind in range(epoch_odd.shape[0]):
        spatially_filtered_odd.append(
            np.dot(epoch_odd[ind, :, :].T, weights.reshape(-1, 1))
        )

    return np.array(spatially_filtered_standard).transpose(0, 2, 1), np.array(
        spatially_filtered_odd
    ).transpose(0, 2, 1)


def peak_vis(
    indiv_peak_standard,
    indiv_peak_odd,
    indiv_latency_standard,
    indiv_latency_odd,
    avg_peak_standard,
    avg_peak_odd,
    avg_latency_standard,
    avg_latency_odd,
    time_range,
    sampling_rate,
):
    """
    Visualize the peak amplitude and latency.
    """

    # peak amplitude
    for ch in range(indiv_peak_standard.shape[0]):
        plt.figure()
        plt.scatter(
            [1] * indiv_peak_standard.shape[1]
            + np.random.normal(0, 0.01, indiv_peak_standard.shape[1]),
            indiv_peak_standard[ch, :],
            label="standard",
        )
        plt.scatter(
            [2] * indiv_peak_odd.shape[1]
            + np.random.normal(0, 0.01, indiv_peak_odd.shape[1]),
            indiv_peak_odd[ch, :],
            label="odd",
        )
        plt.xticks([1, 2], ["standard", "odd"])
        plt.scatter(
            [1],
            avg_peak_standard[ch],
            color="red",
            marker="D",
            s=100,
            label="Standard (average)",
        )
        plt.scatter(
            [2], avg_peak_odd[ch], color="red", marker="D", s=100, label="Odd (average)"
        )

        plt.ylabel("Peak amplitude")
        if indiv_peak_standard.shape[0] > 1:
            plt.title(f"Peak amplitude for channel {ch+1}")
        else:
            plt.title(f"Peak amplitude for weighted sum of channels")
        plt.yticks(np.arange(0, 101e-6, 10e-6), np.arange(0, 101, 10))
        plt.ylim(0, 50e-6)
        plt.legend()
        plt.grid()
        plt.show()

    # peak latency
    indiv_latency_standard = (indiv_latency_standard / sampling_rate) * 1000
    indiv_latency_odd = (indiv_latency_odd / sampling_rate) * 1000
    avg_latency_odd = ((avg_latency_odd / sampling_rate) + time_range[0]) * 1000
    avg_latency_standard = (
        (avg_latency_standard / sampling_rate) + time_range[0]
    ) * 1000

    for ch in range(indiv_latency_standard.shape[0]):
        plt.figure()
        plt.scatter(
            [1] * indiv_latency_standard.shape[1]
            + np.random.normal(0, 0.01, indiv_latency_standard.shape[1]),
            indiv_latency_standard[ch, :],
            label="standard",
        )
        plt.scatter(
            [2] * indiv_latency_odd.shape[1]
            + np.random.normal(0, 0.01, indiv_latency_odd.shape[1]),
            indiv_latency_odd[ch, :],
            label="odd",
        )
        plt.scatter(
            [1],
            avg_latency_standard[ch],
            color="red",
            marker="D",
            s=100,
            label="Standard (average)",
        )
        plt.scatter(
            [2],
            avg_latency_odd[ch],
            color="red",
            marker="D",
            s=100,
            label="Odd (average)",
        )
        plt.xticks([1, 2], ["standard", "odd"])
        plt.ylabel("Peak latency (ms)")
        if indiv_latency_standard.shape[0] > 1:
            plt.title(f"Peak latency for channel {ch+1}")
        else:
            plt.title(f"Peak latency for weighted sum of channels")
        plt.yticks(np.arange(200, 400, 50))
        plt.ylim(200, 400)
        plt.legend()
        plt.grid()
        plt.show()


def main_p300(args):
    epoch_standard, epoch_odd = preprocessing(args.data_path, args.subject_name)
    if args.spatial_filtering:
        print("Spatial filtering")
        spatially_filtered_standard, spatially_filtered_odd = spatial_filtering(
            epoch_standard, epoch_odd
        )
    else:
        spatially_filtered_standard = epoch_standard
        spatially_filtered_odd = epoch_odd

    # initialize the class
    sampling_rate = 250
    P300Vis_standard = P300Vis(spatially_filtered_standard, sampling_rate)
    P300Vis_odd = P300Vis(spatially_filtered_odd, sampling_rate)

    P300Vis_standard.Single_trial_vis(t_min=-0.2, t_max=0.8, label="standard")
    P300Vis_odd.Single_trial_vis(t_min=-0.2, t_max=0.8, label="odd")

    P300Vis_standard.Average_response_vis(t_min=-0.2, t_max=0.8, label="standard")
    P300Vis_odd.Average_response_vis(t_min=-0.2, t_max=0.8, label="odd")

    (
        avg_peak_standard,
        avg_latency_standard,
        indiv_peak_standard,
        indiv_latency_standard,
    ) = P300Vis_standard.peak_detection(t_min=-0.2, time_range=[0.25, 0.35])
    avg_peak_odd, avg_latency_odd, indiv_peak_odd, indiv_latency_odd = (
        P300Vis_odd.peak_detection(t_min=-0.2, time_range=[0.25, 0.35])
    )

    peak_vis(
        indiv_peak_standard,
        indiv_peak_odd,
        indiv_latency_standard,
        indiv_latency_odd,
        avg_peak_standard,
        avg_peak_odd,
        avg_latency_standard,
        avg_latency_odd,
        time_range=[0.25, 0.35],
        sampling_rate=250,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="data/p300", help="Path to the data"
    )
    parser.add_argument("--subject_name", type=str, default="0", help="Subject name")
    parser.add_argument(
        "--spatial_filtering", action="store_true", help="Apply spatial filtering"
    )
    args = parser.parse_args()
    main_p300(args)


if __name__ == "__main__":
    main()
