import pyxdf
import os
import numpy as np
from attention_feature_extraction import AttentionFeatureExtraction
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from argparse import ArgumentParser


# function to load all subject data
def load_all_subjects(folder_path):
    """
    Args:
        folder_path (str): The path to the folder containing the XDF files.
    Returns:
        subjects (list): A list of time series data for each subject.
    """
    subjects = []
    for file in os.listdir(folder_path):
        if file.endswith(".xdf"):
            file_path = os.path.join(folder_path, file)
            streams, header = pyxdf.load_xdf(file_path)  # Load the XDF file
            subjects.append(streams[0]["time_series"])
    return subjects


# function to epoch the data to ? seconds epochs
def epoch_data(data, epoch_length, sampling_rate):
    """
    Args:
        data (numpy.ndarray): The input eeg data.
        epoch_length (float): The length of each epoch in seconds.
        sampling_rate (int): The sampling rate of the data in Hz.

    Returns:
       epochs(list): A list of numpy arrays, where each array represents an epoch of data.

    """
    epoch_samples = int(epoch_length * sampling_rate)
    # check data size
    if data.shape[0] < data.shape[1]:
        data = data.T
    num_epochs = data.shape[0] // epoch_samples
    epochs = np.array_split(data[: num_epochs * epoch_samples, :], num_epochs)
    return epochs


# Funciton to first extract each task data based on event label and then epoch the data


def process_all_subjects(
    folder_path, epoch_length=5, sampling_rate=250, lowcut=1, highcut=40
):
    """
    Args:
        folder_path (str): The path to the folder containing the EEG data files.
        epoch_length (int, optional): The length of each epoch in seconds. Defaults to 5.
        sampling_rate (int, optional): The sampling rate of the EEG data in Hz. Defaults to 250.
        lowcut (int, optional): The lowcut frequency for the bandpass filter. Defaults to 1.
        highcut (int, optional): The highcut frequency for the bandpass filter. Defaults to 40.

    Returns:
        dict: A dictionary containing the computed power spectral density matrices for each task.
              The keys are 'cognitive', 'eyes_closed', and 'eyes_open', and the values are lists
              of relative power matrices for each subject.

    """
    all_psd_matrices = {"cognitive": [], "eyes_closed": [], "eyes_open": []}
    subjects_data = load_all_subjects(folder_path)

    for eeg_data in subjects_data:
        # extract cognitive task section of the data if available
        try:
            start_cognitive = int(
                np.where(eeg_data[:, 4] == 4)[0][0]
            )  # find the index of the event = 4
            end_cognitive = int(
                np.where(eeg_data[:, 4] == 5)[0][0]
            )  # find the index of the event = 5
            cognitive_data = eeg_data[
                start_cognitive:end_cognitive, :4
            ]  # separate the cognitive task data from the rest
        except IndexError:
            print("Cognitive task markers not found for this subject, skipping...")
            continue

        # same for eyes closed data
        try:
            start_ec = int(np.where(eeg_data[:, 4] == 2)[0][0])
            end_ec = int(np.where(eeg_data[:, 4] == 3)[0][0])
            ec_data = eeg_data[start_ec:end_ec, :4]
        except IndexError:
            print("Eyes closed markers not found for this subject, skipping...")
            continue

        # same for eyes open data
        try:
            start_eo = int(np.where(eeg_data[:, 4] == 1)[0][0])
            end_eo = start_ec  # Eyes open is before eyes closed
            eo_data = eeg_data[start_eo:end_eo, :4]
        except IndexError:
            print("Eyes open markers not found for this subject, skipping...")
            continue

        # Epoch the data into 5-second segments
        cognitive_epochs = epoch_data(cognitive_data, epoch_length, sampling_rate)
        ec_epochs = epoch_data(ec_data, epoch_length, sampling_rate)
        eo_epochs = epoch_data(eo_data, epoch_length, sampling_rate)

        # using the attention_feature_extraction class to compute the relative power for each epoch
        for epoch, label in zip(
            [cognitive_epochs, ec_epochs, eo_epochs],
            ["cognitive", "eyes_closed", "eyes_open"],
        ):
            for ep in epoch:
                attention_prep = AttentionFeatureExtraction(
                    ep, sampling_rate=sampling_rate
                )
                attention_prep.filtering(
                    lowcut=lowcut, highcut=highcut
                )  # filtering the epochs in the range of 1-40 Hz
                psd_result, f = attention_prep.compute_power_spectral_density(
                    use_filtered_data=True, lowcut=1, highcut=40, f_res=500
                )

                # compute the relative power matrix for each epoch
                relative_power = attention_prep.relative_power(
                    psd_result, f, label=label
                )
                all_psd_matrices[label].append(relative_power)

    return all_psd_matrices


def test_attention(epoch_length, sampling_rate=250, lowcut=1, highcut=40):
    folder_path = "data/attention/test"
    all_psd_matrices = process_all_subjects(
        folder_path, epoch_length, sampling_rate, lowcut, highcut
    )
    # reformatting the data
    data = []
    labels = []
    for label, rel_power_list in all_psd_matrices.items():
        for rel_power in rel_power_list:
            data.append(rel_power)
            labels.append(label)
    data = np.array(data)
    data_flattened = data.reshape(data.shape[0], -1)

    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_flattened)

    cluster_centers = np.load("cluster_centers.npy")
    distances = np.linalg.norm(data_normalized[:, None] - cluster_centers, axis=-1)
    closest_cluster = np.argmin(distances, axis=-1)
    df_test = pd.DataFrame({"Cluster": closest_cluster, "Task": labels})

    plt.figure()
    for i, task in enumerate(["cognitive", "eyes_closed", "eyes_open"]):
        plt.subplot(3, 1, i + 1)
        task_data = df_test[df_test["Task"] == task]["Cluster"].values

        sns.heatmap(
            [task_data],
            cmap="viridis",
            cbar=True,
            xticklabels=False,
            yticklabels=False,
            vmax=9,
            vmin=0,
        )
        plt.title(f"Clusters (Attention Scores) for {task.capitalize()} Task")
        plt.xlabel("Epoch")
        plt.tight_layout()
    plt.show()


def attention_main(
    folder_path, epoch_length, sampling_rate, lowcut, highcut, is_test=True
):
    all_psd_matrices = process_all_subjects(
        folder_path, epoch_length, sampling_rate, lowcut, highcut
    )

    # clustering the relative power matrices accross all subjects and all epochs (5s) and all tasks (cognitive, eyes closed, eyes open)

    # reformatting the data for clustering
    data = []
    labels = []
    for label, rel_power_list in all_psd_matrices.items():
        for rel_power in rel_power_list:
            data.append(rel_power)
            labels.append(label)
    data = np.array(data)
    data_flattened = data.reshape(
        data.shape[0], -1
    )  # each matrix has a size of 4*4 for kmeans the matrix is flattened to a vector of size 16

    # Map the labels to integers
    label_mapping = {"cognitive": 0, "eyes_closed": 1, "eyes_open": 2}
    label_series = pd.Series(labels)
    label_series = np.array(label_series.map(label_mapping))

    # Normalize the data (prepare for k-means)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_flattened)

    # Apply k-means
    kmeans = KMeans(n_clusters=10)  # 10 clusters corresponding to 10 attentional states
    kmeans.fit(data_normalized)
    klabels = kmeans.labels_

    # Visualizing the clustering results and tasks performed within each cluster

    df = pd.DataFrame({"Cluster": klabels, "Task": label_series})

    task_cluster_counts = pd.crosstab(df["Cluster"], df["Task"], normalize="index")

    task_cluster_counts_sorted = task_cluster_counts.sort_values(by=0)

    sorted_cluster_labels = task_cluster_counts_sorted.index
    sorted_cluster_centers = kmeans.cluster_centers_[sorted_cluster_labels]
    np.save("src/task5/cluster_centers.npy", sorted_cluster_centers)

    task_cluster_counts_sorted.plot(
        kind="bar", stacked=True, figsize=(10, 7), colormap="viridis"
    )

    plt.title("Proportion of tasks in each cluster (attention score)")
    plt.xlabel("Attention cluster")
    plt.xticks([i for i in range(10)], [f"score {i+1}" for i in range(10)], rotation=0)
    plt.ylabel("Proportion of tasks")

    plt.legend(
        title="Task",
        labels=["cognitive", "eyes closed", "eyes open"],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()

    plt.show()

    if is_test:
        test_attention(epoch_length, sampling_rate, lowcut, highcut)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--folder_path",
        type=str,
        default="data/attention",
        help="The path to the folder containing the EEG data files.",
    )
    parser.add_argument(
        "--epoch_length",
        type=int,
        default=5,
        help="The length of each epoch in seconds. Defaults to 5.",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=250,
        help="The sampling rate of the EEG data in Hz. Defaults to 250.",
    )
    parser.add_argument(
        "--lowcut",
        type=int,
        default=1,
        help="The lowcut frequency for the bandpass filter. Defaults to 1.",
    )
    parser.add_argument(
        "--highcut",
        type=int,
        default=40,
        help="The highcut frequency for the bandpass filter. Defaults to 40.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the test function.",
    )

    args = parser.parse_args()

    attention_main(
        args.folder_path,
        args.epoch_length,
        args.sampling_rate,
        args.lowcut,
        args.highcut,
        args.test,
    )


if __name__ == "__main__":
    main()
