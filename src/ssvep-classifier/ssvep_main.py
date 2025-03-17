import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import glob
import os
from tsfresh import extract_relevant_features
from ssvep_feature_extraction import SSVEPFeatureExtraction
from ssvep_classifier import SSVEPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from argparse import ArgumentParser


def data_processing(folder_path="data/ssvep", sampling_rate=250):
    """
    Args:
        folder_path (str): The path to the folder containing the data files (defauelt is "data/ssvep").
        sampling_rate (int): The sampling rate of the data. (default is 250 Hz)

    Returns:
        df_cleaned (pandas.DataFrame): The cleaned dataframe containing the processed data (filtered, nans removed)
        label (pandas.Series): The corresponding labels for each data sample. (1-4 corresponding to [7,5, 8.75, 10, 12] Hz)
    """

    # design the dataframe for dataset and the filter
    df_cleaned = pd.DataFrame()
    label = []
    lowcut = 1
    highcut = 30
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(6, [low, high], btype="band")

    # loading all data files
    parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))

    for subject_count, file_path in enumerate(parquet_files):
        df = pd.read_parquet(file_path)
        # this section finds the event indices for events 1-4 (7.5, 8.75, 10, 12 Hz)
        event_indices = [df.index[df["event_id"] == i].tolist() for i in range(1, 5)]
        epoch_counter = 0
        for event_id, indices in enumerate(event_indices, start=1):
            for i, ind in enumerate(indices):
                temp_df = pd.DataFrame()
                temp_df[["o1", "o2", "t3", "t4"]] = df[["o1", "o2", "t3", "t4"]].iloc[
                    ind + 1 : ind + 1 + 5 * sampling_rate
                ]  # 5 seconds is chosen for length of epochs

                temp_df.interpolate(method="linear", inplace=True)

                temp_df["o1"] = filtfilt(b, a, temp_df["o1"])
                temp_df["o2"] = filtfilt(b, a, temp_df["o2"])
                temp_df["t3"] = filtfilt(b, a, temp_df["t3"])
                temp_df["t4"] = filtfilt(b, a, temp_df["t4"])

                temp_df["epoch_id"] = np.repeat(
                    epoch_counter + subject_count * 100, 5 * sampling_rate
                )  # number of epochs per subject is 100

                epoch_counter += 1
                temp_df.interpolate(method="linear", inplace=True)
                df_cleaned = pd.concat([df_cleaned, temp_df])
                label.append(event_id)

    label = pd.Series(label)

    return df_cleaned, label


def use_tsfresh(df_cleaned, label):
    """
    Extracts relevant features using tsfresh library.
    WARNING: This function may take a long time to run.
    USE ALREADY EXTRACTED FEATURES.

    Args:
        df_cleaned (pandas.DataFrame): The cleaned dataframe containing the data.
        label (str): The name of the target variable column.
    Returns:
        pandas.DataFrame: The dataframe containing the extracted relevant features.
    """
    features_filtered_direct = extract_relevant_features(
        df_cleaned, label, column_id="epoch_id", n_jobs=0
    )

    return features_filtered_direct


def use_extracted_features(file_name="features_filtered_direct.csv"):
    """
    Load the extracted features from a csv file.
    PLEASE UES THIS FUNCTION INSTEAD OF use_tsfresh TO AVOID LONG WAIT TIMES.

    Args:
        file_name (str): The name of the csv file containing the extracted features.
    Returns:
        pandas.DataFrame: The dataframe containing the extracted features.
    """
    features_filtered_direct = pd.read_csv(file_name)
    features_filtered_direct = features_filtered_direct.drop(columns=["Unnamed: 0"])

    return features_filtered_direct


def ssvep_main(tsfreshfeatures=True, data=None, label=None, sampling_rate=250):
    if tsfreshfeatures:
        features = use_extracted_features(
            file_name="src/task1/features_filtered_direct.csv"
        )

    else:
        # feature extraction using SSVEPFeatureExtraction
        freqs = [7.5, 8.75, 10, 12]
        features = []
        for i in range(0, len(data), 5 * sampling_rate):
            data_slice = data.iloc[i : i + 5 * sampling_rate, :-1].values.T
            feature_extractor = SSVEPFeatureExtraction(sampling_rate, data_slice, freqs)
            features.append(feature_extractor.extract_features())
        features = np.array(features)
        features = features.reshape(features.shape[0], -1)

    return features


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/ssvep")
    parser.add_argument("--sampling_rate", type=int, default=250)
    parser.add_argument("--tsfreshfeatures", action="store_true")
    parser.add_argument("--cross_val", action="store_true")

    args = parser.parse_args()

    alldata, label = data_processing(args.data_path, args.sampling_rate)
    features = ssvep_main(
        tsfreshfeatures=args.tsfreshfeatures, data=alldata, label=label
    )

    # Classification using SSVEPClassifier
    model = LinearDiscriminantAnalysis()
    ssvep_classifier = SSVEPClassifier(model, features, label, cross_val=args.cross_val)
    ssvep_classifier.train()
    ssvep_classifier.visualize()

    # Classification using MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    ssvep_classifier = SSVEPClassifier(model, features, label, cross_val=args.cross_val)
    ssvep_classifier.train()
    ssvep_classifier.visualize()


if __name__ == "__main__":
    main()
