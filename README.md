## Overview

This project contains processing and analyzing EEG data across three tasks:

Task 1: Classify SSVEP Responses:
In this task, SSVEP (Steady-State Visual Evoked Potential) responses is classified using EEG data. Five-second epochs are generated, each labeled (1-4) based on the stimulus frequency (7.5 Hz, 8.75 Hz, 10 Hz, 12 Hz). Classification is performed using Linear Discriminant Analysis (LDA) and a Multi-Layer Perceptron (MLP) with two hidden layers (100 and 50 neurons). The results are reported using a confusion matrix and ROC curve.

Task 2: Visualize P300 Responses: 
For each subject, the P300 responses to oddball and standard stimuli are visualized. The individual epochs are displayed, and the average response is shown with a 95% confidence interval in shading. Peak amplitude and latency (identified within 250-350 ms post-stimulus) are also reported.

Task 5: Attention Scores:
In this task, relative power in Delta, Theta, Alpha, and Beta bands is extracted for each epoch and each channel. K-means clustering is used to group the features into 10 clusters, each representing an attention score. The cluster with the most "cognitive" epochs is assigned an attention score of 10. For testing, new epochs are evaluated based on their distance to the cluster centroids, and the closest cluster determines the attention score for test epochs.


## Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Create a Virtual Environment

Create a virtual environment to manage dependencies.

```bash
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment

Activate the environment to use isolated dependencies.

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

Install all the required dependencies listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Running the Tasks

### Task 1: Classify SSVEP Response

To run Task 1, execute the following command:

```bash
python src/ssvep_classifier/ssvep_main.py [--tsfreshfeatures] [--cross_val]
```

### Task 2: P300 Visualization

To run Task 2, execute:

```bash
python src/p300_analysis/p300_main.py --subject_name <subject number> [--spatial_filtering] 
```

- subject_name: Name of the subject. (e.g. 0)
- spatial_filtering: Optional flag to apply spatial filtering.

### Task 5: Define Attention Score

To run Task 5, execute:

```bash
python src/attention_clustering/attention_main.py
```

To run the test, run with the test flag:

```bash
python src/attention_clustering/attention_main.py --test
```