import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class SSVEPClassifier:
    def __init__(
        self,
        model: None,
        featureset: np.ndarray,
        label: np.ndarray,
        cross_val: bool = False,
    ):
        """
        Initialize the SSVEPClassifier class.

        Parameters:
        - model (MLPClassifier or LDA): The machine learning model used for classification.
        - featureset (np.ndarray): The input features for training the model.
        - label (np.ndarray): The corresponding labels for the input features.
        - cross_val (bool, optional): Whether to use cross-validation. Default is False.
        """
        self.model = model
        if model.__class__ not in [MLPClassifier, LinearDiscriminantAnalysis]:
            raise ValueError(
                "Invalid model. Must be MLPClassifier or LinearDiscriminantAnalysis."
            )

        self.cross_val = cross_val
        self.featureset = featureset
        self.label = label

    def stratify(self):
        """
        Stratify the dataset based on whether cross-validation is enabled.

        Returns:
            If cross-validation is enabled:
                X (array-like): The feature set.
                y (array-like): The labels.
                skf (StratifiedKFold): The stratified k-fold object. (k = 5 by default)
            If cross-validation is not enabled:
                X_train (array-like): The training feature set.
                X_test (array-like): The testing feature set. (20% of the data)
                y_train (array-like): The training labels.
                y_test (array-like): The testing labels. (20% of the data)
        """
        if self.cross_val:
            X = self.featureset
            y = self.label
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return X, y, skf
        else:
            X = self.featureset
            y = self.label
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test

    def train(self):
        """
        Train the model and generate the confusion matrix, FPR, and TPR for ROC curve.

        Returns:
            cm (numpy.ndarray): The confusion matrix.
            fpr (numpy.ndarray): The false positive rates for the ROC curve.
            tpr (numpy.ndarray): The true positive rates for the ROC curve.
        """

        fpr = []
        tpr = []
        thresholds = []

        if self.cross_val:
            X, y, skf = self.stratify()
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            self.model.fit(X, y)
            y_pred = cross_val_predict(self.model, X, y, cv=skf)
            cm = confusion_matrix(y, y_pred)
            y_pred_prob = cross_val_predict(
                self.model, X, y, cv=skf, method="predict_proba"
            )

            for l in range(4):
                fpr_l, tpr_l, thresholds_l = roc_curve(
                    y, y_pred_prob[:, l], pos_label=l + 1
                )
                fpr.append(fpr_l)
                tpr.append(tpr_l)
                thresholds.append(thresholds_l)

        else:
            X_train, X_test, y_train, y_test = self.stratify()
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            y_pred_prob = self.model.predict_proba(X_test)

            for l in range(4):
                fpr_l, tpr_l, thresholds_l = roc_curve(
                    y_test, y_pred_prob[:, l], pos_label=l + 1
                )
                fpr.append(fpr_l)
                tpr.append(tpr_l)
                thresholds.append(thresholds_l)

        return cm, fpr, tpr

    def visualize(self):
        """
        Visualize confusion matrix and ROC curve.
        """
        cm, fpr, tpr = self.train()
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["7.5", "8.75", "10", "12"],
            yticklabels=["7.5", "8.75", "10", "12"],
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title("Confusion Matrix - {}".format(self.model.__class__.__name__))
        plt.show()

        plt.plot([0, 1], [0, 1], linestyle="--")
        for l in range(len(fpr)):
            plt.plot(
                fpr[l],
                tpr[l],
                label=f"Label {l} (AUC = {np.trapz(tpr[l], fpr[l]):.2f})",
            )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - {}".format(self.model.__class__.__name__))
        plt.legend()
        plt.show()
