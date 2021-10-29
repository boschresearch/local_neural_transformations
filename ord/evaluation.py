# Local Neural Transformations (LNT) - a self-supervised method for
# anomalous region detection in time series
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from sklearn.metrics import roc_curve, auc
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from .predictor import BaseOutlierRegionPredictor
from torch.utils.data import DataLoader
import inspect


def compute_predictions(predictor: BaseOutlierRegionPredictor, data: DataLoader, epochs=1, numBatches=None, crop_first_steps=0, crop_last_steps=0):
    """
    Computes the outlier predictions for a given predictor on the dataset
    :param predictor: predictor to evaluate
    :param data: evaluation data
    :param epochs: number of epochs to process (default: 1)
    :param numBatches: optional: only process the given number of batches
    :return: outlier_predictions, outlier_labels, outlier_scores
    """

    outlier_labels = []
    outlier_scores = []
    outlier_predictions = []

    for _ in range(epochs):
        for i, (x, y) in tqdm(enumerate(data)):

            # predict the scores
            pred, score = predictor.predict(x)

            if crop_first_steps > 0:
                pred = pred[:, crop_first_steps:-crop_last_steps]
                score = score[:, crop_first_steps:-crop_last_steps]
                y = y[:, crop_first_steps:-crop_last_steps]

            # append score to the stats
            y = y.detach().cpu().numpy().reshape(-1)
            score = score.reshape(-1)
            outlier_scores.append(score)
            outlier_labels.append(y)

            if pred is not None:
                outlier_predictions.append(pred.reshape(-1))

            if numBatches is not None and numBatches == i:
                break

    # stack the stats
    outlier_labels = np.concatenate(outlier_labels)
    outlier_scores = np.concatenate(outlier_scores)

    if outlier_predictions:
        outlier_predictions = np.concatenate(outlier_predictions)

    return outlier_predictions, outlier_labels, outlier_scores


def compute_roc_auc(outlier_labels, outlier_scores):
    """
    Computes the ROC-AUC score for a given sequence of scores and labels
    :param outlier_labels: ground truth labels
    :param outlier_scores: scores
    :return: ROC-AUC
    """
    fpr, tpr, _ = roc_curve(y_true=outlier_labels, y_score=outlier_scores, pos_label=1)
    return auc(fpr, tpr)


def compute_roc(predictors : dict, data : DataLoader, epochs=1):
    """
    Computes the ROC statistics and stores them in a result dictionary
    :param predictors:
    :param data:
    :param epochs:
    :return:
    """

    results = {}
    for name, predictor in predictors.items():
        _, outlier_labels, outlier_scores = compute_predictions(predictor, data, epochs)

        # Compute the ROC curve
        fpr, tpr, thresholds = roc_curve(y_true=outlier_labels, y_score=outlier_scores, pos_label=1)
        results[name] = (fpr, tpr, thresholds)

    return results


def plot_roc_curve(fpr, tpr, label='ROC'):
    """
    Plots an ROC curve with pretty formating
    :param fpr: false positive rate
    :param tpr: true positive rate
    :param label: label of the plot
    :return:
    """
    # compute the area under curve
    a = auc(fpr, tpr)

    # create the plot
    plt.plot(fpr, tpr, color='blue', label=label + " (AUC={0:.2f})".format(float(a)))
    plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), '--', color='red', label='chance (AUC=0.50)')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()


def plot_roc_curves(results):
    """
    Plot multiples ROC curves in a joint figure
    :param results: dictionary with results
    :return:
    """

    plt.figure(figsize=(12,12))
    for (name, (fpr, tpr, _)), color in zip(results.items(), plt.cm.Paired(range(len(results)))):
        # compute the area under curve
        a = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=color, label=name + " (AUC={0:.2f})".format(float(a)))

    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), '--', color='black', label='chance (AUC=0.50)')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc='lower right')
    plt.savefig('rocs.png')
    plt.show()


def compute_metrics(metrics: dict, predictor: BaseOutlierRegionPredictor, data: DataLoader, epochs=1):
    """
    Computes several metrics and stores them in a dictionary
    :param metrics: metric to compute as dictionary
    :param predictor: the predictor to evaluate
    :param data: data used for evaluation
    :param epochs: number of epochs to perform evaluation (defaults to 1)
    :return:
    """
    # compute the predictions first
    outlier_predictions, outlier_labels, outlier_scores = compute_predictions(predictor, data, epochs)

    results = {}
    for metric_name, metric in metrics.items():

        params = {
            'y_pred': outlier_predictions,
            'y_true': outlier_labels,
            'y_score': outlier_scores
        }

        sig = inspect.signature(metric).parameters.keys()
        params = {k: params[k] for k in params.keys() & sig}

        try:
            metric_value = metric(**params)
        except Exception as e:
            print(e)
            metric_value = None
        finally:
            results[metric_name] = metric_value

    return results
