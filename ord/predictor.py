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

class BaseOutlierRegionPredictor:
    """
    Base Class for an Anomaly / Outlier Predictor

    (Every algorithm should implement these methods)
    """

    def __init__(self):
        pass

    def fit(self, train_data, validation_data, ord_evaluation_data):
        """
        Fit the predictor with training data
        :param train_data: training data
        :param validation_data: validation data (not necessary with labeled anomalies)
        :param ord_evaluation_data: data for continuous evaluation with labeled anomalies (optional)
        :return:
        """
        pass

    def predict(self, x):
        """
        Predict the anomaly with the method for a time series at hand
        :param x: time series
        :return: anomaly scores and predictions
        """
        pass

    def save(self, path):
        """
        Save the predictor state to file
        :param path: save file
        :return:
        """
        pass

    def load(self, file):
        """
        Load the (trained) predictor state from file
        :param file:
        :return:
        """
        pass