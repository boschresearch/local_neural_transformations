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

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

from ord.datasets.wadi.wadi import normalize_column

class SWaTDataset(Dataset):
    """
    Secure Water Treatment (SWaT) Dataset Class
    """

    def __init__(self, datadir, window_size, stride=None, attacks=False, relative_split=None, downsampling=1):
        """
        Secure Water Treatment (SWaT) Dataset
        :param datadir: location (directory) of the dataset files (.csv)
        :param window_size: size of the subsequences which are iterated by the dataset
        :param stride: stride of the extracted sub-sequences (see window_size)
        :param attacks: attacks present in the data (Yes/No)
        :param relative_split: tuple to describe the relative subset of the data used, e.g (0, 0.5) is the first half
        :param downsampling: downsample the data by this ratio
        """
        super(SWaTDataset, self).__init__()

        normalization_file = 'normalization.txt'

        if attacks:
            file = "SWaT_Dataset_Attack_v0.csv"
        else:
            file = "SWaT_Dataset_Normal_v0.csv"

        self._path = os.path.join(datadir, file)
        self._normalization_path = os.path.join(datadir, normalization_file)
        self._window_size = window_size
        self._attacks = attacks

        if attacks:
            self._data, self._labels = self.load_data_from_file(skiprows=0, parse_labels=True, recompute_norm=False, downsampling=downsampling)
        else:
            # stabilization time 6h = 21600s
            self._data = self.load_data_from_file(skiprows=1, parse_labels=False, recompute_norm=True, stabilization_time=21600, downsampling=downsampling)

        # keep only the required data split in memory
        if relative_split and isinstance(relative_split, tuple):
            absolute_length = self._data.shape[0]
            self._data = self._data[int(absolute_length * relative_split[0]):int(absolute_length * relative_split[1])]
            self._labels = self._labels[int(absolute_length * relative_split[0]):int(
                absolute_length * relative_split[1])] if attacks else None

        # define a walker to select windows at random
        self._walker = range(0, self._data.shape[0] - window_size, stride if stride else window_size // 2)

    def load_data_from_file(self, skiprows, parse_labels=False, recompute_norm=False, stabilization_time=0, downsampling=1):
        """
        Loads the data from the csv file
        :param skiprows: skip the first rows which contain header data
        :param parse_labels: parse labels (Yes/No)
        :param recompute_norm: should the norm be recomputed or read from file
        :param stabilization_time: remove datapoints at the beginning where the system is unstable
        :param downsampling: downsample the data by this ratio
        :return: the data or data and labels if label are parsed
        """
        # read the data from file
        df = pd.read_csv(self._path, delimiter=';', decimal=',', skiprows=skiprows)

        # get cleaned channel names
        df.columns = map(lambda s: s.strip(), df.columns)
        channels = set(df.columns) - {'Timestamp', 'Normal/Attack'}

        # exclude channels that are constant for both training and testing
        channels = channels - {"P404", "P502", "P603", "P601", "P202", "P401"}

        # Sort the channels in alphabetic order
        channels = sorted(list(channels))
        print(channels)

        # according to the paper it took 5-6h to stabilize
        if stabilization_time > 0:
            df = df.iloc[stabilization_time:]

        # Normalization File
        if os.path.exists(self._normalization_path):
            norm_parameters = SWaTDataset._parse_normalization_file(self._normalization_path)
        else:
            norm_parameters = {}

        for c in channels:
            df[c] = normalize_column(df[c], norm_parameters, recompute=recompute_norm)

        if recompute_norm:
            SWaTDataset._save_normalization_file(self._normalization_path, norm_parameters)

        # convert the data to numpy
        data = np.asarray(df[channels].to_numpy(), dtype=np.float32)[::downsampling, :]

        if parse_labels:
            labels = np.asarray((df['Normal/Attack'].str.lower() == 'attack').to_numpy(), dtype=np.float32)[::downsampling]

            return data, labels
        else:
            return data

    def __getitem__(self, item):
        """
        Iterate the next subsequence in the dataset
        :param item: sequence index
        :return: time series (and labels if present)
        """

        start_idx = self._walker[item]
        end_idx = start_idx + self._window_size
        meta = None

        if self._attacks:
            return self._data[start_idx:end_idx, :].T, self._labels[start_idx:end_idx]
        else:
            return self._data[start_idx:end_idx, :].T

    def __len__(self):
        """
        Length of the dataset
        :return: length
        """
        return len(self._walker)

    def get_raw_data(self):
        """
        Get the raw data matrix
        :return:
        """
        return self._data

    @staticmethod
    def _parse_normalization_file(file):
        """
        Read in the normalization file with stored min/max ranges for all the data channels.
        :param file: file path
        :return: dictionary with data ranges for each channel
        """
        norm = {}
        with open(file, 'r') as f:
            for line in f.readlines():
                name, c_min, c_max = line.split(' ')
                norm[name] = {
                    'min': float(c_min),
                    'max': float(c_max)
                }
        return norm

    @staticmethod
    def _save_normalization_file(file, norm):
        """
        Save the normalization dictionary to file
        :param file: file path
        :param norm: normalization dictionary with value ranges for each channel.
        :return:
        """
        with open(file, 'w') as f:
            for name in norm.keys():
                f.write(f"{name} {norm[name]['min']} {norm[name]['max']}\n")


if __name__ == "__main__":

    data = SWaTDataset(datadir="/home/cit8si/data/swat", window_size=7200, stride=100, attacks=True)
    loader = DataLoader(data, batch_size=32, shuffle=True, drop_last=True)

    print(len(loader))
    print(data.get_raw_data().shape)

    for x, y in loader:
        print(x.shape)
        break