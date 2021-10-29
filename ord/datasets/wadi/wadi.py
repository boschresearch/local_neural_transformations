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


def replace_and_interpolate_zeros(column):
    """
    Interpolates zeros in the signal
    :param column:
    :return:
    """
    return column.replace(0, np.nan).interpolate()


def normalize_column(column, parameters=None, recompute=True):
    """
    Normalize the values in a column to [0,1]
    :param column: the  column to normalize
    :param parameters: dict with normalization parameters
    :param recompute: recompute the normalization boundaries?
    :return:
    """

    if not parameters or recompute:
        c_min = column.min()
        c_max = column.max()
    else:
        assert column.name in parameters
        c_min = parameters[column.name]['min']
        c_max = parameters[column.name]['max']

    if parameters is not None:
        parameters[column.name] = {
            'min': c_min,
            'max': c_max
        }

    norm = (c_max - c_min)

    # Handle the special case of constant values
    if norm == 0:
        norm = 1
        c_min = c_min - 0.5
        c_max = c_max + 0.5

    return (column - c_min) / norm


def get_status_columns(df):
    """
    List of Status columns contained in the dataset
    :param df: dataset
    :return: column list
    """
    return list(filter(lambda c: c.endswith('_STATUS'), df.columns))


def get_pv_columns(df):
    """
    List of PV columns contained in the dataset
    :param df: dataset
    :return: column list
    """
    return list(filter(lambda c: c.endswith('_PV'), df.columns))


def get_al_columns(df):
    """
    List of AL columns contained in the dataset
    :param df:
    :return:
    """
    return list(filter(lambda c: c.endswith('_AL'), df.columns))


def get_co_columns(df):
    """
    List of CO columns contained in the dataset
    :param df:
    :return:
    """
    return list(filter(lambda c: c.endswith('_CO'), df.columns))


def get_sp_columns(df):
    """
    List of SP columns contained in the dataset
    :param df:
    :return:
    """
    return list(filter(lambda c: c.endswith('_SP'), df.columns))


def get_ah_columns(df):
    """
    List of AH columns contained in the dataset
    :param df:
    :return:
    """
    return list(filter(lambda c: c.endswith('_AH'), df.columns))


def get_speed_columns(df):
    """
    List of SPEED columns contained in the dataset
    :param df:
    :return:
    """
    return list(filter(lambda c: c.endswith('_SPEED'), df.columns))


def get_special_columns(df):
    """
    List of special columns in the dataset
    :param df:
    :return:
    """
    return [
        "LEAK_DIFF_PRESSURE",
        "TOTAL_CONS_REQUIRED_FLOW"
    ]


class WaDiDataset(Dataset):
    """
    Water Distribution dataset

    contains 14 days of normal operation
    and 2 days of operation under attack
    """

    def __init__(self, root, window_size, stride=None, attacks=False, version='A1-2017', relative_split=None, improved_preprocessing=False, relabel_testset=False):
        """
        Water Distribution (WaDi) Dataset
        :param root: dataset location (directory)
        :param window_size: window size of the subsequences iterrated
        :param stride: stride of the subsequences (see window_size)
        :param attacks: are attacks present in the data? (Yes/No)
        :param version: version of the dataset used (influences the preprocessing steps due to different data formating)
        :param relative_split: tuple to describe the relative subset of the data used, e.g (0, 0.5) is the first half
        :param improved_preprocessing: use improved preprocessing steps
        :param relabel_testset: relabel the testset as proposed by some lines of work (not recommended!; different setup as in the paper)
        """

        normalization_file = 'normalization.txt'

        if not attacks:
            file = 'WADI_14days.csv'
            cache_file = 'WADI_14days.cache.npz'
        else:
            file = 'WADI_attackdata.csv'
            cache_file = 'WADI_attackdata.cache.npz'

        self._attacks = attacks
        self._improved_preprocessing = improved_preprocessing
        self._path = os.path.join(root, version, file)
        self._cache_path = os.path.join(root, version, cache_file)
        self._normalization_path = os.path.join(root, version, normalization_file)
        self._window_size = window_size

        # Assert the given dataset version exists
        assert version in ["A1-2017", "A2-2019"]

        # compute the number of skip rows since it unfortunately depends on the version
        if version == "A1-2017":
            skip_rows = 0 if attacks else 3
        elif version == "A2-2019":
            skip_rows = 1 if attacks else 0

        # Load the data (and also labels if available)
        if attacks:
            self._data, self._labels = self._load_data_from_file(self._path, self._cache_path, self._normalization_path,
                                                                 recompute_norm=False,
                                                                 skiprows=skip_rows,
                                                                 labels=True,
                                                                 improved_preprocessing=improved_preprocessing,
                                                                 relabel_testset=relabel_testset,
                                                                 version=version)
        else:
            self._data = self._load_data_from_file(self._path, self._cache_path, self._normalization_path,
                                                   recompute_norm=True,
                                                   skiprows=skip_rows,
                                                   improved_preprocessing=improved_preprocessing,
                                                   version=version)

        # keep only the required data split in memory
        if relative_split and isinstance(relative_split, tuple):
            absolute_length = self._data.shape[0]
            self._data = self._data[int(absolute_length * relative_split[0]):int(absolute_length*relative_split[1])]
            self._labels = self._labels[int(absolute_length * relative_split[0]):int(absolute_length*relative_split[1])] if attacks else None

        # define a walker to select windows at random
        self._walker = range(0, self._data.shape[0] - window_size, stride if stride else window_size // 2)

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
                    'min' : float(c_min),
                    'max' : float(c_max)
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


    @staticmethod
    def _load_data_from_file(file, cache_file, normalization_file, recompute_norm=True, skiprows=3, labels=False, improved_preprocessing=False, version="A1-2017", relabel_testset=False):
        """
        Load the data from csv file
        :param file: csv file
        :param cache_file: path to cache preprocessed data (improves speed)
        :param normalization_file: path to normalization file (contains value ranges)
        :param recompute_norm: should the normalization be recomputed
        :param skiprows: skip first rows of the data which contains headers
        :param labels: parse labels from the data (Yes/No)
        :param improved_preprocessing: use improved preprocessing?
        :param version: version of the dataset used
        :param relabel_testset: option for relabeling
        :return: data or data and labels
        """

        # Check whether the data is already cached
        if os.path.exists(cache_file) and improved_preprocessing and not relabel_testset:
            print("Loading WaDi data from cache ...")
            cache = np.load(cache_file)

            if labels:
                return cache['arr_0'], cache['arr_1']
            else:
                return cache['arr_0']

        print("Loading WaDi data ...")

        # Read the WADI dataset
        df = pd.read_csv(open(file, 'r'), sep=',', skiprows=skiprows)
        df.columns = list(map(lambda c: c.split('\\')[-1] if 'WIN-25J4' in c else c, df.columns))
        df.columns = df.columns.str.strip()
        df = df.drop(columns='Row')

        # create a time index for version A1-2017 only
        if version == "A1-2017":
            df.index = pd.to_datetime(df.Date.map(str) + " " + df.Time.map(str))
        df = df.drop(columns=['Date', 'Time'])

        # exclude the columns that only contain NAN values
        excluded_columns = [
            "2_LS_001_AL",
            "2_LS_002_AL",
            "2_P_001_STATUS",
            "2_P_002_STATUS",
            "2_PIC_003_SP",
        ]

        if improved_preprocessing:
            # huge change in value range in the test set (causes many false positive outliers)
            excluded_columns.append("2B_AIT_002_PV")

        df = df.drop(columns=excluded_columns)

        # Normalization File
        if os.path.exists(normalization_file):
            norm_parameters = WaDiDataset._parse_normalization_file(normalization_file)
        else:
            norm_parameters = {}

        # preprocess the columns
        for col in get_pv_columns(df):
            df[col] = replace_and_interpolate_zeros(df[col])
            df[col] = normalize_column(df[col], parameters=norm_parameters, recompute=recompute_norm)

        for col in get_co_columns(df):
            df[col] = normalize_column(df[col], parameters=norm_parameters, recompute=recompute_norm)

        for col in get_sp_columns(df):
            df[col] = normalize_column(df[col], parameters=norm_parameters, recompute=recompute_norm)

        for col in get_speed_columns(df):
            df[col] = normalize_column(df[col], parameters=norm_parameters, recompute=recompute_norm)

        for col in get_status_columns(df):
            df[col] = normalize_column(df[col].interpolate(), parameters=norm_parameters, recompute=recompute_norm)

        for col in get_special_columns(df):
            df[col] = df[col].ewm(span=720).mean()
            df[col] = normalize_column(df[col], parameters=norm_parameters, recompute=recompute_norm)

        if recompute_norm:
            WaDiDataset._save_normalization_file(normalization_file, norm_parameters)

        # fix missing values
        df = df.fillna(0)

        # extract features
        pv_features = df[get_pv_columns(df)].to_numpy()
        al_features = df[get_al_columns(df)].to_numpy()
        co_features = df[get_co_columns(df)].to_numpy()
        sp_features = df[get_sp_columns(df)].to_numpy()
        ah_features = df[get_ah_columns(df)].to_numpy()
        speed_features = df[get_speed_columns(df)].to_numpy()
        status_matrix = df[get_status_columns(df)].interpolate().to_numpy()
        special_features = df[get_special_columns(df)].to_numpy()

        if not improved_preprocessing:
            all_features = np.concatenate([
                pv_features,
                co_features,
                sp_features,
                speed_features,
                al_features,
                ah_features,
                status_matrix
            ], axis=-1).astype(np.float32)
        else:
            all_features = np.concatenate([
                pv_features,
                co_features,
                sp_features,
                speed_features,
                al_features,
                ah_features,
                special_features,
                status_matrix
            ], axis=-1).astype(np.float32)

        if labels:

            if version == "A1-2017":

                # In the first version of the dataset, labels have to be extracted from time domain
                if not relabel_testset:
                    outlier_regions = pd.DataFrame([
                        {"index": 1, "start": "9/10/17 19:25:00", "end": "9/10/17 19:50:16"},
                        {"index": 2, "start": "10/10/17 10:24:10", "end": "10/10/17 10:34:00"},
                        {"index": 3, "start": "10/10/17 10:55:00", "end": "10/10/17 11:24:00"},
                        {"index": 5, "start": "10/10/17 11:30:40", "end": "10/10/17 11:44:50"},
                        {"index": 6, "start": "10/10/17 13:39:30", "end": "10/10/17 13:50:40"},
                        {"index": 7, "start": "10/10/17 14:48:17", "end": "10/10/17 14:59:55"},
                        {"index": 8, "start": "10/10/17 17:40:00", "end": "10/10/17 17:49:40"},
                        {"index": 9, "start": "11/10/17 10:55:00", "end": "11/10/17 10:56:27"},
                        {"index": 10, "start": "11/10/17 11:17:54", "end": "11/10/17 11:31:20"},
                        {"index": 11, "start": "11/10/17 11:36:31", "end": "11/10/17 11:47:00"},
                        {"index": 12, "start": "11/10/17 11:59:00", "end": "11/10/17 12:05:00"},
                        {"index": 13, "start": "11/10/17 12:07:30", "end": "11/10/17 12:10:52"},
                        {"index": 14, "start": "11/10/17 12:16:00", "end": "11/10/17 12:25:36"},
                        {"index": 15, "start": "11/10/17 15:26:30", "end": "11/10/17 15:37:00"}
                    ])
                else:
                    outlier_regions = pd.DataFrame([
                        {"index": 1, "start": "9/10/17 19:25:00", "end": "9/10/17 21:50:16"},
                        {"index": 2, "start": "10/10/17 10:24:10", "end": "10/10/17 12:34:00"},
                        {"index": 3, "start": "10/10/17 10:55:00", "end": "10/10/17 13:24:00"},
                        {"index": 5, "start": "10/10/17 11:30:40", "end": "10/10/17 13:44:50"},
                        {"index": 6, "start": "10/10/17 13:39:30", "end": "10/10/17 15:50:40"},
                        {"index": 7, "start": "10/10/17 14:48:17", "end": "10/10/17 16:59:55"},
                        {"index": 8, "start": "10/10/17 17:40:00", "end": "10/10/17 19:49:40"},
                        {"index": 9, "start": "11/10/17 10:55:00", "end": "11/10/17 12:56:27"},
                        {"index": 10, "start": "11/10/17 11:17:54", "end": "11/10/17 13:31:20"},
                        {"index": 11, "start": "11/10/17 11:36:31", "end": "11/10/17 13:47:00"},
                        {"index": 12, "start": "11/10/17 11:59:00", "end": "11/10/17 14:05:00"},
                        {"index": 13, "start": "11/10/17 12:07:30", "end": "11/10/17 14:10:52"},
                        {"index": 14, "start": "11/10/17 12:16:00", "end": "11/10/17 14:25:36"},
                        {"index": 15, "start": "11/10/17 15:26:30", "end": "11/10/17 17:37:00"}
                    ])

                outlier_regions.start = pd.to_datetime(outlier_regions.start, dayfirst=True)
                outlier_regions.end = pd.to_datetime(outlier_regions.end, dayfirst=True)

                df['label'] = 0
                for _, region in outlier_regions.iterrows():
                    df.loc[(df.index > region.start) & (df.index < region.end), 'label'] = 1

            elif version == "A2-2019":

                # read the labels from the corresponding column
                assert df.columns[-1] == 'Attack LABLE (1:No Attack, -1:Attack)'

                # convert labels to [0,1] range
                df['label'] = (df[df.columns[-1]] * -1 + 1) / 2

            # extract the labels to numpy
            outlier_labels = df.label.to_numpy()
            if improved_preprocessing and not relabel_testset:
                np.savez(cache_file, all_features, outlier_labels)
            return all_features, outlier_labels

        else:
            if improved_preprocessing:
                np.savez(cache_file, all_features)
            return all_features

    def __getitem__(self, item):
        """
        Iterate next subsequence with given index
        :param item: index of the time series
        :return: time series (and labels)
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
        :return:
        """
        return len(self._walker)


if __name__ == "__main__":

    ds = DataLoader(WaDiDataset(root="/home/cit8si/data/wadi", window_size=3600, stride=100), batch_size=32)
    print(len(ds))

    for x in ds:
        print(x.shape)
        break

    ds = DataLoader(WaDiDataset(root="/home/cit8si/data/wadi", window_size=3600, stride=100, attacks=True, improved_preprocessing=True), batch_size=32)
    print(len(ds))

    for x, y in ds:
        print(x.shape)
        print(y.shape)
        break