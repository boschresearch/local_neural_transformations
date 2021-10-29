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

import numpy as np
import os
import torchaudio
from .phonealignment import PhoneAlignment
from torch.utils.data import Dataset
import h5py


class LIBRISpeechDataset(torchaudio.datasets.LIBRISPEECH):
    """
    100h of LIBRI speech dataset
    """

    def __init__(self, root, split_file=None, window_size=20480):
        """
        LIBRI Speech
        :param root: the root data directory
        :param split_file: text file defining the split for the dataset
        :param window_size: the size of the window function applied during batching
        """
        super().__init__(root, url='train-clean-100', download=True)

        # read the split file if available
        if split_file:
            self._walker = sorted(self.read_dataset_split_from_file(split_file))

        # read the aligned phone labels
        self._phone_alignment = PhoneAlignment(phone_label_path=os.path.join(root, "converted_aligned_phones.txt"))

        self._window_size = window_size

    @staticmethod
    def read_dataset_split_from_file(filename):
        """
        Read in the split file for the dataset
        :param filename:
        :return:
        """
        with open(filename, 'r') as f:
            return set(map(str.strip, f.readlines()))

    def __getitem__(self, item):
        """
        Iterate the next time series (with some meta data) in the dataset
        :param item: index of the series
        :return: times series, phone labels and speaker id
        """
        # get the data
        # (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        sample, sample_rate, _, speaker_id, chapter_id, utterance_id = super(LIBRISpeechDataset, self).__getitem__(item)
        phones = self._phone_alignment.get_phone_labels(speaker_id, chapter_id, utterance_id)

        # apply a random window crop
        length = sample.shape[-1] // 160 * 160
        phone_length = phones.shape[-1]

        difference = length // 160 - phone_length
        assert(difference >= 1)
        # print(f"Length difference is {difference}")

        sample_idx = np.random.choice(range(160, length - self._window_size - (difference - 1)*160,160))
        phone_idx = sample_idx // 160

        sample = sample[:, sample_idx:(sample_idx + self._window_size)]
        phones = phones[phone_idx:(phone_idx + self._window_size // 160)]

        return sample, phones, speaker_id

    def __len__(self):
        return super(LIBRISpeechDataset, self).__len__()


class LIBRISpeechHDF5Dataset(Dataset):
    """
    100h of LIBRI speech dataset

    (reads the data from the improved HDF5 file format)
    """

    def __init__(self, root, split_file=None, window_size=20480):
        """
        LIBRI Speech
        :param root: the root data directory
        :param split_file: text file defining the split for the dataset
        :param window_size: the size of the window function applied during batching
        """
        super().__init__()

        # read the split file if available
        if split_file:
            self._walker = sorted(self.read_dataset_split_from_file(split_file))

        # read the aligned phone labels
        self._phone_alignment = PhoneAlignment(phone_label_path=os.path.join(root, "converted_aligned_phones.txt"))

        self._window_size = window_size
        self._h5file = None
        self._h5path = os.path.join(root, "train.h5")

    @staticmethod
    def read_dataset_split_from_file(filename):
        with open(filename, 'r') as f:
            return set(map(str.strip, f.readlines()))

    def __getitem__(self, item):
        """
        Iterate the next time series (with some meta data) in the dataset
        :param item: index of the series
        :return: times series, phone labels and speaker id
        """

        # get the data
        # (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        dataset = self._walker[item]
        speaker_id, chapter_id, utterance_id = map(int, dataset.split('-'))

        phones = self._phone_alignment.get_phone_labels(speaker_id, chapter_id, utterance_id)

        if self._h5file is None:
            self._h5file = h5py.File(self._h5path, "r")

        sample = self._h5file[f"{speaker_id}/{dataset}"]

        # apply a random window crop
        length = sample.shape[-1] // 160 * 160
        phone_length = phones.shape[-1]

        difference = length // 160 - phone_length
        assert(difference >= 1)
        # print(f"Length difference is {difference}")

        sample_idx = np.random.choice(range(160, length - self._window_size - (difference - 1)*160, 160))
        phone_idx = sample_idx // 160

        # actually fetches the data
        sample = sample[:, sample_idx:(sample_idx + self._window_size)]
        phones = phones[phone_idx:(phone_idx + self._window_size // 160)]

        return sample, phones, speaker_id

    def __len__(self):
        """
        Length of the dataset
        :return:
        """
        return len(self._walker)


