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

from .librispeech import LIBRISpeechHDF5Dataset
import torch
import math
import numpy as np


class LIBRISpeechOutlierHDF5Dataset(LIBRISpeechHDF5Dataset):
    """
    Libri Speech Dataset

    with artificial outliers placed in the data
    """

    def __init__(self, root, split_file=None, window_size=20480, output_uncorrupted=False, frequency=[45], width=[4000]):
        """
        Libri Speech Dataset with artificial outliers (= additive pure sine tones)
        :param root: dataset location
        :param split_file: file defining the split of the data
        :param window_size: size of the subsequences iterrated
        :param output_uncorrupted: Should the uncorrupted time series be outputted as well, e.g. for visualization purposes?
        :param frequency: iterable with possible outlier frequencies
        :param width: width of the outlier ranges
        """
        super(LIBRISpeechOutlierHDF5Dataset, self).__init__(root, split_file, window_size)

        self._output_uncorrupted = output_uncorrupted
        self._frequency = frequency
        self._width = width

    def __getitem__(self, item):
        """
        Iterate the next sub-sequence
        :param item: index of the sequence
        :return:
        """

        # get a sample from the super class
        sample, phones, speaker_id = super(LIBRISpeechOutlierHDF5Dataset, self).__getitem__(item)

        # corrupt the sample with outlier
        width = (np.random.choice(self._width) // 2) * 2
        rand_pos = np.random.choice(range(width, self._window_size - width))
        noise = generate_tone(pos=rand_pos, width=width, length=self._window_size, tone_freq=np.random.choice(self._frequency))  #240

        # determine the outlier labels
        labels = np.zeros(self._window_size, dtype=np.int8)
        labels[(rand_pos - width // 2):(rand_pos + width // 2)] = 1

        if not self._output_uncorrupted:
            sample = sample + noise
            return sample, labels
        else:
            return sample, sample + noise, phones, speaker_id, rand_pos


def generate_tone(pos, width, length, sampling_freq=16000, tone_freq=45, amp=0.15):
    """
    Generates an aritifical anomaly that can be placed in the data
    :param pos: position in the sequence
    :param width: with of the anomalous region
    :param length: length of the entire time series
    :param sampling_freq: sampling rate
    :param tone_freq: frequency of the pure sine tone
    :param amp: amplitude of the tone
    :return: anomalous region to be placed in the data
    """
    tone = np.sin(np.arange(width) / sampling_freq * 2 * math.pi * tone_freq) * amp
    out = np.zeros(length)
    out[(pos - width//2):(pos + width // 2)] = tone
    return out.astype(np.float32)
