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

import torch
import numpy as np


class PhoneAlignment:

    phone_labels = {}

    def __init__(self, phone_label_path):
        self.phone_label_path = phone_label_path
        self.phone_labels = self.parse_label_data(self.phone_label_path)

    @staticmethod
    def parse_label_data(path):

        labels = {}

        with open(path, 'r') as file:
            for line in file.readlines():
                data = line.split(sep=' ')
                sample_id = tuple(int(d) for d in data[0].split('-'))
                data = np.array(data[1:]).astype('uint8')
                labels[sample_id] = torch.from_numpy(data)

        return labels

    def get_phone_labels(self, speaker_id, chapter_id, utterance_id):
        return self.phone_labels[(speaker_id, chapter_id, utterance_id)]