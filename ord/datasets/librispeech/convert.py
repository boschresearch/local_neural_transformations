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

import h5py
import os
from tqdm import tqdm
import torchaudio

def convert_data_to_hdf5(datadir, export):
    """
    Converts the dataset to HDF5 format which is better suited for the use on a cluster
    :param datadir: read the data from here
    :param export: export it here
    :return:
    """

    file = h5py.File(export, 'w')
    datadir = os.path.join(datadir, "LibriSpeech", "train-clean-100")

    for reader_id in tqdm(os.listdir(datadir)):

        reader_group = file.create_group(reader_id)
        readerdir = os.path.join(datadir, reader_id)

        for chapter_id in os.listdir(readerdir):

            chapterdir = os.path.join(readerdir, chapter_id)

            for utterance_file in filter(lambda u: u.endswith(".flac"), os.listdir(chapterdir)):

                utterance_id = utterance_file.replace(".flac", "")

                # check if the dataset is already in the database
                if utterance_id in reader_group:
                    continue

                # load the data from file
                waveform, sample_rate = torchaudio.load(os.path.join(chapterdir, utterance_file))
                waveform = waveform.numpy()

                # create hdf5 dataset with attribute
                ds = reader_group.create_dataset(utterance_id, data=waveform)
                ds.attrs['sample_rate'] = sample_rate
