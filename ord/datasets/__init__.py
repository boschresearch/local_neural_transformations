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

import torch.utils.data
import os
from .librispeech import LIBRISpeechHDF5Dataset, LIBRISpeechOutlierHDF5Dataset
from .wadi import WaDiDataset
from .swat import SWaTDataset


def get_training_datasets_from_config(config):
    """

    Reads the dataest used for the experiments from the configuration

    :param config: configuration
    :return: training data, validation_data, test data with anomalies for evaluation
    """

    assert config.dataset, "No dataset specified."

    if config.dataset.lower() == "librispeech":

        datadir = "./data/"

        # download the waveforms
        train_data = LIBRISpeechHDF5Dataset(root=datadir, split_file=os.path.join(datadir, 'train_split.txt'),
                                            window_size=config.window_size)
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=config.num_dataloader_workers)

        validation_data = LIBRISpeechHDF5Dataset(root=datadir, split_file=os.path.join(datadir, 'test_split.txt'))
        validation_loader = torch.utils.data.DataLoader(dataset=validation_data,
                                                        batch_size=config.batch_size,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        num_workers=config.num_dataloader_workers)

        ord_evaluation_data = LIBRISpeechOutlierHDF5Dataset(root=datadir, split_file=os.path.join(datadir, 'test_split.txt'))
        ord_evaluation_loader = torch.utils.data.DataLoader(dataset=ord_evaluation_data,
                                                            batch_size=config.batch_size,
                                                            shuffle=False,
                                                            drop_last=False,
                                                            num_workers=config.num_dataloader_workers)

        return train_loader, validation_loader, ord_evaluation_loader

    elif config.dataset.lower() == "wadi":

        datadir = "./data/wadi"
        version = config.dataset_version if hasattr(config, 'dataset_version') else "A1-2017"

        train_data = WaDiDataset(root=datadir,
                                 window_size=config.window_size,
                                 stride=config.window_stride,
                                 relative_split=(0.0, 0.95),
                                 version=version,
                                 attacks=False,
                                 improved_preprocessing=config.improved_preprocessing)
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=config.num_dataloader_workers)

        validation_data = WaDiDataset(root=datadir,
                                      window_size=config.window_size,
                                      stride=config.window_stride,
                                      relative_split=(0.95, 1.0),
                                      attacks=False,
                                      version=version,
                                      improved_preprocessing=config.improved_preprocessing)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_data,
                                                        batch_size=config.batch_size,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        num_workers=config.num_dataloader_workers)

        ord_evaluation_data = WaDiDataset(root=datadir,
                                          window_size=config.window_size,
                                          stride=config.window_stride,
                                          attacks=True,
                                          version=version,
                                          improved_preprocessing=config.improved_preprocessing)
        ord_evaluation_loader = torch.utils.data.DataLoader(dataset=ord_evaluation_data,
                                                            batch_size=config.batch_size,
                                                            shuffle=True,
                                                            drop_last=True,
                                                            num_workers=config.num_dataloader_workers)

        return train_loader, validation_loader, ord_evaluation_loader

    elif config.dataset.lower() == "swat":

        datadir = "./data/swat"
        downsampling = config.dataset_downsampling if hasattr(config, 'dataset_downsampling') else 1

        train_data = SWaTDataset(datadir=datadir,
                                 window_size=config.window_size,
                                 stride=config.window_stride,
                                 relative_split=(0.0, .95),
                                 downsampling=downsampling,
                                 attacks=False)
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=config.num_dataloader_workers)

        validation_data = SWaTDataset(datadir=datadir,
                                      window_size=config.window_size,
                                      stride=config.window_stride,
                                      relative_split=(0.95, 1.0),
                                      downsampling=downsampling,
                                      attacks=False)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_data,
                                                        batch_size=config.batch_size,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        num_workers=config.num_dataloader_workers)

        ord_evaluation_data = SWaTDataset(datadir=datadir,
                                          window_size=config.window_size,
                                          stride=config.window_stride,
                                          relative_split=(0.5, 1.0),
                                          downsampling=downsampling,
                                          attacks=True)
        ord_evaluation_loader = torch.utils.data.DataLoader(dataset=ord_evaluation_data,
                                                            batch_size=config.batch_size,
                                                            shuffle=False, # True
                                                            drop_last=False, # True
                                                            num_workers=config.num_dataloader_workers)

        return train_loader, validation_loader, ord_evaluation_loader






