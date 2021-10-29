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
import os


def get_device(config):
    """
    Get the device as defined in the config file
    :param config: config file
    :return: device
    """
    if not hasattr(config, 'device') or config.device == "default" or not torch.cuda.is_available():
        device = "cpu"
        print(f"Fallback to default device: {device}")
    else:
        device = config.device

    print(f"Running experiments on device {device}")
    return torch.device(device)


def save_checkpoint(epoch, state_dict, outdir, keep_every_epoch=None):
    """
    Saves a checkpoint to disk
    :parma epoch
    :param state_dict:
    :param outdir:
    :param keep_every_epoch:
    :return:
    """
    file = "checkpoint_{}.tar"
    path = os.path.join(outdir, file.format(epoch))
    path_old = os.path.join(outdir, file.format(epoch - 1))

    # Save the checkpoint
    torch.save(state_dict, path)

    if keep_every_epoch:
        # Check whether the previous checkpoint needs to be removed
        if epoch > 0 and (epoch - 1) % keep_every_epoch != 0:
            os.remove(path_old)


def activation_from_name(name):
    assert hasattr(torch.nn, name), f"Activation {name} not found."
    return getattr(torch.nn, name)
