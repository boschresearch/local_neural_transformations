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

from ord.datasets import get_training_datasets_from_config
from ord import get_predictor_from_config
from utils import load_configuration
import argparse


def training(config):

    # get the required datasets
    train_loader, validation_loader, evaluation_loader = get_training_datasets_from_config(config)

    # setup network and optimizer
    ord = get_predictor_from_config(config)

    # fit the network to the data
    ord.fit(train_loader, validation_loader, evaluation_loader)


if __name__ == "__main__":

    default_config = 'config/config.yaml'

    # Process the cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, default=default_config)
    args = parser.parse_args()

    # Load the configuration first
    config = load_configuration(args.config)

    # Start the training process
    training(config)
