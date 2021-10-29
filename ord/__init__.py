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

from .predictor import BaseOutlierRegionPredictor

import lnt


def get_predictor_from_config(config):
    """
    Get the predictor class used from the config
    :param config: configuration with secified predictor
    :return: instance of the predictor
    """

    assert config.model.type, "No model type specified."

    if config.model.type.lower() in ['neutralard', 'neutralord', 'lnt']:
        return lnt.LNTPredictor(config)

    if config.model.type.lower() == "lstm":
        raise NotImplementedError("Baseline code not inlcuded.")

    if config.model.type.lower() == "thoc":
        raise NotImplementedError("Baseline code not inlcuded.")