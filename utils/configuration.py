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

import yaml
import argparse


def load_configuration(filename):
    """
    Load a YAML configuration file
    :param filename: YAML file
    :return: configuration as nested objects
    """
    print(f"Loading configuration from file: {filename}")
    with open(filename) as file:
        return convert_config(yaml.safe_load(file))


def convert_config(config):
    """
    Convert the config such that nested objects can be parsed
    :param config:
    :return:
    """

    if type(config) is dict:
        config = argparse.Namespace(**config)

    for key in config.__dict__.keys():
        attr = getattr(config, key)

        # account for nested config objects
        if type(attr) is dict:
            setattr(config, key, convert_config(attr))
        if type(attr) is list:
            for i, item in enumerate(attr):
                if type(item) is dict:
                    attr[i] = convert_config(item)
    return config
