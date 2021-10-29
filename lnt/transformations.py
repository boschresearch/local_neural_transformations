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

from torch import nn


def get_transformation_layer(type):
    """
    Get the right Transformation class by type name
    :param type: Name of the transformation type
    :return: transformation class
    """
    return {
        "residual": ResidualTransformation,
        "feedforward": FeedForwardTransformation,
        "multiplicative": MultiplicativeTransformation
    }[type.lower()]


class LearnableTransformation(nn.Module):
    """
    A transformation that is parameterized
    """

    def __init__(self, network):
        super(LearnableTransformation, self).__init__()

        self.network = network

    def forward(self, x):
        return self.network(x)

    @staticmethod
    def get_compatible_activation():
        return nn.Identity


class FeedForwardTransformation(LearnableTransformation):
    """
    Feed Forward Transformation
    simply applies the provided network in a feed forward way
    """

    def __init__(self, network):
        super(FeedForwardTransformation, self).__init__(network)

    def forward(self, x):
        return super(FeedForwardTransformation, self).forward(x)

    @staticmethod
    def get_compatible_activation():
        return nn.Identity


class ResidualTransformation(LearnableTransformation):
    """
    Residual Transformation
    """

    def __init__(self, network):
        super(ResidualTransformation, self).__init__(network)

    def forward(self, x):
        res = self.network(x)
        return x + res

    @staticmethod
    def get_compatible_activation():
        return nn.Tanh


class MultiplicativeTransformation(LearnableTransformation):
    """
    Multiplicative Transformation
    """

    def __init__(self, network):
        super(MultiplicativeTransformation, self).__init__(network)

    def forward(self, x):
        mask = self.network(x)
        return x * mask

    @staticmethod
    def get_compatible_activation():
        return nn.Sigmoid