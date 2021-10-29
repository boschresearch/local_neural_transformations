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
from torch import nn


class ResidualBlock(nn.Module):
    """
    Residual Blocks as used in Neural Transformation Learning
    """

    def __init__(self, in_channels, n_filters, filter_sizes, strides, paddings, activation= nn.ReLU, bias=True):
        super(ResidualBlock, self).__init__()

        assert len(n_filters) == len(filter_sizes) == len(strides) == len(paddings)

        # Create the convolutions
        self.convs = nn.ModuleList()
        hidden_size = in_channels
        for n, f_size, f_stride, f_pad in zip(n_filters, filter_sizes, strides, paddings):
            self.convs.append(
                nn.Conv1d(in_channels=hidden_size, out_channels=n, kernel_size=f_size, stride=f_stride, padding=f_pad, bias=bias),
            )
            hidden_size = n

        self.activation = activation(inplace=True)


    @classmethod
    def from_config(cls, config):
        """
        Parses a residual block from YAML config
        :param config: YAML config (already parsed)
        :return: ResidualBlock instance
        """

        return cls(
            in_channels=config.in_channels,
            n_filters=config.n_filters,
            filter_sizes=config.filter_sizes,
            strides=config.strides,
            paddings=config.paddings,
            activation=nn.ReLU,
            bias=config.bias if hasattr(config, 'bias') else True
        )

    def forward(self, x):

        # Check whether the input needs to be unsqueezed
        if len(x.shape) == 2:
            out = torch.unsqueeze(x, dim=1)
        else:
            out = x

        n = len(self.convs)
        for i, conv in enumerate(self.convs):
            # apply the convolution
            out = conv(out)

            # Residual connection in the last conv layer
            if (i + 1) == n:
                out = self.activation(torch.squeeze(out) + x)
            else:
                out = self.activation(out)

        return out
