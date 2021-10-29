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
import torch.distributions


class GaussianHMM:
    """
    Hidden Markov Model (HMM) with Gaussian Emissions
    """

    def __init__(self, means, variances, transition_mat, prior):

        self._k = len(means)
        self._means = means
        self._variances = variances
        self._transition_mat = transition_mat

        self._transition_dist = torch.distributions.Categorical(
            probs=self._transition_mat
        )

        self._emission_dist = torch.distributions.Normal(
            loc=self._means,
            scale=self._variances
        )

        self._state_prior_dist = torch.distributions.Categorical(
            probs=prior
        )

    def log_prob(self, x):
        pass

    def decode(self, x):
        """
        Viterbi decoding to extract the maximum likelihood sequence of hidden states
        (= Anomalous & Not Anomalous)
        :param x: sequence of observations (= scores of the LNT method)
        :return:
        """

        assert len(x.shape) == 2
        sequence_length = x.shape[-1]
        batch_size = x.shape[0]
        temp = torch.zeros(batch_size, sequence_length, self._k)
        back = torch.zeros(batch_size, sequence_length, self._k, dtype=torch.int)

        # initial values
        temp[:, 0, :] = self._state_prior_dist.logits

        # forward pass
        for i in range(1, sequence_length):

            prob = self._transition_dist.logits \
                      + torch.unsqueeze(temp[:, i - 1, :], dim=2) \
                      + torch.unsqueeze(self._emission_dist.log_prob(x[:, i:i+1]), dim=1)
            temp[:, i, :], back[:, i, :] = torch.max(prob, dim=1)

        # backtrack ml sequence
        ml_sequence = torch.zeros(batch_size, sequence_length, dtype=torch.int64)
        ml_sequence[:, -1] = torch.argmax(temp[:, -1, :], dim=-1)

        for i in reversed(range(1, sequence_length)):
            selection = torch.gather(back[:, i, :], dim=-1, index=ml_sequence[:, i:i+1])
            ml_sequence[:, i - 1] = torch.squeeze(selection)

        likelihood = torch.exp(temp) / torch.sum(torch.exp(temp), dim=2, keepdim=True)

        return ml_sequence, likelihood


