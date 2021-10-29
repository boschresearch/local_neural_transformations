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
import torch


class InfoNCE(nn.Module):
    """
    This class computes the Noise Contrastive Estimation (NCE) loss called InfoNCE.
    It can be viewed as a lower bound to mutual information.

    Implementation follows https://arxiv.org/abs/1807.03748v1
    """

    def __init__(self, enc_emb_size, ar_emb_size, n_prediction_steps):
        """
        InfoNCE instance
        :param enc_emb_size: size of the encoder embeddings
        :param ar_emb_size: size of the embeddings used in auto-regressive compression
        :param n_prediction_steps: number of prediction steps into the future
        """
        super().__init__()

        self._enc_emb_size = enc_emb_size
        self._ar_emb_size = ar_emb_size
        self._n_prediction_steps = n_prediction_steps

        # Create linear future predictors
        self._future_predictor = nn.Linear(in_features=ar_emb_size,
                                           out_features=enc_emb_size * n_prediction_steps, bias=False)

    def n_prediction_steps(self):
        """
        Get the number of prediction steps into the future
        :return:
        """
        return self._n_prediction_steps

    @staticmethod
    def _compute_f_score(z_k, Wc_k):
        """
        Computes the f-score as an inner product between (true) local embeddings z_k and future predictions Wc_k.
        :param z_k: true local embeddings
        :param Wc_k: future predictions
        :return: the f-score
        """
        return torch.squeeze(torch.matmul(z_k, Wc_k), dim=-1)

    def _pack_batch_sequence(self, x):
        """
        Packs batch size and sequence length into the first dimension.
        :param x: batch with shape [B, S, C]
        :return: packed batch with size [B*S, C]
        """
        return torch.reshape(x, (-1, self._enc_emb_size))

    def predict_future(self, c, k=None):
        """
        Predicts the future Wc_t using matrices W
        :param c: global context
        :param k: (optional, default= all steps) number of steps in the future
        :return: Wc
        """

        if not k:
            return self._future_predictor(c)
        else:
            return self._future_predictor(c)[:, :-k, (k - 1) * self._enc_emb_size: k * self._enc_emb_size]

    def enumerate_future(self, z, c):
        """
        Enumerates future prediction steps for each k individually
        :param z:
        :param c:
        :return:
        """

        Wc = self.predict_future(c)

        for k in range(1, self._n_prediction_steps + 1):

            # extract the ground truth future
            z_k = z[:, k:, :]

            # extract the relevant future predictions
            Wc_k = Wc[:, :-k, (k - 1) * self._enc_emb_size: k * self._enc_emb_size]

            yield k, z_k, Wc_k

    def partial_contrastive_loss(self, z_k, Wc_k):
        """
        Computes the contribution to the contrastive loss for a specific k
        :param z_k:
        :param Wc_k:
        :return: (partial) contrastive loss
        """

        # pack the sequence length to the batch size
        z_k = self._pack_batch_sequence(z_k)
        Wc_k = self._pack_batch_sequence(Wc_k)
        Wc_k = Wc_k.permute(1, 0)

        # compute positive f scores
        f_score = self._compute_f_score(z_k, Wc_k)

        f_score_pos = torch.diag(f_score)
        f_score_sum = torch.logsumexp(f_score, dim=[0, 1])

        # loss is the negative log-softmax
        loss = - (f_score_pos - f_score_sum).sum() / f_score.size(0)

        # compute accuracies
        acc = torch.mean(
            torch.eq(torch.argmax(f_score, dim=0),
                     torch.arange(0, f_score.size(0), device=f_score.get_device())).float())

        return loss, acc

    def contrastive_loss(self, z, c):
        """
        Computes the contrastive loss and the accuracy of contrasting future predictions
        for different amount of steps into the future.
        :param z: local embeddings
        :param c: global embeddings
        :return: contrastive loss, accuracy
        """

        total_loss = 0
        accuracies = torch.zeros(self._n_prediction_steps)

        # make predictions for the future
        Wc = self._future_predictor(c)

        # Compute the loss for each number of steps k into the future individually
        for k in range(1, self._n_prediction_steps + 1):

            # extract the ground truth future
            z_k = z[:, k:, :]

            # extract the relevant future predictions
            Wc_k = Wc[:, :-k, (k - 1) * self._enc_emb_size: k * self._enc_emb_size]

            # pack the sequence length to the batch size
            z_k = self._pack_batch_sequence(z_k)
            Wc_k = self._pack_batch_sequence(Wc_k)
            Wc_k = Wc_k.permute(1, 0)

            # compute positive f scores
            f_score = self._compute_f_score(z_k, Wc_k)

            f_score_pos = torch.diag(f_score)
            f_score_sum = torch.logsumexp(f_score, dim=[0,1])

            # loss is the negative log-softmax
            loss = - (f_score_pos - f_score_sum).sum() / f_score.size(0)
            total_loss += loss

            # compute accuracies
            accuracies[k - 1] = torch.mean(
                torch.eq(torch.argmax(f_score, dim=0), torch.arange(0, f_score.size(0), device=f_score.get_device())).float())

        total_loss = total_loss / self._n_prediction_steps

        return total_loss, accuracies

    def forward(self, c):
        return self.predict_future(c)