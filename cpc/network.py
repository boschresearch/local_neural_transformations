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
from cpc.infonce import InfoNCE


def create_strided_convolutional_encoder(
        input_size,
        hidden_size,
        strides=[5, 4, 2, 2, 2],
        filters=[10, 8, 4, 4, 4],
        padding=[2, 2, 2, 2, 1]
):
    """
    Creates a simple convolutional encoder from hyper-parameters
    :param input_size:
    :param hidden_size:
    :param strides:
    :param filters:
    :param padding:
    :return:
    """
    assert(len(strides) == len(filters) == len(padding))

    enc = nn.Sequential()

    for layer, (s, f, p) in enumerate(zip(strides, filters, padding)):
        enc.add_module(name=f'enc_layer{layer}', module=nn.Sequential(
            nn.Conv1d(in_channels=input_size,
                      out_channels=hidden_size,
                      kernel_size=f, stride=s, padding=p),
            nn.ReLU()
        ))
        input_size = hidden_size
    return enc


class CPCNetwork(nn.Module):
    """
    Network used for Contrastive Predictive Coding (CPC)

    Architecture follows ideas from https://arxiv.org/abs/1807.03748v1
    """

    def __init__(self, input_size, enc_emb_size, ar_emb_size, n_prediction_steps, encoder=None, autoregressive=None, encoder_config=None):
        """
        CPC Network instance
        :param input_size: number of channels in the input time series
        :param enc_emb_size: embedding size of the encoder
        :param ar_emb_size: size of the embeddings used in auto-regressive compression
        :param n_prediction_steps: number of prediction steps into the future
        :param encoder: encoder network (optional)
        :param autoregressive: autoregressive network (optional)
        """
        super().__init__()

        self._enc_emb_size = enc_emb_size
        self._ar_emb_size = ar_emb_size

        if not encoder:

            if not encoder_config:
                # create the standard encoder from CPC paper
                self._encoder = create_strided_convolutional_encoder(input_size=input_size,
                                                                     hidden_size=enc_emb_size)
            else:
                # parse the encoder from config file
                self._encoder = create_strided_convolutional_encoder(input_size=input_size,
                                                                     hidden_size=enc_emb_size,
                                                                     strides=encoder_config.strides,
                                                                     filters=encoder_config.filter_sizes,
                                                                     padding=encoder_config.paddings)

        else:
            self._encoder = encoder

        if not autoregressive:
            self._autoregressive = nn.GRU(input_size=enc_emb_size, hidden_size=ar_emb_size, batch_first=True)
        else:
            self._autoregressive = autoregressive

        self._infoNCE = InfoNCE(enc_emb_size, ar_emb_size, n_prediction_steps=n_prediction_steps)

    def unnormalized_cpc_similarity_scores(self, x, device):
        """
        Get the unnormalized similarity scores for the CPC predictors.
        These can be used in a variant for scoring anomalies directly.
        :param x: input time series
        :param device: device where the weights are placed
        :return:
        """

        # get the embeddings first
        z, c = self.get_embeddings(x, device)
        batch_size, seq_len, emb_size = z.shape

        cpc_scores = torch.zeros((batch_size, seq_len), device=device)

        for k, z_k, Wc_k in self._infoNCE.enumerate_future(z, c):

            # pack the sequence length to the batch size
            z_k = self._pack_batch_sequence(z_k)
            Wc_k = self._pack_batch_sequence(Wc_k)
            Wc_k = Wc_k.permute(1, 0)

            # positive f_score similarities can be found on the diagonal
            f_score = - torch.diag(self._infoNCE._compute_f_score(z_k, Wc_k))

            # add up to the final scores
            cpc_scores[:, k:] += f_score.resize(batch_size, seq_len - k)

        return cpc_scores

    def get_embeddings(self, x, device, return_global_context=True):
        """
        Computes the embeddings c_t, z_t
        :param x: input sequence of shape [B, C, L]
        :return:
        """

        # encode to local embeddings z
        z = self._encoder(x)
        z = z.permute(0, 2, 1)

        if not return_global_context:
            return z

        # auto-regress to global embedding c
        batch_size = x.size(0)
        h = torch.zeros((1, batch_size, self._ar_emb_size)).to(device)
        c, h = self._autoregressive(z, h)

        return z, c

    def forward(self, x, device):
        """
        Computes the contrastive loss in a forward pass
        :param x:
        :param device:
        :return:
        """
        z, c = self.get_embeddings(x, device)
        loss = self._infoNCE.contrastive_loss(z, c)
        return loss