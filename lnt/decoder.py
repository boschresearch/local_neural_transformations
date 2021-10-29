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

from lnt.lnt import LNTNetwork, InverseLNTNetwork
from utils import get_device
from tqdm import tqdm


class LNTDecoder:
    """
    The LNT Decoder decodes the transformations learned by the LNT method back to data space where they can be visualized.
    """

    def __init__(self, config):
        """
        Creates a new decoder network from config of an exisiting LNT predictor.
        :param config: configuration
        """

        self._config = config
        self._device = get_device(config)
        self._downsampling_factor = config.model.downsampling_factor if hasattr(config.model,
                                                                                'downsampling_factor') else 160
        self._apply_cpc_after_neutral = config.model.inverse_order if hasattr(config.model, 'inverse_order') else False

        if self._apply_cpc_after_neutral:
            # Create the inverse variant from config
            self._network = InverseLNTNetwork(
                input_size=config.model.input_size,
                enc_emb_size=config.model.enc_embedding_size,
                ar_emb_size=config.model.ar_embedding_size,
                n_prediction_steps=config.model.n_prediction_steps,
                downsampling_factor=self._downsampling_factor,
                neutral_config=config.model.neutral,
                encoder_config=config.model.encoder if hasattr(config.model, 'encoder') else None
            ).to(self._device)

        else:
            # create the neutralord network from config
            self._network = LNTNetwork(
                input_size=config.model.input_size,
                enc_emb_size=config.model.enc_embedding_size,
                ar_emb_size=config.model.ar_embedding_size,
                n_prediction_steps=config.model.n_prediction_steps,
                neutral_config=config.model.neutral,
                encoder_config=config.model.encoder if hasattr(config.model, 'encoder') else None
            ).to(self._device)

        # create the actual conv. decoder network
        self._decoder_network = self.create_decoder_network(encoder_config=config.model.encoder,
                                                            data_size=config.model.input_size,
                                                            emb_size=config.model.enc_embedding_size).to(self._device)

    def create_decoder_network(self, encoder_config, data_size, emb_size):
        """
        Creates a convultional decoder network from the configuration of the encoder used in the corresponding LNT predictor.
        :param encoder_config: encoder configuration used in the corresponding LNT predictor
        :param data_size: size of the data patch to be reconstructed
        :param emb_size: size of the input embedding
        :return:
        """
        dec = []
        in_channels = emb_size

        # invert the order of the convolutions
        for i, (f, s, p) in enumerate(zip(reversed(encoder_config.filter_sizes),
                                     reversed(encoder_config.strides),
                                     reversed(encoder_config.paddings))):
            not_last = i != (len(encoder_config.filter_sizes) - 1)
            out_channels = in_channels if not_last else data_size

            # use a transposed convolution with the same parameters as the corresponding convolution in the encoder
            dec.append(torch.nn.ConvTranspose1d(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=f,
                                                padding=p,
                                                stride=s))
            if not_last:
                # add activation of hidden layers
                dec.append(torch.nn.ReLU())

                # add additional 1x1 convolutions to increase accuracy
                dec.append(torch.nn.ConvTranspose1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=1,
                                                    padding=0,
                                                    stride=1))
                dec.append(torch.nn.ReLU())
            else:
                # output layer
                dec.append(torch.nn.Sigmoid())

        return torch.nn.Sequential(*dec)

    def load(self, file):
        self._network.load_state_dict(torch.load(file))

    def save(self, path):
        raise NotImplementedError("Checkpoints are automatically saved during training.")

    def fit(self, train_data, epochs=500):
        """
        Fit the decoder on a reconstruction loss for the given number of epochs.
        :param train_data: training data
        :param epochs: number of training epochs
        :return:
        """

        params = self._decoder_network.parameters()

        opt = torch.optim.Adam(params, lr=1e-4)
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            epoch_loss = 0
            for X in train_data:

                X = X.to(self._device)

                # reconstruct the signal
                z, c = self._network.get_embeddings(X, device=self._device)
                z = z.permute(0,2,1)
                X_rec = self._decoder_network(z)

                # train on the reconstruction error
                loss = torch.mean(torch.square(X - X_rec))
                loss.backward()
                opt.step()

                epoch_loss += loss.detach().cpu().numpy()
            epoch_loss = epoch_loss / len(train_data)
            pbar.set_description(f"{epoch} : loss={epoch_loss}")

    def get_decoded_transformations(self, X):
        """
        Get the decoded transformations for a given time series X.
        :param X: input time series
        :return: tensor with concatenated transformations in dim 1
        """

        X = X.to(self._device)

        # reconstruct the signal
        z, c = self._network.get_embeddings(X, device=self._device)
        bs, sl, nz = z.shape

        z_trans = self._network.transform(z.reshape(-1, nz))
        z_trans = z_trans.reshape(bs, sl, -1, nz)
        z_trans = z_trans.permute(0, 2, 1, 3)
        z_trans = z_trans.reshape(-1, sl, nz)

        z_trans = z_trans.permute(0, 2, 1)
        z = z.permute(0, 2, 1)
        X_rec = self._decoder_network(z)
        X_trans_rec = self._decoder_network(z_trans)

        bs, nx, sl = X_rec.shape

        X_rec = torch.unsqueeze(X_rec, dim=1)
        X_trans_rec = X_trans_rec.reshape(bs, -1, nx, sl)

        return torch.cat([X_rec, X_trans_rec], dim=1)


