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
import numpy as np

from lnt.lnt import LNTNetwork, InverseLNTNetwork
from lnt import GaussianHMM
from utils import get_device
import ord


class CPCApproximationPredictor(ord.BaseOutlierRegionPredictor):
    """
    Approximate CPC Predictor
    is implemented as a baseline that scores anomalies based on CPC directly (without learned transformations)

    (based on unnormalized CPC similarity measures)
    """

    def __init__(self, config):
        """
        Initialize the Approximate CPC predictor
        :param config: configuration
        """
        super(CPCApproximationPredictor, self).__init__()

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

        # init the HMM for downstream scoring
        # ToDo: replace with dynamic parameters
        self._hmm = GaussianHMM(
            means=torch.tensor([11.0, 16.0]),  # torch.tensor([40.0, 48.0]),
            variances=torch.tensor([3, 5]),
            transition_mat=torch.tensor([[0.99, 0.01], [0.01, 0.99]]),
            prior=torch.tensor([1.0, 0.0])
        )

    def set_hmm(self, hmm):
        self._hmm = hmm

    def load(self, file):
        self._network.load_state_dict(torch.load(file))

    def save(self, path):
        raise NotImplementedError("Checkpoints are automatically saved during training.")

    def predict(self, x, apply_hmm=True, output_loss=False, score_backward=False):

        with torch.no_grad():
            # put the batch to device
            x = x.to(self._device)

            # compute the outlier scores using the approximation to the CPC loss
            scores = self._network.unnormalized_cpc_similarity_scores(x, device=self._device)
            scores = scores.detach().cpu()

            # compensate for sequence length
            sub_len = scores.shape[-1]
            if score_backward:
                l = torch.minimum(sub_len - torch.arange(0, sub_len),
                                  torch.tensor(self._config.model.n_prediction_steps))
            else:
                l = torch.minimum(torch.arange(0, sub_len), torch.tensor(self._config.model.n_prediction_steps))
            scores = scores / l
            scores[:, 0] = 0
            scores[:, -1] = 0

            if apply_hmm:
                ml_pred, prob = self._hmm.decode(scores)
                ml_pred = ml_pred.numpy().repeat(self._downsampling_factor, axis=-1)
                prob = prob[:, :, 1].numpy().repeat(self._downsampling_factor, axis=-1)
                scores = scores.numpy().repeat(self._downsampling_factor, axis=-1)

                prob = np.nan_to_num(prob)

                if not output_loss:
                    return ml_pred, scores
                else:
                    return ml_pred, prob, scores
            else:
                return None, scores.numpy().repeat(self._downsampling_factor, axis=-1)

    def fit(self, train_data, validation_data, ord_evaluation_data):
        raise NotImplementedError("Not implemented - method reads in pre-fitted CPC checkpoint for scoring.")