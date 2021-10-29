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
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from lnt.blocks import ResidualBlock
from lnt.transformations import *
from lnt.hmm import GaussianHMM
from lnt.early_stopping import NoImprovementStoppingCriterion
from cpc import CPCNetwork
import ord
import ord.evaluation
from utils import get_device, save_checkpoint, activation_from_name


class NeuralTransformationNetwork(nn.Module):
    """
    Neural Transformation Learning Network Module
    """

    def __init__(self,input_size, config):
        super(NeuralTransformationNetwork, self).__init__()

        # save the size of the embedding (number of channel is one if not otherwise specified)
        self.input_size = input_size

        resnet_config = config.resnet if hasattr(config, 'resnet') else None
        mlp_config = config.mlp if hasattr(config, 'mlp') else None

        # Create the transformations
        self._transformations = nn.ModuleList()
        self.n_transformations = config.n_transformations
        for l in range(self.n_transformations):

            if resnet_config:
                # Residual blocks
                residual_blocks = []
                for block_conf in resnet_config:
                    residual_blocks.append(ResidualBlock.from_config(block_conf))

                # get the Transformation type
                transformation = get_transformation_layer(config.transformation_type)

                # Build a sequential resnet from blocks
                self._transformations.append(transformation(nn.Sequential(*residual_blocks)))

            elif mlp_config:

                layers = []
                layer_size = input_size
                use_bias = mlp_config.bias if hasattr(mlp_config, 'bias') else True
                act = activation_from_name(mlp_config.activation)

                # init the hidden layers
                for n in mlp_config.n_hidden:
                    layers.append(torch.nn.Linear(in_features=layer_size, out_features=n, bias=use_bias))
                    layers.append(act())
                    layer_size = n

                # get the Transformation type
                transformation = get_transformation_layer(config.transformation_type)

                # output layer
                # must always be the same size as the input
                layers.append(torch.nn.Linear(in_features=layer_size, out_features=input_size, bias=use_bias))
                layers.append(transformation.get_compatible_activation()())

                self._transformations.append(transformation(nn.Sequential(*layers)))

    @staticmethod
    def cosine_similarity(x, x_):
        """
        Computes the cosine similarity between the input vectors
        :param x: input vector
        :param x_: input vector
        :return: cosine similarity
        """

        x = F.normalize(x, p=2, dim=-1)
        x_ = F.normalize(x_, p=2, dim=-2)
        dot = torch.matmul(x, x_)
        return dot

    def deterministic_contrastive_loss(self, z, temperature, average=True):
        """
        Computes the deterministic contrastive loss.
        This method uses the ground truth embedding z_t to contrast against transformations.
        :param z: inputs of shape [BS, ES]
        :param temperature: temperature parameter in the loss
        :param average: average the loss over the batch ?
        :return: DCL
        """

        # Apply the transformation
        z_transformed = self(z)

        # Compute the similarity matrix
        z_all = torch.cat([z.unsqueeze(1), z_transformed], dim=1)  # Shape: [BS, (NT + 1), ES]
        sim = self.cosine_similarity(z_all, z_all.permute(0, 2, 1)) / temperature  # Shape: [BS, (NT + 1), (NT + 1)]
        sim_pos = sim[:, 1:, 0] # Shape: [BS, NT]

        # drop the diagonal (contains self-similarity)
        device = z.get_device()
        batch_size = z.shape[0]
        nt = self.n_transformations + 1
        mask = (torch.ones(nt, nt) - torch.eye(nt)).bool().to(device)
        sim_all = torch.masked_select(sim, mask).view(batch_size, nt, -1)[:, 1:, :]
        normalization_const = torch.logsumexp(sim_all, dim=-1)

        # compute the dcl loss
        loss = - torch.sum(sim_pos - normalization_const, dim=-1)

        if average:
            return torch.mean(loss)  # take the mean over the batch ?
        else:
            return loss

    def dynamic_deterministic_contrastive_loss(self, z_tilde, z=None, z_transformed=None, temperature=1.0, average=True):
        """
        Computes the dynamic deterministic contrastive loss.
        :param z: Ground truth embeddings z
        :param z_tilde: Future forecast embeddings
        :param z_transformed: already transformed z embeddings
        :param temperature:
        :return: DDCL
        """

        assert z is not None or z_transformed is not None

        if z_transformed is None:
            # Apply the transformation
            z_transformed = self(z)

        device = z_transformed.get_device()
        batch_size = z_transformed.shape[0]

        # Compute the similarity matrix
        z_all = torch.cat([z_tilde.unsqueeze(1), z_transformed], dim=1)  # Shape: [BS, (NT + 1), ES]
        sim = self.cosine_similarity(z_all, z_all.permute(0, 2, 1)) / temperature  # Shape: [BS, (NT + 1), (NT + 1)]
        sim_pos = sim[:, 1:, 0]  # Shape: [BS, NT]

        # drop the diagonal (contains self-similarity)
        nt = self.n_transformations + 1
        mask = (torch.ones(nt, nt) - torch.eye(nt)).bool().to(device)
        sim_all = torch.masked_select(sim, mask).view(batch_size, nt, -1)[:, 1:, :]
        normalization_const = torch.logsumexp(sim_all, dim=-1)

        # compute the dcl loss
        loss = - torch.sum(sim_pos - normalization_const, dim=-1)

        if average:
            return torch.mean(loss)  # take the mean over the batch ?
        else:
            return loss

    def forward(self, z, transformation_dim=1):
        """
        Apply the learned transformations in a forward pass
        :param z: input to transform [BS, ES]
        :return: transformed inputs [BS, NT, ES]
        """
        return torch.stack([trans(z) for trans in self._transformations], dim=transformation_dim)


class LNTNetwork(CPCNetwork):
    """
    Implementation of a joint network that applies local neural transformations (LNT) to latent representations acquired from a representation learner, here CPC.
    """

    def __init__(self, input_size, enc_emb_size, ar_emb_size, n_prediction_steps, neutral_config, encoder_config=None):
        """
        Local Neural Transformation Network Instance
        :param input_size: size of the input data
        :param enc_emb_size: size of the ecnoder embedding
        :param ar_emb_size: size of the context embedding
        :param n_prediction_steps: number of prediction steps ahead
        :param neutral_config: transformation learning config
        :param encoder_config: encoder (CPC) config
        """
        super(LNTNetwork, self).__init__(input_size, enc_emb_size, ar_emb_size, n_prediction_steps,
                                         encoder_config=encoder_config)

        self._neuTraL = NeuralTransformationNetwork(input_size=enc_emb_size, config=neutral_config)
        self._temperature = neutral_config.dcl_temperature
        self._detach = neutral_config.detach if hasattr(neutral_config, 'detach') else True
        self._pretrain_representations_for = neutral_config.pretrain_representations_for_epochs if hasattr(neutral_config, 'pretrain_representations_for_epochs') else None

    def _pack_batch_sequence(self, x):
        """
        Packs batch size and sequence length into the first dimension.
        :param x: batch with shape [B, S, C]
        :return: packed batch with size [B*S, C]
        """
        return x.reshape(-1, self._enc_emb_size)

    def score_outlierness(self, x, device, score_backward=False):
        """
        Score the outlierness with the DDCL loss
        :param x: input time series to score
        :param device:
        :param score_backward: use forward or backward scoring ?
        :return:
        """

        # compute the CPC embeddings
        z, c = self.get_embeddings(x, device)
        batch_size, seq_len, emb_size = z.shape

        # compute the losses with an outer loop over all k
        loss_dcl = torch.zeros((batch_size, seq_len), device=device)

        for k, z_k, Wc_k in self._infoNCE.enumerate_future(z, c):

            # compute the DCL loss
            loss_dcl_k = self._neuTraL.dynamic_deterministic_contrastive_loss(
                z=self._pack_batch_sequence(z_k),
                z_tilde=self._pack_batch_sequence(Wc_k),
                temperature= self._temperature,
                average=False)

            # Accumulate losses
            if score_backward:
                loss_dcl[:, :-k] += loss_dcl_k.resize(batch_size, seq_len - k)
            else:
                loss_dcl[:, k:] += loss_dcl_k.resize(batch_size, seq_len - k)

        return loss_dcl

    def transform(self, z):
        """
        Apply learned transformations to the embeddings z
        :param z: embeddings to transform
        :return: transformed embeddings
        """
        return self._neuTraL(z)

    def forward(self, x, device, epoch=None):
        """
        Compute all losses in the forward pass
        :param x:
        :param device:
        :param epoch:
        :return:
        """

        # check detaching criterium
        if epoch is None or self._pretrain_representations_for is None:
            detach = self._detach
        else:
            detach = self._detach if epoch > self._pretrain_representations_for else True

        # compute the CPC embeddings
        z, c = self.get_embeddings(x, device)

        # compute the losses with an outer loop over all k
        loss_cpc = 0
        loss_dcl = 0
        acc_cpc = torch.zeros(self._infoNCE.n_prediction_steps(), device=device)
        for k, z_k, Wc_k in self._infoNCE.enumerate_future(z, c):

            # compute the CPC loss
            loss_cpc_k, acc_cpc_k = self._infoNCE.partial_contrastive_loss(z_k, Wc_k)

            # compute the DCL loss
            batch_size, seq_len, emb_size = z_k.shape
            slice = torch.randint(0, seq_len, (batch_size,), device=device).unsqueeze(1).repeat(1, emb_size).unsqueeze(1)
            loss_dcl_k = self._neuTraL.dynamic_deterministic_contrastive_loss(
                        z=torch.gather(z_k.detach() if detach else z_k, dim=1, index=slice).squeeze(),
                        z_tilde=torch.gather(Wc_k.detach() if detach else Wc_k, dim=1, index=slice).squeeze(),
                        temperature=self._temperature)

            # Accumulate losses and statistics
            loss_cpc += loss_cpc_k
            loss_dcl += loss_dcl_k
            acc_cpc[k-1] = acc_cpc_k

        # take the mean of the losses over all prediction steps
        loss_cpc /= self._infoNCE.n_prediction_steps()
        loss_dcl /= self._infoNCE.n_prediction_steps()

        return loss_cpc, loss_dcl, acc_cpc


class InverseLNTNetwork(CPCNetwork):
    """
    Inverse Variant of the LNT Network that "inverts" the order of representation learning with CPC and application of neural transformations.
    Thus transformations are applied in data space.
    """

    def __init__(self, input_size, enc_emb_size, ar_emb_size, n_prediction_steps, downsampling_factor, neutral_config, encoder_config=None):
        super(InverseLNTNetwork, self).__init__(input_size, enc_emb_size, ar_emb_size, n_prediction_steps,
                                                encoder_config=encoder_config)

        # Initialize the transformation learning
        self._neuTraL = NeuralTransformationNetwork(input_size=input_size, config=neutral_config)

        self._downsampling_factor = downsampling_factor
        self._temperature = neutral_config.dcl_temperature
        self._detach = neutral_config.detach if hasattr(neutral_config, 'detach') else True
        self._pretrain_representations_for = neutral_config.pretrain_representations_for_epochs if hasattr(
            neutral_config, 'pretrain_representations_for_epochs') else None

    def enumerate_transformed_future(self, z, z_trans, c):
        """
        Enumerates the future in CPC fashion but additionaly taking transformations into account
        :param z: local embeddings
        :param c: global context
        :param z_trans: local embeddings of transformed time series
        :return:
        """

        for k, z_k, c_k in self._infoNCE.enumerate_future(z, c):

            # extract the corresponding slice and change batch and transformation dim
            z_trans_k = z_trans[:, :, k:, :].transpose(0,1)

            yield k , z_k, z_trans_k, c_k

    def score_outlierness(self, x, device, score_backward=False):
        """
        Score the outlierness / anomaly of a given time series at hand (with the DDCL loss)
        :param x:
        :param device:
        :param score_backward:
        :return:
        """
        bs, nc, sl = x.shape

        x_unfold = self.unfold_time_distributed(x)
        x_trans = self._neuTraL(x_unfold, transformation_dim=0)

        # fold the time distributed input
        x_trans = x_trans.reshape(self._neuTraL.n_transformations, bs, -1, nc, self._downsampling_factor).transpose(2,3)
        x_trans = x_trans.reshape(self._neuTraL.n_transformations, bs, nc, -1)

        # compute the CPC embeddings
        z, c = self.get_embeddings(x, device=device)
        z_trans = self.get_embeddings(x_trans.reshape(-1, nc, sl),
                                      device=device,
                                      return_global_context=False)

        # reshape the tensors
        # the reduced sequence length is inferred
        z_trans = z_trans.reshape(self._neuTraL.n_transformations, bs, -1, self._enc_emb_size)

        batch_size, seq_len, emb_size = z.shape

        # compute the losses with an outer loop over all k
        loss_dcl = torch.zeros((batch_size, seq_len), device=device)

        for k, z_k, z_trans_k, Wc_k in self.enumerate_transformed_future(z, z_trans, c):

            # compute the DCL loss
            loss_dcl_k = self._neuTraL.dynamic_deterministic_contrastive_loss(
                z_transformed=self._pack_batch_sequence(z_trans_k),
                z_tilde=self._pack_batch_sequence(Wc_k),
                temperature=self._temperature,
                average=False)

            # Accumulate losses
            if score_backward:
                loss_dcl[:, :-k] += loss_dcl_k.resize(batch_size, seq_len - k)
            else:
                loss_dcl[:, k:] += loss_dcl_k.resize(batch_size, seq_len - k)

        return loss_dcl

    def _pack_batch_sequence(self, x):
        """
        Packs batch size and sequence length into the first dimension.
        :param x: batch with shape [B, S, C] or shape [B, T, S, C]
        :return: packed batch with size [B*S, C] or shape [B*S, T, C]
        """
        if len(x.shape) == 3:
            return x.reshape(-1, self._enc_emb_size)
        else:
            return x.transpose(1,2).reshape(-1, self._neuTraL.n_transformations, self._enc_emb_size)

    def unfold_time_distributed(self, x):

        bs, nc, sl = x.shape

        # prepare the unfolded input
        x_unfold = x.unfold(-1, self._downsampling_factor, self._downsampling_factor)
        x_unfold = x_unfold.transpose(1, 2).reshape(-1, nc, self._downsampling_factor)
        return x_unfold

    def fold_time_distributed(self, x_trans):
        pass

    def forward(self, x, device, epoch=None):

        # check detaching criterium
        if epoch is None or self._pretrain_representations_for is None:
            detach = self._detach
        else:
            detach = self._detach if epoch > self._pretrain_representations_for else True

        bs, nc, sl = x.shape

        # apply the transformations to the time series
        x_unfold = self.unfold_time_distributed(x)
        x_trans = self._neuTraL(x_unfold, transformation_dim=0)

        # fold the time distributed input
        x_trans = x_trans.reshape(self._neuTraL.n_transformations, bs, -1, nc, self._downsampling_factor).transpose(2, 3)
        x_trans = x_trans.reshape(self._neuTraL.n_transformations, bs, nc, -1)

        # compute the CPC embeddings
        z, c = self.get_embeddings(x, device=device)
        z_trans = self.get_embeddings(x_trans.reshape(-1, nc, sl),
                                      device=device,
                                      return_global_context=False)

        # reshape the tensors
        # the reduced sequence length is inferred
        z_trans = z_trans.reshape(self._neuTraL.n_transformations, bs, -1, self._enc_emb_size)

        # compute the losses with an outer loop over all k
        loss_cpc = 0
        loss_dcl = 0
        acc_cpc = torch.zeros(self._infoNCE.n_prediction_steps(), device=device)
        for k, z_k, z_trans_k, Wc_k in self.enumerate_transformed_future(z, z_trans, c):

            # compute the CPC loss
            loss_cpc_k, acc_cpc_k = self._infoNCE.partial_contrastive_loss(z_k, Wc_k)

            # sample unbiased subset to reduce memory load
            batch_size, seq_len, emb_size = z_k.shape
            slice = torch.randint(0, seq_len, (batch_size,), device=device).unsqueeze(1).repeat(1, emb_size).unsqueeze(
                1)
            trans_slice = slice.unsqueeze(1).repeat(1, self._neuTraL.n_transformations, 1, 1)

            # compute the DCL loss
            loss_dcl_k = self._neuTraL.dynamic_deterministic_contrastive_loss(
                z_transformed=torch.gather(z_trans_k.detach() if detach else z_trans_k, dim=2, index=trans_slice).squeeze(),
                z_tilde=torch.gather(Wc_k.detach() if detach else Wc_k, dim=1, index=slice).squeeze(),
                temperature=self._temperature)

            # Accumulate losses and statistics
            loss_cpc += loss_cpc_k
            loss_dcl += loss_dcl_k
            acc_cpc[k - 1] = acc_cpc_k

        # take the mean of the losses over all prediction steps
        loss_cpc /= self._infoNCE.n_prediction_steps()
        loss_dcl /= self._infoNCE.n_prediction_steps()

        return loss_cpc, loss_dcl, acc_cpc


class LNTPredictor(ord.BaseOutlierRegionPredictor):
    """
    LNT predictor class
    """

    def __init__(self, config):
        """
        LNT Predictor
        Can either implement the normal LNT or the inverse LNT based on specifications in the config
        :param config: predictor configuration
        """
        super(LNTPredictor, self).__init__()
        
        self._config = config
        self._device = get_device(config)
        self._downsampling_factor = config.model.downsampling_factor if hasattr(config.model, 'downsampling_factor') else 160
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
        self._hmm = GaussianHMM(
            means=torch.tensor([11.0, 16.0]),
            variances=torch.tensor([3, 5]),
            transition_mat=torch.tensor([[0.99, 0.01], [0.01, 0.99]]),
            prior=torch.tensor([1.0, 0.0])
        )

    def set_hmm(self, hmm):
        """
        Sets custom parameters for the HMM
        :param hmm:
        :return:
        """
        self._hmm = hmm

    def load(self, file):
        self._network.load_state_dict(torch.load(file))

    def save(self, path):
        raise NotImplementedError("Checkpoints are automatically saved during training.")

    def get_representations(self, x, transformed=False, k=None):

        if not transformed:
            return self._network.get_embeddings(x, device=self._device)
        else:
            z, c = self._network.get_embeddings(x, device=self._device)
            z_trans = self._network.transform(z[:, k:, :])
            z_tilde = self._network._infoNCE.predict_future(c, k=k)

            return torch.cat([z_tilde.unsqueeze(1), z_trans], dim= 1)

    def predict(self, x, apply_hmm=True, output_loss=False, score_backward=False):

        with torch.no_grad():
            # put the batch to device
            x = x.to(self._device)

            # compute the outlier scores with the network
            scores = self._network.score_outlierness(x, device=self._device, score_backward=score_backward)
            scores = scores.detach().cpu()

            # compensate for sequence length
            sub_len = scores.shape[-1]
            if score_backward:
                l = torch.minimum(sub_len - torch.arange(0, sub_len), torch.tensor(self._config.model.n_prediction_steps))
            else:
                l = torch.minimum(torch.arange(0, sub_len), torch.tensor(self._config.model.n_prediction_steps))
            scores = scores / l
            scores[: , 0] = 0
            scores[: ,-1] = 0

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

        opt = torch.optim.Adam(self._network.parameters(), lr=self._config.learning_rate)

        # init the early stopping criterion
        if hasattr(self._config, 'early_stopping') and self._config.early_stopping.type == "no_improvement":
            early_stopping = NoImprovementStoppingCriterion(no_improvement_for=self._config.early_stopping.no_improvement_for_validations)
        else:
            early_stopping = None

        # Summary writer for Tensorboard
        writer = SummaryWriter(log_dir=self._config.logdir)

        # Training
        epochs = self._config.epochs
        steps_per_epoch = len(train_data)
        print_every_step = self._config.print_every_step
        size_validation_set = len(validation_data) if validation_data else 0
        lambda_weighting = self._config.lambda_weighting if hasattr(self._config, 'lambda_weighting') else 1.0

        global_step = 0
        for epoch in range(epochs):

            cum_loss = 0
            epoch_loss = 0

            for step, X in enumerate(train_data):

                # put the batch on device
                X = X.to(self._device)

                self._network.zero_grad()
                loss_cpc, loss_dcl, acc = self._network(X, self._device, epoch)
                loss = loss_cpc + lambda_weighting * loss_dcl
                loss.backward()
                opt.step()

                cum_loss += loss
                epoch_loss += loss

                if step % print_every_step == 0:
                    print(
                        f"[Epoch {epoch}/{epochs}] Train Step {step}/{steps_per_epoch} \tLoss={cum_loss / print_every_step}"
                    )
                    cum_loss = 0

                    # Log to tensorboard
                    writer.add_scalar("loss", loss, global_step=global_step)
                    writer.add_scalar("loss_cpc", loss_cpc, global_step=global_step)
                    writer.add_scalar("loss_dcl", loss_dcl, global_step=global_step)

                    # Log the accuracies
                    for k in range(acc.size(0)):
                        writer.add_scalar(f"acc/pred_k={k + 1}", acc[k], global_step)

                global_step += 1

            # Log the epoch loss
            writer.add_scalar("loss/epoch", epoch_loss / steps_per_epoch, global_step)

            # save the model after every epoch
            save_checkpoint(epoch, self._network.state_dict(),
                            outdir=self._config.logdir,
                            keep_every_epoch=self._config.keep_checkpoint_every_epoch)

            # Validate the results
            with torch.no_grad():
                if epoch % self._config.validate_every_epoch == 0:

                    if validation_data:
                        # Validate the unsupervised training loss and statistics to prevent overfitting
                        test_loss = 0
                        test_loss_cpc = 0
                        test_loss_dcl = 0
                        test_acc = torch.zeros(self._config.model.n_prediction_steps, device=self._device)
                        for X in tqdm(validation_data, desc="Validation"):
                            X = X.to(self._device)
                            loss_cpc, loss_dcl, acc = self._network(X, self._device)
                            loss = loss_cpc + lambda_weighting * loss_dcl
                            test_loss += loss
                            test_loss_cpc += loss_cpc
                            test_loss_dcl += loss_dcl
                            test_acc += acc

                        writer.add_scalar("loss/test", test_loss / size_validation_set, global_step)
                        writer.add_scalar("loss_cpc/test", test_loss_cpc / size_validation_set, global_step)
                        writer.add_scalar("loss_dcl/test", test_loss_dcl / size_validation_set, global_step)

                        for k in range(test_acc.size(0)):
                            writer.add_scalar(f"acc/test/pred_k={k + 1}", test_acc[k] / size_validation_set, global_step)

                    # Check for Early stopping
                    if early_stopping:
                        early_stopping.inform_value(test_loss / size_validation_set)

                        if early_stopping.is_stop_required():
                            _, outlier_labels, outlier_scores = ord.evaluation.compute_predictions(self,
                                                                                                   ord_evaluation_data,
                                                                                                   epochs=1)
                            roc_auc = ord.evaluation.compute_roc_auc(outlier_labels, outlier_scores)

                            print(f"Final AUC: {roc_auc}")
                            return

                    if ord_evaluation_data:
                        # Evaluate the outlier region detection performance on the fly
                        _, outlier_labels, outlier_scores = ord.evaluation.compute_predictions(self, ord_evaluation_data, epochs=1)
                        roc_auc = ord.evaluation.compute_roc_auc(outlier_labels, outlier_scores)
                        writer.add_scalar("roc_auc", roc_auc, global_step)