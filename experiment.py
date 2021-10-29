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

import os
import numpy as np
import torch
import argparse
import shutil
from utils import load_configuration
from lnt import LNTPredictor
from ord.datasets.librispeech import LIBRISpeechOutlierHDF5Dataset
from ord.datasets.wadi import WaDiDataset
from ord.evaluation import compute_predictions
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


def get_algorithm_by_name(name):

    algs = {
        'lnt': LNTPredictor,
        'ocsvm': None,
        'lstm': None,
        'lof': None,
        'thoc': None,
    }
    return algs[name]


def configure_predictor_from_file(config):

    alg = get_algorithm_by_name(config.type)
    if config.config:
        pred_config = load_configuration(config.config)
        pred = alg(pred_config)
    else:
        pred = alg(**config.kwargs.__dict__)
    pred.load(file=config.checkpoint)

    return pred


def configure_test_data_from_file(config):

    if config.name == "libri":
        datadir = "./data/libri"

        def get_selection(sel):
            if isinstance(sel, str):
                sel = list(range(*map(int, sel.split('-'))))
            elif isinstance(sel, int):
                freq = [sel]
            return sel

        freq = get_selection(config.freq)
        width = get_selection(config.width)

        test_data = LIBRISpeechOutlierHDF5Dataset(root=datadir,
                                                  split_file=os.path.join(datadir, 'test_split.txt'),
                                                  output_uncorrupted=False,
                                                  frequency=freq,
                                                  width=width)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=config.batch_size,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  num_workers=config.num_dataloader_workers)
        return test_loader

    if config.name == "wadi":
        datadir = "./data/wadi"

        test_data = WaDiDataset(root=datadir, window_size=config.window_size, stride=config.window_stride, attacks=True, improved_preprocessing=config.improved_preprocessing)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=config.batch_size,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  num_workers=config.num_dataloader_workers)
        return test_loader


def run_experiment(path, config):

    # load the data
    data = configure_test_data_from_file(config.data)

    for name, pred_config in iter_predictors(config):

        if os.path.exists(get_filename(path, pred_config.checkpoint)):
            continue

        print(f"Evaluating {name}")
        pred = configure_predictor_from_file(pred_config)

        # compute the predictions
        outlier_predictions, outlier_labels, outlier_scores = run_predictor(data, pred, numBatches=config.data.num_batches if hasattr(config.data, 'num_batches') else None)

        # save the predictions
        save(path, pred_config.checkpoint, preds=outlier_predictions, labels=outlier_labels, scores=outlier_scores)


def save(path, config, **kwargs):
    file = get_filename(path, config)
    np.savez_compressed(file, **kwargs)


def get_filename(path, config):
    basename = os.path.basename(os.path.dirname(config))
    file = os.path.join(path, basename + ".npz")
    return file


def run_predictor(data, predictor, numBatches):
    return compute_predictions(predictor=predictor, data=data, epochs=1, numBatches=numBatches)


def iter_predictors(config):

    for pred in config.predictors.__dict__:
        yield pred, getattr(config.predictors, pred)


def setup_experiment_folder(exp_name, config=None):

    basepath = "./experiments/evaluation"
    dest = os.path.join(basepath, exp_name)

    # create the directory
    if not os.path.exists(dest):
        os.makedirs(dest)

    if config:
        # copy the config file
        config_dest = os.path.join(dest, "config.yaml")
        shutil.copy(config, config_dest)

    return dest


def load_runs(path, config):

    for name, pred in iter_predictors(config):

        filename = get_filename(path, pred.checkpoint)
        if not os.path.exists(filename):
            continue

        if hasattr(pred, 'show') and not pred.show:
            continue

        if hasattr(pred, 'pretty_name'):
            yield pred.pretty_name, np.load(filename)
        else:
            yield name, np.load(filename)


def plot_roc(path, config, dest, use_latex=False, figsize=(4.5, 3)):

    if not use_latex:
        from matplotlib import pyplot as plt
    else:
        import matplotlib
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False
        })
        from matplotlib import pyplot as plt

    plt.figure(figsize=figsize)
    plt.grid(c='gray', linestyle='--')

    length = len(list(load_runs(path, config)))
    for (name, data), color in tqdm(zip(load_runs(path, config), plt.cm.tab10(range(length)))):

        fpr, tpr, _ = roc_curve(y_true=data['labels'], y_score=data['scores'])
        a = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={a :.2f})", color=color)

    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), '--', color='black', label='chance (AUC=0.50)')
    plt.legend(loc='lower right')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    # save the figure
    plt.savefig(os.path.join(path, dest), bbox_inches='tight', dpi=600)

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str)
    parser.add_argument('-n', '--name', required=True, type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--dest', type=str)

    args = parser.parse_args()

    if not args.plot:
        assert args.config
        assert os.path.exists(args.config)
    else:
        assert args.dest

    if not args.plot:
        # Run the experiment
        path = setup_experiment_folder(args.name, args.config)
        config = load_configuration(args.config)

        run_experiment(path, config)
    else:
        # Plot the results
        path = setup_experiment_folder(args.name)
        config = load_configuration(os.path.join(path, 'config.yaml'))
        plot_roc(path, config, args.dest, use_latex=args.dest.endswith(".pgf"), figsize=(6.5, 3.5))
