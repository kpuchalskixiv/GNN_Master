import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.mixture import GaussianMixture

from srgnn_model import SRGNN_model
from tagnn_model import TAGNN_model
from utils import fake_parser

torch.set_float32_matmul_precision("medium")


def load_data_model(run_id, tagnn=False):
    with open(f"./wandb/{run_id}/files/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    keys = list(config.keys())
    for k in keys:
        if k not in fake_parser().__dict__.keys():
            del config[k]
        else:
            config[k] = config[k]["value"]

    opt = fake_parser(**config)
    print(opt.__dict__)

    if tagnn:
        model = TAGNN_model.load_from_checkpoint(
            f"./GNN_master/{run_id.split('-')[-1]}/checkpoints/"
            + os.listdir(f"./GNN_master/{run_id.split('-')[-1]}/checkpoints/")[0],
            opt=opt,
        )
    else:
        model = SRGNN_model.load_from_checkpoint(
            f"./GNN_master/{run_id.split('-')[-1]}/checkpoints/"
            + os.listdir(f"./GNN_master/{run_id.split('-')[-1]}/checkpoints/")[0],
            opt=opt,
        )

    if "yoochoose" in opt.dataset:
        items_df = pd.read_csv(f"../datasets/{opt.dataset}/items.csv").drop(
            columns="Unnamed: 0"
        )

    elif "diginetica" in opt.dataset:
        items_df = pd.read_csv("../datasets/diginetica/items.csv").drop(
            columns="Unnamed: 0"
        )

    return items_df, model, opt


def get_items_embedding(model, item_ids: torch.tensor):
    return model.model.embedding(item_ids)


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-id",
    help="Run id of model on which to base lableling",
)
parser.add_argument(
    "--tagnn",
    action="store_true",
    help="Run id of model on which to base lableling",
)

parser.add_argument(
    "--no-clusters",
    type=int,
    default=16,
    help="Number of clusters in GMM",
)
parser.add_argument(
    "--n-init",
    type=int,
    default=2,
    help="Number of times to recalculate EM for GMM",
)
parser.add_argument(
    "--init-params",
    type=str,
    default="k-means++",
    help="Algorithm with which to init means of gausoids",
)

parser = parser.parse_args()


def main():
    items_df, model, opt = load_data_model(parser.run_id, parser.tagnn)
    items_embeddings = (
        get_items_embedding(
            model, torch.arange(items_df.item_number.nunique() + 1, device=model.device)
        )
        .cpu()
        .detach()
        .numpy()
    )
    del items_df
    del model

    gm = GaussianMixture(
        n_components=parser.no_clusters,
        n_init=parser.n_init,
        init_params=parser.init_params,
        weights_init=np.ones(parser.no_clusters) / parser.no_clusters,
    )
    item_labels = gm.fit_predict(items_embeddings)
    print(np.unique(item_labels, return_counts=True))
    with open(
        f"../datasets/{opt.dataset}/item_labels_{gm.n_components}_{parser.init_params}_{opt.hiddenSize}_{parser.run_id.split('-')[-1]}.txt",
        "wb",
    ) as f:
        pickle.dump(item_labels, f)
    with open(
        f"../datasets/{opt.dataset}/cluster_centers_{gm.n_components}_{parser.init_params}_{opt.hiddenSize}_{parser.run_id.split('-')[-1]}.txt",
        "wb",
    ) as f:
        pickle.dump(gm.means_, f)


if __name__ == "__main__":
    main()
