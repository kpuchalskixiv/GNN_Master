import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from srgnn_model import SRGNN_model
from tagnn_model import TAGNN_model
from utils import fake_parser

torch.set_float32_matmul_precision("medium")


def load_model(run_id, tagnn=False):
    with open(f"./wandb/{run_id}/files/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    keys = list(config.keys())
    for k in keys:
        if k == "old_run_id":
            config["augment_old_run_id"] = config[k]["value"]
            del config[k]
        elif k not in fake_parser().__dict__.keys():
            del config[k]
        else:
            config[k] = config[k]["value"]

    opt = fake_parser(**config)

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
    return model, opt


def get_items_embedding(model, item_ids: torch.tensor):
    return model.model.embedding(item_ids)


def load_items(opt):
    if opt.dataset == "diginetica":
        items_df = pd.read_csv("../datasets/diginetica/items.csv").drop(
            columns=["Unnamed: 0"]
        )

    elif "yoochoose" in opt.dataset:
        items_df = pd.read_csv(f"../datasets/{opt.dataset}/items.csv").drop(
            columns=["Unnamed: 0"]
        )
    return items_df


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-id",
    help="Run id of model on which to base lableling",
)
parser.add_argument(
    "--tagnn",
    action='store_true',
    help="Use TAGNN model",
)
parser = parser.parse_args()


def main():
    model, opt = load_model(parser.run_id, parser.tagnn)
    items_df = load_items(opt)

    model.to("cuda")
    base_items_embeddings = (
        get_items_embedding(
            model, torch.arange(items_df.item_number.nunique() + 1, device=model.device)
        )
        .cpu()
        .detach()
        .numpy()
    )
    items_categories = np.array(
        [0]
        + items_df.sort_values(by="item_number")
        .reset_index(drop=True)
        .category.to_list()
    )

    cat_centers = {}
    for cat in tqdm(items_df.category.unique()):
        cat_items = items_df.loc[items_df.category == cat].item_number.values
        cat_centers[cat] = np.average(base_items_embeddings[cat_items], axis=0)
    with open(
        f"../datasets/{opt.dataset}/category_labels_{opt.hiddenSize}_{parser.run_id.split('-')[-1]}.txt",
        "wb",
    ) as f:
        pickle.dump(items_categories, f)
    with open(
        f"../datasets/{opt.dataset}/category_embeddings_{opt.hiddenSize}_{parser.run_id.split('-')[-1]}.txt",
        "wb",
    ) as f:
        pickle.dump(cat_centers, f)


if __name__=='__main__':
    main()