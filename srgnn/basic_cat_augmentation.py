import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import load_model

torch.set_float32_matmul_precision("medium")


def get_items_embedding(model, item_ids: torch.tensor):
    return model.model.embedding(item_ids)


def load_items(opt):
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
    action="store_true",
    help="Use TAGNN model",
)


def main(flags_str=""):

    if flags_str:
        parser_opt = parser.parse_args(flags_str.split())
    else:
        parser_opt = parser.parse_args()

    model, opt = load_model(parser_opt.run_id, parser_opt.tagnn)
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
        f"../datasets/{opt.dataset}/category_labels_{opt.hiddenSize}_{parser_opt.run_id.split('-')[-1]}.txt",
        "wb",
    ) as f:
        pickle.dump(items_categories, f)
    with open(
        f"../datasets/{opt.dataset}/category_embeddings_{opt.hiddenSize}_{parser_opt.run_id.split('-')[-1]}.txt",
        "wb",
    ) as f:
        pickle.dump(cat_centers, f)


if __name__ == "__main__":
    main()
