import argparse
import os
import pickle
from math import ceil
from time import sleep

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from srgnn_datasets import SRGNN_Map_Dataset, SRGNN_sampler
from srgnn_model import SRGNN_model
from utils import fake_parser

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-id", type=str, required=True, help="Run Id on which clustering is based."
)
parser.add_argument(
    "--gmm-clusters", type=int, default=32, help="Number of Gaussoids used in GMM."
)
flags = parser.parse_args()


def main():
    run_id = flags.run_id
    if len(run_id.split("-")) == 1:
        full_run_id = [x for x in os.listdir("./wandb") if run_id in x][0]
        with open(f"./wandb/{full_run_id}/files/config.yaml", "r") as stream:
            config = yaml.safe_load(stream)
    else:
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

    gm_read = False
    while not gm_read:
        try:
            with open(
                f"./GMMs/gmm_val_{flags.gmm_clusters}_k-means++_{opt.hiddenSize}_{opt.dataset}_{opt.augment_matrix}_{run_id.split('-')[-1]}.gmm",
                "rb",
            ) as gmm_file:
                gm = pickle.load(gmm_file)
                gm_read = True
        except FileNotFoundError:
            print("waiting for training to finish...")
            sleep(300)

    torch.set_float32_matmul_precision("medium")
    model = SRGNN_model.load_from_checkpoint(
        f"./GNN_master/{run_id.split('-')[-1]}/checkpoints/"
        + os.listdir(f"./GNN_master/{run_id.split('-')[-1]}/checkpoints/")[0],
        opt=opt,
    )
    train_data = pickle.load(open("../datasets/" + opt.dataset + "/train.txt", "rb"))

    if opt.dataset == "diginetica":
        n_node = 43098
    elif opt.dataset == "yoochoose1_64" or opt.dataset == "yoochoose1_4":
        n_node = 37484

    elif opt.dataset == "yoochoose_custom":
        n_node = 28583
    elif opt.dataset == "yoochoose_custom_augmented":
        n_node = 27809
    elif opt.dataset == "yoochoose_custom_augmented_5050":
        n_node = 27807
    else:
        n_node = 310
    # train_data, valid_data = split_validation(train_data, opt.valid_portion)
    train_dataset = SRGNN_Map_Dataset(train_data, shuffle=False)
    del train_data

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=os.cpu_count(),
        sampler=SRGNN_sampler(
            train_dataset, opt.batchSize, shuffle=False, drop_last=False
        ),
    )
    session_emb = []
    full_sessions = []

    model.to("cuda")
    for batch in tqdm(train_dataloader, total=train_dataset.length // opt.batchSize):
        batch = [b.to("cuda") for b in batch]
        session_emb.append(model.get_session_embeddings(batch).cpu().detach().numpy())
    session_emb = np.concatenate(session_emb)

    session_labels = []
    for i in tqdm(range(ceil(session_emb.shape[0] / opt.batchSize))):
        session_labels.append(
            gm.predict(session_emb[i * opt.batchSize : (i + 1) * opt.batchSize])
        )
    session_labels = np.concatenate(session_labels)

    print(np.unique(session_labels, return_counts=True))

    del train_dataloader
    del train_dataset

    train_data = pickle.load(open("../datasets/" + opt.dataset + "/train.txt", "rb"))
    os.makedirs(
        f"../datasets/{opt.dataset}/gm_train_{flags.gmm_clusters}_splits_{opt.hiddenSize}_{run_id.split('-')[-1]}"
    )
    for cluster in tqdm(np.unique(session_labels)):
        idxs = np.arange(session_labels.shape[0])[session_labels == cluster]
        cluster_sessions = []
        cluster_targets = []
        for i in idxs:
            cluster_sessions.append(train_data[0][i])
            cluster_targets.append(train_data[1][i])
        with open(
            f"../datasets/{opt.dataset}/gm_train_{flags.gmm_clusters}_splits_{opt.hiddenSize}_{run_id.split('-')[-1]}/train_{cluster}.txt",
            "wb",
        ) as cluster_file:
            pickle.dump((cluster_sessions, cluster_targets), cluster_file)


if __name__ == "__main__":
    main()
