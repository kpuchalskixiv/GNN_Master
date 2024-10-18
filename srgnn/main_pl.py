#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from srgnn_datasets import (
    Augment_Matrix_Dataset,
    Clusters_Matrix_Dataset,
    SRGNN_Map_Dataset,
    SRGNN_sampler,
)
from srgnn_model import SRGNN_model
from utils import calculate_embeddings, fake_parser, split_validation, load_model, get_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="sample",
    help="dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample",
)
parser.add_argument("--batchSize", type=int, default=100, help="input batch size")
parser.add_argument("--hiddenSize", type=int, default=100, help="hidden state size")
parser.add_argument(
    "--epoch", type=int, default=30, help="the number of epochs to train for"
)
parser.add_argument(
    "--lr", type=float, default=0.001, help="learning rate"
)  # [0.001, 0.0005, 0.0001]
parser.add_argument("--lr-dc", type=float, default=0.1, help="learning rate decay rate")
parser.add_argument(
    "--lr-dc-step",
    type=int,
    default=3,
    help="the number of steps after which the learning rate decay",
)
parser.add_argument(
    "--lr-milestones",
    type=int,
    default=[2, 5, 8],
    help="Shedule of steps after which the learning rate decay",
    nargs="+",
)
parser.add_argument(
    "--l2", type=float, default=1e-5, help="l2 penalty"
)  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument("--step", type=int, default=1, help="gnn propogation steps")
parser.add_argument(
    "--patience",
    type=int,
    default=10,
    help="the number of epoch to wait before early stop ",
)
parser.add_argument(
    "--nonhybrid", action="store_true", help="only use the global preference to predict"
)
parser.add_argument("--validation", action="store_true", help="validation")
parser.add_argument(
    "--valid-portion",
    type=float,
    default=0.1,
    help="split the portion of training set as validation set",
)
parser.add_argument(
    "--pretrained-embeddings",
    action="store_true",
    help="initialize embeddings using word2vec",
)
parser.add_argument(
    "--unfreeze-epoch",
    type=int,
    default=0,
    help="epoch in which to unfreeze the embeddings layer",
)
parser.add_argument(
    "--gmm",
    default=[],
    nargs="*",
    type=int,
    help="train GM on validation dataset after training",
)
parser.add_argument(
    "--weight-init",
    type=str,
    default="normal",
    help="Type of torch.nn.init wieght initilization to use",
)
parser.add_argument(
    "--augment-matrix",
    action="store_true",
    help="Use version of SRGNN with modified adjacency matrix",
)
parser.add_argument(
    "--augment-clusters",
    action="store_true",
    help="[Depraceted] - use augment-alg instead! Use clusters from GMM to modify adjacency matrix",
)
parser.add_argument(
    "--augment-old-run-id",
    type=str,
    default="",
    help="Full ID of an old run, to use embeddings from",
)
parser.add_argument(
    "--augment-clip",
    type=float,
    default=0,
    help="Max value at which to clip adjacency matrix",
)
parser.add_argument(
    "--augment-normalize",
    action="store_true",
    help="Normalize adjacency matrix as in basic approach",
)
parser.add_argument(
    "--augment-raw",
    action="store_true",
    help="[Depraceted] Raw distances in adjacency matrix",
)
parser.add_argument(
    "--augment-p",
    type=float,
    default=1.0,
    help="Probability of matrix augmentation occuring",
)
parser.add_argument(
    "--augment-noise-p",
    type=float,
    default=0.0,
    help="Probability of matrix augmentation occuring",
)
parser.add_argument(
    "--augment-mean",
    type=float,
    default=0.01,
    help="Mean of gausian noise to inject into A",
)
parser.add_argument(
    "--augment-std",
    type=float,
    default=0.0,
    help="StandardDeviation of gausian noise to inject into A. Value equal to 0 corresponds to no noise injected",
)
parser.add_argument(
    "--augment-categories",
    action="store_true",
    help="[Depraceted] - use augment-alg instead! Use basic categories to modify adjacency matrix",
)
parser.add_argument(
    "--augment-nogmm",
    type=int,
    default=16,
    help="Number of gausoids used in GMM algorithm",
)
parser.add_argument(
    "--augment-gmm-init",
    type=str,
    default="k-means++",
    help="initialization of gausoids used in GMM algorithm",
)
parser.add_argument(
    "--lr-scheduler",
    type=str,
    default="step",
    help="Learning Rate scheduler to use",
)
parser.add_argument(
    "--augment-prenormalize-distances",
    action="store_true",
    help="Use basic categories to modify adjacency matrix",
)
parser.add_argument(
    "--augment-alg",
    type=str,
    default="gmm",
    choices=["gmm", "kmeans", "categories", "raw"],
    help="Augment with clusters distance based on GMM, KMeans or basic dataset item categories, eventually with raw item2item distance.",
)


def train_gm(model, dataset, dataloader, run_id, opt):
    components=opt.gmm
    session_emb = []

    model.to("cuda")
    for batch in tqdm(dataloader, total=dataset.length // opt.batchSize):
        batch = [b.to("cuda") for b in batch]
        session_emb.append(model.get_session_embeddings(batch).cpu().detach().numpy())
    session_emb = np.concatenate(session_emb)

    for n_comp in components:
        gm = GaussianMixture(n_components=n_comp, n_init=2, init_params="k-means++")
        _ = gm.fit_predict(session_emb)
        with open(
            f"./GMMs/gmm_val_{gm.n_components}_{gm.init_params}_{opt.hiddenSize}_{opt.dataset}_{opt.augment_matrix}_{run_id}.gmm",
            "wb",
        ) as gmm_file:
            pickle.dump(gm, gmm_file)


def main(flags_str=""):

    if flags_str:
        opt = parser.parse_args(flags_str.split())
    else:
        opt = parser.parse_args()
    print(opt)

    assert (
        opt.lr_dc_step < opt.patience
    ), "lr decrease patience is bigger or equal to early stopping patience. Please change either or both"

    torch.set_float32_matmul_precision("medium")
    train_data = pickle.load(open("../datasets/" + opt.dataset + "/train.txt", "rb"))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
    else:
        test_data = pickle.load(open("../datasets/" + opt.dataset + "/test.txt", "rb"))

    if opt.dataset == "amazon_cd":
        n_node = 157661 + 1
    elif opt.dataset == "amzaon_Baby_Products":
        n_node = 89555 + 1
    elif opt.dataset == "amzaon_Musical_Instruments":
        n_node = 68465 + 1
    elif opt.dataset == "diginetica":
        n_node = 43098
    elif opt.dataset == "yoochoose1_64" or opt.dataset == "yoochoose1_4":
        n_node = 37484
    elif opt.dataset == "yoochoose_nonspecial":
        n_node = 37853 + 1
    elif opt.dataset == "yoochoose_custom":
        n_node = 28583
    elif opt.dataset == "yoochoose_custom_augmented":
        n_node = 27809
    elif opt.dataset == "yoochoose_custom_augmented_5050":
        n_node = 27807
    else:
        n_node = 310

    embeddings = None
    if opt.pretrained_embeddings:
        clicks_df = pd.read_csv(f"../datasets/{opt.dataset}/w2v_df.csv")
        items_in_train = pickle.load(
            open(f"../datasets/{opt.dataset}/items_in_train.txt", "rb")
        )
        item2id = pickle.load(open(f"../datasets/{opt.dataset}/item2id.txt", "rb"))

        embeddings = calculate_embeddings(
            opt, clicks_df, items_in_train, item2id, n_node, epochs=10
        )
        print("embeddingas calculated")
        del clicks_df
        del items_in_train
        del item2id

    train_dataset=get_dataset(opt, train_data, shuffle=True)
    del train_data
    val_dataset = SRGNN_Map_Dataset(valid_data)
    del valid_data

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=os.cpu_count() - 2,
        sampler=SRGNN_sampler(
            train_dataset, opt.batchSize, shuffle=True, drop_last=True
        ),
    )
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=os.cpu_count() - 2,
        sampler=SRGNN_sampler(
            val_dataset, opt.batchSize, shuffle=False, drop_last=False
        ),
    )

    model = SRGNN_model(
        opt, n_node, init_embeddings=embeddings, name="SRGNN", **(opt.__dict__)
    )
    wandb_logger = pl.loggers.WandbLogger(
        project="GNN_master", entity="kpuchalskixiv", log_model=True
    )

    trainer = pl.Trainer(
        max_epochs=opt.epoch,
        limit_train_batches=train_dataset.length // opt.batchSize,
        limit_val_batches=val_dataset.length // opt.batchSize,
        callbacks=[
            EarlyStopping(
                monitor="val_hit", patience=opt.patience, mode="max", check_finite=True
            ),
            LearningRateMonitor(),
            ModelCheckpoint(monitor="val_loss", mode="min"),
        ],
        logger=wandb_logger,
    )

    wandb.init(project="GNN_master")
    wandb.define_metric("val_loss", summary="min")
    wandb.define_metric("val_hit", summary="max")
    wandb.define_metric("val_mrr", summary="max")
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    run_id = wandb.run.id

    print("Finished training. Run id: ", run_id)
    wandb.finish()
    if opt.gmm:
        del train_dataset
        del train_dataloader
        model = SRGNN_model.load_from_checkpoint(
            f"./GNN_master/{run_id}/checkpoints/"
            + os.listdir(f"./GNN_master/{run_id}/checkpoints/")[0],
            opt=opt,
        )
        train_gm(model, val_dataset, val_dataloader, run_id, opt)
    return run_id


if __name__ == "__main__":
    main()
