#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import os
import pickle
from math import ceil

import yaml
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data_utils
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import fake_parser

import wandb
from srgnn_model import SRGNN_model
from srgnn_datasets import SRGNN_Map_Dataset, SRGNN_sampler, Augment_Matrix_Dataset, GMMClusters_Matrix_Dataset
from utils import calculate_embeddings, split_validation

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
parser.add_argument("--lr_dc", type=float, default=0.1, help="learning rate decay rate")
parser.add_argument(
    "--lr-dc-step",
    type=int,
    default=3,
    help="the number of steps after which the learning rate decay",
)
parser.add_argument(
    "--l2", type=float, default=1e-5, help="l2 penalty"
)  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument("--step", type=int, default=1, help="gnn propogation steps")
parser.add_argument(
    "--patience",
    type=int,
    default=6,
    help="the number of epoch to wait before early stop ",
)
parser.add_argument(
    "--nonhybrid", action="store_true", help="only use the global preference to predict"
)
parser.add_argument("--validation", action="store_true", help="validation")
parser.add_argument(
    "--valid_portion",
    type=float,
    default=0.1,
    help="split the portion of training set as validation set",
)
parser.add_argument(
    "--pretrained_embedings",
    action="store_true",
    help="initialize embeddings using word2vec",
)
parser.add_argument(
    "--unfreeze_epoch",
    type=int,
    default=1,
    help="epoch in which to unfreeze the embeddings layer",
)
parser.add_argument(
    "--gmm", action="store_true", help="train GM on validation dataset after training"
)
parser.add_argument(
    "--weight-init", type=str, default='uniform', help="Raw distances in adjacency matrix"
)
parser.add_argument(
    "--augment-matrix", action="store_true", help="Use version of SRGNN with modified adjacency matrix"
)
parser.add_argument(
    "--augment-clusters", action="store_true", help="Use clusters from GMM to modify adjacency matrix"
)
parser.add_argument(
    "--augment-old-run-id", type=str, help="Full ID of an old run, to use embeddings from"
)
parser.add_argument(
    "--augment-clip", type=float, default=0, help="Max value at which to clip adjacency matrix"
)
parser.add_argument(
    "--augment-normalize", action="store_true", help="Normalize adjacency matrix as in basic approach"
)
parser.add_argument(
    "--augment-raw", action="store_true", help="Raw distances in adjacency matrix"
)
parser.add_argument(
    "--augment-p", type=float, default=1.0, help="Probability of matrix augmentation occuring"
)
parser.add_argument(
    "--augment-mean", type=float, default=0.01, help="Mean of gausian noise to inject into A"
)
parser.add_argument(
    "--augment-std", type=float, default=0.0, help="StandardDeviation of gausian noise to inject into A. Value equal to 0 corresponds to no noise injected"
)

opt = parser.parse_args()
print(opt)


def train_gm(model, dataset, dataloader):
    session_emb = []

    model.to("cuda")
    for batch in tqdm(dataloader, total=dataset.length // opt.batchSize):
        batch = [b.to("cuda") for b in batch]
        session_emb.append(model.get_session_embeddings(batch).cpu().detach().numpy())
    session_emb = np.concatenate(session_emb)

    gm = GaussianMixture(n_components=32, n_init=2, init_params="k-means++")
    _ = gm.fit_predict(session_emb)
    with open(
        f"./GMMs/gmm_val_{gm.n_components}_{gm.init_params}_{opt.hiddenSize}_{opt.dataset}_{opt.augment_matrix}.gmm",
        "wb",
    ) as gmm_file:
        pickle.dump(gm, gmm_file)


def main():
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
    if opt.pretrained_embedings:
        clicks_df = pickle.load(open(f"../datasets/{opt.dataset}/yoo_df.txt", "rb"))
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

    if opt.augment_matrix:

        if opt.augment_clusters:
            with open(f'../datasets/{opt.dataset}/item_labels_16_{opt.hiddenSize}_{opt.augment_old_run_id.split('-')[-1]}.txt', 
                      'rb') as f:
                item_labels=pickle.load(f)
            with open(f'../datasets/{opt.dataset}/cluster_centers_16_{opt.hiddenSize}_{opt.augment_old_run_id.split('-')[-1]}.txt', 
                      'rb') as f:
                cluster_centers=pickle.load(f)

            train_dataset = GMMClusters_Matrix_Dataset(
                                    item_labels, 
                                    cluster_centers,
                                    clip=opt.augment_clip, 
                                    normalize=opt.augment_normalize, 
                                    raw=opt.augment_raw,
                                    p=opt.augment_p,
                                    noise_mean=opt.augment_mean,
                                    noise_std=opt.augment_std,
                                    data=train_data, shuffle=True)
            del train_data

            val_dataset = SRGNN_Map_Dataset(
                                 #   item_labels, 
                                  #  cluster_centers, 
                                   # clip=opt.augment_clip, 
                                    #normalize=opt.augment_normalize, 
                                    #raw=opt.augment_raw,
                                    data=valid_data)
            del valid_data
        else:
            with open(f"./wandb/{opt.augment_old_run_id}/files/config.yaml", "r") as stream:
                    config=yaml.safe_load(stream)

            keys=list(config.keys())
            for k in keys:
                if k not in fake_parser().__dict__.keys():
                    del config[k]
                else:
                    config[k]=config[k]['value']

            old_opt=fake_parser(**config)
            assert old_opt.dataset==opt.dataset, f'Different datasets used in old ({old_opt.dataset}) and current ({opt.dataset}) models!'
            assert old_opt.hiddenSize==opt.hiddenSize, f'Different hidden size used in old ({old_opt.hiddenSize}) and current ({opt.hiddenSize}) models!'

            emb_model=SRGNN_model.load_from_checkpoint(f"./GNN_master/{opt.augment_old_run_id.split('-')[-1]}/checkpoints/"+
                                        os.listdir(f"./GNN_master/{opt.augment_old_run_id.split('-')[-1]}/checkpoints/")[0], opt=old_opt).model.embedding
            train_dataset = Augment_Matrix_Dataset(emb_model, 
                                                clip=opt.augment_clip, 
                                                normalize=opt.augment_normalize, 
                                                raw=opt.augment_raw,
                                                p=opt.augment_p,
                                                noise_mean=opt.augment_mean,
                                                noise_std=opt.augment_std,
                                                data=train_data, shuffle=True)
            del train_data
            val_dataset = SRGNN_Map_Dataset(
                                                #emb_model, 
                                                #clip=opt.augment_clip, 
                                                #normalize=opt.augment_normalize, 
                                                #raw=opt.augment_raw,
                                                data=valid_data)
            del valid_data
    else:
        train_dataset = SRGNN_Map_Dataset(train_data, shuffle=True)
        del train_data
        val_dataset = SRGNN_Map_Dataset(valid_data)
        del valid_data

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=os.cpu_count() - 2,
        sampler=SRGNN_sampler(
            train_dataset, opt.batchSize, shuffle=True, drop_last=False
        ),
    )
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=os.cpu_count() - 2,
        sampler=SRGNN_sampler(
            val_dataset, opt.batchSize, shuffle=False, drop_last=False
        ),
    )

    model = SRGNN_model(opt, n_node, init_embeddings=embeddings, **(opt.__dict__))
    wandb_logger = pl.loggers.WandbLogger(
        project="GNN_master", entity="kpuchalskixiv", log_model=True
    )
    trainer = pl.Trainer(
        max_epochs=60,
        limit_train_batches=train_dataset.length // opt.batchSize,
        limit_val_batches=val_dataset.length // opt.batchSize,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=opt.patience, mode="min", check_finite=True
            ),
            LearningRateMonitor(),
            ModelCheckpoint(monitor="val_loss", mode="min"),
        ],
        logger=wandb_logger,
    )
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
        train_gm(model, val_dataset, val_dataloader)


if __name__ == "__main__":
    main()
