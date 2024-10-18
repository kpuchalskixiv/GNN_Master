#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import os
import pickle
from typing import List

import numpy as np
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
from tagnn_model import TAGNN_model
from utils import calculate_embeddings, fake_parser, split_validation
from parser import parser 


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
            with open(
                f"../datasets/{opt.dataset}/item_labels_{opt.augment_nogmm}_{opt.augment_gmm_init}_{opt.hiddenSize}_{opt.augment_old_run_id.split('-')[-1]}.txt",
                "rb",
            ) as f:
                item_labels = pickle.load(f)
            with open(
                f"../datasets/{opt.dataset}/cluster_centers_{opt.augment_nogmm}_{opt.augment_gmm_init}_{opt.hiddenSize}_{opt.augment_old_run_id.split('-')[-1]}.txt",
                "rb",
            ) as f:
                cluster_centers = pickle.load(f)

            train_dataset = Clusters_Matrix_Dataset(
                item_labels,
                cluster_centers,
                clip=opt.augment_clip,
                normalize=opt.augment_normalize,
                raw=opt.augment_raw,
                p=opt.augment_p,
                noise_mean=opt.augment_mean,
                noise_std=opt.augment_std,
                data=train_data,
                shuffle=True,
            )
            del train_data

            val_dataset = SRGNN_Map_Dataset(
                #   item_labels,
                #  cluster_centers,
                # clip=opt.augment_clip,
                # normalize=opt.augment_normalize,
                # raw=opt.augment_raw,
                data=valid_data
            )
            del valid_data
        elif opt.augment_categories:
            with open(
                f"../datasets/{opt.dataset}/category_labels_{opt.hiddenSize}_{opt.augment_old_run_id.split('-')[-1]}.txt",
                "rb",
            ) as f:
                item_labels = pickle.load(f)
            with open(
                f"../datasets/{opt.dataset}/category_embeddings_{opt.hiddenSize}_{opt.augment_old_run_id.split('-')[-1]}.txt",
                "rb",
            ) as f:
                cluster_centers = pickle.load(f)

            train_dataset = Clusters_Matrix_Dataset(
                item_labels,
                cluster_centers,
                clip=opt.augment_clip,
                normalize=opt.augment_normalize,
                raw=opt.augment_raw,
                p=opt.augment_p,
                noise_mean=opt.augment_mean,
                noise_std=opt.augment_std,
                data=train_data,
                shuffle=True,
            )
            del train_data
            val_dataset = SRGNN_Map_Dataset(data=valid_data)
            del valid_data

        else:
            with open(
                f"./wandb/{opt.augment_old_run_id}/files/config.yaml", "r"
            ) as stream:
                config = yaml.safe_load(stream)

            keys = list(config.keys())
            for k in keys:
                if k not in fake_parser().__dict__.keys():
                    del config[k]
                else:
                    config[k] = config[k]["value"]

            old_opt = fake_parser(**config)
            assert (
                old_opt.dataset == opt.dataset
            ), f"Different datasets used in old ({old_opt.dataset}) and current ({opt.dataset}) models!"
            assert (
                old_opt.hiddenSize == opt.hiddenSize
            ), f"Different hidden size used in old ({old_opt.hiddenSize}) and current ({opt.hiddenSize}) models!"

            emb_model = TAGNN_model.load_from_checkpoint(
                f"./GNN_master/{opt.augment_old_run_id.split('-')[-1]}/checkpoints/"
                + os.listdir(
                    f"./GNN_master/{opt.augment_old_run_id.split('-')[-1]}/checkpoints/"
                )[0],
                opt=old_opt,
            ).model.embedding
            train_dataset = Augment_Matrix_Dataset(
                emb_model,
                clip=opt.augment_clip,
                normalize=opt.augment_normalize,
                raw=opt.augment_raw,
                p=opt.augment_p,
                noise_mean=opt.augment_mean,
                noise_std=opt.augment_std,
                data=train_data,
                shuffle=True,
            )
            del train_data
            val_dataset = SRGNN_Map_Dataset(
                # emb_model,
                # clip=opt.augment_clip,
                # normalize=opt.augment_normalize,
                # raw=opt.augment_raw,
                data=valid_data
            )
            del valid_data
    else:
        train_dataset = SRGNN_Map_Dataset(
            train_data,
            shuffle=True,
            p=opt.augment_p,
            noise_mean=opt.augment_mean,
            noise_std=opt.augment_std,
        )
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

    model = TAGNN_model(
        opt, n_node, init_embeddings=embeddings, name="TAGNN", **(opt.__dict__)
    )
    wandb_logger = pl.loggers.WandbLogger(
        project="GNN_master", entity="kpuchalskixiv", log_model=True
    )

    trainer = pl.Trainer(
        max_epochs=60,
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
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    run_id = wandb.run.id
    wandb.define_metric("val_loss", summary="min")
    wandb.define_metric("val_hit", summary="max")
    wandb.define_metric("val_mrr", summary="max")
    print("Finished training. Run id: ", run_id)
    wandb.finish()
    if opt.gmm:
        del train_dataset
        del train_dataloader
        model = TAGNN_model.load_from_checkpoint(
            f"./GNN_master/{run_id}/checkpoints/"
            + os.listdir(f"./GNN_master/{run_id}/checkpoints/")[0],
            opt=opt,
        )
        train_gm(model, val_dataset, val_dataloader)


if __name__ == "__main__":
    main()
