import argparse
import os
import pickle

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

import wandb
from srgnn_datasets import SRGNN_Map_Dataset, SRGNN_sampler
from srgnn_model import SRGNN_model
from utils import calculate_embeddings, fake_parser, split_validation

parser = argparse.ArgumentParser()
parser.add_argument("--finetune", action="store_true", help="finetune")
parser.add_argument(
    "--run-id", type=str, required=True, help="Run Id on which clustering is based."
)
parser.add_argument(
    "--gmm-clusters", type=int, default=32, help="Number of Gaussoids used in GMM."
)
flags = parser.parse_args()


def get_datasets_and_dataloaders(opt, cluster, dir):
    with open(dir + f"train_{cluster}.txt", "rb") as cluster_file:
        train_data = pickle.load(cluster_file)
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
    train_dataset = SRGNN_Map_Dataset(train_data, shuffle=True)
    del train_data
    val_dataset = SRGNN_Map_Dataset(valid_data, shuffle=False)
    del valid_data

    val_batch_size = min(opt.batchSize, val_dataset.length)
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=os.cpu_count(),
        sampler=SRGNN_sampler(
            train_dataset, opt.batchSize, shuffle=True, drop_last=True
        ),
    )
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=os.cpu_count(),
        sampler=SRGNN_sampler(
            val_dataset, val_batch_size, shuffle=False, drop_last=False
        ),
    )
    return train_dataset, val_dataset, train_dataloader, val_dataloader


def main():
    torch.set_float32_matmul_precision("medium")

    ## run_id of the global model to use as reference/finetune
    #  run_id='run-20240302_233004-xh5dmcet'
    run_id = flags.run_id  # "run-20240404_162708-ekuo66ei"
    ## same params as global model
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
    if flags.finetune:
        opt.pretrained_embedings = False
        opt.unfreeze_epoch = 1
        opt.lr /=2
    ## decrease validation ratio and increase patinece, as we have much fewer data per model
    opt.valid_portion = 0.1
    #opt.patience = 8
    #opt.lr_dc_step = 2
    print(opt.__dict__)

    if opt.dataset == "diginetica":
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
    if not flags.finetune and opt.pretrained_embedings:
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

    print("Start modelling clusters!")
    no_clusters = (
        flags.gmm_clusters
    )  # max([f.split('_') for f in os.listdir(f'../datasets/{opt.dataset}/gm_all_splits_{opt.hiddenSize}/')])

    for cluster in range(no_clusters):
        try:
            (
                train_dataset,
                val_dataset,
                train_dataloader,
                val_dataloader,
            ) = get_datasets_and_dataloaders(
                opt,
                cluster,
                f'../datasets/{opt.dataset}/gm_train_{no_clusters}_splits_{opt.hiddenSize}_{run_id.split("-")[-1]}/',
            )
        except FileNotFoundError:
            print("File not found for cluster = ", cluster, " Continue...")
            continue
        print("Train sessions: ", train_dataset.length)
        print("Validation sessions: ", val_dataset.length)

        if flags.finetune:
            model = SRGNN_model.load_from_checkpoint(
                f"./GNN_master/{run_id.split('-')[-1]}/checkpoints/"
                + os.listdir(f"./GNN_master/{run_id.split('-')[-1]}/checkpoints/")[0],
                opt=opt,
            )
            run_name = f"gm_finetune_cluster_{cluster}"
        else:
            # train from scratch
            model = SRGNN_model(
                opt, n_node, init_embeddings=embeddings, **(opt.__dict__)
            )
            run_name = f"gm_all_cluster_{cluster}"

        if opt.unfreeze_epoch > 0:
            model.freeze_embeddings()

        model.hparams.dataset = opt.dataset + "_cluster_" + str(no_clusters)
        model.hparams.name += "_cluster_" + str(no_clusters)

        model.lr *= model.hparams.lr_dc
        model.hparams.lr_dc_step += 1

        model.save_hyperparameters(ignore=["opt", "init_embeddings"])
        wandb_logger = pl.loggers.WandbLogger(
            project="GNN_master", entity="kpuchalskixiv", name=run_name, log_model=True
        )

        trainer = pl.Trainer(
            max_epochs=opt.epoch,
            limit_train_batches=train_dataset.length // opt.batchSize,
            limit_val_batches=max(1, val_dataset.length // opt.batchSize),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=opt.patience,
                    mode="min",
                    check_finite=True,
                ),
                LearningRateMonitor(),
                #  ModelCheckpoint(monitor="val_loss", mode="min"),
            ],
            logger=wandb_logger,
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        wandb.finish()


if __name__ == "__main__":
    main()
