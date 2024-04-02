import os
import pickle
from math import ceil

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from srgnn_pl import SRGNN_Map_Dataset, SRGNN_model, SRGNN_sampler
from utils import fake_parser

torch.set_float32_matmul_precision("medium")


def main():
    torch.set_float32_matmul_precision("medium")
    # run_id of a model to test/evaluate on
    run_id = "run-20240215_105643-8xqvam4u"

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

    model = SRGNN_model.load_from_checkpoint(
        f"./GNN_master/{run_id.split('-')[-1]}/checkpoints/"
        + os.listdir(f"./GNN_master/{run_id.split('-')[-1]}/checkpoints/")[0],
        opt=opt,
    )

    test_data = pickle.load(open("../datasets/" + opt.dataset + "/test.txt", "rb"))

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

    test_dataset = SRGNN_Map_Dataset(test_data, shuffle=False)

    test_dataloader = DataLoader(
        test_dataset,
        num_workers=os.cpu_count(),
        sampler=SRGNN_sampler(
            test_dataset, opt.batchSize, shuffle=False, drop_last=False
        ),
        drop_last=False,
    )

    print(opt.__dict__)
    trainer = pl.Trainer(
        limit_test_batches=ceil(test_dataset.length / opt.batchSize),
        limit_predict_batches=ceil(test_dataset.length / opt.batchSize),
    )
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
