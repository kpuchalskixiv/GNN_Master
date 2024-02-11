import pickle
from utils import split_validation, fake_parser, calculate_embeddings
from srgnn_pl import SRGNN_model, SRGNN_Map_Dataset, SRGNN_sampler
import torch
import os

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pickle
import yaml

def get_datasets_and_dataloaders(opt, cluster):
    with open(f'../datasets/{opt.dataset}/gm_splits_{opt.hiddenSize}/train_{cluster}.txt', 'rb') as cluster_file:
        train_data = pickle.load(cluster_file)
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
    train_dataset=SRGNN_Map_Dataset(train_data, shuffle=True)
    del train_data
    val_dataset=SRGNN_Map_Dataset(valid_data)
    del valid_data

    train_dataloader=DataLoader(train_dataset, 
                                #batch_size=opt.batchSize, 
                                num_workers=os.cpu_count(),  
                            #  worker_init_fn=worker_init_fn, 
                                sampler=SRGNN_sampler(train_dataset, opt.batchSize, shuffle=True, drop_last=True)
                                #drop_last=
                                )
    #del train_dataset
    val_dataloader=DataLoader(val_dataset, 
                            #batch_size=opt.batchSize, 
                            num_workers=os.cpu_count(), 
                            sampler=SRGNN_sampler(val_dataset, opt.batchSize, shuffle=False, drop_last=False)

                            #  worker_init_fn=worker_init_fn
                            )
    return train_dataset, val_dataset, train_dataloader, val_dataloader

def main():
    torch.set_float32_matmul_precision('medium')
    run_id='run-20240209_162656-h23ej73g'

    ## same params as global model
    with open(f"./wandb/{run_id}/files/config.yaml", "r") as stream:
            config=yaml.safe_load(stream)

    keys=list(config.keys())
    for k in keys:
        if k not in fake_parser().__dict__.keys():
            del config[k]
        else:
            config[k]=config[k]['value']

    opt=fake_parser(**config)
    ## maybe not the same
    ## decrease validation ratio as we have much fewer data per model
    opt.valid_portion=0.1
    print(opt.__dict__)

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'yoochoose_custom':
        n_node = 28583
    elif opt.dataset == 'yoochoose_custom_augmented':
        n_node = 27809
    elif opt.dataset == 'yoochoose_custom_augmented_5050':
        n_node = 27807
    else:
        n_node = 310

    embeddings=None
    if opt.pretrained_embedings:
        clicks_df=pickle.load(open(f'../datasets/{opt.dataset}/yoo_df.txt', 'rb'))
        items_in_train=pickle.load(open(f'../datasets/{opt.dataset}/items_in_train.txt', 'rb'))
        item2id=pickle.load(open(f'../datasets/{opt.dataset}/item2id.txt', 'rb'))

        embeddings = calculate_embeddings(opt, clicks_df, items_in_train, item2id, n_node, epochs=10)
        print('embeddingas calculated')
        del clicks_df
        del items_in_train
        del item2id

    print('Start modelling clusters!')
    no_clusters=sum(['train' in f for f in os.listdir(f'../datasets/{opt.dataset}/gm_splits_{opt.hiddenSize}/')])
    for cluster in range(no_clusters):
        train_dataset, val_dataset, train_dataloader, val_dataloader = get_datasets_and_dataloaders(opt, cluster)

        model=SRGNN_model(opt, n_node, 
                    init_embeddings=embeddings,
                    **(opt.__dict__))

        if opt.unfreeze_epoch>0:
            model.freeze_embeddings()

        wandb_logger = pl.loggers.WandbLogger(project='GNN_master',entity="kpuchalskixiv", name=f'gm_cluster_{cluster}',
                                        log_model="all")
        
        trainer=pl.Trainer(max_epochs=20,
                    limit_train_batches=train_dataset.length//opt.batchSize,
                    limit_val_batches=val_dataset.length//opt.batchSize,
                    callbacks=[
                                EarlyStopping(monitor="val_loss", patience=opt.patience, mode="min", check_finite=True)],
                    logger=wandb_logger
                    )
        trainer.fit(model=model, 
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
            )
        
        wandb.finish()


if __name__=='__main__':
    main()