#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np
from item2vec import Item2Vec
import pandas as pd
from tqdm import tqdm
from math import ceil


class fake_parser():
    def __init__(self,
                 dataset='yoochoose1_64',
                 batchSize=128,
                 hiddenSize=128,
                 epoch=30,
                 lr=1e-3,
                 lr_dc=0.1,
                 lr_dc_step=3,
                 l2=1e-5,
                 step=1,
                 patience=6,
                 nonhybrid=False,
                 validation=True,
                 valid_portion=0.1,
                 pretrained_embedings=False,
                 unfreeze_epoch=1):
        self.dataset=dataset
        self.batchSize=batchSize
        self.hiddenSize=hiddenSize
        self.epoch=epoch
        self.lr=lr
        self.lr_dc=lr_dc
        self.lr_dc_step=lr_dc_step
        self.l2=l2
        self.step=step
        self.patience=patience
        self.nonhybrid=nonhybrid
        self.validation=validation
        self.valid_portion=valid_portion
        self.pretrained_embedings=pretrained_embedings
        self.unfreeze_epoch=unfreeze_epoch


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    print('data masking start')
    us_lens = np.asarray([len(upois) for upois in all_usr_pois])
    print('data masking 1')
    len_max = max(us_lens)
    print('data masking 2')
    us_pois=np.asarray([upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois,
                                                                                us_lens)])
    print('data masking 3')
    us_msks=np.asarray([[1] * le + [0] * (len_max - le) for le in us_lens])
    del us_lens
    print('done masking')
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def normalize(embeddings, new_mean=0, new_std=1):
    mean, std = np.mean(embeddings), np.std(embeddings)
    embeddings += (new_mean - mean)
    embeddings *= (new_std / std)
    return embeddings


def calculate_embeddings(opt, clicks_pdf, items_in_train, item2id, n_node, epochs=3):
    embedding_model = Item2Vec(vector_size=opt.hiddenSize)
    embedding_model.train(clicks_pdf, item_col='item_id', user_col='session_id', epochs=epochs)
    embeddings_pdf = embedding_model.generate_item_embeddings()
    embeddings_pdf = embeddings_pdf.loc[embeddings_pdf.index.isin(items_in_train)]
    embeddings_pdf = pd.DataFrame(normalize(embeddings_pdf.values), index=embeddings_pdf.index)

    embeddings_pdf.index = map(lambda item_id: item2id[str(item_id)], embeddings_pdf.index)

    embeddings = np.random.standard_normal((n_node, opt.hiddenSize))
    embeddings[embeddings_pdf.index] = embeddings_pdf.values

    return embeddings