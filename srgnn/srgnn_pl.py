#!/usr/bin/env python38
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
from typing import Any, Optional
import numpy as np

import networkx as nx
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.utils.data as data_utils
import pytorch_lightning as pl
from tqdm import tqdm
from itertools import batched
from math import ceil

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super().__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
      #  self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
      #  self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        
        
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores
    
    def session_embedding(self,  hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        return a

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


class SRGNN_model(pl.LightningModule):
    def __init__(self, opt=None, n_node=0, init_embeddings=None, **kwargs):
        super().__init__()
        self.lr=opt.lr
        self.save_hyperparameters(ignore=['opt', 'init_embeddings'])
        self.model=SessionGraph(opt, n_node)
        if init_embeddings is not None:
            self.model.embedding=nn.Embedding.from_pretrained(torch.FloatTensor(init_embeddings))
        self.unfreeze_epoch=opt.unfreeze_epoch
    
    def forward(self, x):
        for i in range(len(x)):
            x[i]=x[i].squeeze(dim=0)
            

        alias_inputs, A, items, mask = x
        items=items.to(torch.int32)
        A=A.to(torch.float32)
        hidden = self.model(items, A)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        return self.model.compute_scores(seq_hidden, mask)

    def training_step(self, batch, batch_idx):
        x=batch[:-1]
        targets = batch[-1].squeeze()
        scores=self.forward(x)

        loss = self.model.loss_function(scores, targets - 1)
        self.log("train_loss", loss, prog_bar=True)

        return loss
    
    def evaluate(self, batch, stage=None):
        x=batch[:-1]
        targets = batch[-1].squeeze()
        scores=self.forward(x)
        loss = self.model.loss_function(scores, targets - 1)

        # get metrics @20        
        # hit is recall/precision, that is
        # proportion of cases having the desired item amongst the top-20 items
        sub_scores = scores.topk(20)[1]
        hit,mrr=[],[]
        for score, target in zip(sub_scores, targets):
            correct_pred=torch.isin(target - 1, score)
            hit.append(correct_pred)
            if not correct_pred:
                mrr.append(0)
            else:
                mrr.append(1 / (torch.where(score == target - 1)[0][0] + 1))
        hit = 100*sum(hit)/targets.shape[0]
        mrr = 100*sum(mrr)/targets.shape[0]
        if stage:
            self.log(stage+"_loss", loss, prog_bar=True)
            self.log(stage+"_hit", hit, prog_bar=True)
            self.log(stage+"_mrr", mrr, prog_bar=True)

    def validation_step(self, batch, *args, **kwargs):
        return self.evaluate(batch, 'val')
    
    def test_step(self, batch, *args, **kwargs):
        return self.evaluate(batch, 'test')
    
    def predict_step(self, batch, *args, **kwargs):
        x=batch[:-1]
        targets = batch[-1]
        scores=self.forward(x)
        sub_scores = scores.topk(20)[1]
        return sub_scores, targets
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.hparams.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                    optimizer, 
                                                    patience=self.hparams.lr_dc_step, 
                                                    factor=self.hparams.lr_dc,
                                                    cooldown=1)
        return {'optimizer': optimizer, 
                'lr_scheduler':{'scheduler': scheduler,
                                'monitor': 'val_loss',
                                "interval": "epoch",
                                "frequency": 1,
                                'name': 'scheduler_lr'
                                }
                }
    
    def get_raw_embeddings(self, batch):
        items = batch[2]
        embs=self.model.embedding(items)
        return embs
    
    def get_session_embeddings(self, batch):
        x=batch[:-1]
        for i in range(len(x)):
            x[i]=x[i].squeeze()
        alias_inputs, A, items, mask = x
        items=items.to(torch.int32)
        A=A.to(torch.float32)
        hidden = self.model(items, A)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        return self.model.session_embedding(seq_hidden, mask)

    def get_gnn_embeddings(self, batch):
        _, A, items, _ = x
        A=A.to(torch.float32)
        hidden = self.model(items, A)
        return hidden
    
    def unfreeze_embeddings(self):
        self.model.embedding.requires_grad_(True)

    def freeze_embeddings(self):
        self.model.embedding.requires_grad_(False)

    def on_train_epoch_end(self):
        if self.current_epoch==self.unfreeze_epoch:
            self.unfreeze_embeddings()
    

def data_masks(all_usr_pois, item_tail):
    print('data masking start')
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)

    no_batches=64
    batch_size=ceil(len(us_lens)/no_batches)

    all_usr_pois=batched(all_usr_pois, batch_size)
    us_lens=batched(us_lens, batch_size)
    us_msks=[]
    us_pois=[]

    for all_usr_pois_batch, us_lens_batch in tqdm(zip(all_usr_pois, us_lens), total=no_batches):
        us_pois.append(np.asarray([upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois_batch,us_lens_batch)], dtype=np.uint16))
        us_msks.append(np.asarray([[1] * le + [0] * (len_max - le) for le in us_lens_batch], dtype=np.bool_))

    del all_usr_pois
    del us_lens

    us_pois=np.concatenate(us_pois)
    us_msks=np.concatenate(us_msks)
    print('done masking')
    return us_pois, us_msks, len_max


class SRGNN_Dataset(data_utils.IterableDataset):
    def __init__(self, data, shuffle=False, graph=None):
        super().__init__()
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

        self.start=0
        self.end=self.length

    def reinit(self):
        if self.end-self.start!=self.length:
            self.length=self.end-self.start
            self.inputs=self.inputs[self.start:self.end]
            self.mask=self.mask[self.start:self.end]
            self.targets=self.targets[self.start:self.end]
        else:
            self.end=self.length
            self.start=0
        
    def __iter__(self):
#        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        #items, , A, alias_inputs = [], [], [], []
        assert self.start<=self.end
        assert self.end-self.start==self.length
        order=np.arange(self.length)
        if self.shuffle:
            order=np.random.permutation(order)
        n_node=[]
        for u_input in self.inputs[order]:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        #print(data_utils.get_worker_info().id, self.start, self.end)
        for i, u_input in enumerate(self.inputs[order]):
           # print(i)
            node = np.unique(u_input)
            items=(node.tolist())
            u_A = np.zeros((len(node), len(node)))
            for j in np.arange(len(u_input) - 1):
                if u_input[j + 1] == 0:
                    break
                u = np.where(node == u_input[j])[0][0]
                v = np.where(node == u_input[j + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A=np.pad(u_A, ((0, max_n_node-node.shape[0]), 
                           (0, 2*(max_n_node-node.shape[0]))))
            alias_inputs=np.asarray([np.where(node == i)[0][0] for i in u_input])
            
            items=np.pad(items, (0, max_n_node-node.shape[0]))
            
            yield alias_inputs, A, items, self.mask[i], self.targets[i]


class SRGNN_Map_Dataset(data_utils.Dataset):
    def __init__(self, data, shuffle=False, graph=None):
        super().__init__()
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
       # self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

        self.start=0
        self.end=self.length

    def reinit(self):
        if self.end-self.start!=self.length:
            self.length=self.end-self.start
            self.inputs=self.inputs[self.start:self.end]
            self.mask=self.mask[self.start:self.end]
            self.targets=self.targets[self.start:self.end]
        else:
            self.end=self.length
            self.start=0
        
    def __len__(self):
        return self.length

    def __getitem__(self, idxs):
       # print(idxs)
        if isinstance(idxs, int):
            idxs=[idxs]
       # print(idxs)
        inputs, mask, targets = self.inputs[idxs], self.mask[idxs], self.targets[idxs]
        non_zero_cols=(mask!=0).sum(axis=0)!=0
        inputs=inputs[:,non_zero_cols]
        mask=mask[:,non_zero_cols]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node) # length of the longest session in batch

        for u_input in inputs:
            node = np.unique(u_input)
            items.append(np.concatenate([node, np.zeros(max_n_node - len(node))]))
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return np.asarray(alias_inputs), np.asarray(A), np.asarray(items), np.asarray(mask), targets

def worker_init_fn(worker_id):
    worker_info = data_utils.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = 0
    overall_end = dataset.length
    # configure the dataset to only process the split workload
    per_worker = int(ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
    dataset.reinit()

class SRGNN_sampler(data_utils.Sampler):
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.drop_last=drop_last

    def __len__(self):
        return self.dataset.length
    
    def __iter__(self):
        order=np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(order)
        if len(self)%self.batch_size:
            for i in range(0, len(self)-self.batch_size, self.batch_size):
                yield order[i:i+self.batch_size]
            if not self.drop_last:
                yield order[-(len(self)%self.batch_size):]
        else:
            for i in range(0, len(self), self.batch_size):
                yield order[i:i+self.batch_size]
       # raise IndexError('Done iterating')