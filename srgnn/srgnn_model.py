#!/usr/bin/env python38
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import Module, Parameter


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
        input_in = (
            torch.matmul(A[:, :, : A.shape[1]], self.linear_edge_in(hidden))
            + self.b_iah
        )
        input_out = (
            torch.matmul(
                A[:, :, A.shape[1] : 2 * A.shape[1]], self.linear_edge_out(hidden)
            )
            + self.b_oah
        )
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
        self.linear_transform = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=True
        )
        self.loss_function = nn.CrossEntropyLoss()
        #  self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        #  self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters(opt.weight_init)

    def reset_parameters(self, weight_init):
        if weight_init == "uniform":
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                nn.init.uniform_(weight, -stdv, stdv)
        elif weight_init == "normal":
            for weight in self.parameters():
                nn.init.normal_(weight, 0, 0.1)
        elif weight_init == "xavier_normal":
            for weight in self.parameters():
                if len(weight.shape) < 2:
                    nn.init.normal_(weight, 0, 0.1)
                else:
                    nn.init.xavier_normal_(weight)
        else:
            raise ValueError(
                f"Weight initialization of type {weight_init} not implemented!"
            )

    def compute_scores(self, hidden, mask):
        ht = hidden[
            torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1
        ]  # batch_size x latent_size

        q1 = self.linear_one(ht).view(
            ht.shape[0], 1, ht.shape[1]
        )  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def session_embedding(self, hidden, mask):
        ht = hidden[
            torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1
        ]  # batch_size x latent_size

        q1 = self.linear_one(ht).view(
            ht.shape[0], 1, ht.shape[1]
        )  # batch_size x 1 x latent_size
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
        self.lr = opt.lr
        self.save_hyperparameters(ignore=["opt", "init_embeddings"])
        self.save_hyperparameters({'name': 'SRGNN'})
        self.model = SessionGraph(opt, n_node)
        if init_embeddings is not None:
            self.model.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(init_embeddings)
            )
        self.unfreeze_epoch = opt.unfreeze_epoch

    def forward(self, x):
        for i in range(len(x)):
            x[i] = x[i].squeeze(dim=0)

        alias_inputs, A, items, mask = x
        items = items.to(torch.int32)
        A = A.to(torch.float32)
        hidden = self.model(items, A)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack(
            [get(i) for i in torch.arange(len(alias_inputs)).long()]
        )
        return self.model.compute_scores(seq_hidden, mask)

    def training_step(self, batch, batch_idx):
        x = batch[:-1]
        targets = batch[-1].squeeze()
        scores = self.forward(x)

        loss = self.model.loss_function(scores, targets - 1)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def evaluate(self, batch, stage=None):
        x = batch[:-1]
        targets = batch[-1].squeeze()
        scores = self.forward(x)
        loss = self.model.loss_function(scores, targets - 1)

        # get metrics @20
        # hit is recall/precision, that is
        # proportion of cases having the desired item amongst the top-20 items
        sub_scores = scores.topk(20)[1]
        hit, mrr = [], []
        for score, target in zip(sub_scores, targets):
            correct_pred = torch.isin(target - 1, score)
            hit.append(correct_pred)
            if not correct_pred:
                mrr.append(0)
            else:
                mrr.append(1 / (torch.where(score == target - 1)[0][0] + 1))
        hit = 100 * sum(hit) / targets.shape[0]
        mrr = 100 * sum(mrr) / targets.shape[0]
        if stage:
            self.log(stage + "_loss", loss, prog_bar=True)
            self.log(stage + "_hit", hit, prog_bar=True)
            self.log(stage + "_mrr", mrr, prog_bar=True)

    def validation_step(self, batch, *args, **kwargs):
        return self.evaluate(batch, "val")

    def test_step(self, batch, *args, **kwargs):
        return self.evaluate(batch, "test")

    def predict_step(self, batch, *args, **kwargs):
        x = batch[:-1]
        targets = batch[-1]
        scores = self.forward(x)
        sub_scores = scores.topk(20)[1]
        return sub_scores, targets

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.hparams.l2
        )
        if self.hparams.lr_scheduler == "plateu":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.hparams.lr_dc_step,
                factor=self.hparams.lr_dc,
                cooldown=1,
            )
        elif self.hparams.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.lr_dc_step,
                gamma=self.hparams.lr_dc,
            )
        elif self.hparams.lr_scheduler == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.hparams.lr_milestones,
                gamma=self.hparams.lr_dc,
            )
        else:
            raise ValueError('Unknown (or not implemented) learning rate sheduler')
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "name": "scheduler_lr",
            },
        }

    def get_raw_embeddings(self, batch):
        items = batch[2]
        embs = self.model.embedding(items)
        return embs

    def get_session_embeddings(self, batch):
        x = batch[:-1]
        for i in range(len(x)):
            x[i] = x[i].squeeze()
        alias_inputs, A, items, mask = x
        items = items.to(torch.int32)
        A = A.to(torch.float32)
        hidden = self.model(items, A)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack(
            [get(i) for i in torch.arange(len(alias_inputs)).long()]
        )
        return self.model.session_embedding(seq_hidden, mask)

    def get_gnn_embeddings(self, batch):
        x = batch[:-1]
        _, A, items, _ = x
        A = A.to(torch.float32)
        hidden = self.model(items, A)
        return hidden

    def unfreeze_embeddings(self):
        self.model.embedding.requires_grad_(True)

    def freeze_embeddings(self):
        self.model.embedding.requires_grad_(False)

    def on_train_epoch_end(self):
        if self.current_epoch == self.unfreeze_epoch:
            self.unfreeze_embeddings()
