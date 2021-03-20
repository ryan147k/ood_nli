#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/03/19 Fri
# TIME: 18:15:26
# DESCRIPTION:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
from data import train_data_snli, val_data_snli, test_data_snli
from data import SNLIDataset, VOCAB, _collate_fn
from model import WordAvg
from tqdm import tqdm
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch_num', type=int, default=1000)

args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# data
snli_train_dataset = SNLIDataset(train_data_snli, VOCAB)
snli_val_dataset = SNLIDataset(val_data_snli, VOCAB)
snli_test_dataset = SNLIDataset(test_data_snli, VOCAB)

snli_train_dataloader = DataLoader(dataset=snli_train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=_collate_fn)
snli_val_dataloader = DataLoader(dataset=snli_val_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=_collate_fn)
snli_test_dataloader = DataLoader(dataset=snli_test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=_collate_fn)

# model
model = WordAvg(num_embeddings=len(VOCAB),
                embedding_dim=100,
                in_features=128,
                out_features=3)
model.embedding.weight.data.copy_(VOCAB.vectors)
model.embedding.requires_grad_ = False
# 多GPU运行
model = nn.DataParallel(model)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)


# train
def val(dataloader, val=True):
    """
    val: true为验证集
         flase为测试集
    """
    with torch.no_grad():
        loss_sum = 0
        correct = 0
        total = 0
        for batch in dataloader:
            premise, hypothesis, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            # forward
            y_hat = model.forward(premise, hypothesis).to(device)
            loss = loss_fn(y_hat, label)

            loss_sum += loss.item()
            prediction = torch.max(y_hat, dim=-1)[1]
            correct += torch.sum(prediction == label).item()
            total += len(label)
        
        loss = loss / total
        acc = correct / total

        s = 'val  ' if val else 'test '
        print("\t {} loss: {:.4} val_acc: {:.3}".format(s, loss, acc))


for epoch in range(1, args.epoch_num+1):
    loss_sum = 0
    correct = 0
    total = 0
    for batch in snli_train_dataloader:
        premise, hypothesis, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        # forward
        y_hat = model.forward(premise, hypothesis).to(device)
        loss = loss_fn(y_hat, label)

        loss_sum += loss.item()
        prediction = torch.max(y_hat, dim=-1)[1]
        correct += torch.sum(prediction == label).item()
        total += len(label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()
    
    loss = loss_sum / total
    acc = correct / total
    print("\nEpoch {}".format(epoch))
    print("\t correct: {} total: {}".format(correct, total))
    print("\t trian loss: {:.4} acc: {:.3}".format(loss, acc))
    val(snli_val_dataloader)
    val(snli_test_dataloader, val=False)

    if epoch % 20 == 0:
        torch.save(model.module.state_dict(), 'ckpts/{}_{}_{}.pt'.format(model.module.__class__.__name__, epoch, time.time()))
