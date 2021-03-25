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
import torchtext.vocab as vocab
from data import SNLIVocab, _collate_fn
from data import SNLIDataset, MNLIDataset, XNLIDataset
from model import WordAvg, BiLSTM, DiSANN
from tqdm import tqdm
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--epoch_num', type=int, default=1000)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--load_ckpt', type=bool, default=True)

args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # data

    VOCAB = SNLIVocab(vectors=vocab.GloVe(name='6B', dim=100))

    train_dataset = SNLIDataset(VOCAB, train=True, val=False)
    val_dataset = SNLIDataset(VOCAB, train=False, val=True)
    test_dataset = SNLIDataset(VOCAB, train=False, val=False)

    train_dataloader = DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=_collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=_collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=_collate_fn)


    # model

    # model = WordAvg(num_embeddings=len(VOCAB), embedding_dim=100, in_features=256, num_class=3)
    # model = BiLSTM(num_embeddings=len(VOCAB), embedding_dim=100, hidden_size=256, num_class=3)
    model = DiSANN(num_embeddings=len(VOCAB), embedding_dim=300, dropout=args.dropout, hidden_size=300, num_class=3, device=device)
    model.embedding.weight.data.copy_(VOCAB.vectors)
    # Embedding层是否更新
    model.embedding.requires_grad_ = False
    # 多GPU运行
    model = nn.DataParallel(model)
    model = model.to(device)
    print(model.module)

    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=args.lr, 
                                        weight_decay=args.weight_decay)


    # train

    best_val_acc, best_val_epoch = 0, None      # 记录全局最优信息
    best_test_acc, best_test_epoch = 0, None
    save_model = False
    loss_last = 1e10

    for epoch in range(1, args.epoch_num+1):
        model.train()
        loss_sum = 0
        correct = 0
        total = 0
        for batch in train_dataloader:
            (premise, pre_length), (hypothesis, hyp_length), label = batch
            premise = premise.to(device)
            pre_length = pre_length.to(device)
            hypothesis = hypothesis.to(device)
            hyp_length = hyp_length.to(device)
            label = label.to(device)

            # forward
            y_hat = model.forward(premise, pre_length, hypothesis, hyp_length).to(device)
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
        print("\t trian loss: {:.4} acc: {:.4}".format(loss, acc))
        val_acc = val(model, val_dataloader, val=True)
        test_acc = val(model, test_dataloader, val=False)
        
        # 更新全局最优信息
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            save_model = True
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch
            save_model = True
        if save_model:
            # 保存模型
            torch.save(model.module.state_dict(), 'ckpts/{}_{}_{}.pt'.format(model.module.__class__.__name__, epoch, int(time.time())))
            save_model = False

        # loss不下降则停止训练
        if loss_last - loss < 1e-6:
            break
        loss_last = loss    # 更新loss_last

    print('\n')
    print("Best Val Acc: {:.4} Epoch: {}".format(best_val_acc, best_val_epoch))
    print("Best Test Acc: {:.4} Epoch: {}".format(best_test_acc, best_test_epoch))


def val(model, dataloader, val=True):
        """
        val: true为验证集
            flase为测试集
        """
        model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        

        with torch.no_grad():
            loss_sum = 0
            correct = 0
            total = 0
            for batch in dataloader:
                (premise, pre_length), (hypothesis, hyp_length), label = batch
                premise = premise.to(device)
                pre_length = pre_length.to(device)
                hypothesis = hypothesis.to(device)
                hyp_length = hyp_length.to(device)
                label = label.to(device)
                # forward
                if 'DiSAN' in model.module.__class__.__name__:
                    y_hat = model.forward(premise, pre_length, hypothesis, hyp_length).to(device)
                else:
                    y_hat = model.forward(premise, hypothesis).to(device)
                loss = loss_fn(y_hat, label)

                loss_sum += loss.item()
                prediction = torch.max(y_hat, dim=-1)[1]
                correct += torch.sum(prediction == label).item()
                total += len(label)
            
            loss = loss / total
            acc = correct / total

            s = 'val  ' if val else 'test '
            print("\t {} loss: {:.4} acc: {:.4}".format(s, loss, acc))
            return acc


if __name__ == "__main__":
    # data
    VOCAB = SNLIVocab(vectors=vocab.GloVe(name='6B', dim=300))

    # train_dataset = SNLIDataset(VOCAB, train=True, val=False)
    # val_dataset = SNLIDataset(VOCAB, train=False, val=True)
    # test_dataset = SNLIDataset(VOCAB, train=False, val=False)

    test_dataset = MNLIDataset(VOCAB, val=False)

    test_dataloader = DataLoader(dataset=test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=_collate_fn)

    # model
    # model = BiLSTM(len(VOCAB), 100, 256, 3)
    model = DiSANN(len(VOCAB), 300, args.dropout, 300, 3, device)
    if args.load_ckpt:
        model.load_state_dict(torch.load('./ckpts/DiSANN_6_1616508017.035337.pt'))
    model = nn.DataParallel(model)
    model = model.to(device)
    print(model.module)

    # val(model, val_dataloader, val=True)
    val(model, test_dataloader, val=False)
