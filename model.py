#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/03/19 Fri
# TIME: 19:12:23
# DESCRIPTION:
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchtext


class WordAvg(nn.Module):
    """
    sequence每个token的embedding求平均
    送到MLP输出分类结果
    """
    def __init__(self, num_embeddings, embedding_dim, in_features, out_features):
        super(WordAvg, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                        embedding_dim=embedding_dim)
        self.linear = nn.Linear(in_features=embedding_dim,
                                out_features=in_features)
        self.predict = nn.Linear(in_features=in_features,
                                    out_features=out_features)

    def forward(self, premise, hypothesis):
        x = torch.cat((premise, hypothesis), dim=-1)
        embedding = self.embedding(x)
        x = torch.mean(embedding, dim=1)
        x = F.relu(self.linear(x))
        x = self.predict(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        