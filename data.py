#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/03/18 Thu
# TIME: 21:45:46
# DESCRIPTION:
import torch
import torchtext
from torchtext import vocab
import torchtext.legacy as legacy


TEXT = legacy.data.Field(sequential=True, lower=True)
LABLE = legacy.data.Field()

train_data_snli, val_data_snli, test_data_snli = legacy.datasets.SNLI.splits(text_field=TEXT,
                                                                                label_field=LABLE,
                                                                                root='./data')

def _merge(dataset):
    """
    将dataset每一个example的“前提”和“假设”合并成一个token列表
    """
    return [example.premise + example.hypothesis for example in dataset]


tokens = []
tokens += _merge(train_data_snli) + _merge(val_data_snli) + _merge(test_data_snli)
_vocab = vocab.build_vocab_from_iterator(tokens)

VOCAB = vocab.Vocab(_vocab.freqs, min_freq=2, vectors=vocab.GloVe(name='6B', dim=100))


class SNLIDataset(torch.utils.data.Dataset):
    def __init__(self, data, vocab):
        super(SNLIDataset, self).__init__()
        self.data = data
        self.vocab = vocab

    def __getitem__(self, index):
        premise, hypothesis = self.data[index].premise, self.data[index].hypothesis
        label = self.data[index].label[0]
        # 转成tensor
        label = torch.LongTensor([self._label2idx(label)])
        premise = torch.LongTensor(self._token2idx(premise))
        hypothesis = torch.LongTensor(self._token2idx(hypothesis))
        return premise, hypothesis, label

    def __len__(self):
        return len(self.data)

    def _token2idx(self, tokens):
        return [self.vocab.stoi[token] for token in tokens]

    def _label2idx(self, label):
        if label == 'contradiction':
            return 0
        elif label == 'neutral':
            return 1
        else:
            return 2


def _collate_fn(batch):
    def _pad(seq):
        max_len = max([len(t) for t in seq])
        res = []
        for t in seq:
            pad = [VOCAB.stoi['<pad>']] * (max_len - len(t))
            t = torch.hstack((t, torch.LongTensor(pad)))
            res.append(t)
        res_tensor = torch.stack(res)
        return res_tensor

    premise, hypothesis, label = zip(*batch)
    
    premise_pad = _pad(premise)
    hypothesis_pad = _pad(hypothesis)
    label = torch.hstack(label)

    return premise_pad, hypothesis_pad, label


# snli_dataset = SNLIDataset(train_data_snli, VOCAB)
# snli_dataloader = torch.utils.data.DataLoader(snli_dataset, batch_size=4, shuffle=True, collate_fn=_collate_fn)