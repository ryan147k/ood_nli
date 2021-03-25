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
import pandas as pd
import os


DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')


class SNLIVocab(vocab.Vocab):
    """
    SNLI数据集词表
    """
    def __init__(self, vectors, min_freq=2):
        TEXT = legacy.data.Field(sequential=True, lower=True)
        LABLE = legacy.data.Field()
        train_data_snli, val_data_snli, test_data_snli = legacy.datasets.SNLI.splits(text_field=TEXT,
                                                                                        label_field=LABLE,
                                                                                        root=DATA_ROOT)
        tokens = []
        tokens += self._merge(train_data_snli) + self._merge(val_data_snli) + self._merge(test_data_snli)
        _vocab = vocab.build_vocab_from_iterator(tokens)
        
        super(SNLIVocab, self).__init__(_vocab.freqs, min_freq=min_freq, vectors=vectors)
        

    def _merge(self, dataset):
        """
        将dataset每一个example的“前提”和“假设”合并成一个token列表
        """
        return [example.premise + example.hypothesis for example in dataset]


class SNLIDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, train=True, val=False):
        """
        train和val控制 train/val/test
        """
        super(SNLIDataset, self).__init__()
        TEXT = legacy.data.Field(sequential=True, lower=True)
        LABLE = legacy.data.Field()

        if train:
            data, = legacy.datasets.SNLI.splits(text_field=TEXT,
                                                label_field=LABLE,
                                                root=DATA_ROOT,
                                                validation=None,
                                                test=None)
        elif val:
            data, = legacy.datasets.SNLI.splits(text_field=TEXT,
                                                label_field=LABLE,
                                                root=DATA_ROOT,
                                                train=None,
                                                test=None)
        else:
            data, = legacy.datasets.SNLI.splits(text_field=TEXT,
                                                label_field=LABLE,
                                                root=DATA_ROOT,
                                                train=None,
                                                validation=None)

        self.vocab = vocab
        self.data = data

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


class MNLIDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, train=True, val=False):
        """
        train和val控制 train/val/test
        """
        super(MNLIDataset, self).__init__()
        TEXT = legacy.data.Field(sequential=True, lower=True)
        LABLE = legacy.data.Field()

        if train:
            data, = legacy.datasets.MultiNLI.splits(text_field=TEXT,
                                                    label_field=LABLE,
                                                    root=DATA_ROOT,
                                                    validation=None,
                                                    test=None)
        elif val:
            data, = legacy.datasets.MultiNLI.splits(text_field=TEXT,
                                                    label_field=LABLE,
                                                    root=DATA_ROOT,
                                                    train=None,
                                                    test=None)
        else:
            data, = legacy.datasets.MultiNLI.splits(text_field=TEXT,
                                                    label_field=LABLE,
                                                    root=DATA_ROOT,
                                                    train=None,
                                                    validation=None)

        self.vocab = vocab
        self.data = data

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


class XNLIDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, val=True):
        """
        XNLI 只有 val set 和 test set
        val控制 val/test
        """
        # super(XNLIDataset, self).__init__()
        # TEXT = legacy.data.Field(sequential=True, lower=True)
        # LABLE = legacy.data.Field()

        # if val:
        #     data, = legacy.datasets.XNLI.splits(text_field=TEXT,
        #                                             label_field=LABLE,
        #                                             root=DATA_ROOT,
        #                                             test=None)
        # else:
        #     data, = legacy.datasets.XNLI.splits(text_field=TEXT,
        #                                             label_field=LABLE,
        #                                             root=DATA_ROOT,
        #                                             validation=None)
        if val:
            data_path = os.path.join(DATA_ROOT, 'xnli/XNLI-1.0/xnli.dev.tsv')
        else:
            data_path = os.path.join(DATA_ROOT, 'xnli/XNLI-1.0/xnli.test.tsv')

        # pandas读取数据并过滤出英文数据
        df = pd.read_csv(data_path, header=0, sep='\t')
        df = df[df['language'] == 'en']

        self.vocab = vocab
        self.data = df

    def __getitem__(self, index):
        premise, hypothesis = self.data.iloc[index].sentence1.split(), self.data.iloc[index].sentence2.split()
        label = self.data.iloc[index].gold_label
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
        length = [] # 记录每个t padding前的长度
        for t in seq:
            length.append(len(t))
            pad = [1] * (max_len - len(t)) # 1是<pad>的index
            t = torch.hstack((t, torch.LongTensor(pad)))
            res.append(t)
        res_tensor = torch.stack(res)
        length = torch.LongTensor(length)
        return res_tensor, length

    premise, hypothesis, label = zip(*batch)
    
    premise_pad = _pad(premise)
    hypothesis_pad = _pad(hypothesis)
    label = torch.hstack(label)

    return premise_pad, hypothesis_pad, label


if __name__ == "__main__":
    VOCAB = SNLIVocab(vectors=vocab.GloVe(name='6B', dim=300))
    xnli = XNLIDataset(VOCAB, val=True)
