# @project : pythonProject
# -*- coding: utf-8 -*-
# @Time    : 2022/8/28 18:25
# @Author  : leejack
# @File    : bilstm_crf.py
# @Description :
import torch
from torch import nn
from torch.utils.data import Dataset
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2index, embedding_dim, hidden_size, padding_idx, device):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tagset_size = len(tag2index)
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.padding_index = padding_idx
        self.device = device

        # 词嵌入层
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        # LSTM层
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        # Dense层
        self.dense = nn.Linear(in_features=hidden_size * 2, out_features=self.tagset_size)
        # CRF层
        self.crf = CRF(num_tags=self.tagset_size)

        # 隐藏层
        self.hidden = None

    def forward(self, texts, tags=None, masks=None):
        # texts:[batch, seq_len] tags:[batch, seq_len] masks:[batch, seq_len]
        x = self.embed(texts).permute(1, 0, 2)  # [seq_len, batch, embedding_dim]
        # 初始化隐藏层参数
        self.hidden = (torch.randn(2, x.size(1), self.hidden_size).to(self.device),
                       torch.randn(2, x.size(1), self.hidden_size).to(self.device))  # [num_directions , batch, hidden_size]
        out, self.hidden = self.lstm(x, self.hidden)  # out:[seq_len, batch, num_directions * hidden_size]
        lstm_feats = self.dense(out)  # lstm_feats:[seq_len, batch, tagset_size]

        if tags is not None:
            tags = tags.permute(1, 0)
        if masks is not None:
            masks = masks.permute(1, 0)
        # 计算损失值和概率
        if tags is not None:
            loss = self.neg_log_likelihood(lstm_feats, tags, masks, 'mean')
            predictions = self.crf.decode(emissions=lstm_feats, mask=masks)  # [batch, 任意数]
            return loss, predictions
        else:
            predictions = self.crf.decode(emissions=lstm_feats, mask=masks)
            return predictions

    # 负对数似然损失函数
    def neg_log_likelihood(self, emissions, tags=None, mask=None, reduction=None):
        return -1 * self.crf(emissions=emissions, tags=tags, mask=mask, reduction=reduction)


class NerDataset(Dataset):
    def __init__(self, texts, tags, masks):
        super(NerDataset, self).__init__()
        self.texts = texts
        self.tags = tags
        self.masks = masks

    def __getitem__(self, index):
        return {
            "texts": self.texts[index],
            "tags": self.tags[index] if self.tags is not None else None,
            "masks": self.masks[index]
        }

    def __len__(self):
        return len(self.texts)


class NerDatasetTest(Dataset):
    def __init__(self, texts, masks):
        super(NerDatasetTest, self).__init__()
        self.texts = texts
        self.masks = masks

    def __getitem__(self, index):
        return {
            "texts": self.texts[index],
            "masks": self.masks[index]
        }

    def __len__(self):
        return len(self.texts)
