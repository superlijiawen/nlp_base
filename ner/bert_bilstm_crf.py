# @project : pythonProject
# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 22:15
# @Author  : leejack
# @File    : bert_bilstm_crf.py
# @Description : 学习Bert-BiLSTM-CRF模型
import torch
from torch import nn
from torchcrf import CRF
from transformers import BertModel
from torch.utils.data import Dataset


class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, tag2index):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tagset_size = len(tag2index)

        # bert层
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # config = self.bert.config
        # lstm层
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        # dropout层
        self.dropout = nn.Dropout(p=0.1)
        # Dense层
        self.dense = nn.Linear(in_features=256, out_features=self.tagset_size)
        # CRF层
        self.crf = CRF(num_tags=self.tagset_size)

        # 隐藏层
        self.hidden = None

    # 负对数似然损失函数
    def neg_log_likelihood(self, emissions, tags=None, mask=None, reduction=None):
        return -1 * self.crf(emissions=emissions, tags=tags, mask=mask, reduction=reduction)

    def forward(self, token_texts, tags):
        """
        token_texts:{"input_size": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                    "token_type_ids": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                     "attention_mask": tensor  [batch, 1, seq_len]->[batch, seq_len]->[seq_len, batch]
                     }
        tags:  [batch, seq_len]->[seq_len, batch]
        bert_out:  [batch, seq_len, hidden_size(768)]->[seq_len, batch, hidden_size]
        self.hidden:  [num_layers * num_directions, hidden_size(128)]
        out:  [seq_len, batch, hidden_size * 2(256)]
        lstm_feats:  [seq_len, batch, tagset_size]
        loss:  tensor
        predictions:  [batch, num]
        """
        texts, token_type_ids, masks = token_texts['input_ids'], token_texts['token_type_ids'], token_texts['attention_mask']
        texts = texts.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
        masks = masks.squeeze(1)
        bert_out = self.bert(input_ids=texts, attention_mask=masks, token_type_ids=token_type_ids)[0]
        bert_out = bert_out.permute(1, 0, 2)
        # 检测设备
        device = bert_out.device
        # 初始化隐藏层参数
        self.hidden = (torch.randn(2, bert_out.size(0), 128).to(device),
                       torch.randn(2, bert_out.size(0), 128).to(device))
        out, self.hidden = self.lstm(bert_out, self.hidden)
        lstm_feats = self.dense(out)

        # 格式转换
        masks = masks.permute(1, 0)
        masks = masks.clone().detach().bool()
        # masks = torch.tensor(masks, dtype=torch.uint8)
        # 计算损失值和预测值
        if tags is not None:
            tags = tags.permute(1, 0)
            loss = self.neg_log_likelihood(lstm_feats, tags, masks, 'mean')
            predictions = self.crf.decode(emissions=lstm_feats, mask=masks)  # [batch, 任意数]
            return loss, predictions
        else:
            predictions = self.crf.decode(emissions=lstm_feats, mask=masks)
            return predictions


class NerDataset(Dataset):
    def __init__(self, token_texts, tags):
        super(NerDataset, self).__init__()
        self.token_texts = token_texts
        self.tags = tags

    def __getitem__(self, index):
        return {
            "token_texts": self.token_texts[index],
            "tags": self.tags[index] if self.tags is not None else None,
        }

    def __len__(self):
        return len(self.token_texts)


class NerDatasetTest(Dataset):
    def __init__(self, token_texts):
        super(NerDatasetTest, self).__init__()
        self.token_texts = token_texts

    def __getitem__(self, index):
        return {
            "token_texts": self.token_texts[index],
            "tags": 0
        }

    def __len__(self):
        return len(self.token_texts)
