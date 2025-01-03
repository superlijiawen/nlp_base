# @project : pythonProject
# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 19:58
# @Author  : leejack
# @File    : bert_crf.py
# @Description :  Bert-CRF
from torch import nn
from torchcrf import CRF
from transformers import BertModel


class Bert_CRF(nn.Module):
    def __init__(self, tag2index):
        super(Bert_CRF, self).__init__()
        self.tagset_size = len(tag2index)

        # bert层
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # dense层
        self.dense = nn.Linear(in_features=768, out_features=self.tagset_size)
        # CRF层
        self.crf = CRF(num_tags=self.tagset_size)

        # 隐藏层
        self.hidden = None

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
        feats:  [seq_len, batch, tagset_size]
        loss:  tensor
        predictions:  [batch, num]
        """
        texts, token_type_ids, masks = token_texts.values()
        texts = texts.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
        masks = masks.squeeze(1)
        bert_out = self.bert(input_ids=texts, attention_mask=masks, token_type_ids=token_type_ids)[0]
        bert_out = bert_out.permute(1, 0, 2)
        feats = self.dense(bert_out)

        # 格式转换
        masks = masks.permute(1, 0)
        masks = masks.clone().detach().bool()
        # 计算损失之和预测值
        if tags is not None:
            tags = tags.permute(1, 0)
            loss = self.neg_log_likelihood(feats, tags, masks, 'mean')
            predictions = self.crf.decode(emissions=feats, mask=masks)
            return loss, predictions
        else:
            predictions = self.crf.decode(emissions=feats, mask=masks)
            return predictions
