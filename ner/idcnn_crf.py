# @project : pythonProject
# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 15:55
# @Author  : leejack
# @File    : idcnn_crf.py
# @Description : IDCNN-CRF模型
import torch
from torch import nn
from torch.utils.data import Dataset
from torchcrf import CRF


class IDCNN_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx, filters: int, kernel_size, tagset_size):
        super(IDCNN_CRF, self).__init__()

        # 词嵌入层
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        # IDCNN
        self.conv1_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2, dilation=(1,))
        self.conv1_2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2, dilation=(1,))
        self.conv1_3 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2 + 1, dilation=(2,))

        self.conv2_1 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2, dilation=(1,))
        self.conv2_2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2, dilation=(1,))
        self.conv2_3 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2 + 1, dilation=(2,))

        self.conv3_1 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2, dilation=(1,))
        self.conv3_2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2, dilation=(1,))
        self.conv3_3 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2 + 1, dilation=(2,))

        self.conv4_1 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2, dilation=(1,))
        self.conv4_2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2, dilation=(1,))
        self.conv4_3 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=kernel_size // 2 + 1, dilation=(2,))

        # 归一化层
        self.norm = nn.LayerNorm(filters, elementwise_affine=False)
        # 全连接层
        self.dense = nn.Linear(in_features=filters, out_features=tagset_size)
        # CRF层
        self.crf = CRF(num_tags=tagset_size)

    def forward(self, texts, tags, masks):
        # [8, 60, 120]
        texts = self.embed(texts).permute(0, 2, 1)

        x = torch.relu(self.conv1_1(texts)).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        x = torch.relu(self.conv1_2(x)).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        x = torch.relu(self.conv1_3(x)).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        x = torch.relu(self.conv2_1(x)).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        x = torch.relu(self.conv2_2(x)).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        x = torch.relu(self.conv2_3(x)).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        x = torch.relu(self.conv3_1(x)).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        x = torch.relu(self.conv3_2(x)).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        x = torch.relu(self.conv3_3(x)).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        x = torch.relu(self.conv4_1(x)).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        x = torch.relu(self.conv4_2(x)).permute(0, 2, 1)
        x = self.norm(x).permute(0, 2, 1)
        x = torch.relu(self.conv4_3(x)).permute(0, 2, 1)
        x = self.norm(x)
        out = x.permute(1, 0, 2)
        feats = self.dense(out)

        if tags is not None:
            tags = tags.permute(1, 0)
        if masks is not None:
            masks = masks.permute(1, 0)
        # 计算损失值和概率
        if tags is not None:
            loss = self.neg_log_likelihood(feats, tags, masks, 'mean')
            predictions = self.crf.decode(emissions=feats, mask=masks)  # [batch, 任意数]
            return loss, predictions
        else:
            predictions = self.crf.decode(emissions=feats, mask=masks)
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


# if __name__ == '__main__':
#     model = IDCNN_CRF(vocab_size=200, embedding_dim=200, filters=10, kernel_size=3, tagset_size=10)
#     print(model)

