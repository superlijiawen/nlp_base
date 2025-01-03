# @project : pythonProject
# -*- coding: utf-8 -*-
# @Time    : 2022/8/28 17:04
# @Author  : leejack
# @File    : study_bilstm_crf.py
# @Description : 学习BiLSTM-CRF
import pandas as pd
import torch
import pickle
import os
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from bilstm_crf import BiLSTM_CRF, NerDataset, NerDatasetTest
from sklearn.metrics import f1_score

# 路径
TRAIN_PATH = './dataset/train_data_public.csv'
TEST_PATH = './dataset/test_public.csv'
VOCAB_PATH = './data/vocab.txt'
WORD2INDEX_PATH = './data/word2index.pkl'
MODEL_PATH = './model/bilstm_crf.pkl'

# 超参数
MAX_LEN = 60
BATCH_SIZE = 8
EMBEDDING_DIM = 120
HIDDEN_DIM = 12
EPOCH = 5

# 预设
# 设备
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# tag2index
tag2index = {
    "O": 0,  # 其他
    "B-BANK": 1, "I-BANK": 2,  # 银行实体
    "B-PRODUCT": 3, "I-PRODUCT": 4,  # 产品实体
    "B-COMMENTS_N": 5, "I-COMMENTS_N": 6,  # 用户评论，名词
    "B-COMMENTS_ADJ": 7, "I-COMMENTS_ADJ": 8  # 用户评论，形容词
}
# 预设标签
unk_flag = '[UNK]'  # 未知
pad_flag = '[PAD]'  # 填充
start_flag = '[STA]'  # 开始
end_flag = '[END]'  # 结束


# 生成word2index词典
def create_word2index():
    if not os.path.exists(WORD2INDEX_PATH):  # 如果文件不存在，则生成word2index
        word2index = dict()
        with open(VOCAB_PATH, 'r', encoding='utf8') as f:
            for word in f.readlines():
                word2index[word.strip()] = len(word2index) + 1
        with open(WORD2INDEX_PATH, 'wb') as f:
            pickle.dump(word2index, f)


# 数据准备
def data_prepare():
    # 加载训练集、测试集
    train_dataset = pd.read_csv(TRAIN_PATH, encoding='utf8')
    test_dataset = pd.read_csv(TEST_PATH, encoding='utf8')
    return train_dataset, test_dataset


# 文本、标签转化为索引（训练集）
def text_tag_to_index(dataset, word2index):
    # 预设索引
    unk_index = word2index.get(unk_flag)
    pad_index = word2index.get(pad_flag)
    start_index = word2index.get(start_flag, 2)
    end_index = word2index.get(end_flag, 3)

    texts, tags, masks = [], [], []
    n_rows = len(dataset)  # 行数
    for row in tqdm(range(n_rows)):
        text = dataset.iloc[row, 1]
        tag = dataset.iloc[row, 2]

        # 文本对应的索引
        text_index = [start_index] + [word2index.get(w, unk_index) for w in text] + [end_index]
        # 标签对应的索引
        tag_index = [0] + [tag2index.get(t) for t in tag.split()] + [0]

        # 填充或截断句子至标准长度
        if len(text_index) < MAX_LEN:  # 句子短，填充
            pad_len = MAX_LEN - len(text_index)
            text_index += pad_len * [pad_index]
            tag_index += pad_len * [0]
        elif len(text_index) > MAX_LEN:  # 句子长，截断
            text_index = text_index[:MAX_LEN - 1] + [end_index]
            tag_index = tag_index[:MAX_LEN - 1] + [0]

        # 转换为mask
        def _pad2mask(t):
            return 0 if t == pad_index else 1

        # mask
        mask = [_pad2mask(t) for t in text_index]

        # 加入列表中
        texts.append(text_index)
        tags.append(tag_index)
        masks.append(mask)

    # 转换为tensor
    texts = torch.LongTensor(texts)
    tags = torch.LongTensor(tags)
    masks = torch.tensor(masks, dtype=torch.uint8)
    return texts, tags, masks


# 文本转化为索引（测试集）
def text_to_index(dataset, word2index):
    # 预设索引
    unk_index = word2index.get(unk_flag)
    pad_index = word2index.get(pad_flag)
    start_index = word2index.get(start_flag, 2)
    end_index = word2index.get(end_flag, 3)

    texts, masks = [], []
    n_rows = len(dataset)  # 行数
    for row in tqdm(range(n_rows)):
        text = dataset.iloc[row, 1]

        # 文本对应的索引
        text_index = [start_index] + [word2index.get(w, unk_index) for w in text] + [end_index]

        # 填充或截断句子至标准长度
        if len(text_index) < MAX_LEN:  # 句子短，填充
            pad_len = MAX_LEN - len(text_index)
            text_index += pad_len * [pad_index]
        elif len(text_index) > MAX_LEN:  # 句子长，截断
            text_index = text_index[:MAX_LEN - 1] + [end_index]

        # 转换为mask
        def _pad2mask(t):
            return 0 if t == pad_index else 1

        # mask
        mask = [_pad2mask(t) for t in text_index]
        masks.append(mask)

        # 加入列表中
        texts.append(text_index)

    # 转换为tensor
    texts = torch.LongTensor(texts)
    masks = torch.tensor(masks, dtype=torch.uint8)
    return texts, masks


# 训练
def train(train_dataloader, model, optimizer, epoch):
    for i, batch_data in enumerate(train_dataloader):
        texts = batch_data['texts'].to(DEVICE)
        tags = batch_data['tags'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        loss, predictions = model(texts, tags, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            micro_f1 = get_f1_score(tags, masks, predictions)
            print(f'Epoch:{epoch} | i:{i} | loss:{loss.item()} | Micro_F1:{micro_f1}')


# 计算f1值
def get_f1_score(tags, masks, predictions):
    final_tags = []
    final_predictions = []
    tags = tags.to('cpu').data.numpy().tolist()
    masks = masks.to('cpu').data.numpy().tolist()
    for index in range(BATCH_SIZE):
        length = masks[index].count(1)  # 未被mask的长度
        final_tags += tags[index][1:length - 1]  # 去掉头和尾，有效tag，最大max_len-2
        final_predictions += predictions[index][1:length - 1]  # 去掉头和尾，有效mask，最大max_len-2

    f1 = f1_score(final_tags, final_predictions, average='micro')  # 取微平均
    return f1


# 执行流水线
def execute():
    # 读取数据集
    train_dataset, test_dataset = data_prepare()
    # 生成word2index字典
    create_word2index()
    with open(WORD2INDEX_PATH, 'rb') as f:
        word2index = pickle.load(f)

    # 文本、标签转化为索引
    texts, tags, masks = text_tag_to_index(train_dataset, word2index)
    print('text')
    print(texts[0])
    print('tags')
    print(tags[0])
    print('masks')
    print(masks)
    # breakpoint()
    # 数据集装载
    train_dataset = NerDataset(texts, tags, masks)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # 构建模型
    model = BiLSTM_CRF(vocab_size=len(word2index), tag2index=tag2index, embedding_dim=EMBEDDING_DIM,
                       hidden_size=HIDDEN_DIM, padding_idx=1, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    print(f"GPU_NAME:{torch.cuda.get_device_name()} | Memory_Allocated:{torch.cuda.memory_allocated()}")
    # 模型训练
    for i in range(EPOCH):
        train(train_dataloader, model, optimizer, i)

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)


# 测试集预测实体标签
def test():
    # 加载数据集
    test_dataset = pd.read_csv(TEST_PATH, encoding='utf8')
    # 加载word2index文件
    with open(WORD2INDEX_PATH, 'rb') as f:
        word2index = pickle.load(f)
    # 文本、标签转化为索引
    texts, masks = text_to_index(test_dataset, word2index)
    # 装载测试集
    dataset_test = NerDatasetTest(texts, masks)
    test_dataloader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False)
    # 构建模型
    model = BiLSTM_CRF(vocab_size=len(word2index), tag2index=tag2index, embedding_dim=EMBEDDING_DIM,
                       hidden_size=HIDDEN_DIM, padding_idx=1, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    # 模型预测
    model.eval()
    predictions_list = []
    for i, batch_data in enumerate(test_dataloader):
        texts = batch_data['texts'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        predictions = model(texts, None, masks)
        # print(len(texts), len(predictions))
        predictions_list.extend(predictions)
    print(len(predictions_list))
    print(len(test_dataset['text']))

    # 将预测结果转换为文本格式
    entity_tag_list = []
    index2tag = {v: k for k, v in tag2index.items()}  # 反转字典
    for i, (text, predictions) in enumerate(zip(test_dataset['text'], predictions_list)):
        # 删除首位和最后一位
        predictions.pop()
        predictions.pop(0)
        text_entity_tag = []
        for c, t in zip(text, predictions):
            if t != 0:
                text_entity_tag.append(c + index2tag[t])
        entity_tag_list.append(" ".join(text_entity_tag))  # 合并为str并加入列表中

    print(len(entity_tag_list))
    result_df = pd.DataFrame(data=entity_tag_list, columns=['result'])
    result_df.to_csv('./data/result_df.csv')


if __name__ == '__main__':
    execute()
    # test()
