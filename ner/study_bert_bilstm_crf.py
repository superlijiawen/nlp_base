# @project : pythonProject
# -*- coding: utf-8 -*-
# @Time    : 2022/8/28 17:04
# @Author  : leejack
# @File    : study_bilstm_crf.py
# @Description : 学习Bert-BiLSTM-CRF
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from bert_bilstm_crf import Bert_BiLSTM_CRF, NerDataset, NerDatasetTest
from bert_crf import Bert_CRF
from transformers import AutoTokenizer, BertTokenizer
from seqeval.metrics import f1_score

# 路径
TRAIN_PATH = './dataset/train_data_public.csv'
TEST_PATH = './dataset/test_public.csv'
MODEL_PATH1 = './model/bert_bilstm_crf.pkl'
MODEL_PATH2 = '../model/bert_crf.pkl'

# 超参数
MAX_LEN = 64
BATCH_SIZE = 16
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
index2tag = {v: k for k, v in tag2index.items()}


# 训练
def train(train_dataloader, model, optimizer, epoch):
    for i, batch_data in enumerate(train_dataloader):
        token_texts = batch_data['token_texts'].to(DEVICE)
        tags = batch_data['tags'].to(DEVICE)
        loss, predictions = model(token_texts, tags)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            micro_f1 = get_f1_score(tags, predictions)
            print(f'Epoch:{epoch} | i:{i} | loss:{loss.item()} | Micro_F1:{micro_f1}')


# 计算f1值
def get_f1_score(tags, predictions):
    tags = tags.to('cpu').data.numpy().tolist()
    temp_tags = []
    final_tags = []
    for index in range(BATCH_SIZE):
        # predictions先去掉头，再去掉尾
        predictions[index].pop()
        length = len(predictions[index])
        temp_tags.append(tags[index][1:length])
        predictions[index].pop(0)
        # 格式转化，转化为List(str)
        temp_tags[index] = [index2tag[x] for x in temp_tags[index]]
        predictions[index] = [index2tag[x] for x in predictions[index]]
        final_tags.append(temp_tags[index])

    f1 = f1_score(final_tags, predictions, average='micro')
    return f1


# 预处理
def data_preprocessing(dataset, is_train):
    # 数据str转化为list
    dataset['text_split'] = dataset['text'].apply(list)
    # token
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    texts = dataset['text_split'].array.tolist()
    token_texts = []
    for text in tqdm(texts):
        tokenized = tokenizer.encode_plus(text=text,
                                          max_length=MAX_LEN,
                                          return_token_type_ids=True,
                                          return_attention_mask=True,
                                          return_tensors='pt',
                                          padding='max_length',
                                          truncation=True)
        token_texts.append(tokenized)

    # 训练集有tag，测试集没有tag
    tags = None
    if is_train:
        dataset['tag'] = dataset['BIO_anno'].apply(lambda x: x.split(sep=' '))
        tags = []
        for tag in tqdm(dataset['tag'].array.tolist()):
            index_list = [0] + [tag2index[t] for t in tag] + [0]
            if len(index_list) < MAX_LEN:  # 填充
                pad_length = MAX_LEN - len(index_list)
                index_list += [tag2index['O']] * pad_length
            if len(index_list) > MAX_LEN:  # 裁剪
                index_list = index_list[:MAX_LEN-1] + [0]
            tags.append(index_list)
        tags = torch.LongTensor(tags)

    return token_texts, tags


# 执行流水线
def execute():
    # 加载训练集
    train_dataset = pd.read_csv(TRAIN_PATH, encoding='utf8')
    # 数据预处理
    token_texts, tags = data_preprocessing(train_dataset, is_train=True)
    # 数据集装载
    train_dataset = NerDataset(token_texts, tags)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # 构建模型
    # model = Bert_BiLSTM_CRF(tag2index=tag2index).to(DEVICE)
    model = Bert_CRF(tag2index=tag2index).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-6)
    print(f"GPU_NAME:{torch.cuda.get_device_name()} | Memory_Allocated:{torch.cuda.memory_allocated()}")
    # 模型训练
    for i in range(EPOCH):
        train(train_dataloader, model, optimizer, i)

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH2)


# 测试集预测实体标签
def test():
    # 加载数据集
    test_dataset = pd.read_csv(TEST_PATH, encoding='utf8')
    # 数据预处理
    token_texts, _ = data_preprocessing(test_dataset, is_train=False)
    # 装载测试集
    dataset_test = NerDatasetTest(token_texts)
    test_dataloader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # 构建模型
    # model = Bert_BiLSTM_CRF(tag2index).to(DEVICE)
    model = Bert_CRF(tag2index).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH2))
    # 模型预测
    model.eval()
    predictions_list = []
    with torch.no_grad():
        for i, batch_data in enumerate(test_dataloader):
            token_texts = batch_data['token_texts'].to(DEVICE)
            predictions = model(token_texts, None)
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
    result_df.to_csv('./data/result_df3.csv')


if __name__ == '__main__':
    # execute()
    test()
