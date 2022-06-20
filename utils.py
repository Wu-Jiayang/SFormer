import re
import os
from typing import Dict
import paddle
import json
import random
from tqdm import tqdm
import numpy as np


class PretrainDataset(paddle.io.Dataset):
    def __init__(self, tokenizer, file_root: str, batch_size=32, steps=10000, max_data_epoch=16):
        super(PretrainDataset, self).__init__()
        self.mask_id = 30000
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.steps = steps
        self.max_data_epoch = max_data_epoch
        self.file_list = [os.path.join(file_root, file) for file in os.listdir(file_root)]
        self.random_weight = []
        print('正在计算数据集权重......')
        for file in tqdm(self.file_list):
            with open(file, 'r', encoding='utf-8') as f:
                self.random_weight.append(len(f.read().strip().split('\n')))

    def __len__(self):
        # return int(np.ceil(len(self.data) / float(self.batch_size)))
        return self.steps

    def __getitem__(self, item) -> Dict:
        file = random.choices(self.file_list, self.random_weight)[0]
        with open(file, 'r', encoding='utf-8') as f:
            data = f.read().strip().split('\n')
        data = random.choices(data, k=self.batch_size)
        data = [json.loads(i)[:self.max_data_epoch] for i in data]  # 将每条数据集轮次压在max_data_epoch之内
        data_epochs = len(data[0])
        for d in data:
            assert len(d) == data_epochs
        masked_inputs = []
        initial_inputs = []
        labels = []
        for i in range(data_epochs):
            input_ids = []
            masked_ids = []
            y = []
            token_type_ids = []
            position_ids = []
            attention_mask_temp = []
            for d in data:  # batch
                encoded_input_temp = self.tokenizer.dialogue_encode(
                    d[i],
                    return_tensors=False,
                    is_split_into_words=False
                )
                input_ids.append(encoded_input_temp['input_ids'])
                masked_id, label = self.random_mask(encoded_input_temp['input_ids'])
                masked_ids.append(masked_id)
                y.append(label)
                token_type_ids.append(encoded_input_temp['token_type_ids'])
                position_ids.append(encoded_input_temp['position_ids'])
                attention_mask_temp.append(encoded_input_temp['attention_mask'])     # numpy, shape=(seq_len, seq_len)
            input_ids = self.pad_batch_data(input_ids)
            masked_ids = self.pad_batch_data(masked_ids)
            y = self.pad_batch_data(y)
            token_type_ids = self.pad_batch_data(token_type_ids)
            position_ids = self.pad_batch_data(position_ids)
            attention_mask = np.zeros((input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1]))
            for j in range(input_ids.shape[0]):
                raw_seq_len = attention_mask_temp[j].shape[0]
                attention_mask[j, 0, :raw_seq_len, :raw_seq_len] = attention_mask_temp[j]
                attention_mask[j, ..., raw_seq_len:] = -1e9
            attention_mask = paddle.to_tensor(attention_mask, dtype=paddle.float32)
            initial_inputs.append({
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask
            })
            masked_inputs.append({
                'input_ids': masked_ids,
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask
            })
            labels.append(y)
        return {
            'initial_inputs': initial_inputs,
            'masked_inputs': masked_inputs,
            'labels': labels
        }

    def random_mask(self, seq: list) -> tuple:
        """
        :param seq: 一维数组
        """
        x, y = [], []
        for id in seq:
            num = random.random()
            if num >= 0.85:
                y.append(id)
                if num >= 1 - 0.15 * 0.1:
                    x.append(id)
                elif num >= 1 - 0.15 * 0.2:
                    x.append(random.choice(range(1, len(self.tokenizer.vocab))))
                else:
                    x.append(self.mask_id)
            else:
                x.append(id)
                y.append(0)
        return x, y

    def pad_batch_data(self, batch):
        """Pad the instances to the max sequence length in batch. """
        max_len = max(map(len, batch))
        batch_data = paddle.to_tensor(
            [
                list(data) + [0] * (max_len - len(data))
                for data in batch
            ],
            dtype='int64')
        return batch_data


def split(text: str) -> list:
    texts = text.strip().split('\r')
    return [item for text in texts for item in cut_sent(text) if item != '']


def cut_sent(para: str) -> list:
    para = re.sub('([。！？\?]+)([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?]+[”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")
