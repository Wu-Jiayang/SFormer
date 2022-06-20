from utils import split
import random
import json
import os
from tqdm import tqdm

with open('baike_qa2019/baike_qa_train.json', 'r', encoding='utf-8') as f:
    data = f.read().strip().split('\n')
data = [json.loads(i) for i in data]
data = [['category: ' + i['category'], 'title: ' + i['title'],
         'desc: ' + i['desc'], 'answer: ' + i['answer']] for i in data]
print(len(data))
s = random.choice(data)
print(s)
s = [j for i in s for j in split(i)]
print(s)

result = []
for item in tqdm(data):
    l = [j for i in item for j in split(i)]
    while len(l) > len(result):
        result.append([])
    result[len(l) - 1].append(l)

l = {i + 1: len(j) for i, j in enumerate(result)}
print(l)
result = [i for i in result[1:] if len(i) >= 32]
l = {len(i[0]): len(i) for i in result}
print(l)
'''
for item in result:
    with open('corpus/%d.txt' % len(item[0]), 'a', encoding='utf-8') as f:
        s = [json.dumps(i, ensure_ascii=False) for i in item]
        f.write('\n'.join(s) + '\n')
'''