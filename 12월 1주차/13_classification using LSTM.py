import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F

#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename=r"data\ratings_train.txt")
#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename=r"data\ratings_test.txt")

train_data = pd.read_table(r"data\ratings_train.txt")
test_data = pd.read_table(r"data\ratings_test.txt")


#  data 구조
'''
          id                           document  label
0   9976970                아 더빙.. 진짜 짜증나네요 목소리      0
1   3819312  흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나      1
2  10265843                  너무재밓었다그래서보는것을추천한다      0

'''

train_data.drop_duplicates(subset=['document'], inplace=True)       # document 항에서 중복되는 내용 제거, inplace : 원본 수정

train_data.loc[:,'document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", '', regex = True)

train_data.loc[:,'document'] = train_data['document'].str.replace("^ +", '', regex = True)
train_data['document'] = train_data['document'].replace('', np.nan)

train_data = train_data.dropna(how = 'any')

test_data.drop_duplicates(subset=['document'], inplace=True)       # document 항에서 중복되는 내용 제거, inplace : 원본 수정

test_data.loc[:,'document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", '', regex = True)

test_data.loc[:,'document'] = test_data['document'].str.replace("^ +", '', regex = True)
test_data['document'] = test_data['document'].replace('', np.nan)

test_data = test_data.dropna(how = 'any')

stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
'''
okt = Okt()

X_train = []

for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence)
    stopwords_removed_sentence = [word for word in tokenized_sentence if word not in stopwords]
    X_train.append(stopwords_removed_sentence)


X_test = []
for sentence in tqdm(test_data['document']):
    tokenized_sentence = okt.morphs(sentence)
    stopwords_removed_sentence = [word for word in tokenized_sentence if word not in stopwords]
    X_test.append(stopwords_removed_sentence)


pd.Series(X_train).to_csv('X_train_pandas.csv', index=False, header=False)
pd.Series(X_test).to_csv('X_test_pandas.csv', index=False, header=False)
'''

X_train = pd.read_csv(r'data\X_train_pandas.csv', header=None).values.flatten()
X_test = pd.read_csv(r'data\X_test_pandas.csv', header=None).values.flatten()

X_train = [ast.literal_eval(row) for row in X_train]
X_test = [ast.literal_eval(row) for row in X_test]

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)    # stratify : 레이블의 균형 비율을 유지하면서 분리

word_list = []

for sent in X_train:
    for word in sent:
        word_list.append(word)

word_counts = Counter(word_list)

vocab = sorted(word_counts, key = word_counts.get, reverse=True)        # .get() method는 각 키의 빈도수 가져오는 것 ex) .get('apple') = 3

threshold = 3
total_cnt = len(vocab)
rare_cnt = 0
total_rare = 0

rare_freq=0
total_freq = 0

for word, cnt in word_counts.items():
    total_freq += cnt

    if cnt < threshold:
        rare_cnt +=1
        rare_freq += cnt

vocab_size = total_cnt - rare_cnt
vocab = vocab[:vocab_size]

word2idx = dict((word, idx+2) for idx, word in enumerate(vocab))
word2idx['<pad>'] = 0
word2idx['<unk>'] = 1

def text_to_sequence(tokenized_X_data, word2idx):
    encoded_X_data = []

    for sent in tokenized_X_data:
        encoded_sent = []
        for word in sent:
            try:
                encoded_sent.append(word2idx[word])
            except KeyError:
                encoded_sent.append(word2idx['<unk>'])
        encoded_X_data.append(encoded_sent)

    return encoded_X_data

encoded_X_train = text_to_sequence(X_train, word2idx)
encoded_X_valid = text_to_sequence(X_valid, word2idx)
encoded_X_test = text_to_sequence(X_test, word2idx)

def how_many_below_threshold(threshold, encoded_X_train):
    cnt = 0

    for sent in encoded_X_train:
        if len(sent) < threshold:
            cnt += 1

    return cnt / len(encoded_X_train) * 100

# threshold : 30

max_len = 30

def pad_sequences(sentences, max_len):
    padded_data = np.zeros((len(sentences), max_len), dtype=int)

    for idx, sentence in enumerate(sentences):
        padded_data[idx, :len(sentence)] = np.array(sentence[:max_len])         # numpy의 경우 원래 크기보다 큰 범위를 불러와도, 자동으로 제한해주는 시스템 존재

    return padded_data

padded_X_train = pad_sequences(encoded_X_train, max_len)
padded_X_test = pad_sequences(encoded_X_test, max_len)
padded_X_valid = pad_sequences(encoded_X_valid, max_len)


train_label = torch.tensor(y_train)
test_label = torch.tensor(y_test)
valid_label = torch.tensor(y_valid)


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size, hidden_dim):
        super(TextClassifier, self).__init__()
        self.output_dim = 2
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        out = self.lstm(out)
