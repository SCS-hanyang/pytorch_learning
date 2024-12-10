
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import torch
import urllib.request
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv(r"IMDB Dataset.csv")

# column : review, sentiment, 개수 : 5만개

# df['sentiment'].value_counts().plot(kind='bar')     label distribution 판단

df.loc[:,'sentiment'] = df['sentiment'].map({"positive": 1, "negative": 0})

X_data = df['review']
y_data = df['sentiment'].astype(int)


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.4 ,random_state=0, stratify=y_data)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

def tokenize(sentences):
    stopword = set(stopwords.words('english'))

    tokenized = []
    for sentence in tqdm(sentences):
        words = word_tokenize(sentence)
        tokenized_sentence = [word for word in words if word not in stopword]
        tokenized.append(tokenized_sentence)

    return tokenized

tokenized_X_train = tokenize(X_train)
tokenized_X_valid = tokenize(X_valid)
#tokenized_X_test = tokenize(X_test)

word_list = []

for sentence in tokenized_X_train:
    for word in sentence:
        word_list.append(word)

word_counts = Counter(word_list)

vocab = sorted(word_counts, key=word_counts.get, reverse=True)

threshold = 3

total_cnt = len(word_counts)
rare_cnt = 0
rare_freq = 0
total_freq = 0


for key, value in word_counts.items():

    total_freq += value

    if value < threshold + 1 :
        rare_cnt += 1
        rare_freq += value

vocab = vocab[:total_cnt - rare_cnt]

word2idx = dict([word, idx+2] for idx, word in enumerate(vocab))
word2idx["<pad>"] = 0
word2idx["<unk>"] = 1

vocab_size = len(word2idx)

def text_to_sequence(text, word2idx):
    embedded_sentences = []

    for sent in text:
        embedded_sentence=[]
        for word in sent:
            try:
                embedded_sentence.append(word2idx[word])
            except KeyError:
                embedded_sentence.append(word2idx['<unk>'])
        embedded_sentences.append(embedded_sentence)

    return embedded_sentences

embedded_X_train = text_to_sequence(tokenized_X_train, word2idx)
embedded_X_valid = text_to_sequence(tokenized_X_valid, word2idx)
#embedded_X_test = text_to_sequence(tokenized_X_test, word2idx)

def how_much_below_threshold(text):
    threshold = 500

    cnt = 0
    for sent in text:
        if len(sent) < threshold:
            cnt += 1

    print(f"{threshold} 단어 보다 짧은 문장은 전체의 {cnt / len(text) * 100}% 이다")

def padding_sequence(data, threshold):

    sequences = np.zeros((len(data), threshold), dtype=int)

    for idx, sent in enumerate(data):
        if len(sent) != 0:
            sequences[idx,:len(sent)] = np.array(sent)[:threshold]

    return sequences

padded_X_train = padding_sequence(embedded_X_train, 400)
padded_X_valid = padding_sequence(embedded_X_valid, 400)
#padded_X_test = padding_sequence(embedded_X_test, 400)

X_train = torch.tensor(padded_X_train).to(torch.int64)
X_valid = torch.tensor(padded_X_valid).to(torch.int64)
#X_test = torch.tensor(padded_X_test, dtype=torch.int64)

y_train = torch.LongTensor(np.array(y_train))
y_valid = torch.LongTensor(np.array(y_valid))
#y_test = torch.LongTensor(np.array(y_test))


class GRU(nn.Module):
    def __init__(self, vocab_size, embedded_size, hidden_size, out_dim):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedded_size, padding_idx=0)
        self.GRU = nn.GRU(embedded_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.GRU(embedded)        # hidden 정보를 사용하는 이유 : hidden의 정보는 시퀀스 전체의 정보를 담고 있는 요약본 느낌이다
                                                    # 만약 단어 단위의 정보가 필요하다면, gru_out을 사용하는 것이 맞지만, 문장 전체에 대한 간단한 정보를 얻고자 한다면
                                                    # hidden의 정보를 사용하는 것이 유용할 수도 있다.
        last_hidden = hidden.squeeze(0)
        logits = self.fc(last_hidden)
        return logits

def calculate_accuracy(logits, labels):
    prediction = logits.argmax(dim=1)

    accuracy = (labels == prediction).sum().item()      # item() : torch 안에 값 리턴
    rate = accuracy / len(prediction)

    return rate

def evaluate(model, valid_dataloader, criterion):

    model.eval()
    val_loss = 0
    val_accuracy = 0
    total_size = 0

    with torch.no_grad():                        # 연산 도중 parameter 업데이트 방지하고, grad 계산 안해서 계산 속도 빠르게 하려고
        for dataset in valid_dataloader:
            x = dataset[0].to(device)
            y = dataset[1].to(device)

            prediction = model(x)

            loss = criterion(prediction, y)

            val_loss += loss
            val_accuracy += calculate_accuracy(prediction, y) * x.size(0)
            total_size += x.size(0)

    val_accuracy = val_accuracy / total_size
    val_loss = val_loss / total_size

    return val_accuracy, val_loss

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32, drop_last=True)

valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=32)

#test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
#test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=1)

nb_epoch = 100
embedding_size = 100
hidden_size = 128
output_dim = 2

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

model = GRU(vocab_size, embedding_size, hidden_size, output_dim)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')


for epoch in range(nb_epoch):
    avg_loss = 0
    avg_correct = 0
    total_size = 0
    model.train()

    for data in tqdm(train_dataloader):
        input = data[0].to(device)
        label = data[1].to(device)

        prediction = model(input)
        loss = criterion(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()        # loss 결과값은 tensor
        avg_correct += calculate_accuracy(prediction, label) * input.size(0)
        total_size += input.size(0)

    avg_loss = avg_loss / total_size
    avg_correct = avg_correct / total_size
    val_accuracy, val_loss = evaluate(model, valid_dataloader, criterion)

    print(f"epoch {epoch+1 / nb_epoch} /// loss : {avg_loss:.3f}  /// accuracy : {avg_correct * 100:.1f}")
    print(f"/// val_loss : {val_loss:.3f}  /// val_accuracy : {val_accuracy * 100:.1f}")

    if val_loss < best_val_loss:
        print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. 체크포인트를 저장합니다.')
        best_val_loss = val_loss
        torch.save(model.state_dict(), r'best_model_checkpoint.pth')


