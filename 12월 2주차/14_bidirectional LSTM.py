import torch
from nltk import word_tokenize, pos_tag, ne_chunk

import urllib.request
import numpy as np
from tqdm import tqdm
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

'''
Named Entity Recognition

    named entity(이름을 가진 개체)를 인식 -> 어떤 이름을 의미하는 단어를 보고 그 단어가 어떤 유형인지 인식
    
    ex) 예를 들어 유정이는 2018년에 골드만삭스에 입사했다. 라는 문장이 있을 때, 
    사람(person), 조직(organization), 시간(time)에 대해 개체명 인식을 수행하는 모델이라면 다음과 같은 결과를 보여줍니다. 
        유정 - 사람  
        2018년 - 시간  
        골드만삭스 - 조직

    
'''

sentence = "James is working at Disney in London"

tokenized_sentence = pos_tag(word_tokenize(sentence))                       # 품사 태깅

ner_sentence = ne_chunk(tokenized_sentence)                                 # 품사 태깅된 객체를 대상으로 NER

#urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20RNN%20Sequence%20Labeling/dataset/train.txt", filename=r"data\train.txt")

with open(r'data\train.txt', 'r') as f:

    tagged_sentences,sentence  = [], []

    for line in f:
        line = line.strip()
        if not line or line.startswith('-DOCSTART-'):
            if sentence:
                tagged_sentences.append(sentence)

            sentence = []
            continue

        word = line.split()
        sentence.append([word[0].lower(), word[-1]])

sentences, ner_tags = [] , []

for tagged_sentence in tagged_sentences:
    sentence, ner_tag = zip(*tagged_sentence)

    sentences.append(list(sentence))
    ner_tags.append(list(ner_tag))

x_train, x_test, y_train, y_test = train_test_split(sentences, ner_tags, test_size=0.2, random_state=0)

words = []
for sentence in x_train:
    for word in sentence:
        words.append(word)

vocab = Counter(words)

vocab = sorted(vocab, key = vocab.get, reverse=True)

word2idx = dict((word, idx+2) for idx, word in enumerate(vocab))
word2idx['<pad>'] = 0
word2idx['<unk>'] = 1

def text_to_sequences(text, word2idx):

    embedded_sentences=[]

    for sentence in text:
        list_sentence = []
        for word in sentence:
            try:
                list_sentence.append(word2idx[word])
            except KeyError:
                list_sentence.append(word2idx['<unk>'])
        embedded_sentences.append(list_sentence)

    return embedded_sentences

embedded_x_train = text_to_sequences(x_train, word2idx)
embedded_x_test = text_to_sequences(x_test, word2idx)

flatten_tag = [tag for tags in y_train for tag in tags]
tag_vocab = list(set(flatten_tag))

tag2idx = dict((tag, idx+1) for idx, tag in enumerate(tag_vocab))
tag2idx['<pad>'] = 0

def encoding_labels(labels, tag2idx):
    encoded_labels = []

    for label in labels:
        encoded_labels.append([tag2idx[tag] for tag in label])

    return encoded_labels

embedded_y_train = encoding_labels(y_train, tag2idx)
embedded_y_test = encoding_labels(y_test, tag2idx)

'''
embedded x train

    최대 길이 : 113
    평균 길이 : 14.467770655270655

'''

max_len = 40

def padding_sequence(sequences, max_len):
    padded_sequences = np.zeros((len(sequences), max_len), dtype=int)

    for idx, sequence in enumerate(sequences):
        padded_sequences[idx, :len(sequence)] = np.array(sequence, dtype=int)[:max_len]

    return padded_sequences

padded_x_train = padding_sequence(embedded_x_train, max_len)
padded_x_test = padding_sequence(embedded_x_test, max_len)

padded_y_train = padding_sequence(embedded_y_train, max_len)
padded_y_test = padding_sequence(embedded_y_test, max_len)

USE_CUDA = torch.cuda.is_available()

device = torch.device('cuda' if USE_CUDA else "cpu")

class NERTagger(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers):
        super(NERTagger, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)
        self.gru = nn.GRU(
            input_size= embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=True)
        self.fc = nn.Linear(in_features = hidden_size * 2, out_features=output_size)

    def forward(self, input):
        embedding = self.embedding(input)
        gru_out, _hidden = self.gru(embedding)
        output = self.fc(gru_out)
        return output

vocab_size = len(word2idx)
embedding_size = 100
hidden_size = 256
output_size = len(tag2idx)
learning_rate = 0.01
num_epochs = 10
num_layers = 2


model = NERTagger(vocab_size, embedding_size, hidden_size, output_size, num_layers)
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def calculate_accuracy(logits, labels, ignore_idx = 0):

    # logits : (batch, sequence, labels)

    prediction = torch.argmax(logits, dim=2)

    mask = (labels != ignore_idx)
    correct = (prediction == labels)[mask].sum().item()
    total_size = mask.sum().item()

    return correct/total_size, total_size

def evaluate(model, criterion, test_dataloader):
    model.eval()
    avg_loss = 0
    total_accuracy = 0
    total_size = 0

    with torch.no_grad():
        for x,y in test_dataloader:
            x, y = x.to(device), y.to(device)

            prediction = model(x)

            loss = criterion(prediction.view(-1, prediction.size(-1)), y.view(-1))

            avg_loss += loss.item()
            accuracy, size = calculate_accuracy(prediction, y)
            total_accuracy += accuracy * size
            total_size += size

        total_accuracy = total_accuracy / total_size
        avg_loss = avg_loss / len(test_dataloader)

    return avg_loss, total_accuracy

train_dataset = torch.tensor(padded_x_train, dtype = torch.long)
test_dataset = torch.tensor(padded_x_test, dtype = torch.long)

train_labels = torch.tensor(padded_y_train, dtype = torch.long)
test_labels = torch.tensor(padded_y_test, dtype = torch.long)

Tensor_train_dataset = TensorDataset(train_dataset, train_labels)
Tensor_test_dataset = TensorDataset(test_dataset, test_labels)

train_dataloader = DataLoader(Tensor_train_dataset, batch_size = 32, shuffle=True, drop_last = True)
test_dataloader = DataLoader(Tensor_test_dataset, batch_size = 32, shuffle=True, drop_last = True)

best_val_loss = float('inf')

for epoch in range(num_epochs):
    avg_loss = 0
    avg_accuracy = 0
    model.train()

    for x, y in tqdm(train_dataloader):
        x, y = x.to(device), y.to(device)

        prediction = model(x)

        loss = criterion(prediction.view(-1, prediction.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy, _ = calculate_accuracy(prediction, y)
        avg_accuracy += accuracy
        avg_loss += loss.item()

    avg_accuracy /= len(train_dataloader)
    avg_loss /= len(train_dataloader)

    val_loss, val_accuracy = evaluate(model, criterion, test_dataloader)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    if val_loss < best_val_loss:
        print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. 체크포인트를 저장합니다.')
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_checkpoint.pth')

# LSTM과 GRU 동시에 해본 결과 LSTM이 학습 속도는 느리지만 성능은 더 좋음
# max_len아 40 이하인 sample이 96% 이지만 max_len을 40으로 했을 경우 정답률이 94%까지 올라가다가 90%에서 정체됨
# 50 이하가 99% 인데도 90%에서 정체됨