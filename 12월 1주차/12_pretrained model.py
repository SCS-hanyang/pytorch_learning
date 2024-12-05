import numpy as np
from collections import Counter
import gensim
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality', 'bad', 'highly respectable']
y_train = [1, 0, 0, 1, 1, 0, 1]

tokenized_sentences = [word_tokenize(sentence, language='english') for sentence in sentences]

word_list = []
for sentence in tokenized_sentences:
    for word in sentence:
        word_list.append(word)

word_counts = Counter(word_list)

vocab = sorted(word_list, key=word_counts.get, reverse=True)

word2idx = dict((word, i+2) for i, word in enumerate(vocab))
word2idx['<pad>'] = 0
word2idx['<unk>'] = 1

vocab_size = len(word2idx)

'''
def text_to_sequence(tokenized_X_data, word2idx):
    encoded_X_data = []

    for sentence in tokenized_X_data:
        idx_list = []
        for word in sentence:
            try:
                idx_list.append(word2idx[word])
            except KeyError:
                idx_list.append(word2idx["<unk>"])
        encoded_X_data.append(idx_list)

    return encoded_X_data



def pad_sequences(encoded_X_data):


    features = np.zeros((len(encoded_X_data), max_len), dtype=int)
    for idx, sentence in enumerate(encoded_X_data):
        if len(sentence) != 0:
            features[idx,:len(sentence)] = np.array(sentence)[:max_len]
    return features

sequence = text_to_sequence(tokenized_sentences, word2idx)
max_len = max(len(l) for l in sequence)

x_train = pad_sequences(sequence)
y_train = np.array(y_train)

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embedding_size*max_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        output = self.embedding(x)
        output = self.flatten(output)
        output = self.fc(output)

        return self.sigmoid(output)

embedding_size = 100
model = SimpleModel(vocab_size=vocab_size, embedding_size=embedding_size)

creterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

nb_epoch = 15

train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
train_dataloader = DataLoader(train_dataset, batch_size=2)

for epoch in range(nb_epoch):
    for input, output in train_dataloader:
        prediction = model(input).view(-1)
        loss = creterion(prediction, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'epoch : {epoch+1} / {nb_epoch}, loss : {loss}')
    

'''

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

embedding_matrix = np.zeros((vocab_size, 300))

def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None

print(get_vector['great'])