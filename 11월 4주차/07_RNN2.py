<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sympy import sequence
import tqdm

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = sorted(list(set(sentence)))
char_dic = dict((c,i) for i, c in enumerate(char_set))
dic_size = len(char_dic)

hidden_size = dic_size
sequence_length = 10
learning_rate = 0.1

x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i+sequence_length]
    y_str = sentence[i+1 : i+sequence_length+1]

    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])


X = F.one_hot(torch.tensor(x_data), num_classes=dic_size).float()
Y = torch.LongTensor(y_data)


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

model = RNN(dic_size, hidden_size, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

nb_epoches = 300

for epoch in range(nb_epoches):
    prediction = model(X)
    loss = criterion(prediction.view(-1, dic_size), Y.view(-1))        # CrossEntropy에서 입력 텐서의 형식은 (N, C, d1, d2...)로 이루어져야 하고
                                                            # 정답 텐서는 (N, d1, d2...) 이런 식이다. 하지만 이 코드의 경우 입력 텐서가
                                                            # (N, d1, C)로 이루어져서 오류가 발생했다. 이를 수정하면 된다.

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    results = prediction.argmax(dim=2)
    result_str = ''

    for i, value in enumerate(results):
        if i == 0:
            result_str += ''.join([char_set[c] for c in value])
        else :
            result_str += char_set[value[-1]]

    print(f'epoch : {epoch+1}, loss : {loss}')
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sympy import sequence
import tqdm

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = sorted(list(set(sentence)))
char_dic = dict((c,i) for i, c in enumerate(char_set))
dic_size = len(char_dic)

hidden_size = dic_size
sequence_length = 10
learning_rate = 0.1

x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i+sequence_length]
    y_str = sentence[i+1 : i+sequence_length+1]

    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])


X = F.one_hot(torch.tensor(x_data), num_classes=dic_size).float()
Y = torch.LongTensor(y_data)


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

model = RNN(dic_size, hidden_size, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

nb_epoches = 300

for epoch in range(nb_epoches):
    prediction = model(X)
    loss = criterion(prediction.view(-1, dic_size), Y.view(-1))        # CrossEntropy에서 입력 텐서의 형식은 (N, C, d1, d2...)로 이루어져야 하고
                                                            # 정답 텐서는 (N, d1, d2...) 이런 식이다. 하지만 이 코드의 경우 입력 텐서가
                                                            # (N, d1, C)로 이루어져서 오류가 발생했다. 이를 수정하면 된다.

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    results = prediction.argmax(dim=2)
    result_str = ''

    for i, value in enumerate(results):
        if i == 0:
            result_str += ''.join([char_set[c] for c in value])
        else :
            result_str += char_set[value[-1]]

    print(f'epoch : {epoch+1}, loss : {loss}')
>>>>>>> 95b60bdff058f123cb273be7163370ffb90b989b
    print(result_str)