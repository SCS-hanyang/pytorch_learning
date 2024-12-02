import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

input_str = "apple"
label_str = "pple!"

char_vocab = sorted(list(set(input_str + label_str)))
vocab_size = len(char_vocab)

input_size = vocab_size
hidden_size = 5
output_size = 5
lr = 0.2

char_to_index = dict((c, i) for (i, c) in enumerate(char_vocab))
index_to_char = dict((i, c) for (i, c) in enumerate(char_vocab))

x_data = [char_to_index[c] for c in input_str]
y_data = [char_to_index[c] for c in label_str]

x_data = F.one_hot(torch.tensor(x_data), num_classes=vocab_size).float().unsqueeze(
    0)  # RNN은 기본적으로 3차원 텐서를 입력으로 받아서(1은 batch size가 될것)
y_data = torch.tensor(y_data, dtype=torch.float32).unsqueeze(0).long()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size,
                          batch_first=True)  # batch_first : RNN, LSTM, GRU input의 기본 구조는 input = (seq_len, batch_size, input_size)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)  # 인데, 여기서 batch_size를 가장 앞에 놓기 위해 사용

    def forward(self, x):
        x, _status = self.rnn(x)  # nn.rnn은 output으로 (output, hidden)을 출력한다
        x = self.fc(x)  # output은 각 타임스탭에서의 출력이고, hidden은 마지막 타임스탭에서의 은닉상태이다
        return x


model = RNN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = lr)

for i in range(100):
    outputs = model(x_data)

    loss = criterion(outputs.view(-1, input_size), y_data.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis = 2)
    result_str = ''.join(index_to_char[c] for c in result.squeeze())      # str.join(iterable) iterable한 데이터를 하나의 str로 합침

    print(f'loss : {loss.item():.3f}, result : {result_str}')

    if result_str == label_str:
        print(f'{i+1}번째 만에 정답인 {result_str} 출력함!')
        break
