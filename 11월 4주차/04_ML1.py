import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


# softmax function 구현

t = torch.tensor([[1.,2.,3.],
                  [4.,5.,6.],
                  [7.,8.,9.],
                  [10.,11.,12.]], dtype=float)

print(F.softmax(t, dim=1))

a = torch.tensor([[-1.3301, -1.8084, -1.6846, -1.3530, -2.0584],
        [-1.4147, -1.8174, -1.4602, -1.6450, -1.7758],
        [-1.5025, -1.6165, -1.4586, -1.8360, -1.6776]],dtype=torch.float32)
b = torch.tensor([[1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0.]], dtype=torch.float32)

print((a*b).sum(dim=1).mean())

# 구현

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

dataset = TensorDataset(x_train, y_train)

train_data = DataLoader(dataset, shuffle=True, batch_size=2)


class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5,3)

    def forward(self, x):
        x1 = self.linear1(x)
        a1 = self.relu(x1)
        x2 = self.linear2(a1)

        return x2

model = myModel()

optimizer = optim.SGD(model.parameters(), lr = 0.01)

nb_epoches = 2000

for epoch in range(nb_epoches+1):
    for idx, data in enumerate(train_data):
        x_data, y_data = data

        prediction = model(x_data)

        loss = F.cross_entropy(prediction, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch : {epoch} / {nb_epoches}, loss : {loss:.6f}")
