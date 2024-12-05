import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 다중 변수 선형 회귀
# 훈련 데이터

x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w1,w2,w3,b], lr=1e-5)

nb_epochs = 1000
'''
for epoch in range(nb_epochs + 1):

    Hx = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    cost = torch.mean((Hx - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch : {epoch} / {nb_epochs}, cost : {cost:.6f} ")

print(f'w1 : {w1.item():.3f}, w2 : {w2.item():.3f}, w3 : {w3.item():.3f}, b : {b.item():.3f}')
'''
# 행렬 연산 다중 변수 선형 회귀

x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  80],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

Hx = x_train.matmul(W) + b

print(W.shape)