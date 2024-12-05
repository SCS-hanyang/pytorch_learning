<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

'''
선형 회귀 모델에 대한 nn 함수

model = nn.Linear(input_dim, output_dim)

cost = F.mse_loss(prediction, y_train)
'''

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1,1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # optimizer 선언 방법

nb_epochs = 1000

for epoch in range(nb_epochs):

    # Hx 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch : {epoch} / {nb_epochs}, cost : {cost:.6f}')

print(f"w : {model.weight}, b : {model.bias}")

# 모델에서 weight, bias 추출 방법 : model.weight, model.bias

# nn.Module로 선언한 클래스의 경우 __call__에 forward 함수를 실행하라는 함수가 포함되어있기 때문에
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

'''
선형 회귀 모델에 대한 nn 함수

model = nn.Linear(input_dim, output_dim)

cost = F.mse_loss(prediction, y_train)
'''

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1,1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # optimizer 선언 방법

nb_epochs = 1000

for epoch in range(nb_epochs):

    # Hx 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch : {epoch} / {nb_epochs}, cost : {cost:.6f}')

print(f"w : {model.weight}, b : {model.bias}")

# 모델에서 weight, bias 추출 방법 : model.weight, model.bias

# nn.Module로 선언한 클래스의 경우 __call__에 forward 함수를 실행하라는 함수가 포함되어있기 때문에
>>>>>>> 95b60bdff058f123cb273be7163370ffb90b989b
# model(x)를 하게 되면 model.forward(x)함수가 실행되게 된다.