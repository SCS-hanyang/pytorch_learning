import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)        # 난수 생성기의 시드

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

hypothesis = x_train * W + b

cost = torch.mean((hypothesis - y_train) ** 2)

print(f"W : {W}, b : {b}")
print(f"Cost : {cost}")

optimizer = optim.SGD([W,b], lr=0.01)

optimizer.zero_grad()

cost.backward()

optimizer.step()

cost = torch.mean((hypothesis - y_train) ** 2)
print(f"W : {W}, b : {b}")
print(f"Cost : {cost}")


# requires_grad=True인 텐서
W = torch.tensor([2.0], requires_grad=True)

# 첫 번째 연산과 backward
loss1 = W * 3  # 연산 1
loss1.backward()  # Gradient 계산
print("W.grad after first backward:", W.grad)  # 첫 번째 backward의 gradient

# 두 번째 연산과 backward
loss2 = W * 5  # 연산 2
loss2.backward()  # Gradient 계산 (누적됨)
print("W.grad after second backward:", W.grad)  # 누적된 gradient

'''
!!! .backward() 함수는 계산한 .grad값을 누적해서 저장한다. 그래서 각 epoch당 optimizer.zero_grad()가 필요한 이유
'''

torch.manual_seed(1)        # 난수 생성기의 시드

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr = 0.01)

n_epoch = 100
'''
for epoch in range(n_epoch):
    hypothesis = x_train * W + b

    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print(f"epoch : {epoch+1}회 //// W:{W.item():.3f} //// b:{b.item():.3f} //// cost:{cost.item():.6f}")
'''

t = torch.Tensor([1,2,3,4])

print(t)
print(t.tolist())