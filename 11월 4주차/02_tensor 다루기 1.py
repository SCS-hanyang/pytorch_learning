# torch.autograd :자동 미분을 위한 함수들이 포함되어져 있습니다. 자동 미분의 on/off를 제어하는 콘텍스트 매니저(enable_grad/no_grad)나
# 자체 미분 가능 함수를 정의할 때 사용하는 기반 클래스인 'Function' 등이 포함되어져 있습니다.

# torch.nn : 신경망을 구축하기 위한 레이어 정의

# torch.optim : 파라미터 최적화 알고리즘

# torch.utils.data : 미니 배치용 유틸 함수

# torch.onnx : 서로 다른 딥 러닝 프레임워크 간에 모델 공유할 때 사용

import torch
import numpy as np

t = torch.FloatTensor([[0.,1.],[2.,3.],[4.,5.],])
print(t.dim())
print(t.shape)
print(t.size())

t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])

print(t[:,:-1])

# BroadCasting : tensor간에 사이즈가 맞지 않을 때 자동으로 연산해 주는 기능
#                (1,2)인 tensor와 (2,1)인 tensor가 존재할 경우 더하면 (2,2)로 연산 결과가 나옴


# 행렬곱 : m1.matmul(m2)

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1

# element-wise  : 행렬의 동일한 위치에 있는 원소끼리 곱
#               : m1.mul(m2), m1*m2
#               : Broadcasting 적용

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1, 2], [2, 3]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.mul(m2)) # 2 x 1
print(m1*m2)

# 평균    : t.mean()
#        : 이때 dim=은 해당 차원의 평균을 구하겠다는 의미 ex) 여기에서는 dim=0이면 행의 평균을 구하겠다는 의미

print(m1.mean())
print(m1.mean(dim=0))
print(m1.mean(dim=1))

# 덧셈    : t.sum()
#        : dim=역시 적용

print(m1.sum())
print(m1.sum(dim=0))
print(m1.sum(dim=1))

# max, argmax : max는 원소의 최댓값 리턴, argmax는 최댓값을 가진 인덱스를 리턴
#             : dim= 적용

# t.view : numpy에서 reshape와 같은 역할
# t.view([5,3]) 등으로 변환, -1은 주어진 상황에 맞춰서 자동 설정됨, 이때 t.view(-1, size, size)는 2번째, 3번째 차원이 size인 방식으로 view를 reshpe하라는 뜻

t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)

view1t = ft.view([3,4])
view2t = ft.view([-1,1,3])

print(ft)
print(view1t)
print(view2t.size())

# t.squeeze() : 1인 차원 제거

ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
print(ft.squeeze())
print(ft.squeeze().shape)

# t.unsqueeze(num) : num 위치에 1인 차원 추가, -1 넣으면 맨 뒤에 만듬

ft = torch.Tensor([0, 1, 2])
print(ft.unsqueeze(0).shape)
print(ft.unsqueeze(1).shape)

# 타입 캐스팅
# t.float(), t.long() 등등

# concatenate 하기

x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0))
print(torch.cat([x,y], dim=1))

# stacking : cat과 unsqueeze같은 많은 연산 포함

x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))

# torch.one_like(t), torch.zero_like(t) : t 와 동일한 shape의 1, 0 tensor 생성

# requires_grad=True : tensor 생성 시 pytorch에서 해당 텐서의 연산 그래프 생성 하고 gradient 추적

# scalar tensor 값 비교할 때는 tensor끼리