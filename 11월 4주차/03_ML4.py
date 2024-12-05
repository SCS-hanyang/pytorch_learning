import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


'''

torch.utils.data.TensorDataset : 여러 텐서를 하나의 데이터셋으로 묶어 관리
    1. 텐서들의 크기가 동일한 첫 번째 차원 수(N, 샘플수)여야함
    2. 묶인 텐서를 인덱싱(__getitem__)을 통해 샘플 단위로 쉽게 접근 가능
    3. DataLoader와 같이 쓰임
    4. param[num]을 통해 관리하는 모든 텐서의 num index를 불러올 수 있음

'''

# 입력 텐서와 레이블 텐서
x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float32)

# TensorDataset으로 묶기
dataset = TensorDataset(x, y)


# 인덱싱을 통해 데이터 접근
print(dataset[0])  # 첫 번째 샘플 (x[0], y[0])

# 첫 번째 샘플의 x만 가져오기
x_sample = dataset[0][0]  # (x[0], y[0]) 중 x[0] 선택
print("X Sample:", x_sample)

# .tensors를 통해 각각의 tensor불러오기 가능

print("x tensor:",dataset.tensors[0])
print("y tensor:",dataset.tensors[1])

'''

torch.utils.data.DataLoader

데이터셋에서 미니배치 단위로 데이터를 로드할때 쓰임
    1. 데이터셋을 배치 크기로 나눔
    2. shuffle=True로 설정하면 데이터셋의 순서를 섞어서 로드
    3. num_workers를 설정하면 여러 프로세스를 사용하여 데이터를 병렬로 로드
'''


# DataLoader로 배치 처리
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# DataLoader에서 미니배치 반복
'''
for batch in dataloader:
    print(batch)
'''

# 입력 데이터와 레이블
x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float32)

# TensorDataset으로 데이터셋 생성
dataset = TensorDataset(x, y)

# DataLoader로 배치 처리 설정
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 학습 루프
'''
for epoch in range(2):  # 2번 반복
    print(f"Epoch {epoch + 1}")
    for batch in dataloader:
        batch_x, batch_y = batch    # 다음과 같이 tensor를 같이 불러와서 사용할 때 편함
        print("Batch x:", batch_x)
        print("Batch y:", batch_y)

'''

x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  90],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)

    def forward(self,x):
        return self.linear(x)


model = Model()
optimizer = optim.SGD(model.parameters(), lr=1e-6)
nb_epochs = 2000

for epoch in range(nb_epochs+1):
    for sample in dataloader:

        x_train, y_train = sample

        prediction = model(x_train)

        loss = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch : {epoch} / {nb_epochs}, loss : {loss:.6f}")


# 데이터 셋 커스텀

class CustomDataset(torch.utils.data.Dataset):
  def __init__(self):
      pass
  # 데이터셋의 전처리를 해주는 부분

  def __len__(self):
      return 5
  # 데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분

  def __getitem__(self, idx):
      return 5
  # 데이터셋에서 특정 1개의 샘플을 가져오는 함수

'''
len(dataset)을 했을 때 데이터셋의 크기를 리턴할 len
dataset[i]을 했을 때 i번째 샘플을 가져오도록 하는 인덱싱을 위한 get_item
'''

# Dataset 상속
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70],]
    self.y_data = [[152], [185], [180], [196], [142]]

  # 총 데이터의 개수를 리턴
  def __len__(self):
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y

dataset1 = CustomDataset()

print(len(dataset1))
print(dataset1[2])