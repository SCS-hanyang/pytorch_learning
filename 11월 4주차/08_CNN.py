<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else "cpu"

torch.manual_seed(777)



learning_rate = 0.001
training_epoch = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

indices = list(range(len(mnist_train)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(mnist_train, train_indices)
val_dataset = torch.utils.data.Subset(mnist_train, val_indices)

train_data = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        drop_last=True)

val_data = DataLoader(dataset=val_dataset,
                      batch_size=batch_size,
                      drop_last=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_dim = (batch_size, 14, 14, 32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_dim = (batch_size, 7, 7, 64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output_dim = (batch_size, 4, 4, 128)
        )

        self.fc1 = nn.Linear(in_features=4 * 4 * 128, out_features=625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)  # 어떤 네트워크를 사용하고, input과 ooutput의 형태에 따라 최적의 초기화 방법이 다르다
        # nn.Linear의 가중치 초기화 방식의 디폴트는 He 초기화
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob)
        )

        self.fc2 = nn.Linear(in_features=625, out_features=10, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)  # uniform과 uniform_의 차이 : 전자는 새로운 tensor생성 후자는 기존의 값 변경(후자가 보편적)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out


model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(train_data)

for epoch in range(training_epoch):
    avg_loss = 0

    with tqdm(train_data, desc=f"Epoch {epoch + 1}/{training_epoch}", unit="batch") as pbar:
        for inputs, labels in pbar:

            output = model(inputs)

            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            avg_loss += loss / total_batch

    with torch.no_grad():
        total_correct = 0
        total_samples = 0

        for sample, answer in val_data:
            sample = sample
            answer = answer  # 추가
            prediction = model(sample)
            prediction = torch.argmax(prediction, dim=1)

            total_correct += (prediction == answer).sum().item()
            total_samples += len(answer)

        accuracy = total_correct / total_samples * 100

    print(f'epoch : {epoch + 1} / {training_epoch}, loss : {avg_loss}, accuracy : {accuracy}%')
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else "cpu"

torch.manual_seed(777)



learning_rate = 0.001
training_epoch = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

indices = list(range(len(mnist_train)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(mnist_train, train_indices)
val_dataset = torch.utils.data.Subset(mnist_train, val_indices)

train_data = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        drop_last=True)

val_data = DataLoader(dataset=val_dataset,
                      batch_size=batch_size,
                      drop_last=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_dim = (batch_size, 14, 14, 32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_dim = (batch_size, 7, 7, 64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # output_dim = (batch_size, 4, 4, 128)
        )

        self.fc1 = nn.Linear(in_features=4 * 4 * 128, out_features=625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)  # 어떤 네트워크를 사용하고, input과 ooutput의 형태에 따라 최적의 초기화 방법이 다르다
        # nn.Linear의 가중치 초기화 방식의 디폴트는 He 초기화
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob)
        )

        self.fc2 = nn.Linear(in_features=625, out_features=10, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)  # uniform과 uniform_의 차이 : 전자는 새로운 tensor생성 후자는 기존의 값 변경(후자가 보편적)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out


model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(train_data)

for epoch in range(training_epoch):
    avg_loss = 0

    with tqdm(train_data, desc=f"Epoch {epoch + 1}/{training_epoch}", unit="batch") as pbar:
        for inputs, labels in pbar:

            output = model(inputs)

            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            avg_loss += loss / total_batch

    with torch.no_grad():
        total_correct = 0
        total_samples = 0

        for sample, answer in val_data:
            sample = sample
            answer = answer  # 추가
            prediction = model(sample)
            prediction = torch.argmax(prediction, dim=1)

            total_correct += (prediction == answer).sum().item()
            total_samples += len(answer)

        accuracy = total_correct / total_samples * 100

    print(f'epoch : {epoch + 1} / {training_epoch}, loss : {avg_loss}, accuracy : {accuracy}%')
>>>>>>> 95b60bdff058f123cb273be7163370ffb90b989b
