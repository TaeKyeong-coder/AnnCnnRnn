import torch
from torch import nn
import numpy
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
 
train_data = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())

#torchvision에서 MNIST데이터셋 가져오기
#각 데이터 [이미지28x28 텐서, 레이블 값]
lt.imshow(train_data[0][0].reshape(28, 28), cmap='gist_yarg')

#이미지 확인
#plt.show()

#배치 크기를 정해주고 가져오기
batch_size = 100
 
train_batch = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_batch = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

class ModelANN(nn.Module):
    def __init__(self, size_in, size_out, size_hidden):
        super().__init__()
        self.hidden1 = nn.Linear(size_in, size_hidden[0])
        self.hidden2 = nn.Linear(size_hidden[0], size_hidden[1])
        self.out = nn.Linear(size_hidden[1], size_out)
 
    def forward(self, x):
        h1 = nn.functional.relu(self.hidden1(x))
        h2 = nn.functional.relu(self.hidden2(h1))
        y = nn.functional.log_softmax(self.out(h2), dim=1)
        return y

#모델 생성
model = ModelANN(784, 10, [200, 100])
 
learning_rate = 0.0001
epochs = 10
#어떤 손실값을 줄일지 정해주는 함수 entropy loss는 확률에 대한 손실을 줄임.
criterion = nn.CrossEntropyLoss()
#최적화를 할 때 가중치의 경로를 잡아주는?함수 같음.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#학습 및 테스트
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_result = 0
    test_result = 0
    for index, (x_train, y_train) in enumerate(train_batch):
        index += 1
        train_output = model(x_train.view(100, -1))
        answer = torch.max(train_output.data, 1)[1]
        batch_result = (answer == y_train).sum()
        train_result += batch_result
        train_loss = criterion(train_output, y_train)
        if index % 100 == 0:
            print('Loss of {}.{} : {} ({}%)'.format(epoch, index,
                                                    train_loss.item(), train_result.item() / index))
        train_loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for index, (x_test, y_test) in enumerate(test_batch):
            test_output = model(x_test.view(100, -1))
            answer = torch.max(test_output.data, 1)[1]
            test_result += (answer == y_test).sum()
        loss = criterion(test_output, y_test)
        print('test {} : {}%'.format(epoch, test_result.item() / 100))
