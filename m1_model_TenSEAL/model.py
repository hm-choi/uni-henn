import torch
from torchvision import datasets, transforms
import numpy as np

from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
training_epochs = 15
batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./../Data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./../Data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print("train dataset:", train_dataset.data.shape)
print("test dataset :", test_dataset.data.shape)

class CNN(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(CNN, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=3, padding=0)
        self.FC1 = torch.nn.Linear(9 * 9 * 8, 64)
        self.FC2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.Conv1(x)
        x = x * x
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        x = x * x
        x = self.FC2(x)
        return x
    
def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        hypothesis = model(image)
        loss = criterion(hypothesis, label)
        loss.backward()
        optimizer.step()

        train_loss += loss / len(train_loader)

    return train_loss

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output, label)
            test_loss += loss / len(test_loader)

            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

# 모델 생성 및 손실 함수, 옵티마이저 설정
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# loss value list
loss_keeper = {'train':[], 'test':[]}

for epoch in range(training_epochs):
    train_loss = 0.0
    test_loss = 0.0

    '''
    Training phase
    '''
    train_loss = train(model, train_loader, optimizer)
    train_loss = train_loss.item()
    loss_keeper['train'].append(train_loss)
    
    '''
    Test phase
    '''
    test_loss, test_accuracy = evaluate(model, test_loader)
    test_loss = test_loss.item()
    loss_keeper['test'].append(test_loss)


    print("Epoch:%2d/%2d.. Training loss: %f.. Test loss: %f.. Test Accuracy: %f" 
          %(epoch + 1, training_epochs, train_loss, test_loss, test_accuracy))
    
train_loss_data = loss_keeper['train']
test_loss_data = loss_keeper['test']

plt.plot(train_loss_data, label = "Training loss")
plt.plot(test_loss_data, label = "Test loss")

plt.legend(), plt.grid(True)
plt.xlim(-2,training_epochs+3)
plt.show()

print(model)
torch.save(model.state_dict(), 'MNIST_test1.pth')
