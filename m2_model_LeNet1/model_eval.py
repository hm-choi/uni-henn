import numpy as np
import torch
from torchvision import datasets, transforms

class Square(torch.nn.Module):
    def __init__(self):
        super().__init__

    def forward(self, t):
        return torch.pow(t, 2)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0)
        self.Square1 = Square()
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.Conv2 = torch.nn.Conv2d(in_channels=4, out_channels=12, kernel_size=5, stride=1, padding=0)
        self.Square2 = Square()
        self.AvgPool2 = torch.nn.AvgPool2d(kernel_size=2)
        self.FC1 = torch.nn.Linear(192, 10)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Tanh(x)
        x = self.AvgPool1(x)
        x = self.Conv2(x)
        x = self.Tanh(x)
        x = self.AvgPool2(x)
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        return x

# model_cnn = CNN()
model_cnn = torch.load('./LeNet1_Approx.pt', map_location=torch.device('cpu'))
# model_cnn.load_state_dict(torch.load('./LeNet1_Approx.pt', map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

# MNIST 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(root='./../Data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)

            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy

print(model_cnn)

for i in range(5):
    # test_accuracy = test_net(model_cnn, device)
    test_accuracy = evaluate(model_cnn, test_loader)

    print("%dth Test Accuracy: %f" %(i+1, test_accuracy))

