import numpy as np
import torch
from torchvision import datasets, transforms

class CNN(torch.nn.Module):
    def __init__(self, hidden=84, output=10):
        super(CNN, self).__init__()
        # L1 Image shape=(?, 32, 32, 1)
        #    Conv     -> (?, 28, 28, 6)
        #    Pool     -> (?, 14, 14, 6)
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size = 2)
        # L2 Image shape=(?, 14, 14, 6)
        #    Conv     -> (?, 10, 10, 16)
        #    Pool     -> (?, 5, 5, 16)
        self.Conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.AvgPool2 = torch.nn.AvgPool2d(kernel_size = 2)
        # L2 Image shape=(?, 5, 5, 16)
        #    Conv     -> (?, 1, 1, 120)
        self.Conv3 = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.FC1 = torch.nn.Linear(120, hidden)
        self.FC2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.Conv1(x)
        x = x * x
        x = self.AvgPool1(x)

        x = self.Conv2(x)
        x = x * x
        x = self.AvgPool2(x)
        
        x = self.Conv3(x)
        x = x * x
        
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        x = x * x
        x = self.FC2(x)
        return x

model_cnn = CNN()
model_cnn.load_state_dict(torch.load('./MNIST_LeNet5.pth', map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

# MNIST 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
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

print()

model = CNN().to(device)
# print(model)

for i in range(5):
    test_accuracy = evaluate(model_cnn, test_loader)

    print("%dth Test Accuracy: %f" %(i+1, test_accuracy))

