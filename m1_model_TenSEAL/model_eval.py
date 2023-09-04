import numpy as np
import torch
from torchvision import datasets, transforms

# 간단한 컨볼루션 신경망(CNN) 모델 정의
class CNN(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(CNN, self).__init__()
        # L1 Image shape=(?, 28, 28, 1)
        #    Conv     -> (?, 8, 8, 4)
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=7, stride=3, padding=0)
        self.FC1 = torch.nn.Linear(8 * 8 * 4, hidden)
        self.FC2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.Conv1(x)
        x = x * x
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        x = x * x
        x = self.FC2(x)
        return x

model_cnn = CNN()
model_cnn.load_state_dict(torch.load('./MNIST_test1.pth', map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

# MNIST 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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
print(model)

for i in range(5):
    test_accuracy = evaluate(model_cnn, test_loader)

    print("%dth Test Accuracy: %f" %(i+1, test_accuracy))

