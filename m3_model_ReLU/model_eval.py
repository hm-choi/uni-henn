import numpy as np
import torch
from torchvision import datasets, transforms

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # L1 Image shape=(?, 28, 28, 1)
        #    Conv     -> (?, 26, 26, 6)
        #    Pool     -> (?, 13, 13, 6)
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.FC1 = torch.nn.Linear(1014, 120)
        self.FC2 = torch.nn.Linear(120, 10)

    def forward(self, x):
        x = self.Conv1(x)
        x = 0.117071*x**2 + 0.5*x + 0.375373
        x = self.AvgPool1(x)
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        x = 0.117071*x**2 + 0.5*x + 0.375373
        x = self.FC2(x)
        return x

model_cnn = CNN()
model_cnn.load_state_dict(torch.load('./M3_model.pth', map_location=torch.device('cpu')))

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
# print(model)

for i in range(5):
    test_accuracy = evaluate(model_cnn, test_loader)

    print("%dth Test Accuracy: %f" %(i+1, test_accuracy))

