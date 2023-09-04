import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # L1 Image shape=(?, 16, 16, 1)
        #    Conv     -> (?, 14, 14, 6)
        #    Pool     -> (?, 7, 7, 6)
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.FC1 = torch.nn.Linear(294, 64)
        self.FC2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.Conv1(x)
        x = x * x
        x = self.AvgPool1(x)
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        x = x * x
        x = self.FC2(x)
        return x

model_cnn = CNN()
model_cnn.load_state_dict(torch.load('./CNN_USPS.pth', map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.USPS(root='./../Data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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


model = CNN().to(device)
# print(model)

for i in range(5):
    test_accuracy = evaluate(model_cnn, test_loader)

    print("%dth Test Accuracy: %f" %(i+1, test_accuracy))

