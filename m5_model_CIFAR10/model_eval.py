import numpy as np
import torch
from torchvision import datasets, transforms

class CNN(torch.nn.Module):
    def __init__(self, output=10):
        super(CNN, self).__init__()
        # L1 Image shape=(?, 32, 32, 1)
        #    Conv     -> (?, 30, 30, 16)
        #    Pool     -> (?, 15, 15, 16)
        self.Conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size = 2)
        # L2 Image shape=(?, 15, 15, 16)
        #    Conv     -> (?, 12, 12, 64)
        #    Pool     -> (?, 6, 6, 64)
        self.Conv2 = torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=4, stride=1, padding=0)
        self.AvgPool2 = torch.nn.AvgPool2d(kernel_size = 2)
        # L2 Image shape=(?, 6, 6, 64)
        #    Conv     -> (?, 4, 4, 128)
        #    Pool     -> (?, 1, 1, 128)
        self.Conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.AvgPool3 = torch.nn.AvgPool2d(kernel_size = 4)
        self.FC1 = torch.nn.Linear(1*1*128, output)

    def forward(self, x):
        x = self.Conv1(x)
        x = x * x
        x = self.AvgPool1(x)
  
        x = self.Conv2(x)
        x = x * x
        x = self.AvgPool2(x)
        
        x = self.Conv3(x)
        x = x * x
        x = self.AvgPool3(x)
        
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        return x

model_cnn = CNN()
model_cnn.load_state_dict(torch.load('./CIFAR10_model1.pth', map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = datasets.CIFAR10(root='./../Data', train=False, transform=transform, download=True)
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

