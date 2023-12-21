from uni_henn import *
from models.model_structures import * 

from seal import *
from torchvision import datasets, transforms
import numpy as np
import torch
import math

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
    
model = M1()
model = torch.load(current_dir + '/models/M1_model.pth', map_location=torch.device('cpu'))

print(model)
print('='*40)

for layer_name, layer in model.named_children():
    layer_params = getattr(model, layer_name)
    print(f"layer: {layer_params}")

    print(layer.__class__.__name__)

    if layer.__class__.__name__ == 'Conv2d':
        print(f"in_channels: {layer_params.in_channels}")
        print(f"out_channels: {layer_params.out_channels}")
        print(f"weight shape: {layer_params.weight.shape}")
        print(f"bias shape: {layer_params.bias.shape}")
        print(f"kernel: {layer_params.kernel_size}")
        print(f"stride: {layer_params.stride}")
        print(f"padding: {layer_params.padding}")
    elif layer.__class__.__name__ == 'AvgPool2d':
        print(f"kernel: {layer_params.kernel_size}")
        print(f"stride: {layer_params.stride}")
        print(f"padding: {layer_params.padding}")
    elif layer.__class__.__name__ == 'Linear':
        print(f"in_features: {layer_params.in_features}")
        print(f"out_features: {layer_params.out_features}")
        print(f"weight shape: {layer_params.weight.shape}")
        print(f"bias shape: {layer_params.bias.shape}")

    print(f"layer 전체: {layer}")  # 레이어의 전체 정보 출력
    print('='*40)

print()

if __name__ == "__main__":
    Img = Cuboid(1, 28, 28)