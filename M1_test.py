from uni_henn import *
from uni_henn.he_cnn import HE_CNN
from models.model_structures import M1, Square, Flatten

from seal import *
from torchvision import datasets, transforms
import numpy as np
import torch
import time

import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    m1_model = M1()
    m1_model = torch.load(current_dir + '/models/M1_model.pth', map_location=torch.device('cpu'))

    MNIST_Img = Cuboid(1, 28, 28)
    context = Context()

    HE_m1 = HE_CNN(m1_model, MNIST_Img, context)
    print(HE_m1)
    print('='*40)

    num_of_data = int(NUMBER_OF_SLOTS // HE_m1.data_size)
   
    test_dataset = datasets.MNIST(
        root=current_dir + '/Data', 
        train=False, 
        transform=TRANSFORM,
        download=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_data, shuffle=True, drop_last=True)

    data, _label = next(iter(test_loader))
    data, _label = np.array(data), _label.tolist()
    
    new_data = []
    for i in range(num_of_data):
        new_data.extend(data[i].flatten())
        new_data.extend([0] * (HE_m1.data_size - MNIST_Img.size2d()))
    data = torch.Tensor(data)
    new_data = torch.Tensor(new_data)

    ciphertext_list = HE_m1.encrypt([new_data])
 
    result_ciphertext = HE_m1(ciphertext_list)

    result_plaintext = HE_m1.decrypt(result_ciphertext)

    for i in range(num_of_data):
        max_data_idx = -1
        dataList = m1_model(data)[i].flatten().tolist()
        max_data_idx = dataList.index(max(dataList))

        max_ctxt_idx = -1
        max_ctxt = -1e10
        sum = 0
        for j in range(10):
            ctxt_data = result_plaintext[j+HE_m1.data_size*i]

            sum = sum + np.abs(dataList[j] - ctxt_data)

            if(max_ctxt < ctxt_data):
                max_ctxt = ctxt_data
                max_ctxt_idx = j
        
        print(i+1, 'th result')
        print("Error          |", sum)
        print("original label |", max_data_idx)
        print("HE label       |", max_ctxt_idx)
        print("real label     |", _label[i])
        print("="*30)
